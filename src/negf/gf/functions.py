from __future__ import annotations
import multiprocessing as mp
import numpy as np
import scipy.sparse as sp
import scipy.constants as spc
from typing import List, Tuple, Iterable, Callable, Any
from contextlib import contextmanager


from ..utils.ozaki import (
    get_ozaki_poles_residues,
    fermi_cfr,
    fermi_derivative_cfr_abs_batch,
)
from .recursive_greens_functions import recursive_gf
from hamiltonian.base.block_tridiagonalization import split_into_subblocks_optimized
from hamiltonian.base.hamiltonian_core import Hamiltonian
from negf.gf.recursive_greens_functions import gf_inverse
from negf.utils.block_partition import compute_optimal_block_sizes
from negf.utils.common import chandrupatla

class GFFunctions:
    def __init__(self, ham : Hamiltonian, energy_grid: np.ndarray, k_space: np.ndarray | None = None,
                 self_energy_method: str = "sancho_rubio", use_variable_blocks: bool = True):
        self.ham = ham
        self.energy_grid = np.atleast_1d(energy_grid)
        self.dE = (self.energy_grid[1] - self.energy_grid[0]) if self.energy_grid.size > 1 else 1.0
        self.k_space = np.atleast_1d(k_space) if k_space is not None else np.array([0])
        self.self_energy_method = self_energy_method

        # Caching and execution controls
        self.LDOS_cache = {}
        self._force_serial = False

        ham_new, hL0, hLC, hR0, hRC, h_periodic = self.ham.get_hamiltonians()
        self.ham_device = ham_new
        self.H00 = hL0
        self.H01 = hLC
        self.h_k_lead = h_periodic
        self.h_k_device = sp.block_diag([h_periodic] * (int)(ham_new.shape[0] / self.H00.shape[0]), format='csc')
        self.block_size = self.H00.shape[0]
        if use_variable_blocks is None:
            use_variable_blocks = bool(getattr(self.ham, "enable_variable_blocks", False))
        self.use_variable_blocks: bool = bool(use_variable_blocks)
        self.block_size_list: np.ndarray | None = None
        self.use_variable_blocks: bool = getattr(self.ham, "enable_variable_blocks", False)
        self.block_size_list: np.ndarray | None = None
        if self.use_variable_blocks:
            self.block_size_list = compute_optimal_block_sizes(self.ham_device, self.H01)
        if (self.ham.periodic_dirs == None or self.ham.periodic_dirs == "x"):
            self.transverse_periodic = False
        else:
            self.transverse_periodic = True
            
        self.kbT_eV = spc.Boltzmann / spc.elementary_charge * 300

        # Atom/orbital mapping helpers
        # Offsets come from BasisTB: cumulative start indices per atom into the orbital DOFs
        # Build robust offsets from per-atom orbital counts (length n_atoms+1)
        labels = list(self.ham.atom_list.keys())
        counts = [self.ham.orbitals_dict[label].num_of_orbitals for label in labels]
        self.n_atoms: int = len(labels)
        self.atom_offsets = np.concatenate(([0], np.cumsum(np.asarray(counts, dtype=int))))
        # Determine number of orbitals from device size; adjust last offset if mismatch
        self.n_orbitals: int = int(self.ham_device.shape[0])
        if int(self.atom_offsets[-1]) != self.n_orbitals:
            # Adjust last atom count to match total DOFs; prevents OOB and aligns mapping
            delta = self.n_orbitals - int(self.atom_offsets[-1])
            self.atom_offsets[-1] = self.n_orbitals
        # Precompute orbital->atom index map
        self._orb2atom = np.empty(self.n_orbitals, dtype=int)
        for a in range(self.n_atoms):
            s, e = int(self.atom_offsets[a]), int(self.atom_offsets[a+1])
            if s >= e:
                continue
            self._orb2atom[s:e] = a

        if self.use_variable_blocks:
            self.block_size_list = compute_optimal_block_sizes(
                self.ham_device,
                self.H01,
                atom_offsets=self.atom_offsets,
            )

    def _aggregate_orbital_to_atom(self, vec_orb: np.ndarray) -> np.ndarray:
        """Sum orbital-resolved vector into atom-resolved vector.

        Parameters
        ----------
        vec_orb : (n_orbitals,) array_like
            Orbital-diagonal quantity (e.g., diag(G), density per orbital).

        Returns
        -------
        (n_atoms,) ndarray
            Sum over each atom's orbitals.
        """
        v = np.asarray(vec_orb)
        out = np.zeros(self.n_atoms, dtype=v.dtype)
        # Fast path with bincount for numeric dtypes
        try:
            out = np.bincount(self._orb2atom, weights=v, minlength=self.n_atoms)
            return out
        except Exception:
            pass
        for a in range(self.n_atoms):
            s, e = int(self.atom_offsets[a]), int(self.atom_offsets[a+1])
            if s < e:
                out[a] = np.sum(v[s:e])
        return out

    def _expand_atom_to_orbital(self, vec_atom: np.ndarray) -> np.ndarray:
        """Expand an atom-wise vector to orbitals by repeating values across that atom's orbitals."""
        v = np.atleast_1d(vec_atom)
        if v.size == self.n_orbitals:
            return v
        if v.size != self.n_atoms:
            # Best-effort: broadcast scalar
            if v.size == 1:
                return np.full(self.n_orbitals, float(v[0]), dtype=float)
            # Trim or pad
            v = v[:self.n_atoms]
        out = np.empty(self.n_orbitals, dtype=float)
        for a in range(self.n_atoms):
            s, e = int(self.atom_offsets[a]), int(self.atom_offsets[a+1])
            if s < e:
                out[s:e] = v[a]
        return out

    def _call_gf_inverse(self, E, ham_mat, H00_mat, H01_mat=None, *, processes: int = 1):
        H01_use = H01_mat if H01_mat is not None else self.H01
        block_size = H01_use.shape[0]
        return _gf_inverse_dispatch(
            E,
            ham_mat,
            H00_mat,
            H01_use,
            block_size=block_size,
            block_size_list=self.block_size_list,
            use_variable_blocks=self.use_variable_blocks,
            processes=processes,
        )
        
    
    def _ldos_key(self, E, ky) -> tuple[float, float]:
        return (round(float(np.real(E)), 12), round(float(ky), 12))

    def clear_ldos_cache(self):
        """Clear the LDOS cache (works for plain dict or Manager proxy)."""
        try:
            self.LDOS_cache.clear()
        except Exception:
            # Fall back to replacing with a new dict
            self.LDOS_cache = {}

    @contextmanager
    def serial_mode(self):
        """Context manager to force serial execution within the block."""
        old = self._force_serial
        self._force_serial = True
        try:
            yield
        finally:
            self._force_serial = old

    def _ensure_shared_cache(self, processes: int | None):
        """Promote LDOS_cache to a multiprocessing.Manager dict during parallel sections.

        Returns the Manager instance if promotion occurred; caller must finalize with
        _finalize_shared_cache(manager) after the parallel work.
        """
        if processes is not None and processes > 1 and not hasattr(self.LDOS_cache, '_callmethod'):
            manager = mp.Manager()
            shared = manager.dict()
            try:
                for k, v in self.LDOS_cache.items():
                    shared[k] = v
            except Exception:
                pass
            self.LDOS_cache = shared
            return manager
        return None

    def _finalize_shared_cache(self, manager):
        """Copy proxy cache back to a plain dict and shutdown the manager."""
        if manager is None:
            return
        try:
            self.LDOS_cache = dict(self.LDOS_cache)
        finally:
            try:
                manager.shutdown()
            except Exception:
                pass


    def _get_ldos_cached(self, E, ky =0) -> np.ndarray:
        key = self._ldos_key(E, ky)
        try:
            if key in self.LDOS_cache:
                return np.asarray(self.LDOS_cache[key])
        except Exception:
            # If cache proxy not behaving like a normal mapping, ignore and compute
            pass
        ham_device = self.ham_device + self.h_k_device * np.exp(1j * ky) + self.h_k_device.T * np.exp(-1j * ky)
        H00 = self.H00 + self.h_k_lead * np.exp(1j * ky) + self.h_k_lead.T * np.exp(-1j * ky)
        G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R = self._call_gf_inverse(E, ham_device, H00, self.H01)
        ldos_vec = -1.0 / np.pi * np.imag(G_R_diag)
        try:
            self.LDOS_cache[key] = ldos_vec
        except Exception:
            # If cache is not writable (e.g., during shutdown), skip caching
            pass
        return ldos_vec

    
    
    
    def dos(self, E, ky=0) -> np.ndarray:
        return self._get_ldos_cached(E, ky)

    def total_dos(self, processes: int = 1) -> np.ndarray:
        """Compute total DOS(E) = sum_sites LDOS(E, site) averaged over k.

        Parallel strategy: partition the energy grid contiguously across workers
        (process 0 gets the first block, etc.). Each worker returns its DOS slice
        which is concatenated preserving order.
        """
        E_grid = self.energy_grid
        nE = E_grid.size
        norm_k = max(1, self.k_space.size)
        if getattr(self, '_force_serial', False):
            processes = 1
        if processes <= 1 or nE == 1:
            out = np.zeros(nE, dtype=float)
            for i, E in enumerate(E_grid):
                acc = 0.0
                for ky in self.k_space:
                    acc += float(np.sum(self._get_ldos_cached(E, ky)))
                out[i] = acc / norm_k
            return out
        P = min(processes, nE)
        # Contiguous partition indices
        bounds = np.linspace(0, nE, P + 1, dtype=int)
        payloads = []
        for p in range(P):
            sl = slice(bounds[p], bounds[p+1])
            if sl.start == sl.stop:
                continue
            # Pass shared cache proxy to worker if available
            cache_ref = self.LDOS_cache if hasattr(self.LDOS_cache, '_callmethod') else None
            payloads.append((E_grid[sl], self.ham_device, self.H00, self.H01, self.k_space, norm_k,
                              self.block_size_list, self.use_variable_blocks, cache_ref))
        _mgr = self._ensure_shared_cache(P)
        try:
            ctx = mp.get_context('fork') if hasattr(mp, 'get_context') else mp
            with ctx.Pool(processes=P) as pool:
                parts = pool.map(_dos_slice_worker_static, payloads)
        except Exception as exc:
            print(f"[total_dos] Parallel fallback to serial due to: {exc}")
            self._finalize_shared_cache(_mgr)
            return self.total_dos(processes=1)
        finally:
            self._finalize_shared_cache(_mgr)
        return np.concatenate(parts)

    def fermi(self, E, V_vec: np.ndarray, Efn_vec: np.ndarray, cutoff=60) -> np.ndarray:
        poles, residues = get_ozaki_poles_residues(cutoff, self.kbT_eV, getattr(self.ham, 'T', 300))
        return fermi_cfr(np.atleast_1d(E), None, poles, residues, self.kbT_eV, V_vec=V_vec, Efn_vec=Efn_vec)[0]

    def get_n(self, V: np.ndarray, Efn: np.ndarray, *, processes: int = 1,
              ky_avg: bool = True, conduction_only: bool = True,
              Ec: float | np.ndarray = 0.0, ozaki_cutoff: int = 60,
              force_recompute_ozaki: bool = False) -> np.ndarray:
        """Carrier density per site using ONLY Ozaki CFR expansion.

        Strategy: For each k-point we (optionally in parallel) build LDOS(E,site)
        on the existing uniform ``energy_grid`` (using cache), then evaluate the
        Ozaki rational approximation of Fermi-Dirac simultaneously for all sites.

        Parameters
        ----------
        V : array_like (n_sites,)
            Electrostatic potential per site (eV).
        Efn : array_like (n_sites,)
            Quasi-Fermi level per site (eV).
        processes : int
            Number of k-point workers (>= number of k-points is usually overkill).
        ky_avg : bool
            Average over k-points (True) or return sum over k (False).
        conduction_only : bool
            If True, only energies >= Ec (per site or scalar) contribute.
        Ec : float or array_like
            Conduction band edge (scalar or per-site).
        ozaki_cutoff : int
            Number of poles/residues for Ozaki CFR.
        force_recompute_ozaki : bool
            Clear cached poles/residues.
        """
        V = np.atleast_1d(V).astype(float)
        Efn = np.atleast_1d(Efn).astype(float)
        # Expand atom-wise inputs to orbital-wise if needed
        V_orb = self._expand_atom_to_orbital(V)
        Efn_orb = self._expand_atom_to_orbital(Efn)
        n_orb = V_orb.size
        Ec_arr = np.atleast_1d(Ec).astype(float)
        if Ec_arr.size == 1:
            Ec_arr = np.full(n_orb, float(Ec_arr[0]))
        elif Ec_arr.size != n_orb:
            # Accept either atom-sized or orbital-sized arrays and expand if needed
            if Ec_arr.size == self.n_atoms:
                Ec_arr = self._expand_atom_to_orbital(Ec_arr)
            else:
                Ec_arr = Ec_arr[:n_orb]

        if force_recompute_ozaki:
            try:
                get_ozaki_poles_residues.cache_clear()
            except Exception:
                pass
        poles, residues = get_ozaki_poles_residues(ozaki_cutoff, self.kbT_eV, getattr(self.ham,'T',300))

        E_grid = self.energy_grid
        dE = self.dE
        k_points = self.k_space if self.k_space.size > 0 else np.array([0])

        if getattr(self, '_force_serial', False):
            processes = 1
        if processes <= 1 or k_points.size == 1:
            densities = []
            for ky in k_points:
                densities.append(GFFunctions._density_k_worker_static((float(ky), E_grid, self.ham_device, self.H00, self.H01,
                                                           poles, residues, V_orb, Efn_orb, conduction_only, Ec_arr, dE,
                                                           self.H00.shape[0], self.block_size_list, self.use_variable_blocks, None)))
        else:
            cache_ref = self.LDOS_cache if hasattr(self.LDOS_cache, '_callmethod') else None
            payloads = [(float(ky), E_grid, self.ham_device, self.H00, self.H01,
                         poles, residues, V_orb, Efn_orb, conduction_only, Ec_arr, dE, self.H00.shape[0],
                         self.block_size_list, self.use_variable_blocks, cache_ref)
                        for ky in k_points]
            _mgr = self._ensure_shared_cache(processes)
            try:
                ctx = mp.get_context('fork') if hasattr(mp, 'get_context') else mp
                with ctx.Pool(processes=min(processes, k_points.size)) as pool:
                    densities = pool.map(GFFunctions._density_k_worker_static, payloads)
            except Exception as exc:
                print(f"[get_n] Parallel fallback to serial due to: {exc}")
                densities = []
                for ky in k_points:
                    densities.append(GFFunctions._density_k_worker_static((float(ky), E_grid, self.ham_device, self.H00, self.H01,
                                                               poles, residues, V_orb, Efn_orb, conduction_only, Ec_arr, dE,
                                                               self.H00.shape[0], self.block_size_list, self.use_variable_blocks, None)))
            finally:
                self._finalize_shared_cache(_mgr)

        dens_stack = np.vstack(densities)  # shape: (nk, n_orb)
        orb_density = np.mean(dens_stack, axis=0) if ky_avg else np.sum(dens_stack, axis=0)
        # Aggregate to atoms
        return self._aggregate_orbital_to_atom(orb_density)

    def diff_rho_poisson(self, V: np.ndarray, Efn: np.ndarray, Ec: float | None = None,
                          ozaki_cutoff: int = 60) -> np.ndarray:
        V = np.atleast_1d(V)
        Efn = np.atleast_1d(Efn)
        n_sites = V.size
        poles, residues = get_ozaki_poles_residues(ozaki_cutoff, self.kbT_eV, getattr(self.ham, 'T', 300))
        deriv = np.zeros(n_sites)
        for E in self.energy_grid:
            ldos_acc = np.zeros(n_sites)
            for ky in self.k_space:
                ldos_acc += self._get_ldos_cached(E, ky)
            ldos_acc /= max(1, self.k_space.size)
            dfdx_vec = fermi_derivative_cfr_abs_batch(np.array([E]), V, Efn, poles, residues, self.kbT_eV)[0]
            deriv += ldos_acc * (dfdx_vec / self.kbT_eV) * self.dE
        return deriv

    def transmission_worker(self, E):
        """Serial transmission helper using direct inversion."""
        G_R, _, Gamma_L, Gamma_R = gf_inverse(E, self.ham_device, self.H00, self.H01, method="direct")
        G_A = sp.csc_matrix(G_R.T.conj())
        G_R = sp.csc_matrix(G_R)
        return float(np.real(np.trace(Gamma_L @ G_R @ Gamma_R @ G_A)))

    def transmission(self, processes: int = 1, average_over_k: bool = True) -> np.ndarray:
        """Compute transmission T(E) averaged over k (if ``average_over_k``).

        Currently the internal ``transmission_worker`` ignores ky because the
        Hamiltonian supplied to ``gf_inverse`` is already the device (possibly
        k-parametrized externally). If k-dependence is later added, extend
        the worker to accept (E,ky).
        """
        E_grid = self.energy_grid
        # Serial fast path
        if getattr(self, '_force_serial', False):
            processes = 1
        if processes <= 1:
            return np.array([self.transmission_worker(E) for E in E_grid], dtype=float)

        # Parallel path: build lightweight payload without embedding self (which is not picklable due to lambdas in Hamiltonian)
        payloads = [(float(E), self.ham_device, self.H00, self.H01) for E in E_grid]
        _mgr = self._ensure_shared_cache(processes)
        try:
            # Prefer 'fork' when available to reduce serialization overhead
            ctx = mp.get_context('fork') if hasattr(mp, 'get_context') else mp
            with ctx.Pool(processes=processes) as pool:
                vals = pool.map(_transmission_worker_static, payloads)
            return np.asarray(vals, dtype=float)
        except Exception as exc:
            print(f"[transmission] Parallel fallback to serial due to: {exc}")
            return np.array([self.transmission_worker(E) for E in E_grid], dtype=float)
        finally:
            self._finalize_shared_cache(_mgr)
        
    def current_worker(self, E: float) -> np.ndarray:
        _, _, G_lesser_offdiag_right, _, _ = self._call_gf_inverse(
            E,
            self.ham_device,
            self.H00,
            self.H01,
        )
        if self.use_variable_blocks and self.block_size_list is not None:
            offsets = np.cumsum(np.concatenate(([0], self.block_size_list.astype(int))))
        else:
            step = self.block_size
            offsets = np.arange(0, self.ham_device.shape[0] + 1, step)
        num_blocks = offsets.size - 1
        bond_current_integrands = []
        for i in range(num_blocks - 1):
            start_row = int(offsets[i])
            end_row = int(offsets[i + 1])
            start_col = int(offsets[i + 1])
            end_col = int(offsets[i + 2])
            H_ij = self.ham_device[start_row:end_row, start_col:end_col]
            G_lesser_ji = G_lesser_offdiag_right[i]
            product_matrix = H_ij @ G_lesser_ji
            trace_val = np.trace(product_matrix)
            bond_current_integrands.append(np.imag(trace_val))
            
        return np.array(bond_current_integrands)

    def compute_charge_density(self, *, method: str = "lesser", processes: int = 1) -> np.ndarray:
        """Compute electron number per site via integration over energy.

        When method == 'lesser', integrates using the diagonal of G^< returned by
        the recursive Green's function, partitioning the energy grid into contiguous
        blocks (like total_dos) for parallel execution.

        Returns
        -------
        np.ndarray
            Density vector (n_sites,), real part.
        """
        if method != "lesser":
            raise ValueError("Only method='lesser' is currently supported in compute_charge_density().")

        if getattr(self, '_force_serial', False):
            processes = 1

        E_grid = self.energy_grid
        nE = E_grid.size
        n_orb = self.ham_device.shape[0]
        norm_k = max(1, self.k_space.size)
        block_size = self.H00.shape[0]
        dE = self.dE

        # Serial fast path
        if processes <= 1 or nE == 1:
            density_orb = np.zeros(n_orb, dtype=float)
            for E in E_grid:
                accum_k = np.zeros(n_orb, dtype=complex)
                for ky in self.k_space:
                    ham_k = self.ham_device + self.h_k_device * np.exp(1j * ky) + self.h_k_device.T * np.exp(-1j * ky)
                    H00_k = self.H00 + self.h_k_lead * np.exp(1j * ky) + self.h_k_lead.T * np.exp(-1j * ky)
                    # Use recursive method to obtain G^< diagonal
                    _, G_lesser_diag, _, _, _ = self._call_gf_inverse(E, ham_k, H00_k, self.H01)
                    G_n_diag = -1j * G_lesser_diag
                    accum_k += (dE * G_n_diag) / (2.0 * np.pi)
                density_orb += np.real(accum_k) / norm_k
            # Aggregate orbitals -> atoms
            return self._aggregate_orbital_to_atom(density_orb)

        # Parallel path: partition energy grid contiguously
        P = min(processes, nE)
        bounds = np.linspace(0, nE, P + 1, dtype=int)
        payloads = []
        for p in range(P):
            sl = slice(bounds[p], bounds[p+1])
            if sl.start == sl.stop:
                continue
            payloads.append((E_grid[sl], self.ham_device, self.H00, self.H01, self.h_k_device, self.h_k_lead,
                             self.k_space, block_size, self.block_size_list, self.use_variable_blocks, dE))

        try:
            ctx = mp.get_context('fork') if hasattr(mp, 'get_context') else mp
            with ctx.Pool(processes=P) as pool:
                parts = pool.map(_density_lesser_slice_worker_static, payloads)
        except Exception as exc:
            print(f"[compute_charge_density] Parallel fallback to serial due to: {exc}")
            return self.compute_charge_density(method=method, processes=1)

        # Sum contributions from slices and average over k-points
        total_orb = np.sum(np.vstack(parts), axis=0) / norm_k
        return self._aggregate_orbital_to_atom(total_orb)

 
    def _density_k_worker_static(payload: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                bool, np.ndarray, float, int, Any | None]) -> np.ndarray:
        """Compute carrier density contribution for a single k-point using Ozaki CFR.

        Returns density vector (n_sites,).
        """
        (ky, E_grid, ham_device, H00, H01, poles, residues, V, Efn,
        conduction_only, Ec_arr, dE, block_size, block_size_list, use_var_blocks, cache_ref) = payload
        n_sites = V.size
        nE = E_grid.size
        ldos_mat = np.zeros((nE, n_sites), dtype=float)
        for i, E in enumerate(E_grid):
            ldos_vec = None
            if cache_ref is not None:
                key = (round(float(np.real(E)), 12), round(float(ky), 12))
                try:
                    if key in cache_ref:
                        ldos_vec = np.asarray(cache_ref[key])
                except Exception:
                    ldos_vec = None
            if ldos_vec is None:
                G_R_diag, _, _, _, _ = _gf_inverse_dispatch(
                    E,
                    ham_device,
                    H00,
                    H01,
                    block_size=block_size,
                    block_size_list=block_size_list,
                    use_variable_blocks=use_var_blocks,
                )
                ldos_vec = -1/np.pi * np.imag(G_R_diag)
                try:
                    if cache_ref is not None:
                        cache_ref[key] = ldos_vec
                except Exception:
                    pass
            ldos_mat[i, :] = ldos_vec
        f_mat = fermi_cfr(E_grid, None, poles, residues, spc.Boltzmann / spc.elementary_charge * 300, V_vec=V, Efn_vec=Efn)
        # Note: original fermi_cfr signature used self.kbT_eV; we rely on poles/residues already built for correct kT.
        if conduction_only:
            mask = (E_grid[:, None] >= Ec_arr[None, :])
            f_mat = f_mat * mask
        return np.sum(ldos_mat * f_mat, axis=0) * dE


    def fermi_level(self, V: np.ndarray, lower_bound=None, upper_bound=None, Ec=0, verbose=False,
                     f_tol=None,processes = 1,  allow_unbracketed=True,
                     plateau_f_tol=1e-8, auto_shrink_gap=True):
        """Alias to fermi_level with optional Ec and auto_shrink_gap handling similar to test_systems.rgf.

        Note: This method computes target charge density using G^< integration and then solves for Efn such that
        Ozaki-based carrier density equals the target. Inputs V and Ec may be atom-wise or orbital-wise; both are supported.
        """
        V = np.atleast_1d(V).astype(float)
        if not isinstance(lower_bound, np.ndarray):
            lower_bound = np.full(self.n_atoms, -1.0, dtype=float) if V.size != self.n_atoms else np.full_like(V, -1.0)
        if not isinstance(upper_bound, np.ndarray):
            upper_bound = np.full(self.n_atoms, 2.0, dtype=float) if V.size != self.n_atoms else np.full_like(V, 2.0)


        with self.serial_mode():
            target_density = self.compute_charge_density(processes=processes)

        # Optional gap-plateau early return: if DOS ~ 0 in window, Efn ~ mid
        if auto_shrink_gap:
            with self.serial_mode():
                dos_vec = self.total_dos(processes=processes)
            E_min = float(np.min(lower_bound))
            E_max = float(np.max(upper_bound))
            in_window = (self.energy_grid >= E_min) & (self.energy_grid <= E_max)
            if np.any(in_window) and np.max(dos_vec[in_window]) < 1e-12:
                return 0.5 * (lower_bound + upper_bound)

        def func(x):
            with self.serial_mode():
                return self.get_n(V=V, Efn=x, conduction_only=False, processes=processes) - target_density

        roots = chandrupatla(func, lower_bound, upper_bound, verbose=verbose,
                              allow_unbracketed=allow_unbracketed, f_tol=f_tol,
                              plateau_f_tol=plateau_f_tol)
        return roots
            

def _transmission_worker_static(payload: tuple[float, np.ndarray, np.ndarray, np.ndarray]) -> float:
    """Top-level picklable worker for transmission(E).

    Parameters
    ----------
    payload : tuple
        (E, ham_device, H00, H01)
    Returns
    -------
    float
        Transmission value at energy E.
    """
    E, ham_device, H00, H01 = payload
    G_R, _, Gamma_L, Gamma_R = gf_inverse(E, ham_device, H00, H01, method="direct")
    G_A = sp.csc_matrix(G_R.T.conj())
    G_R = sp.csc_matrix(G_R)
    return float(np.real(np.trace(Gamma_L @ G_R @ Gamma_R @ G_A)))


def _dos_slice_worker_static(payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray | None, bool, Any | None]) -> np.ndarray:
    """Worker computing DOS for a contiguous slice of energies.

    Parameters
    ----------
    payload : tuple
        (E_slice, ham_device, H00, H01, k_space, norm_k, cache_ref)
    """
    E_slice, ham_device, H00, H01, k_space, norm_k, block_size_list, use_var_blocks, cache_ref = payload
    out = np.zeros(E_slice.size, dtype=float)
    for i, E in enumerate(E_slice):
        acc = 0.0
        for ky in k_space:
            # LDOS computation with optional shared cache across processes
            ldos_vec = None
            if cache_ref is not None:
                key = (round(float(np.real(E)), 12), round(float(ky), 12))
                try:
                    if key in cache_ref:
                        ldos_vec = np.asarray(cache_ref[key])
                except Exception:
                    ldos_vec = None
            if ldos_vec is None:
                G_R_diag, _, _, _, _ = _gf_inverse_dispatch(
                    E,
                    ham_device,
                    H00,
                    H01,
                    block_size=H00.shape[0],
                    block_size_list=block_size_list,
                    use_variable_blocks=use_var_blocks,
                )
                ldos_vec = -1/np.pi * np.imag(G_R_diag)
                try:
                    if cache_ref is not None:
                        cache_ref[key] = ldos_vec
                except Exception:
                    pass
            acc += float(np.sum(ldos_vec))
        out[i] = acc / norm_k
    return out


def _density_lesser_slice_worker_static(payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray, np.ndarray, int,
                                                       np.ndarray | None, bool, float]) -> np.ndarray:
    """Worker that integrates density over a contiguous slice of energies using G^<.

    Parameters
    ----------
    payload : tuple
        (E_slice, ham_device, H00, H01, h_k_device, h_k_lead, k_space, block_size, dE)

    Returns
    -------
    np.ndarray
        Partial density vector (n_sites,) integrated over the slice; caller averages over k.
    """
    (E_slice, ham_device, H00, H01, h_k_device, h_k_lead, k_space, block_size,
     block_size_list, use_var_blocks, dE) = payload
    n_sites = ham_device.shape[0]
    out = np.zeros(n_sites, dtype=float)
    for E in E_slice:
        accum_k = np.zeros(n_sites, dtype=complex)
        for ky in k_space:
            ham_k = ham_device + h_k_device * np.exp(1j * ky) + h_k_device.T * np.exp(-1j * ky)
            H00_k = H00 + h_k_lead * np.exp(1j * ky) + h_k_lead.T * np.exp(-1j * ky)
            _, G_lesser_diag, _, _, _ = _gf_inverse_dispatch(
                E,
                ham_k,
                H00_k,
                H01,
                block_size=block_size,
                block_size_list=block_size_list,
                use_variable_blocks=use_var_blocks,
            )
            G_n_diag = -1j * G_lesser_diag
            accum_k += (dE * G_n_diag) / (2.0 * np.pi)
        out += np.real(accum_k)
    return out


def _gf_inverse_dispatch(E, ham_device, H00, H01, *, block_size: int, block_size_list: np.ndarray | None,
                         use_variable_blocks: bool, processes: int = 1):
    if use_variable_blocks and block_size_list is not None:
        return gf_inverse(
            E,
            ham_device,
            H00,
            H01,
            block_size_list=block_size_list,
            method="var_recursive",
            processes=processes,
        )
    return gf_inverse(
        E,
        ham_device,
        H00,
        H01,
        block_size=block_size,
        method="recursive",
    )

