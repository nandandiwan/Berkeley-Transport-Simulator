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
from negf.self_energy.greens_functions import surface_greens_function_nn
from negf.utils.block_partition import (
    compute_block_sizes_block_tridiagonal,
    compute_block_sizes_metis,
)
from negf.utils.common import chandrupatla

class GFFunctions:
    def __init__(self, ham : Hamiltonian, energy_grid: np.ndarray, k_space: np.ndarray | None = None,
                 self_energy_method: str = "sancho_rubio", use_variable_blocks: bool = True,
                 inverse_method: str = "auto"):
        self.ham = ham
        self.energy_grid = np.atleast_1d(energy_grid)
        self.dE = (self.energy_grid[1] - self.energy_grid[0]) if self.energy_grid.size > 1 else 1.0
        self.k_space = np.atleast_1d(k_space) if k_space is not None else np.array([0])
        self.self_energy_method = self_energy_method

        # Caching and execution controls
        self.LDOS_cache = {}
        self._force_serial = False

        ham_new, hL0, hLC, hR0, hRC, h_periodic = self.ham.get_hamiltonians()
        self.ham_device = np.array(ham_new, copy=True, dtype=complex)
        self._ham_device_base = np.array(self.ham_device, copy=True)
        self._ham_diag_idx = np.diag_indices(self.ham_device.shape[0])
        self.H00 = hL0
        self.H00_right = hR0 if hR0 is not None else hL0
        self.H01 = hLC
        if h_periodic is not None and h_periodic.shape[0] > 0:
            self.h_k_lead = h_periodic
            tiles = int(ham_new.shape[0] / self.H00.shape[0]) if self.H00.shape[0] > 0 else 0
            if tiles > 0:
                self.h_k_device = sp.block_diag([h_periodic] * tiles, format='csc')
            else:
                self.h_k_device = None
        else:
            self.h_k_lead = None
            self.h_k_device = None
        self.block_size = self.H00.shape[0]
        
        self.use_variable_blocks = use_variable_blocks
        self.inverse_method = inverse_method.lower()
        self.mu_left = 0.0
        self.mu_right = 0.0
        self.block_size_list: np.ndarray | None = None
        self.occupation_mode: str = "fermi_dirac"
        self.occupation_prefactor: float | None = None
        self.occupation_kbT: float | None = None
        self.self_energy_damping: float = 1e-7
        
            
        if (self.ham.periodic_dirs == None or self.ham.periodic_dirs == "x"):
            self.transverse_periodic = False
        else:
            self.transverse_periodic = True
            
        self.kbT_eV = spc.Boltzmann / spc.elementary_charge * 300 * 0

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
            try:
                self.block_size_list = compute_block_sizes_metis(
                    self.ham_device,
                    self.H00,
                    self.H00_right,
                    atom_offsets=self.atom_offsets,
                    min_block_orbitals=self.H00.shape[0],
                )
                print(self.block_size_list)
            except Exception as exc:
                print(f"[GFFunctions] METIS partition fell back to heuristic: {exc}")
        if self.block_size_list is None:
            self.block_size_list = compute_block_sizes_block_tridiagonal(self.ham_device)

        self._update_block_offsets()

        # if self.use_variable_blocks:
        #     self.block_size_list = compute_block_sizes_block_tridiagonal(
        #         self.ham_device,
        #         self.H01,
        #         atom_offsets=self.atom_offsets,
        #     )

    def update_hamiltonian(self, potential):
        pot = np.asarray(potential, dtype=float)
        self.set_fermi_levels(pot[0], pot[-1])
        
        self.ham_device += np.diag(pot)
        
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

    def set_fermi_levels(self, mu_left: float, mu_right: float):
        self.mu_left = float(mu_left)
        self.mu_right = float(mu_right)

    def set_occupation_profile(self, mode: str = "fermi_dirac", *, prefactor: float | None = None,
                               kbT: float | None = None):
        mode_l = mode.lower()
        if mode_l not in {"fermi_dirac", "fd_half"}:
            raise ValueError("mode must be 'fermi_dirac' or 'fd_half'")
        if mode_l == "fd_half" and (prefactor is None or kbT is None):
            raise ValueError("fd_half mode requires prefactor and kbT values")
        self.occupation_mode = mode_l
        if prefactor is None:
            self.occupation_prefactor = None
        else:
            pref_arr = np.asarray(prefactor, dtype=float)
            if pref_arr.size != 1:
                raise ValueError("prefactor must be scalar for fd_half occupation")
            self.occupation_prefactor = float(pref_arr)
        self.occupation_kbT = None if kbT is None else float(kbT)

        def set_self_energy_damping(self, eta: float | None):
            """Configure the imaginary energy shift (in eV) applied when evaluating surface self-energies.

            Parameters
            ----------
            eta : float or None
                Positive value specifies the magnitude of the complex shift ``E -> E + i*eta``.
                If ``None`` or non-positive, the damping reverts to the default value (1e-7 eV).
            """
            if eta is None:
                self.self_energy_damping = 1e-7
                return
            try:
                eta_val = abs(float(eta))
            except Exception as exc:  # pragma: no cover - defensive programming
                raise ValueError("eta must be convertible to float") from exc
            if eta_val == 0.0:
                self.self_energy_damping = 1e-12
            else:
                self.self_energy_damping = eta_val

    def _resolve_inverse_method(self) -> str:
        method = getattr(self, "inverse_method", "auto")
        if method == "auto":
            if self.use_variable_blocks and self.block_size_list is not None:
                return "var_recursive"
            return "recursive"
        return method

    def _call_gf_inverse(self, E, ham_mat, H00_mat, H01_mat=None, H00_right_mat=None, *, processes: int = 1):
        H01_use = H01_mat if H01_mat is not None else self.H01
        H00_right_use = H00_right_mat if H00_right_mat is not None else self.H00_right
        block_size = H01_use.shape[0]
        method = self._resolve_inverse_method()
        result = _gf_inverse_dispatch(
            E,
            ham_mat,
            H00_mat,
            H01_use,
            block_size=block_size,
            block_size_list=self.block_size_list,
            use_variable_blocks=self.use_variable_blocks,
            processes=processes,
            method=method,
            mu_left=self.mu_left,
            mu_right=self.mu_right,
            H00_right=H00_right_use,
            occupation_mode=self.occupation_mode,
            occupation_prefactor=self.occupation_prefactor,
            occupation_kbT=self.occupation_kbT,
            self_energy_damp=self.self_energy_damping,
        )
        if method == "direct":
            G_R_full, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R = result
            return np.diag(G_R_full), G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R
        return result

    def _update_block_offsets(self):
        if self.use_variable_blocks and self.block_size_list is not None:
            self._block_offsets = np.cumsum(np.concatenate(([0], self.block_size_list.astype(int))))
        else:
            step = int(self.H00.shape[0]) if self.H00 is not None else 1
            total = self.ham_device.shape[0]
            if step <= 0:
                step = 1
            self._block_offsets = np.arange(0, total + 1, step, dtype=int)

    def _bond_current_integrand(self, G_lesser_offdiag_right):
        offsets = getattr(self, "_block_offsets", None)
        if offsets is None or offsets.size < 2:
            return np.zeros(0, dtype=float)
        num_blocks = offsets.size - 1
        if num_blocks <= 1:
            return np.zeros(0, dtype=float)
        bond_vals = np.zeros(num_blocks - 1, dtype=float)
        for i in range(num_blocks - 1):
            start_row = int(offsets[i])
            end_row = int(offsets[i + 1])
            start_col = int(offsets[i + 1])
            end_col = int(offsets[i + 2])
            H_ij = self.ham_device[start_row:end_row, start_col:end_col]
            G_lesser_ji = G_lesser_offdiag_right[i]
            if G_lesser_ji.shape[0] == H_ij.shape[0] and G_lesser_ji.shape[1] == H_ij.shape[1]:
                G_lesser_ji = G_lesser_ji.T
            trace_val = np.trace(H_ij @ G_lesser_ji)
            bond_vals[i] = np.imag(trace_val)
        return bond_vals
        
    
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
        ham_device = np.array(self.ham_device, copy=True, dtype=complex)
        if self.h_k_device is not None:
            h_dev = self.h_k_device.toarray() if sp.issparse(self.h_k_device) else np.asarray(self.h_k_device)
            ham_device += h_dev * np.exp(1j * ky)
            ham_device += h_dev.T * np.exp(-1j * ky)

        H00_left = np.array(self.H00, copy=True, dtype=complex)
        if self.h_k_lead is not None:
            h_lead = self.h_k_lead.toarray() if sp.issparse(self.h_k_lead) else np.asarray(self.h_k_lead)
            H00_left = H00_left + h_lead * np.exp(1j * ky) + h_lead.T * np.exp(-1j * ky)

        H00_right = np.array(self.H00_right, copy=True, dtype=complex)
        if self.h_k_lead is not None:
            H00_right = H00_right + h_lead * np.exp(1j * ky) + h_lead.T * np.exp(-1j * ky)
        G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R = self._call_gf_inverse(
            E, ham_device, H00_left, self.H01, H00_right_mat=H00_right
        )
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
            payloads.append((E_grid[sl], self.ham_device, self.H00, self.H00_right, self.H01, self.k_space, norm_k,
                             self.block_size_list, self.use_variable_blocks, cache_ref,
                             self.h_k_device, self.h_k_lead, self.self_energy_damping))
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
                densities.append(GFFunctions._density_k_worker_static((float(ky), E_grid, self.ham_device, self.H00,
                                                           self.H00_right, self.H01, poles, residues, V_orb, Efn_orb,
                                                           conduction_only, Ec_arr, dE, self.H00.shape[0],
                                                           self.block_size_list, self.use_variable_blocks, None,
                                                           self.h_k_device, self.h_k_lead)))
        else:
            cache_ref = self.LDOS_cache if hasattr(self.LDOS_cache, '_callmethod') else None
            payloads = [(float(ky), E_grid, self.ham_device, self.H00, self.H00_right, self.H01,
                         poles, residues, V_orb, Efn_orb, conduction_only, Ec_arr, dE, self.H00.shape[0],
                         self.block_size_list, self.use_variable_blocks, cache_ref,
                         self.h_k_device, self.h_k_lead)
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
                    densities.append(GFFunctions._density_k_worker_static((float(ky), E_grid, self.ham_device, self.H00,
                                                               self.H00_right, self.H01, poles, residues, V_orb, Efn_orb,
                                                               conduction_only, Ec_arr, dE, self.H00.shape[0],
                                                               self.block_size_list, self.use_variable_blocks, None,
                                                               self.h_k_device, self.h_k_lead)))
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
        return _caroli_transmission(
            E,
            self.ham_device,
            self.H00,
            self.H01,
            self.H00_right,
            getattr(self, "H01_right", None),
            self.self_energy_damping,
        )

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
        payloads = [
            (
                float(E),
                self.ham_device,
                self.H00,
                self.H01,
                self.H00_right,
                getattr(self, "H01_right", None),
                self.self_energy_damping,
            )
            for E in E_grid
        ]
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
        if len(G_lesser_offdiag_right) == 0:
            return np.zeros(0, dtype=float)
        return self._bond_current_integrand(G_lesser_offdiag_right)

    def bond_current(self, processes: int = 1, *, return_energy_resolved: bool = False):
        if getattr(self, '_force_serial', False):
            processes = 1
        offsets = getattr(self, '_block_offsets', None)
        if offsets is None or offsets.size < 2:
            return (np.zeros(0, dtype=float), np.zeros((self.energy_grid.size, 0), dtype=float)) if return_energy_resolved else np.zeros(0, dtype=float)
        num_blocks = offsets.size - 1
        if num_blocks <= 1:
            return (np.zeros(0, dtype=float), np.zeros((self.energy_grid.size, 0), dtype=float)) if return_energy_resolved else np.zeros(0, dtype=float)

        E_grid = self.energy_grid
        nE = E_grid.size
        integrand = np.zeros((nE, num_blocks - 1), dtype=float)

        for idx, E in enumerate(E_grid):
            integrand[idx, :] = self.current_worker(E)

        energy_joules = E_grid * spc.elementary_charge
        currents = np.zeros(num_blocks - 1, dtype=float)
        for bond in range(num_blocks - 1):
            currents[bond] = (2.0 * spc.elementary_charge / spc.h) * np.trapz(integrand[:, bond], energy_joules)

        if return_energy_resolved:
            return currents, integrand
        return currents

    def compute_charge_density(
        self,
        *,
        method: str = "lesser",
        processes: int = 1,
        density_scale: float | np.ndarray | None = None,
    ) -> np.ndarray:
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
                    ham_k = np.array(self.ham_device, copy=True, dtype=complex)
                    if self.h_k_device is not None:
                        h_dev = self.h_k_device.toarray() if sp.issparse(self.h_k_device) else np.asarray(self.h_k_device)
                        ham_k += h_dev * np.exp(1j * ky)
                        ham_k += h_dev.T * np.exp(-1j * ky)

                    H00_k = np.array(self.H00, copy=True, dtype=complex)
                    if self.h_k_lead is not None:
                        h_lead = self.h_k_lead.toarray() if sp.issparse(self.h_k_lead) else np.asarray(self.h_k_lead)
                        H00_k = H00_k + h_lead * np.exp(1j * ky) + h_lead.T * np.exp(-1j * ky)

                    H00_right_k = np.array(self.H00_right, copy=True, dtype=complex)
                    
                    if self.h_k_lead is not None:
                        H00_right_k = H00_right_k + h_lead * np.exp(1j * ky) + h_lead.T * np.exp(-1j * ky)
                    # Use recursive method to obtain G^< diagonal
                    _, G_lesser_diag, _, _, _ = self._call_gf_inverse(E, ham_k, H00_k, self.H01, H00_right_mat=H00_right_k)
                    
                    
                    G_n_diag = -1j * G_lesser_diag
                    
                    
                    accum_k += (dE * G_n_diag) / (2.0 * np.pi)
                density_orb += np.real(accum_k) / norm_k
            # Aggregate orbitals -> atoms
            density_sites = self._aggregate_orbital_to_atom(density_orb)
            if density_scale is not None:
                scale = np.asarray(density_scale, dtype=float)
                if scale.size == 1:
                    density_sites = density_sites * float(scale)
                elif scale.shape == density_sites.shape:
                    density_sites = density_sites * scale
                else:
                    raise ValueError("density_scale must be a scalar or match the density vector length.")
            return density_sites

        # Parallel path: partition energy grid contiguously
        P = min(processes, nE)
        bounds = np.linspace(0, nE, P + 1, dtype=int)
        payloads = []
        for p in range(P):
            sl = slice(bounds[p], bounds[p+1])
            if sl.start == sl.stop:
                continue
            payloads.append((E_grid[sl], self.ham_device, self.H00, self.H00_right, self.H01, self.h_k_device, self.h_k_lead,
                             self.k_space, block_size, self.block_size_list, self.use_variable_blocks, dE,
                             float(self.mu_left), float(self.mu_right), self.occupation_mode,
                             self.occupation_prefactor, self.occupation_kbT, self.self_energy_damping))

        try:
            ctx = mp.get_context('fork') if hasattr(mp, 'get_context') else mp
            with ctx.Pool(processes=P) as pool:
                parts = pool.map(_density_lesser_slice_worker_static, payloads)
        except Exception as exc:
            print(f"[compute_charge_density] Parallel fallback to serial due to: {exc}")
            return self.compute_charge_density(method=method, processes=1, density_scale=density_scale)

        # Sum contributions from slices and average over k-points
        total_orb = np.sum(np.vstack(parts), axis=0) / norm_k
        density_sites = self._aggregate_orbital_to_atom(total_orb)
        if density_scale is not None:
            scale = np.asarray(density_scale, dtype=float)
            if scale.size == 1:
                density_sites = density_sites * float(scale)
            elif scale.shape == density_sites.shape:
                density_sites = density_sites * scale
            else:
                raise ValueError("density_scale must be a scalar or match the density vector length.")
        return density_sites

 
    def _density_k_worker_static(payload: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                bool, np.ndarray, float, int, np.ndarray | None,
                                                bool, Any | None, np.ndarray | None, np.ndarray | None]) -> np.ndarray:
        """Compute carrier density contribution for a single k-point using Ozaki CFR.

        Returns density vector (n_sites,).
        """
        (ky, E_grid, ham_device, H00_left, H00_right, H01, poles, residues, V, Efn,
         conduction_only, Ec_arr, dE, block_size, block_size_list, use_var_blocks, cache_ref,
         h_k_device, h_k_lead) = payload
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
                ham_k = np.array(ham_device, copy=True, dtype=complex)
                if h_k_device is not None:
                    h_dev = h_k_device.toarray() if sp.issparse(h_k_device) else np.asarray(h_k_device)
                    ham_k += h_dev * np.exp(1j * ky)
                    ham_k += h_dev.T * np.exp(-1j * ky)

                H00_k = np.array(H00_left, copy=True, dtype=complex)
                h_lead_dense = None
                if h_k_lead is not None:
                    h_lead_dense = h_k_lead.toarray() if sp.issparse(h_k_lead) else np.asarray(h_k_lead)
                    H00_k = H00_k + h_lead_dense * np.exp(1j * ky) + h_lead_dense.T * np.exp(-1j * ky)

                H00_right_k = np.array(H00_right, copy=True, dtype=complex)
                
                if h_lead_dense is not None:
                    H00_right_k = H00_right_k + h_lead_dense * np.exp(1j * ky) + h_lead_dense.T * np.exp(-1j * ky)

                G_R_diag, _, _, _, _ = _gf_inverse_dispatch(
                    E,
                    ham_k,
                    H00_k,
                    H01,
                    block_size=block_size,
                    block_size_list=block_size_list,
                    use_variable_blocks=use_var_blocks,
                    H00_right=H00_right_k,
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
            

def _transmission_worker_static(
    payload: tuple[
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ]
) -> float:
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
    E, ham_device, H00, H01, H00_right, H01_right, self_energy_damp = payload
    return _caroli_transmission(
        E,
        ham_device,
        H00,
        H01,
        H00_right,
        H01_right,
        self_energy_damp,
    )


def _caroli_transmission(
    energy: float,
    ham_device: np.ndarray,
    H00_left: np.ndarray,
    H01_left: np.ndarray,
    H00_right: np.ndarray | None,
    H01_right: np.ndarray | None,
    self_energy_damp: float,
) -> float:
    H_dev = ham_device.toarray() if sp.issparse(ham_device) else np.asarray(ham_device, dtype=complex)
    H00_L = H00_left.toarray() if sp.issparse(H00_left) else np.asarray(H00_left, dtype=complex)
    tau_L = H01_left.toarray() if sp.issparse(H01_left) else np.asarray(H01_left, dtype=complex)
    H00_R = H00_right if H00_right is not None else H00_L
    tau_R = H01_right if H01_right is not None else tau_L
    H00_R = H00_R.toarray() if sp.issparse(H00_R) else np.asarray(H00_R, dtype=complex)
    tau_R = tau_R.toarray() if sp.issparse(tau_R) else np.asarray(tau_R, dtype=complex)

    damp = 1j * abs(float(self_energy_damp)) if self_energy_damp is not None else 1e-7j

    sigma_L_raw = surface_greens_function_nn(energy, tau_L, H00_L, tau_L.conj().T, damp=damp)
    sigma_R_raw = surface_greens_function_nn(energy, tau_R, H00_R, tau_R.conj().T, damp=damp)

    if isinstance(sigma_L_raw, tuple):
        sigma_L = sigma_L_raw[1]
    else:
        sigma_L = sigma_L_raw
    if isinstance(sigma_R_raw, tuple):
        sigma_R = sigma_R_raw[0]
    else:
        sigma_R = sigma_R_raw

    sigma_L = sigma_L.toarray() if sp.issparse(sigma_L) else np.asarray(sigma_L, dtype=complex)
    sigma_R = sigma_R.toarray() if sp.issparse(sigma_R) else np.asarray(sigma_R, dtype=complex)

    gamma_L = 1j * (sigma_L - sigma_L.conj().T)
    gamma_R = 1j * (sigma_R - sigma_R.conj().T)

    n = H_dev.shape[0]
    left_dim = gamma_L.shape[0]
    right_dim = gamma_R.shape[0]

    sigma_L_full = np.zeros((n, n), dtype=complex)
    sigma_R_full = np.zeros((n, n), dtype=complex)
    sigma_L_full[:left_dim, :left_dim] = sigma_L
    sigma_R_full[-right_dim:, -right_dim:] = sigma_R

    eta = 1e-12
    A = (energy + 1j * eta) * np.eye(n, dtype=complex) - (H_dev + sigma_L_full + sigma_R_full)
    G_R = np.linalg.solve(A, np.eye(n, dtype=complex))

    G_lr = G_R[:left_dim, n - right_dim :]
    T_val = np.real(np.trace(gamma_L @ G_lr @ gamma_R @ G_lr.conj().T))
    return float(T_val)


def _dos_slice_worker_static(payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            np.ndarray, int, np.ndarray | None, bool, Any | None,
                                            np.ndarray | None, np.ndarray | None, float]) -> np.ndarray:
    """Worker computing DOS for a contiguous slice of energies.

    Parameters
    ----------
    payload : tuple
        (E_slice, ham_device, H00_left, H00_right, H01, k_space, norm_k, block_size_list,
         use_var_blocks, cache_ref, h_k_device, h_k_lead)
    """
    (E_slice, ham_device, H00_left, H00_right, H01, k_space, norm_k,
     block_size_list, use_var_blocks, cache_ref, h_k_device, h_k_lead,
     self_energy_damp) = payload
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
                ham_k = np.array(ham_device, copy=True, dtype=complex)
                if h_k_device is not None:
                    h_dev = h_k_device.toarray() if sp.issparse(h_k_device) else np.asarray(h_k_device)
                    ham_k += h_dev * np.exp(1j * ky)
                    ham_k += h_dev.T * np.exp(-1j * ky)

                H00_k = np.array(H00_left, copy=True, dtype=complex)
                h_lead_dense = None
                if h_k_lead is not None:
                    h_lead_dense = h_k_lead.toarray() if sp.issparse(h_k_lead) else np.asarray(h_k_lead)
                    H00_k = H00_k + h_lead_dense * np.exp(1j * ky) + h_lead_dense.T * np.exp(-1j * ky)

                H00_right_k = np.array(H00_right, copy=True, dtype=complex)
                if h_lead_dense is not None:
                    H00_right_k = H00_right_k + h_lead_dense * np.exp(1j * ky) + h_lead_dense.T * np.exp(-1j * ky)

                G_R_diag, _, _, _, _ = _gf_inverse_dispatch(
                    E,
                    ham_k,
                    H00_k,
                    H01,
                    block_size=H00_left.shape[0],
                    block_size_list=block_size_list,
                    use_variable_blocks=use_var_blocks,
                    method="recursive",
                    H00_right=H00_right_k,
                    self_energy_damp=self_energy_damp,
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
                                                       np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray,
                                                       int, np.ndarray | None, bool, float, float, float, float]) -> np.ndarray:
    """Worker that integrates density over a contiguous slice of energies using G^<.

    Parameters
    ----------
    payload : tuple
        (E_slice, ham_device, H00_left, H00_right, H01, h_k_device, h_k_lead, k_space,
         block_size, block_size_list, use_var_blocks, dE)

    Returns
    -------
    np.ndarray
        Partial density vector (n_sites,) integrated over the slice; caller averages over k.
    """
    (E_slice, ham_device, H00_left, H00_right, H01, h_k_device, h_k_lead, k_space, block_size,
     block_size_list, use_var_blocks, dE, mu_left, mu_right, occupation_mode,
     occupation_prefactor, occupation_kbT, self_energy_damp) = payload
    n_sites = ham_device.shape[0]
    out = np.zeros(n_sites, dtype=float)
    for E in E_slice:
        accum_k = np.zeros(n_sites, dtype=complex)
        for ky in k_space:
            ham_k = np.array(ham_device, copy=True, dtype=complex)
            if h_k_device is not None:
                h_dev = h_k_device.toarray() if sp.issparse(h_k_device) else np.asarray(h_k_device)
                ham_k += h_dev * np.exp(1j * ky)
                ham_k += h_dev.T * np.exp(-1j * ky)

            H00_k = np.array(H00_left, copy=True, dtype=complex)
            h_lead_dense = None
            if h_k_lead is not None:
                h_lead_dense = h_k_lead.toarray() if sp.issparse(h_k_lead) else np.asarray(h_k_lead)
                H00_k = H00_k + h_lead_dense * np.exp(1j * ky) + h_lead_dense.T * np.exp(-1j * ky)

            H00_right_k = np.array(H00_right, copy=True, dtype=complex)
            if h_lead_dense is not None:
                H00_right_k = H00_right_k + h_lead_dense * np.exp(1j * ky) + h_lead_dense.T * np.exp(-1j * ky)
            _, G_lesser_diag, _, _, _ = _gf_inverse_dispatch(
                E,
                ham_k,
                H00_k,
                H01,
                block_size=block_size,
                block_size_list=block_size_list,
                use_variable_blocks=use_var_blocks,
                method="var_recursive",
                H00_right=H00_right_k,
                mu_left=mu_left,
                mu_right=mu_right,
                occupation_mode=occupation_mode,
                occupation_prefactor=occupation_prefactor,
                occupation_kbT=occupation_kbT,
                self_energy_damp=self_energy_damp,
            )
            G_n_diag = -1j * G_lesser_diag
            accum_k += (dE * G_n_diag) / (2.0 * np.pi)
        out += np.real(accum_k)
    return out


def _gf_inverse_dispatch(E, ham_device, H00, H01, *, block_size: int, block_size_list: np.ndarray | None,
                         use_variable_blocks: bool, processes: int = 1, method: str | None = None,
                         mu_left: float = 0.0, mu_right: float = 0.0,
                         H00_right: np.ndarray | None = None,
                         occupation_mode: str = "fermi_dirac",
                         occupation_prefactor: float | None = None,
                         occupation_kbT: float | None = None,
                         self_energy_damp: float = 1e-7):
    selected_method = method
    if selected_method is None or selected_method == "auto":
        if use_variable_blocks and block_size_list is not None:
            selected_method = "var_recursive"
        else:
            selected_method = "recursive"

    if selected_method == "direct":
        return gf_inverse(
            E,
            ham_device,
            H00,
            H01,
            mu_left,
            mu_right,
            block_size=block_size,
            block_size_list=block_size_list,
            method="direct",
            processes=processes,
            H00_right=H00_right,
            occupation_mode=occupation_mode,
            occupation_prefactor=occupation_prefactor,
            occupation_kbT=occupation_kbT,
            self_energy_damp=self_energy_damp,
        )

    if selected_method == "var_recursive":
        
        return gf_inverse(
            E,
            ham_device,
            H00,
            H01,
            mu_left,
            mu_right,
            block_size_list=block_size_list,
            method="var_recursive",
            processes=processes,
            H00_right=H00_right,
            occupation_mode=occupation_mode,
            occupation_prefactor=occupation_prefactor,
            occupation_kbT=occupation_kbT,
            self_energy_damp=self_energy_damp,
        )

    if selected_method == "recursive":
        return gf_inverse(
            E,
            ham_device,
            H00,
            H01,
            mu_left,
            mu_right,
            block_size=block_size,
            method="recursive",
            H00_right=H00_right,
            occupation_mode=occupation_mode,
            occupation_prefactor=occupation_prefactor,
            occupation_kbT=occupation_kbT,
            self_energy_damp=self_energy_damp,
        )

    raise ValueError(f"Unsupported inversion method '{selected_method}'.")

