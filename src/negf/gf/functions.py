from __future__ import annotations
from itertools import product
import multiprocessing as mp
import numpy as np
import scipy.sparse as sp
import scipy.constants as spc
from typing import List, Tuple, Iterable, Callable, Any


from ..utils.ozaki import (
    get_ozaki_poles_residues,
    fermi_cfr,
    fermi_derivative_cfr_abs_batch,
)
from .recursive_greens_functions import recursive_gf
from hamiltonian.base.block_tridiagonalization import split_into_subblocks_optimized
from hamiltonian.base.hamiltonian_core import Hamiltonian
from negf.gf.recursive_greens_functions import gf_inverse
class GFFunctions:
    def __init__(self, ham : Hamiltonian, energy_grid: np.ndarray, k_space: np.ndarray | None = None,
                 self_energy_method: str = "sancho_rubio"):
        self.ham = ham
        self.energy_grid = np.atleast_1d(energy_grid)
        self.dE = (self.energy_grid[1] - self.energy_grid[0]) if self.energy_grid.size > 1 else 1.0
        self.k_space = np.atleast_1d(k_space) if k_space is not None else np.array([0])
        self.self_energy_method = self_energy_method
        
        self.LDOS_cache: dict[tuple[float, float], np.ndarray] = {}
        self._force_serial = False
        
        ham_new, hL0, hLC, hR0, hRC = self.ham.get_hamiltonians()
        self.ham_device = ham_new
        self.H00 = hL0
        self.H01 = hLC
        self.block_size = self.H00.shape[0]
        if (self.ham.periodic_dirs == None or self.ham.periodic_dirs == "x"):
            self.transverse_periodic = False
        else:
            self.transverse_periodic = True
        
    
    def _ldos_key(self, E, ky) -> tuple[float, float]:
        return (round(float(np.real(E)), 12), round(float(ky), 12))

    def clear_ldos_cache(self):
        self.LDOS_cache.clear()


    def _get_ldos_cached(self, E, ky =0) -> np.ndarray:
        key = self._ldos_key(E, ky)
        if key in self.LDOS_cache:
            return self.LDOS_cache[key]
        G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R = gf_inverse(E, self.ham_device, self.H00, self.H01,block_size=150, method="recursive")
        ldos_vec = -1.0 / np.pi * np.imag(G_R_diag)
        self.LDOS_cache[key] = ldos_vec
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
            payloads.append((E_grid[sl], self.ham_device, self.H00, self.H01, self.k_space, norm_k))
        try:
            ctx = mp.get_context('fork') if hasattr(mp, 'get_context') else mp
            with ctx.Pool(processes=P) as pool:
                parts = pool.map(_dos_slice_worker_static, payloads)
        except Exception as exc:
            print(f"[total_dos] Parallel fallback to serial due to: {exc}")
            return self.total_dos(processes=1)
        return np.concatenate(parts)

    def fermi(self, E, V_vec: np.ndarray, Efn_vec: np.ndarray, cutoff=60) -> np.ndarray:
        poles, residues = get_ozaki_poles_residues(cutoff, self.ham.kbT_eV, getattr(self.ham, 'T', 300))
        return fermi_cfr(np.atleast_1d(E), None, poles, residues, self.ham.kbT_eV, V_vec=V_vec, Efn_vec=Efn_vec)[0]

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
        n_sites = V.size
        Ec_arr = np.atleast_1d(Ec).astype(float)
        if Ec_arr.size == 1:
            Ec_arr = np.full(n_sites, Ec_arr[0])
        elif Ec_arr.size != n_sites:
            Ec_arr = Ec_arr[:n_sites]

        if force_recompute_ozaki:
            try:
                get_ozaki_poles_residues.cache_clear()
            except Exception:
                pass
        poles, residues = get_ozaki_poles_residues(ozaki_cutoff, self.ham.kbT_eV, getattr(self.ham,'T',300))

        E_grid = self.energy_grid
        dE = self.dE
        k_points = self.k_space if self.k_space.size > 0 else np.array([0])

        if processes <= 1 or k_points.size == 1:
            densities = []
            for ky in k_points:
                densities.append(_density_k_worker_static((float(ky), E_grid, self.ham_device, self.H00, self.H01,
                                                           poles, residues, V, Efn, conduction_only, Ec_arr, dE, self.H00.shape[0])))
        else:
            payloads = [(float(ky), E_grid, self.ham_device, self.H00, self.H01,
                         poles, residues, V, Efn, conduction_only, Ec_arr, dE, self.H00.shape[0])
                        for ky in k_points]
            try:
                ctx = mp.get_context('fork') if hasattr(mp, 'get_context') else mp
                with ctx.Pool(processes=min(processes, k_points.size)) as pool:
                    densities = pool.map(_density_k_worker_static, payloads)
            except Exception as exc:
                print(f"[get_n] Parallel fallback to serial due to: {exc}")
                densities = []
                for ky in k_points:
                    densities.append(_density_k_worker_static((float(ky), E_grid, self.ham_device, self.H00, self.H01,
                                                               poles, residues, V, Efn, conduction_only, Ec_arr, dE, self.H00.shape[0])))

        dens_stack = np.vstack(densities)
        if ky_avg:
            return np.mean(dens_stack, axis=0)
        return np.sum(dens_stack, axis=0)

    def diff_rho_poisson(self, V: np.ndarray, Efn: np.ndarray, Ec: float | None = None,
                          ozaki_cutoff: int = 60) -> np.ndarray:
        V = np.atleast_1d(V)
        Efn = np.atleast_1d(Efn)
        n_sites = V.size
        poles, residues = get_ozaki_poles_residues(ozaki_cutoff, self.ham.kbT_eV, getattr(self.ham, 'T', 300))
        deriv = np.zeros(n_sites)
        for E in self.energy_grid:
            ldos_acc = np.zeros(n_sites)
            for ky in self.k_space:
                ldos_acc += self._get_ldos_cached(E, ky)
            ldos_acc /= max(1, self.k_space.size)
            dfdx_vec = fermi_derivative_cfr_abs_batch(np.array([E]), V, Efn, poles, residues, self.ham.kbT_eV)[0]
            deriv += ldos_acc * (dfdx_vec / self.ham.kbT_eV) * self.dE
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
        if processes <= 1:
            return np.array([self.transmission_worker(E) for E in E_grid], dtype=float)

        # Parallel path: build lightweight payload without embedding self (which is not picklable due to lambdas in Hamiltonian)
        payloads = [(float(E), self.ham_device, self.H00, self.H01) for E in E_grid]
        try:
            # Prefer 'fork' when available to reduce serialization overhead
            ctx = mp.get_context('fork') if hasattr(mp, 'get_context') else mp
            with ctx.Pool(processes=processes) as pool:
                vals = pool.map(_transmission_worker_static, payloads)
            return np.asarray(vals, dtype=float)
        except Exception as exc:
            print(f"[transmission] Parallel fallback to serial due to: {exc}")
            return np.array([self.transmission_worker(E) for E in E_grid], dtype=float)
        
    def current_worker(self, E: float) -> np.ndarray:
        _, _, G_lesser_offdiag_right, _, _ = gf_inverse(
            E, self.ham_device, self.H00, self.H01, 
            block_size=self.block_size, method="recursive"
        )
        num_blocks = self.ham_device.shape[0] // self.block_size
        bond_current_integrands = []
        for i in range(num_blocks - 1):
            start_row = i * self.block_size
            end_row = (i + 1) * self.block_size
            start_col = (i + 1) * self.block_size
            end_col = (i + 2) * self.block_size
            H_ij = self.ham_device[start_row:end_row, start_col:end_col]
            G_lesser_ji = G_lesser_offdiag_right[i]
            product_matrix = H_ij @ G_lesser_ji
            trace_val = np.trace(product_matrix)
            bond_current_integrands.append(np.imag(trace_val))
            
        return np.array(bond_current_integrands)

    def bond_currents(self, processes: int = 1) -> np.ndarray:
        """Energy-integrated bond currents between consecutive principal layers.

        Returns
        -------
        np.ndarray (num_blocks-1,)
            Integrated currents (A). Prefactor uses (q/ħ); spin degeneracy, if
            desired, should be applied externally.
        """
        q = getattr(self.ham, 'q', spc.elementary_charge)
        hbar = spc.hbar
        pref = q / hbar  # Meir-Wingreen style (assuming already symmetrized lesser GF)
        dE = self.dE
        E_grid = self.energy_grid

        # Serial path
        if processes <= 1 or E_grid.size == 1:
            acc = None
            for E in E_grid:
                vals = _bond_current_energy_worker_static((float(E), self.ham_device, self.H00, self.H01, self.block_size))
                if acc is None:
                    acc = np.zeros_like(vals, dtype=float)
                acc += vals
            return pref * dE * acc

        payloads = [(float(E), self.ham_device, self.H00, self.H01, self.block_size) for E in E_grid]
        try:
            ctx = mp.get_context('fork') if hasattr(mp, 'get_context') else mp
            with ctx.Pool(processes=min(processes, E_grid.size)) as pool:
                all_vals = pool.map(_bond_current_energy_worker_static, payloads)
        except Exception as exc:
            print(f"[bond_currents] Parallel fallback to serial due to: {exc}")
            all_vals = [ _bond_current_energy_worker_static(pl) for pl in payloads ]
        acc = np.sum(all_vals, axis=0)
        return pref * dE * acc


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


def _dos_slice_worker_static(payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]) -> np.ndarray:
    """Worker computing DOS for a contiguous slice of energies.

    Parameters
    ----------
    payload : tuple
        (E_slice, ham_device, H00, H01, k_space, norm_k)
    """
    E_slice, ham_device, H00, H01, k_space, norm_k = payload
    out = np.zeros(E_slice.size, dtype=float)
    for i, E in enumerate(E_slice):
        acc = 0.0
        for ky in k_space:
            # Direct LDOS computation (no cache across processes)
            G_R_diag, _, _, _, _ = gf_inverse(E, ham_device, H00, H01, block_size=H00.shape[0], method="recursive")
            acc += float(np.sum(-1/np.pi * np.imag(G_R_diag)))
        out[i] = acc / norm_k
    return out


def _density_k_worker_static(payload: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            bool, np.ndarray, float, int]) -> np.ndarray:
    """Compute carrier density contribution for a single k-point using Ozaki CFR.

    Returns density vector (n_sites,).
    """
    (ky, E_grid, ham_device, H00, H01, poles, residues, V, Efn,
     conduction_only, Ec_arr, dE, block_size) = payload
    n_sites = V.size
    nE = E_grid.size
    ldos_mat = np.zeros((nE, n_sites), dtype=float)
    for i, E in enumerate(E_grid):
        G_R_diag, _, _, _, _ = gf_inverse(E, ham_device, H00, H01, block_size=block_size, method="recursive")
        ldos_mat[i, :] = -1/np.pi * np.imag(G_R_diag)
    f_mat = fermi_cfr(E_grid, None, poles, residues, # type: ignore
                      spc.Boltzmann * getattr(spc, 'zero_Celsius', 273.15) / spc.elementary_charge if False else  # placeholder never used
                      None, V_vec=V, Efn_vec=Efn)
    # Note: original fermi_cfr signature used self.ham.kbT_eV; we rely on poles/residues already built for correct kT.
    if conduction_only:
        mask = (E_grid[:, None] >= Ec_arr[None, :])
        f_mat = f_mat * mask
    return np.sum(ldos_mat * f_mat, axis=0) * dE


def _bond_current_energy_worker_static(payload: tuple[float, np.ndarray, np.ndarray, np.ndarray, int]) -> np.ndarray:
    """Compute bond current integrand array for a single energy."""
    E, ham_device, H00, H01, block_size = payload
    G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R = gf_inverse(E, ham_device, H00, H01, block_size=block_size, method="recursive")
    num_blocks = ham_device.shape[0] // block_size
    vals = []
    for i in range(num_blocks - 1):
        start_row = i * block_size
        end_row = (i + 1) * block_size
        start_col = (i + 1) * block_size
        end_col = (i + 2) * block_size
        H_ij = ham_device[start_row:end_row, start_col:end_col]
        G_lesser_ji = G_lesser_offdiag_right[i]
        trace_val = np.trace(H_ij @ G_lesser_ji)
        vals.append(np.imag(trace_val))
    return np.asarray(vals, dtype=float)