"""High-level Green's function utilities built atop block-recursive solver.

Provides: DOS(E), get_n(V,Efn,...), diff_rho_poisson(...), and fermi utilities.
Maintains an LDOS cache keyed by (E, ky) to avoid recomputation.

Uses Hamiltonian.get_hamiltonians() and the updated recursive_greens_functions.
"""
from __future__ import annotations
from itertools import product
import multiprocessing as mp
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple

from ..self_energy.surface import LeadSelfEnergy
from ..utils.ozaki import (
    get_ozaki_poles_residues,
    fermi_cfr,
    fermi_derivative_cfr_abs_batch,
)
from .recursive_greens_functions import recursive_gf
from hamiltonian.base.block_tridiagonalization import split_into_subblocks_optimized


class GFFunctions:
    def __init__(self, ham, energy_grid: np.ndarray, k_space: np.ndarray | None = None,
                 self_energy_method: str = "sancho_rubio"):
        self.ham = ham
        self.energy_grid = np.atleast_1d(energy_grid)
        self.dE = (self.energy_grid[1] - self.energy_grid[0]) if self.energy_grid.size > 1 else 1.0
        self.k_space = np.atleast_1d(k_space) if k_space is not None else np.array([0])
        self.self_energy_method = self_energy_method
        self.leads = LeadSelfEnergy(ham)
        self.LDOS_cache: dict[tuple[float, float], np.ndarray] = {}
        self._force_serial = False

    def _ldos_key(self, E, ky) -> tuple[float, float]:
        return (round(float(np.real(E)), 12), round(float(ky), 12))

    def clear_ldos_cache(self):
        self.LDOS_cache.clear()

    def _get_blocks_and_sigmas(self, E, ky) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """Build principal-layer blocks and lead sigmas for energy E, ky.

        Returns (H_ii, H_l, H_r, Sigma_L, Sigma_R).
        """
        Hl, H0, Hr = self.ham.get_hamiltonians()
        try:
            H_ii, H_l, H_r, _ = split_into_subblocks_optimized(H0, Hl, Hr)
        except Exception:
            H_ii, H_l, H_r, _ = split_into_subblocks_optimized(H0, 1, 1)
        # Ensure dense
        H_ii = [blk.toarray() if sp.issparse(blk) else blk for blk in H_ii]
        H_l = [blk.toarray() if sp.issparse(blk) else blk for blk in H_l]
        H_r = [blk.toarray() if sp.issparse(blk) else blk for blk in H_r]
        nb = len(H_ii)
        if nb <= 1:
            H00_L = H0
            H01_L = Hr
            H00_R = H0
            H01_R = Hl.conj().T
        else:
            H00_L = H_ii[0]
            H01_L = H_r[0]
            H00_R = H_ii[-1]
            H01_R = H_l[-1].conj().T
        self.leads.set_lead_matrices('left', H00_L, H01_L)
        self.leads.set_lead_matrices('right', H00_R, H01_R)
        try:
            sigma_L = self.leads.self_energy('left', E, ky, self.self_energy_method)
            sigma_R = self.leads.self_energy('right', E, ky, self.self_energy_method)
        except Exception:
            n0 = H_ii[0].shape[0] if nb > 0 else H0.shape[0]
            sigma_L = np.zeros((n0, n0), dtype=complex)
            sigma_R = np.zeros((n0, n0), dtype=complex)
        return H_ii, H_l, H_r, sigma_L, sigma_R

    def _get_ldos_cached(self, E, ky) -> np.ndarray:
        key = self._ldos_key(E, ky)
        if key in self.LDOS_cache:
            return self.LDOS_cache[key]
        H_ii, H_l, H_r, sigma_L, sigma_R = self._get_blocks_and_sigmas(E, ky)
        # Use recursive_gf to compute diagonal blocks of G^R and concatenate their diagonals
        _, grd, _, _, _ = recursive_gf(float(np.real(E)), H_l.copy(), H_ii.copy(), H_r.copy(),
                        left_se=sigma_L, right_se=sigma_R, damp=0.000001j)
        G_R_diag = np.concatenate([np.diag(block) for block in grd]) if len(grd) > 0 else np.array([])
        ldos_vec = -1.0 / np.pi * np.imag(G_R_diag)
        self.LDOS_cache[key] = ldos_vec
        return ldos_vec

    # -------- Public integrals --------
    def dos(self, E, ky=0) -> np.ndarray:
        return self._get_ldos_cached(E, ky)

    def total_dos(self) -> np.ndarray:
        # Sum over sites and k for each energy
        out = np.zeros(self.energy_grid.size)
        norm_k = max(1, self.k_space.size)
        for i, E in enumerate(self.energy_grid):
            acc = 0.0
            for ky in self.k_space:
                acc += np.sum(self._get_ldos_cached(E, ky))
            out[i] = acc / norm_k
        return out

    def fermi(self, E, V_vec: np.ndarray, Efn_vec: np.ndarray, cutoff=60) -> np.ndarray:
        poles, residues = get_ozaki_poles_residues(cutoff, self.ham.kbT_eV, getattr(self.ham, 'T', 300))
        return fermi_cfr(np.atleast_1d(E), None, poles, residues, self.ham.kbT_eV, V_vec=V_vec, Efn_vec=Efn_vec)[0]

    def get_n(self, V: np.ndarray, Efn: np.ndarray, ky_avg: bool = True, conduction_only: bool = False, Ec: float | np.ndarray = 0.0,
              method: str = 'ozaki_cfr', ozaki_cutoff: int = 60, processes: int = 1) -> np.ndarray:
        V = np.atleast_1d(V)
        Efn = np.atleast_1d(Efn)
        Ec_arr = np.atleast_1d(Ec)
        n_sites = V.size
        k_norm = max(1, self.k_space.size) if ky_avg else 1
        density = np.zeros(n_sites)
        poles, residues = get_ozaki_poles_residues(ozaki_cutoff, self.ham.kbT_eV, getattr(self.ham, 'T', 300))
        for E in self.energy_grid:
            ldos_acc = np.zeros(n_sites)
            for ky in self.k_space:
                ldos_acc += self._get_ldos_cached(E, ky)
            ldos_acc /= k_norm
            f_vec = fermi_cfr(np.array([E]), None, poles, residues, self.ham.kbT_eV, V_vec=V, Efn_vec=Efn)[0]
            if conduction_only:
                if Ec_arr.size == 1:
                    if E < Ec_arr[0]:
                        continue
                else:
                    f_vec = np.where(E >= Ec_arr, f_vec, 0.0)
            density += ldos_acc * f_vec * self.dE
        return density

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
