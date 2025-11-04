"""Lightweight NEGF helpers tailored for the Poisson coupling workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import scipy.constants as spc
import scipy.sparse as spa
import scipy.sparse.linalg as spla


from negf.self_energy.surface import surface_greens_function
from negf.self_energy.greens_functions import surface_greens_function_nn
KB_OVER_Q = spc.Boltzmann / spc.elementary_charge


def build_tridiagonal_hamiltonian(onsite: Sequence[float], hoppings: Sequence[float]) -> np.ndarray:
    """Construct a tight-binding tridiagonal Hamiltonian matrix."""

    onsite_arr = np.asarray(onsite, dtype=float)
    hop_arr = np.asarray(hoppings, dtype=float)
    if onsite_arr.ndim != 1:
        raise ValueError("Onsite energies must form a 1D array")
    if hop_arr.ndim != 1:
        raise ValueError("Hopping array must be 1D")
    if hop_arr.size != onsite_arr.size - 1:
        raise ValueError("Hopping array must have length N-1 for N onsite entries")
    n = onsite_arr.size
    H = np.zeros((n, n), dtype=np.complex128)
    np.fill_diagonal(H, onsite_arr)
    for i in range(n - 1):
        hop = hop_arr[i]
        H[i, i + 1] = hop
        H[i + 1, i] = hop
    return H


def effective_mass_1d_tb(hopping: float, lattice_constant: float) -> float:
    """Return the 1D tight-binding effective mass in kilograms.

    A nearest-neighbour dispersion ``E(k)=E0+2t cos(ka)`` gives
    ``m* = ħ² / (2 |t| a² e)`` (with ``t`` in eV).
    """

    if lattice_constant <= 0:
        raise ValueError("Lattice constant must be positive")
    t_abs = abs(float(hopping))
    if t_abs == 0:
        raise ValueError("Hopping cannot be zero when computing effective mass")
    return (spc.hbar ** 2) / (2.0 * t_abs * spc.elementary_charge * lattice_constant ** 2)


@dataclass(slots=True)
class Lead1D:
    """Semi-infinite 1D or block lead supporting Sancho-Rubio or nanonet self-energies."""

    onsite: Sequence[float] | np.ndarray
    hopping: Sequence[float] | np.ndarray
    attach_index: int
    coupling: Optional[Sequence[float] | np.ndarray] = None
    method: str = "sancho_rubio"
    eta: float = 1e-12
    iteration_max: int = 200
    tolerance: float = 1e-10
    H00: np.ndarray = field(init=False)
    H01: np.ndarray = field(init=False)
    V_couple: np.ndarray = field(init=False)
    block_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.H00 = np.atleast_2d(np.asarray(self.onsite, dtype=complex))
        if self.H00.shape[0] != self.H00.shape[1]:
            raise ValueError("Lead onsite block must be square")
        hop_arr = np.asarray(self.hopping, dtype=complex)
        if hop_arr.ndim == 0:
            self.H01 = np.array([[hop_arr]])
        else:
            self.H01 = np.atleast_2d(hop_arr)
        if self.H01.shape != self.H00.shape:
            raise ValueError("Lead hopping block must match onsite block dimensions")
        if self.coupling is None:
            self.V_couple = self.H01.copy()
        else:
            coup = np.asarray(self.coupling, dtype=complex)
            if coup.ndim == 0:
                self.V_couple = np.array([[coup]])
            else:
                self.V_couple = np.atleast_2d(coup)
        if self.V_couple.shape[1] != self.H00.shape[0]:
            raise ValueError("Coupling matrix must have width equal to lead block dimension")
        self.block_dim = self.V_couple.shape[0]

    def self_energy(self, energy: float, size: int) -> np.ndarray:
        idx = self.attach_index if self.attach_index >= 0 else size + self.attach_index
        if idx < 0 or idx + self.block_dim > size:
            raise IndexError("Lead attach index outside device range")
        if self.method == "nanonet":
            sigma_block = self._nanonet_sigma(float(energy), idx == 0)
        elif self.block_dim == 1 and self.method == "sancho_rubio":
            sigma_block = np.array([[self._analytic_sigma(float(energy))]])
        else:
            g_surface = surface_greens_function(
                energy + 1j * self.eta,
                self.H00,
                self.H01,
                method=self.method,
                iteration_max=self.iteration_max,
                tolerance=self.tolerance,
            )
            sigma_block = self.V_couple @ g_surface @ self.V_couple.conjugate().T
        mat = np.zeros((size, size), dtype=np.complex128)
        mat[idx : idx + self.block_dim, idx : idx + self.block_dim] = sigma_block
        return mat

    def _analytic_sigma(self, energy: float) -> complex:
        z = energy + 1j * self.eta - self.H00[0, 0]
        t = self.H01[0, 0]
        disc = z * z - 4.0 * (t ** 2)
        root = np.lib.scimath.sqrt(disc)
        if np.imag(root) < 0:
            root = -root
        g_surface = (z - root) / (2.0 * (t ** 2))
        v = self.V_couple[0, 0]
        return (v ** 2) * g_surface

    def _nanonet_sigma(self, energy: float, is_left: bool) -> np.ndarray:
        energy_real = float(np.real(energy))
        sigma_right, sigma_left = surface_greens_function_nn(
            energy_real,
            self.H01,
            self.H00,
            self.H01.conjugate().T,
            damp=1j * self.eta,
        )
        # sigma_right = self._ensure_retarded(sigma_right)
        # sigma_left = self._ensure_retarded(sigma_left)
        sigma_block = sigma_right if is_left else sigma_left
        if sigma_block.shape[0] != self.block_dim:
            raise ValueError("Lead self-energy block size mismatch for nanonet method")
        return sigma_block

    def _ensure_retarded(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.array(sigma, dtype=np.complex128, copy=True)
        if sigma.shape == (1, 1):
            val = sigma[0, 0]
            sigma[0, 0] = val.real - 1j * abs(val.imag)
            return sigma
        gamma = 1j * (sigma - sigma.conjugate().T)
        gamma = 0.5 * (gamma + gamma.conjugate().T)
        min_eval = np.min(np.linalg.eigvalsh(gamma))
        if min_eval < -self.tolerance:
            sigma = sigma.conjugate().T
            gamma = 1j * (sigma - sigma.conjugate().T)
            gamma = 0.5 * (gamma + gamma.conjugate().T)
            min_eval = np.min(np.linalg.eigvalsh(gamma))
            if min_eval < -self.tolerance:
                raise RuntimeError("Failed to obtain retarded self-energy from nanonet output")
        return sigma


@dataclass(slots=True)
class OzakiNEGF:
    """Minimal NEGF calculator providing LDOS and carrier densities."""

    onsite: Sequence[float]
    hoppings: Sequence[float]
    energy_grid: Sequence[float]
    left_lead: Lead1D
    right_lead: Lead1D
    temperature: float = 300.0
    eta: float = 1e-12
    ozaki_cutoff: int = 60
    size: int = field(init=False)
    kbT: float = field(init=False)
    _uniform_dE: Optional[float] = field(init=False, default=None)
    _nonuniform: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.onsite = np.asarray(self.onsite, dtype=float)
        self.hoppings = np.asarray(self.hoppings, dtype=float)
        self.energy_grid = np.asarray(self.energy_grid, dtype=float)
        if self.onsite.ndim != 1:
            raise ValueError("Onsite array must be 1D")
        if self.hoppings.size != self.onsite.size - 1:
            raise ValueError("Hopping array length must be N-1")
        if self.energy_grid.ndim != 1 or self.energy_grid.size < 2:
            raise ValueError("Energy grid must be 1D with at least two points")
        self.size = self.onsite.size
        self.kbT = KB_OVER_Q * self.temperature
        diffs = np.diff(self.energy_grid)
        if np.allclose(diffs, diffs[0]):
            self._uniform_dE = float(diffs[0])
            self._nonuniform = False
        else:
            self._uniform_dE = None
            self._nonuniform = True

    def _hamiltonian(self, potential: Sequence[float] | float) -> np.ndarray:
        pot = np.asarray(potential, dtype=float)
        if pot.size == 1:
            pot = np.full(self.size, float(pot), dtype=float)
        if pot.size != self.size:
            raise ValueError("Potential must match device size")
        diag = self.onsite - pot
        H = np.zeros((self.size, self.size), dtype=np.complex128)
        np.fill_diagonal(H, diag)
        for i in range(self.size - 1):
            hop = self.hoppings[i]
            H[i, i + 1] = hop
            H[i + 1, i] = hop
        return H

    def _greens(self, energy: float, potential: Sequence[float] | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = self._hamiltonian(potential)
        Sigma_L = self.left_lead.self_energy(energy, self.size)
        Sigma_R = self.right_lead.self_energy(energy, self.size)
        identity = np.eye(self.size, dtype=np.complex128)
        mat = (energy + 1j * self.eta) * identity - H - Sigma_L - Sigma_R
        I = spa.eye(mat.shape[0])
        mat = spa.csc_matrix(mat)
        G_R = spla.spsolve(mat, I)
        return G_R, Sigma_L, Sigma_R

    def _gamma(self, sigma: np.ndarray) -> np.ndarray:
        return 1j * (sigma - sigma.conjugate().T)

    def _fermi(self, energy: float, mu: float) -> float:
        return 1.0 / (1.0 + np.exp((energy - mu) / self.kbT))

    def ldos_vector(self, energy: float, potential: Sequence[float] | float) -> np.ndarray:
        G_R, _, _ = self._greens(energy, potential)
        diag = -1.0 / np.pi * np.imag(np.diag(G_R))
        return diag.astype(float)

    def ldos_matrix(self, potential: Sequence[float] | float) -> np.ndarray:
        ldos = np.zeros((self.energy_grid.size, self.size), dtype=float)
        for i, E in enumerate(self.energy_grid):
            ldos[i, :] = self.ldos_vector(E, potential)
        return ldos

    def electron_density_lesser(self, potential: Sequence[float] | float, mu: float) -> np.ndarray:
        return self.electron_density_non_equilibrium(potential, mu, mu)

    def electron_density_ozaki(
        self,
        potential: Sequence[float] | float,
        mu: Sequence[float] | float,
        *,
        conduction_only: bool = True,
        Ec: Sequence[float] | float = 0.0,
    ) -> np.ndarray:
        mu_arr = np.asarray(mu, dtype=float)
        if mu_arr.ndim == 0 or mu_arr.size == 1:
            mu_scalar = float(np.reshape(mu_arr, (-1,))[0])
            mu_arr = np.full(self.size, mu_scalar, dtype=float)
        if mu_arr.size != self.size:
            raise ValueError("Fermi level array must match device size")
        ldos = self.ldos_matrix(potential)
        f_mat = 1.0 / (1.0 + np.exp((self.energy_grid[:, None] - mu_arr[None, :]) / self.kbT))
        if conduction_only:
            Ec_arr = np.asarray(Ec, dtype=float)
            if Ec_arr.ndim == 0 or Ec_arr.size == 1:
                Ec_scalar = float(np.reshape(Ec_arr, (-1,))[0])
                Ec_arr = np.full(self.size, Ec_scalar, dtype=float)
            if Ec_arr.size != self.size:
                raise ValueError("Ec array must match device size")
            mask = self.energy_grid[:, None] >= Ec_arr[None, :]
        else:
            mask = 1.0
        integrand = ldos * f_mat * mask
        if self._nonuniform:
            densities = np.trapz(integrand, self.energy_grid, axis=0)
        else:
            densities = integrand.sum(axis=0) * self._uniform_dE
        return densities.astype(float)

    def electron_density_non_equilibrium(
        self,
        potential: Sequence[float] | float,
        mu_left: float,
        mu_right: float,
    ) -> np.ndarray:
        samples = np.zeros((self.energy_grid.size, self.size), dtype=float)
        for i, E in enumerate(self.energy_grid):
            G_R, Sigma_L, Sigma_R = self._greens(E, potential)
            Gamma_L = self._gamma(Sigma_L)
            Gamma_R = self._gamma(Sigma_R)
            G_A = G_R.conjugate().T
            fL = self._fermi(E, mu_left)
            fR = self._fermi(E, mu_right)
            coupling = Gamma_L * fL + Gamma_R * fR
            G_less = 1j * (G_R @ coupling @ G_A)
            samples[i, :] = np.imag(np.diag(G_less))
        if self._nonuniform:
            densities = np.trapz(samples, self.energy_grid, axis=0) / (2.0 * np.pi)
        else:
            densities = samples.sum(axis=0) * (self._uniform_dE / (2.0 * np.pi))
        return densities.astype(float)

    def transmission(
        self,
        potential: Sequence[float] | float,
    ) -> np.ndarray:
        trans = np.zeros(self.energy_grid.size, dtype=float)
        for i, E in enumerate(self.energy_grid):
            trans[i] = self.transmission_at_energy(E, potential)
        return trans

    def transmission_at_energy(
        self,
        energy: float,
        potential: Sequence[float] | float,
    ) -> float:
        G_R, Sigma_L, Sigma_R = self._greens(energy, potential)
        Gamma_L = self._gamma(Sigma_L)
        Gamma_R = self._gamma(Sigma_R)
        G_A = G_R.conjugate().T
        val = np.trace(Gamma_L @ G_R @ Gamma_R @ G_A)
        return float(np.real(val))

    def current(
        self,
        potential: Sequence[float] | float,
        mu_left: float,
        mu_right: float,
        transmission: Optional[np.ndarray] = None,
    ) -> float:
        if transmission is None:
            transmission = self.transmission(potential)
        else:
            transmission = np.asarray(transmission, dtype=float)
        f_left = 1.0 / (1.0 + np.exp((self.energy_grid - mu_left) / self.kbT))
        f_right = 1.0 / (1.0 + np.exp((self.energy_grid - mu_right) / self.kbT))
        integrand = transmission * (f_left - f_right)
        if self._nonuniform:
            integral = np.trapz(integrand, self.energy_grid)
        else:
            integral = integrand.sum() * self._uniform_dE
        prefactor = 2.0 * spc.elementary_charge / spc.h
        return prefactor * integral * spc.elementary_charge


@dataclass(slots=True)
class NEGFChargeProvider:
    """Bridge between NEGF densities and Poisson charge updates."""

    negf: OzakiNEGF
    q: float = spc.elementary_charge

    def density_from_lesser(self, potential: Sequence[float] | float, mu: float) -> np.ndarray:
        return self.negf.electron_density_lesser(potential, mu)

    def density_from_contacts(
        self,
        potential: Sequence[float] | float,
        mu_left: float,
        mu_right: float,
    ) -> np.ndarray:
        return self.negf.electron_density_non_equilibrium(potential, mu_left, mu_right)

    def density_from_integral(
        self,
        potential: Sequence[float] | float,
        mu: Sequence[float] | float,
        *,
        conduction_only: bool = True,
        Ec: Sequence[float] | float = 0.0,
    ) -> np.ndarray:
        return self.negf.electron_density_ozaki(
            potential,
            mu,
            conduction_only=conduction_only,
            Ec=Ec,
        )

    def charge_density(
        self,
        potential: Sequence[float] | float,
        mu_left: float,
        *,
        mu_right: Optional[float] = None,
        net_doping: Optional[Sequence[float] | float] = None,
        extra_charge: Optional[Sequence[float] | float] = None,
    ) -> np.ndarray:
        if mu_right is None:
            n = self.density_from_lesser(potential, mu_left)
        else:
            n = self.density_from_contacts(potential, mu_left, mu_right)
        rho = self.q * (-n)
        if net_doping is not None:
            dop = np.asarray(net_doping, dtype=float)
            if dop.ndim == 0 or dop.size == 1:
                dop_scalar = float(np.reshape(dop, (-1,))[0])
                dop = np.full(n.shape, dop_scalar)
            rho = rho + self.q * dop
        if extra_charge is not None:
            add = np.asarray(extra_charge, dtype=float)
            if add.ndim == 0 or add.size == 1:
                add_scalar = float(np.reshape(add, (-1,))[0])
                add = np.full(n.shape, add_scalar)
            rho = rho + add
        return rho


__all__ = [
    "Lead1D",
    "OzakiNEGF",
    "NEGFChargeProvider",
    "build_tridiagonal_hamiltonian",
    "effective_mass_1d_tb",
]
