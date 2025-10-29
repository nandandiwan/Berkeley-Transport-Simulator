"""Fermi-level estimation utilities used for Poisson/NEGF coupling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.constants as spc
from scipy.optimize import brentq

try:
	from .utils import chandrupatla
except ImportError:  # Fallback if package layout differs (e.g., direct script execution)
	chandrupatla = None  # type: ignore


KB_OVER_Q = spc.Boltzmann / spc.elementary_charge


def _effective_density_of_states(m_eff_rel: float, temperature: float) -> float:
	"""Return Nc or Nv (per m^3) for a parabolic band with mass m_eff_rel * m0."""

	if m_eff_rel <= 0:
		raise ValueError("Effective mass must be positive")
	m_eff = m_eff_rel * spc.m_e
	prefactor = 2.0 * ((2.0 * np.pi * m_eff * spc.Boltzmann * temperature) / (spc.h ** 2)) ** 1.5
	return float(prefactor)


@dataclass(slots=True)
class EffectiveMassFermiEstimator:
	"""Boltzmann/degenerate hybrid estimator for the electron quasi-Fermi level.

	Parameters
	----------
	m_effective_e : float
		Conduction-band effective mass (relative to the electron rest mass).
	temperature : float, optional
		Lattice temperature in Kelvin.
	m_effective_h : float, optional
		Valence-band effective mass (relative to the electron rest mass). Required
		if p-type regions need a valence-based estimate.
	min_density : float, optional
		Floor applied to densities (m^-3) when evaluating logarithms.
	"""

	m_effective_e: float
	temperature: float = 300.0
	m_effective_h: Optional[float] = None
	min_density: float = 1e6
	kBT: float = field(init=False)
	Nc: float = field(init=False)
	Nv: Optional[float] = field(init=False, default=None)

	def __post_init__(self) -> None:
		self.kBT = KB_OVER_Q * self.temperature
		self.Nc = _effective_density_of_states(self.m_effective_e, self.temperature)
		self.Nv = (
			_effective_density_of_states(self.m_effective_h, self.temperature)
			if self.m_effective_h is not None
			else None
		)

	def estimate(
		self,
		net_doping: Sequence[float] | np.ndarray,
		Ec: Sequence[float] | np.ndarray | float,
		*,
		Ev: Optional[Sequence[float] | np.ndarray | float] = None,
	) -> np.ndarray:
		"""Return an initial quasi-Fermi level array (eV) for the given doping.

		Positive ``net_doping`` denotes donor-like regions; negative values denote
		acceptor-like regions.  ``Ec`` and ``Ev`` may be scalar or nodal arrays.
		"""

		net = np.asarray(net_doping, dtype=float)
		Ec_arr = np.asarray(Ec, dtype=float)
		if Ec_arr.ndim == 0 or Ec_arr.size == 1:
			Ec_scalar = float(np.reshape(Ec_arr, (-1,))[0])
			Ec_arr = np.full(net.shape, Ec_scalar)
		if Ec_arr.shape != net.shape:
			Ec_arr = np.broadcast_to(Ec_arr, net.shape)

		result = np.empty_like(net, dtype=float)
		floor = max(self.min_density, 1.0)

		donor_mask = net >= 0
		if np.any(donor_mask):
			n = np.maximum(net[donor_mask], floor)
			result[donor_mask] = Ec_arr[donor_mask] + self.kBT * np.log(n / self.Nc)

		acceptor_mask = ~donor_mask
		if np.any(acceptor_mask):
			if self.Nv is None or Ev is None:
				# Fall back to mid-gap estimate if valence data unavailable.
				gap_shift = 0.5 * (Ec_arr[acceptor_mask] - np.min(Ec_arr))
				result[acceptor_mask] = Ec_arr[acceptor_mask] - gap_shift
			else:
				Ev_arr = np.asarray(Ev, dtype=float)
				if Ev_arr.ndim == 0 or Ev_arr.size == 1:
					Ev_scalar = float(np.reshape(Ev_arr, (-1,))[0])
					Ev_arr = np.full(net.shape, Ev_scalar)
				if Ev_arr.shape != net.shape:
					Ev_arr = np.broadcast_to(Ev_arr, net.shape)
				p = np.maximum(-net[acceptor_mask], floor)
				result[acceptor_mask] = Ev_arr[acceptor_mask] - self.kBT * np.log(p / self.Nv)

		return result


@dataclass(slots=True)
class ChandrupatlaFermiSolver:
	"""Vectorised Chandrupatla root finder for quasi-Fermi corrections."""

	estimator: EffectiveMassFermiEstimator
	bracket_width: float = 0.25
	rtol: float = 1e-6
	atol: float = 1e-10
	max_iters: int = 60

	def solve(
		self,
		density_operator: Callable[..., np.ndarray],
		*,
		target_density: Sequence[float] | np.ndarray,
		Ec: Sequence[float] | np.ndarray | float,
		net_doping: Optional[Sequence[float] | np.ndarray] = None,
		Ev: Optional[Sequence[float] | np.ndarray | float] = None,
		initial_guess: Optional[Sequence[float] | np.ndarray] = None,
		extra_args: tuple = (),
	) -> np.ndarray:
		"""Solve for the quasi-Fermi level by matching a density operator to targets.

		Parameters
		----------
		density_operator : callable
			Function mapping a quasi-Fermi level array to a carrier density array.
		target_density : array_like
			Desired carrier density derived from the lesser Green's function (or
			another reference).
		Ec : array_like or float
			Conduction-band edge used by the estimator when an initial guess is
			required.
		net_doping : array_like, optional
			Net doping profile supplied to the estimator when ``initial_guess`` is
			omitted.
		Ev : array_like or float, optional
			Valence band edge for p-type estimation.
		initial_guess : array_like, optional
			Explicit starting values for the quasi-Fermi level.
		extra_args : tuple, optional
			Additional arguments forwarded to ``density_operator``.
		"""

		target = np.asarray(target_density, dtype=float)
		if initial_guess is None:
			if net_doping is None:
				raise ValueError("net_doping must be provided when initial_guess is None")
			guess = self.estimator.estimate(net_doping, Ec, Ev=Ev)
		else:
			guess = np.asarray(initial_guess, dtype=float)
		lower = guess - self.bracket_width
		upper = guess + self.bracket_width

		def residual(Efn: np.ndarray, *args) -> np.ndarray:
			return density_operator(Efn, *args) - target

		if chandrupatla is not None:
			roots = chandrupatla(
				residual,
				lower,
				upper,
				rtol=self.rtol,
				atol=self.atol,
				maxiter=self.max_iters,
				args=extra_args,
			)
			return np.asarray(roots, dtype=float)

		roots = np.empty_like(guess)

		def scalar_func_factory(index: tuple[int, ...]):
			def scalar_func(x: float) -> float:
				probe = guess.copy()
				probe[index] = x
				return float(residual(probe, *extra_args)[index])

			return scalar_func

		it = np.nditer(guess, flags=["multi_index"])
		for _ in it:
			idx = it.multi_index
			f_scalar = scalar_func_factory(idx)
			lower_i = float(lower[idx])
			upper_i = float(upper[idx])
			f_lower = f_scalar(lower_i)
			f_upper = f_scalar(upper_i)
			if f_lower == 0.0:
				roots[idx] = lower_i
				continue
			if f_upper == 0.0:
				roots[idx] = upper_i
				continue
			expansions = 0
			while f_lower * f_upper > 0 and expansions < 6:
				width = upper_i - lower_i
				lower_i -= width
				upper_i += width
				f_lower = f_scalar(lower_i)
				f_upper = f_scalar(upper_i)
				expansions += 1
			if f_lower * f_upper > 0:
				raise RuntimeError(
					"Failed to bracket quasi-Fermi root; residual retains sign even after expansions"
				)
			roots[idx] = brentq(
				f_scalar,
				lower_i,
				upper_i,
				extol=self.atol,
				rtol=self.rtol,
				maxiter=self.max_iters,
			)

		return roots


__all__ = [
	"EffectiveMassFermiEstimator",
	"ChandrupatlaFermiSolver",
]
