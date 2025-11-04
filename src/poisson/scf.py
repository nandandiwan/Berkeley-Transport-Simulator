"""Self-consistent Poisson–NEGF driver with Fermi-level refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .solver import NonlinearPoissonSolver, PoissonSolveResult
from .fermi import ChandrupatlaFermiSolver
from .negf_coupling import NEGFChargeProvider


def _ensure_vector(arr: Sequence[float] | np.ndarray | float, length: int) -> np.ndarray:
    vec = np.asarray(arr, dtype=float)
    if vec.size == 1:
        return np.full(length, float(vec.reshape(-1)[0]), dtype=float)
    if vec.size != length:
        raise ValueError(f"Expected array of length {length}, got {vec.size}")
    return vec.astype(float, copy=False)


@dataclass(slots=True)
class SCFIterationRecord:
    iteration: int
    delta_potential: float
    delta_fermi: float
    poisson_iterations: int
    poisson_residual: float


@dataclass(slots=True)
class SCFResult:
    converged: bool
    iterations: int
    potential: np.ndarray
    fermi_level: np.ndarray
    electron_density: np.ndarray
    poisson_result: PoissonSolveResult
    history: List[SCFIterationRecord]


class PoissonNEGFSCFSolver:
    """Orchestrate the outer SCF loop between Poisson and NEGF solvers."""

    def __init__(
        self,
        poisson_solver: NonlinearPoissonSolver,
        negf_provider: NEGFChargeProvider,
        fermi_solver: ChandrupatlaFermiSolver,
        *,
        net_doping: Sequence[float] | np.ndarray | float = 0.0,
        conduction_band_edge: Sequence[float] | np.ndarray | float = 0.0,
        valence_band_edge: Optional[Sequence[float] | np.ndarray | float] = None,
        potential_mixing: float = 1.0,
        tolerance: float = 1e-4,
        max_iterations: int = 30,
        enable_fermi_solver: bool = True,
        conduction_only: bool = True,
        density_scheme: str = "lesser",
    ) -> None:
        self.poisson = poisson_solver
        self.negf = negf_provider
        self.fermi_solver = fermi_solver
        ndofs = self.poisson.solution.x.array.size
        self.net_doping = _ensure_vector(net_doping, ndofs)
        self.conduction_band = _ensure_vector(conduction_band_edge, ndofs)
        self.valence_band = (
            _ensure_vector(valence_band_edge, ndofs)
            if valence_band_edge is not None
            else None
        )
        self.potential_mixing = float(potential_mixing)
        self.tolerance = float(tolerance)
        self.max_iterations = max_iterations
        self.enable_fermi_solver = enable_fermi_solver
        self.conduction_only = conduction_only
        valid_schemes = {"lesser", "integral"}
        if density_scheme not in valid_schemes:
            raise ValueError(f"density_scheme must be one of {valid_schemes}")
        self.density_scheme = density_scheme

        self._frozen_charge = np.zeros(ndofs, dtype=float)
        self.poisson.set_charge_callback(self._charge_callback)
        self._contact_fermi: Optional[tuple[float, float]] = None

    def _charge_callback(self, _u: np.ndarray, _coords: np.ndarray) -> np.ndarray:
        return self._frozen_charge

    def _compute_density(
        self,
        potential: Sequence[float] | np.ndarray | float,
        fermi_level: Sequence[float] | np.ndarray | float,
    ) -> np.ndarray:
        fermi_arr = _ensure_vector(fermi_level, self.poisson.solution.x.array.size)
        mu_scalar = float(np.mean(fermi_arr))
        if self._contact_fermi is not None:
            mu_left, mu_right = self._contact_fermi
            return self.negf.density_from_contacts(potential, mu_left, mu_right)
        if self.density_scheme == "lesser":
            return self.negf.density_from_lesser(potential, mu_scalar)
        return self.negf.density_from_integral(
            potential,
            fermi_arr,
            conduction_only=self.conduction_only,
            Ec=self.conduction_band,
        )

    def run(
        self,
        *,
        initial_potential: Optional[Sequence[float] | np.ndarray | float] = None,
        initial_fermi: Optional[Sequence[float] | np.ndarray | float] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        potential_mixing: Optional[float] = None,
        contact_fermi: Optional[tuple[float, float]] = None,
    ) -> SCFResult:
        ndofs = self.poisson.solution.x.array.size

        if initial_potential is not None:
            vec = _ensure_vector(initial_potential, ndofs)
            self.poisson.solution.x.array[:] = vec

        potential = self.poisson.solution.x.array.copy()

        if initial_fermi is not None:
            fermi = _ensure_vector(initial_fermi, ndofs)
        else:
            fermi = self.fermi_solver.estimator.estimate(
                self.net_doping,
                self.conduction_band,
                Ev=self.valence_band,
            )

        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        tol = tolerance if tolerance is not None else self.tolerance
        mixing = potential_mixing if potential_mixing is not None else self.potential_mixing
        self._contact_fermi = contact_fermi

        history: List[SCFIterationRecord] = []
        converged = False
        poisson_result: Optional[PoissonSolveResult] = None
        electron_density = np.zeros_like(potential)

        for iteration in range(1, max_iter + 1):
            fermi_prev = fermi.copy()

            electron_density = self._compute_density(potential, fermi)
            target_density = electron_density

            if self.enable_fermi_solver:

                def density_operator(Efn: np.ndarray, pot: np.ndarray) -> np.ndarray:
                    return self.negf.density_from_integral(
                        pot,
                        Efn,
                        conduction_only=self.conduction_only,
                        Ec=self.conduction_band,
                    )

                fermi = self.fermi_solver.solve(
                    density_operator,
                    target_density=target_density,
                    Ec=self.conduction_band,
                    net_doping=self.net_doping,
                    Ev=self.valence_band,
                    initial_guess=fermi,
                    extra_args=(potential,),
                )

            electron_density = self._compute_density(potential, fermi)
            if self._contact_fermi is not None:
                mu_left, mu_right = self._contact_fermi
                charge_density = self.negf.charge_density(
                    potential,
                    mu_left,
                    mu_right=mu_right,
                    net_doping=self.net_doping,
                )
            else:
                charge_density = self.negf.charge_density(
                    potential,
                    float(np.mean(fermi)),
                    net_doping=self.net_doping,
                )

            self._frozen_charge = charge_density

            poisson_result = self.poisson.solve(initial_guess=potential)

            potential_new = self.poisson.solution.x.array.copy()
            if mixing < 1.0:
                potential_new = mixing * potential_new + (1.0 - mixing) * potential
                self.poisson.solution.x.array[:] = potential_new

            delta_potential = np.linalg.norm(potential_new - potential) / np.sqrt(ndofs)
            delta_fermi = np.linalg.norm(fermi - fermi_prev) / np.sqrt(ndofs)

            history.append(
                SCFIterationRecord(
                    iteration=iteration,
                    delta_potential=delta_potential,
                    delta_fermi=delta_fermi,
                    poisson_iterations=poisson_result.iterations,
                    poisson_residual=poisson_result.residual_norm,
                )
            )

            potential = potential_new

            if delta_potential < tol and delta_fermi < tol:
                converged = True
                break

        if poisson_result is None:
            poisson_result = PoissonSolveResult(self.poisson.solution, False, 0, float("inf"))

        electron_density = self._compute_density(self.poisson.solution.x.array, fermi)
        self._contact_fermi = None

        return SCFResult(
            converged=converged,
            iterations=len(history),
            potential=self.poisson.solution.x.array.copy(),
            fermi_level=fermi,
            electron_density=electron_density,
            poisson_result=poisson_result,
            history=history,
        )


__all__ = [
    "PoissonNEGFSCFSolver",
    "SCFIterationRecord",
    "SCFResult",
]
