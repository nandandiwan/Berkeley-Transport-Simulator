"""Reusable Poisson/NEGF coupling utilities."""

try:
    from .solver import (
        DirichletBCSpec,
        NonlinearPoissonSolver,
        PoissonSolveResult,
        VACUUM_PERMITTIVITY,
    )
except ModuleNotFoundError:  # petsc4py/dolfinx optional when using NEGF-only utilities
    DirichletBCSpec = None  # type: ignore[assignment]
    NonlinearPoissonSolver = None  # type: ignore[assignment]
    PoissonSolveResult = None  # type: ignore[assignment]
    VACUUM_PERMITTIVITY = None  # type: ignore[assignment]

# from .fermi import EffectiveMassFermiEstimator, ChandrupatlaFermiSolver
# from .negf_coupling import (
#     Lead1D,
#     OzakiNEGF,
#     NEGFChargeProvider,
#     build_tridiagonal_hamiltonian,
#     effective_mass_1d_tb,
# )
# try:
#     from .scf import PoissonNEGFSCFSolver, SCFIterationRecord, SCFResult
# except ModuleNotFoundError:  # allow NEGF-only usage without PETSc
#     PoissonNEGFSCFSolver = None  # type: ignore[assignment]
#     SCFIterationRecord = None  # type: ignore[assignment]
#     SCFResult = None  # type: ignore[assignment]

__all__ = [
    # "DirichletBCSpec",
    # "NonlinearPoissonSolver",
    # "PoissonSolveResult",
    # "VACUUM_PERMITTIVITY",
    # "EffectiveMassFermiEstimator",
    # "ChandrupatlaFermiSolver",
    # "Lead1D",
    # "OzakiNEGF",
    # "NEGFChargeProvider",
    # "build_tridiagonal_hamiltonian",
    # "effective_mass_1d_tb",
    # "PoissonNEGFSCFSolver",
    # "SCFIterationRecord",
    # "SCFResult",
]
