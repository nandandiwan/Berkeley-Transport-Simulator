from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector

VACUUM_PERMITTIVITY = 8.854187817e-12 


ChargeCallback = Callable[[np.ndarray, np.ndarray], np.ndarray]
BackgroundCharge = np.ndarray | float | Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DirichletBCSpec:
	"""Specification for a Dirichlet boundary condition on the scalar space."""

	value: float | Callable[[np.ndarray], np.ndarray] | fem.Function
	marker: Optional[Callable[[np.ndarray], np.ndarray]] = None
	dofs: Optional[np.ndarray] = None

	def instantiate(
		self,
		V: fem.FunctionSpace,
	) -> tuple[fem.DirichletBC, fem.Function, np.ndarray]:
		"""Create a dolfinx DirichletBC and return it with the backing function."""

		if self.dofs is not None:
			dof_idx = np.asarray(self.dofs, dtype=np.int32)
		else:
			if self.marker is None:
				raise ValueError("DirichletBCSpec requires either dofs or marker")
			fdim = V.mesh.topology.dim - 1
			facets = mesh.locate_entities_boundary(V.mesh, fdim, self.marker)
			dof_idx = fem.locate_dofs_topological(V, fdim, facets)

		if isinstance(self.value, fem.Function):
			backing = self.value
		else:
			backing = fem.Function(V)
			if callable(self.value):
				backing.interpolate(self.value)
			else:
				backing.interpolate(lambda X: np.full(X.shape[1], float(self.value)))

		bc = fem.dirichletbc(backing, dof_idx)
		return bc, backing, dof_idx


@dataclass(slots=True)
class PoissonSolveResult:
	potential: fem.Function
	converged: bool
	iterations: int
	residual_norm: float


class NonlinearPoissonSolver:
	"""Matrix-free SNES wrapper for the nonlinear Poisson equation."""

	def __init__(
		self,
		*,
		domain: mesh.Mesh,
		permittivity: ufl.core.expr.Expr | float,
		dirichlet_bcs: Sequence[DirichletBCSpec],
		degree: int = 1,
		eps0: float = VACUUM_PERMITTIVITY,
		charge_callback: Optional[ChargeCallback] = None,
		background_charge: Optional[BackgroundCharge] = None,
		snes_options_prefix: str = "poisson_",
	) -> None:
		self.mesh = domain
		self.comm: MPI.Intracomm = domain.comm
		self.V = fem.functionspace(domain, ("Lagrange", degree))
		self.eps_r = permittivity if isinstance(permittivity, ufl.core.expr.Expr) else ufl.as_ufl(permittivity)
		self.eps0 = float(eps0)
		self._charge_callback: ChargeCallback = (
			charge_callback if charge_callback is not None else lambda u, xy: np.zeros_like(u)
		)
		self.solution = fem.Function(self.V, name="potential")
		self.rho_fn = fem.Function(self.V, name="charge_density")

		self._dof_coords = self._tabulate_dof_coords()
		self._background_charge = background_charge
		self._background_values = self._evaluate_background(background_charge)

		u = ufl.TrialFunction(self.V)
		v = ufl.TestFunction(self.V)
		self._a_form = fem.form(ufl.inner(self.eps_r * ufl.grad(u), ufl.grad(v)) * ufl.dx)
		self._b_form = fem.form((self.rho_fn / self.eps0) * v * ufl.dx)

		self.K = assemble_matrix(self._a_form)
		self.K.assemble()
		self.R = self.K.createVecRight()
		self.X = self.K.createVecRight()
		self.b_vec = create_vector(self._b_form)

		self._bc_functions: list[fem.Function] = []
		self._dirichlet_sets: list[np.ndarray] = []
		self.bcs: list[fem.DirichletBC] = []
		for spec in dirichlet_bcs:
			bc, backing, dofs = spec.instantiate(self.V)
			self.bcs.append(bc)
			self._bc_functions.append(backing)
			self._dirichlet_sets.append(np.asarray(dofs, dtype=np.int32))

		self.P = self.K.copy()
		self._apply_dirichlet_identity(self.P)
		self.P.assemble()

		self._snes: Optional[PETSc.SNES] = None
		self._snes_prefix = snes_options_prefix

	def _tabulate_dof_coords(self) -> np.ndarray:
		coords = self.V.tabulate_dof_coordinates()
		if coords.ndim == 1:
			coords = coords.reshape((-1, self.mesh.geometry.dim))
		return coords[:, : self.mesh.geometry.dim]

	def set_charge_callback(self, callback: ChargeCallback) -> None:
		self._charge_callback = callback

	def set_background_charge(self, background: Optional[BackgroundCharge]) -> None:
		self._background_charge = background
		self._background_values = self._evaluate_background(background)

	def _evaluate_background(
		self,
		background: Optional[BackgroundCharge],
	) -> Optional[np.ndarray]:
		if background is None:
			return None
		n = self.solution.x.array.size
		if callable(background):
			values = np.asarray(background(self._dof_coords), dtype=float)
		else:
			values = np.asarray(background, dtype=float)
		if values.size == 1:
			return np.full(n, float(values), dtype=float)
		if values.size == n:
			return values.astype(float, copy=False)
		raise ValueError("Background charge array has incompatible size")

	def update_initial_potential(self, initializer: float | Callable[[np.ndarray], np.ndarray]) -> None:
		if callable(initializer):
			self.solution.interpolate(initializer)
		else:
			self.solution.interpolate(lambda X: np.full(X.shape[1], float(initializer)))

	def _ensure_snes(self) -> PETSc.SNES:
		if self._snes is not None:
			return self._snes
		snes = PETSc.SNES().create(self.comm)
		snes.setFunction(self._residual_callback, self.R)
		snes.setJacobian(None, P=self.P)
		snes.setUseKSP(True)
		snes.setUseMF(True)
		snes.setOptionsPrefix(self._snes_prefix)
		ksp = snes.getKSP()
		ksp.setType("gmres")
		ksp.getPC().setType("hypre")
		snes.getLineSearch().setType("bt")
		snes.setTolerances(rtol=1e-8, atol=1e-10, max_it=50)
		snes.setFromOptions()
		self._snes = snes
		return snes

	def _apply_dirichlet_identity(self, mat: PETSc.Mat) -> None:
		im = self.V.dofmap.index_map
		owned = im.size_local
		owned_dofs: list[np.ndarray] = []
		for dofs in self._dirichlet_sets:
			if dofs.size == 0:
				continue
			owned_mask = dofs < owned
			if np.any(owned_mask):
				owned_dofs.append(dofs[owned_mask].astype(np.int32, copy=False))
		if not owned_dofs:
			return
		owned_concat = np.unique(np.concatenate(owned_dofs))
		mat.zeroRowsLocal(owned_concat, 1.0)

	def _apply_dirichlet_residual(self, residual: PETSc.Vec, X_vec: PETSc.Vec) -> None:
		with residual.localForm() as Rl, X_vec.localForm() as Xl:
			arr_r = Rl.array
			arr_x = Xl.array_r
			for dofs, backing in zip(self._dirichlet_sets, self._bc_functions):
				if dofs.size == 0:
					continue
				target = backing.x.array
				arr_r[dofs] = arr_x[dofs] - target[dofs]

	def _update_charge_density(self) -> None:
		u_vals = self.solution.x.array.copy()
		rho_vals = np.asarray(self._charge_callback(u_vals, self._dof_coords), dtype=float)
		if rho_vals.shape[0] != u_vals.shape[0]:
			raise ValueError("Charge callback must return an array matching the number of DOFs")
		if self._background_values is not None:
			rho_vals = rho_vals + self._background_values
		self.rho_fn.x.array[:] = rho_vals

	def _residual_callback(self, snes: PETSc.SNES, X_in: PETSc.Vec, F_out: PETSc.Vec) -> int:
		with X_in.localForm() as Xl:
			x_loc = Xl.array_r
		with self.solution.x.petsc_vec.localForm() as Ul:
			Ul.array[:] = x_loc
		self.solution.x.scatter_forward()
		self._update_charge_density()
		self.b_vec.zeroEntries()
		assemble_vector(self.b_vec, self._b_form)
		self.b_vec.assemble()
		self.K.mult(X_in, F_out)
		F_out.axpy(-1.0, self.b_vec)
		self._apply_dirichlet_residual(F_out, X_in)
		return 0

	def solve(
		self,
		*,
		initial_guess: Optional[float | np.ndarray | Callable[[np.ndarray], np.ndarray]] = None,
		max_it: Optional[int] = None,
		rtol: Optional[float] = None,
		atol: Optional[float] = None,
		monitor: Optional[Callable[[int, float], None]] = None,
	) -> PoissonSolveResult:
		if initial_guess is not None:
			if callable(initial_guess):
				self.solution.interpolate(initial_guess)
			else:
				arr = np.asarray(initial_guess, dtype=float)
				if arr.size == 1:
					self.solution.x.array[:] = float(arr[0])
				elif arr.size == self.solution.x.array.size:
					self.solution.x.array[:] = arr
				else:
					raise ValueError("Initial guess size mismatch")

		with self.solution.x.petsc_vec.localForm() as Ul, self.X.localForm() as Xl:
			Xl.array[:] = Ul.array_r

		snes = self._ensure_snes()
		if max_it is not None or rtol is not None or atol is not None:
			snes.setTolerances(
				max_it=max_it if max_it is not None else snes.getTolerances()[2],
				rtol=rtol if rtol is not None else snes.getTolerances()[0],
				atol=atol if atol is not None else snes.getTolerances()[1],
			)

		if monitor is not None:
			snes.setMonitor(lambda snes, its, rnorm: monitor(its, rnorm))

		snes.solve(None, self.X)

		with self.X.localForm() as Xl, self.solution.x.petsc_vec.localForm() as Ul:
			Ul.array[:] = Xl.array_r
		self.solution.x.scatter_forward()
		self._update_charge_density()

		iterations = snes.getIterationNumber()
		converged = snes.getConvergedReason() > 0
		residual_norm = snes.getFunctionNorm()
		return PoissonSolveResult(self.solution, converged, iterations, residual_norm)


__all__ = ["DirichletBCSpec", "NonlinearPoissonSolver", "PoissonSolveResult", "VACUUM_PERMITTIVITY"]
