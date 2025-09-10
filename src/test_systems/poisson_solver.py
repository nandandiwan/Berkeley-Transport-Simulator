from __future__ import annotations
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from hamiltonian import Hamiltonian  
from rgf import GreensFunction  
import utils


q = 1.602e-19        
k_B = 1.380e-23        
T = 300.0        
epsilon_0 = 8.854e-12 
epsilon_r = 11.7       
eps = epsilon_r * epsilon_0
n_i = 1.0e16        
N_D = 1.0e21        
thermal_voltage = k_B * T / q 


def build_interval_space(mesh_size: int, device_length: float):
    """Create 1D interval mesh and linear Lagrange function space."""
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, mesh_size, [0.0, device_length])
    V_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    return mesh, V_space


class PoissonBoltzmannProblem(dolfinx.fem.petsc.NonlinearProblem):
    """Nonlinear Poisson with Boltzmann electron density (analytic exponential)."""
    def __init__(self, F, J, u, rho, drho_dv, bcs):
        super().__init__(F, u, bcs=bcs, J=J)
        self._rho = rho
        self._drho_dv = drho_dv
        self._u = u

    def F(self, x: PETSc.Vec, b: PETSc.Vec):  # type: ignore[override]
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self._u.x.petsc_vec)
        self._u.x.scatter_forward()
        Vvals = self._u.x.array  # Potential in volts
        # Charge density rho = q (N_D - n_i exp(V / V_T)) (C/m^3)
        self._rho.x.array[:] = q * (N_D - n_i * np.exp(Vvals / thermal_voltage))
        self._rho.x.scatter_forward()
        super().F(x, b)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):  # type: ignore[override]
        Vvals = self._u.x.array
        # d rho / d V = - q * (n_i / V_T) * exp(V / V_T)
        self._drho_dv.x.array[:] = -q * (n_i / thermal_voltage) * np.exp(Vvals / thermal_voltage)
        self._drho_dv.x.scatter_forward()
        super().J(x, A)


def solve_poisson_boltzmann(mesh_size: int = 200, device_length: float = 100e-9,
                             V_left: float = 0.25, V_right: float = -0.25,
                             rtol: float = 1e-8, max_it: int = 50,
                             return_fields: bool = False):
    """Solve 1D nonlinear Poisson (Boltzmann) and optionally plot.

    Potentials are in volts. Boundary values are Dirichlet.
    """
    mesh, V_space = build_interval_space(mesh_size, device_length)
    Vh = fem.Function(V_space, name="Potential")
    rho_h = fem.Function(V_space, name="ChargeDensity")
    d_rho_dV_h = fem.Function(V_space, name="ChargeDensityDerivative")

    v = ufl.TestFunction(V_space)
    F_form = eps * ufl.dot(ufl.grad(Vh), ufl.grad(v)) * ufl.dx - rho_h * v * ufl.dx
    J_form = ufl.derivative(F_form, Vh, ufl.TrialFunction(V_space))

    # Boundary conditions
    fdim = mesh.topology.dim - 1
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], device_length))
    left_dofs = fem.locate_dofs_topological(V_space, fdim, left_facets)
    right_dofs = fem.locate_dofs_topological(V_space, fdim, right_facets)
    bc_left = fem.dirichletbc(PETSc.ScalarType(V_left), left_dofs, V_space)
    bc_right = fem.dirichletbc(PETSc.ScalarType(V_right), right_dofs, V_space)
    bcs = [bc_left, bc_right]

    problem = PoissonBoltzmannProblem(F_form, J_form, Vh, rho_h, d_rho_dV_h, bcs)
    solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = rtol
    solver.max_it = max_it
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    opts[f"{prefix}ksp_type"] = "preonly"
    opts[f"{prefix}pc_type"] = "lu"
    opts["snes_linesearch_type"] = "basic"
    ksp.setFromOptions()

    # Linear initial guess
    Vh.interpolate(lambda x: V_left + (V_right - V_left) * x[0] / device_length)
    its, converged = solver.solve(Vh)
    if MPI.COMM_WORLD.rank == 0:
        msg = f"Boltzmann Poisson converged in {its} iterations." if converged else f"Boltzmann Poisson NOT converged (iters={its})."
        print(msg)
    if return_fields:
        return mesh, V_space, Vh, rho_h
    if MPI.COMM_WORLD.rank == 0:
        _plot_poisson_solution(V_space, Vh, rho_h, device_length, title_prefix="Boltzmann")
    return Vh



def _plot_poisson_solution(V_space, Vh, rho_h, device_length, title_prefix=""):
    """Utility plotting (rank 0 only)."""
    x_coords = V_space.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(x_coords)
    x_sorted = x_coords[order]
    V_sorted = Vh.x.array[order]
    rho_sorted = rho_h.x.array[order]
    linear_ref = V_sorted[0] + (V_sorted[-1] - V_sorted[0]) * (x_sorted - x_sorted[0]) / (x_sorted[-1] - x_sorted[0])
    diff = V_sorted - linear_ref
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    ax1.plot(x_sorted * 1e9, V_sorted, lw=2)
    ax1.set_title(f"{title_prefix} Potential (V)")
    ax1.set_xlabel("x (nm)")
    ax1.grid(True, ls=':')
    ax2.plot(x_sorted * 1e9, rho_sorted, lw=2, color='tab:red')
    ax2.set_title("Charge density (C/m^3)")
    ax2.set_xlabel("x (nm)")
    ax2.grid(True, ls=':')
    ax3.plot(x_sorted * 1e9, diff, lw=2, color='tab:green')
    ax3.set_title("V - linear(x)")
    ax3.set_xlabel("x (nm)")
    ax3.grid(True, ls=':')
    fig.tight_layout()
    plt.show()


# ----------------------------- CLI / Example ----------------------------- #
class PoissonNEGFProblem(dolfinx.fem.petsc.NonlinearProblem):
    """Nonlinear Poisson problem where charge density from NEGF GF depends on potential.

    NOTE: As in the Boltzmann implementation, the Jacobian assembled via dolfinx.derivative
    does not include d(rho)/dV; we approximate Newton by updating rho at each nonlinear
    iteration (a quasi-Newton / Picard scheme). For improved convergence one could build
    a custom UFL form including an approximation to drho/dV using self._drho_dv.
    """
    def __init__(self, F, J, u, rho, drho_dv, bcs, gf: GreensFunction, ham: Hamiltonian,
                 site_positions: np.ndarray, dof_x: np.ndarray, Efn: float, Ec: float):
        super().__init__(F, u, bcs=bcs, J=J)
        self._rho = rho
        self._drho_dv = drho_dv
        self._u = u
        self.gf = gf
        self.ham = ham
        self.site_positions = site_positions
        self.dof_x = dof_x
        self.Efn = Efn
        self.Ec = Ec

    def update_Efn(self, Efn: float):
        self.Efn = Efn

    def _map_nodes_to_sites(self, V_nodes: np.ndarray) -> np.ndarray:
        return np.interp(self.site_positions, self.dof_x, V_nodes)

    def _map_sites_to_nodes(self, arr_sites: np.ndarray) -> np.ndarray:
        
        return np.interp(self.dof_x, self.site_positions, arr_sites)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):  # type: ignore[override]
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self._u.x.petsc_vec)
        self._u.x.scatter_forward()
        V_nodes = self._u.x.array
        V_sites = self._map_nodes_to_sites(V_nodes)
        # Compute electron density (#/m^3) from NEGF
        
        n_sites = self.gf.get_n(V=V_sites, Efn=self.Efn, Ec=self.Ec, processes=1)
        
        n_nodes = self._map_sites_to_nodes(n_sites)
        rho_nodes = q * (- n_nodes)*1e23
        self._rho.x.array[:] = rho_nodes
        self._rho.x.scatter_forward()
        super().F(x, b)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):  # type: ignore[override]
        # Optional: could incorporate approximate derivative; currently placeholder
        V_nodes = self._u.x.array
        V_sites = self._map_nodes_to_sites(V_nodes)
        
        dn_dV_sites = self.gf.diff_rho_poisson(Efn=self.Efn, V=V_sites, Ec=self.Ec, processes=1)

        # Charge derivative: d rho / d V = - q * dn/dV
        drho_dV_nodes = -q * self._map_sites_to_nodes(dn_dV_sites)*1e23
        self._drho_dv.x.array[:] = drho_dV_nodes
        self._drho_dv.x.scatter_forward()
        super().J(x, A)


def _compute_fermi_energy(gf: GreensFunction, V_sites: np.ndarray, Ec: float) -> float:
    try:
        fe = gf.fermi_energy
        if callable(fe):
            val = fe(V_sites, Ec=Ec)
        else:
            val = fe
        # Coerce to scalar float (handles ndarray / list / 0-d array)
        val = np.asarray(val)
        if val.ndim > 0:
            val = val.ravel()[0]
        return float(val)
    except Exception:
        return 0.0


def solve_poisson_negf_nonlinear(gf: GreensFunction, ham: Hamiltonian, mesh_size: int = 400,
                                 device_length: float = 100e-9, V_left: float = 0.25,
                                 V_right: float = -0.25, Ec: float = -2.0,
                                 newton_rtol: float = 1e-6, newton_max_it: int = 30,
                                 scf_max: int = 50, scf_tol: float = 1e-4,
                                 mix_alpha: float = 0.0,  # optional linear mixing of Newton output
                                 verbose: bool = True, plot: bool = True,
                                 return_history: bool = False):
    """Outer SCF updating Fermi level; inner Newton solves nonlinear Poisson with NEGF density.

    Parameters:
      mix_alpha: if >0, apply linear mixing between previous and Newton solution (stability aid).
    """
    mesh, V_space = build_interval_space(mesh_size, device_length)
    Vh = fem.Function(V_space, name="Potential")
    rho_h = fem.Function(V_space, name="ChargeDensity")
    d_rho_dV_h = fem.Function(V_space, name="ChargeDensityDerivative")
    Vh.interpolate(lambda x: V_left + (V_right - V_left) * x[0] / device_length)
    dof_x = V_space.tabulate_dof_coordinates()[:, 0]
    site_positions = np.linspace(0.0, device_length, ham.N)
    V_sites = np.interp(site_positions, dof_x, Vh.x.array)
    Efn = _compute_fermi_energy(gf, V_sites, Ec)

    v = ufl.TestFunction(V_space)
    F_form = eps * ufl.dot(ufl.grad(Vh), ufl.grad(v)) * ufl.dx - rho_h * v * ufl.dx
    J_form = ufl.derivative(F_form, Vh, ufl.TrialFunction(V_space))

    # Dirichlet BCs
    fdim = mesh.topology.dim - 1
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], device_length))
    left_dofs = fem.locate_dofs_topological(V_space, fdim, left_facets)
    right_dofs = fem.locate_dofs_topological(V_space, fdim, right_facets)
    bc_left = fem.dirichletbc(PETSc.ScalarType(V_left), left_dofs, V_space)
    bc_right = fem.dirichletbc(PETSc.ScalarType(V_right), right_dofs, V_space)
    bcs = [bc_left, bc_right]

    problem = PoissonNEGFProblem(F_form, J_form, Vh, rho_h, d_rho_dV_h, bcs,
                                 gf=gf, ham=ham, site_positions=site_positions,
                                 dof_x=dof_x, Efn=Efn, Ec=Ec)
    newton = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
    newton.convergence_criterion = "incremental"
    newton.rtol = newton_rtol
    newton.max_it = newton_max_it
    ksp = newton.krylov_solver
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    opts[f"{prefix}ksp_type"] = "preonly"
    opts[f"{prefix}pc_type"] = "lu"
    opts["snes_linesearch_type"] = "basic"
    ksp.setFromOptions()

    history = []
    Efn_history = []
    V_prev = Vh.x.array.copy()
    Efn_prev = Efn
    for scf in range(1, scf_max + 1):
        # Inner Newton solve at fixed Efn
        its, converged = newton.solve(Vh)
        if verbose and MPI.COMM_WORLD.rank == 0:
            print(f"SCF {scf:02d} Newton: iters={its}, converged={converged}")
        if mix_alpha > 0:
            Vh.x.array[:] = (1 - mix_alpha) * V_prev + mix_alpha * Vh.x.array
        # Map potential to sites and update Hamiltonian potential energy (negative sign for electrons)
        V_sites = np.interp(site_positions, dof_x, Vh.x.array)
        try:
            ham.set_potential(V_sites)
        except Exception:
            ham.set_potential(-V_sites)
        try:
            gf.clear_ldos_cache()
        except Exception:
            pass
        # Update Fermi level with new potential
        Efn = float(_compute_fermi_energy(gf, V_sites, Ec))
        problem.update_Efn(Efn)
        # Convergence metrics
        dV = float(np.max(np.abs(Vh.x.array - V_prev)))
        dE = float(abs(Efn - Efn_prev))
        history.append(dV)
        Efn_history.append(Efn)
        if verbose and MPI.COMM_WORLD.rank == 0:
            print(f"SCF {scf:02d}: max|dV|={dV:.3e}, |dEfn|={dE:.3e}")
        if dV < scf_tol and dE < max(scf_tol * 1e-1, 1e-6):
            if verbose and MPI.COMM_WORLD.rank == 0:
                print("NEGF nonlinear Poisson converged.")
            break
        V_prev = Vh.x.array.copy()
        Efn_prev = Efn
    else:
        if verbose and MPI.COMM_WORLD.rank == 0:
            print("WARNING: NEGF nonlinear Poisson did not fully converge.")

    if plot and MPI.COMM_WORLD.rank == 0:
        _plot_poisson_solution(V_space, Vh, rho_h, device_length, title_prefix="NEGF-Nonlinear")
        if len(history) > 1:
            plt.figure(figsize=(5, 3.2))
            plt.semilogy(history, marker='o')
            plt.xlabel('SCF iteration')
            plt.ylabel('max|dV| (V)')
            plt.title('NEGF Nonlinear Poisson SCF')
            plt.grid(True, which='both', ls=':')
            plt.tight_layout()
            plt.show()
    result = {
        'V_nodes': Vh.x.array.copy(),
        'x_nodes': dof_x.copy(),
        'Efn': Efn,
        'history': history,
        'Efn_history': Efn_history
    }
    return result if return_history else result['V_nodes']


def solve_poisson_negf(gf: GreensFunction, ham: Hamiltonian, **kwargs):
        """Wrapper implementing requested NEGF nonlinear Poisson SCF loop.

        Steps per outer iteration:
            1. Solve nonlinear Poisson with current Efn (inner Newton) -> V
            2. Set Hamiltonian potential
            3. Clear LDOS cache
            4. Recompute Efn via gf.fermi_energy
            5. Check convergence and repeat

        Delegates to solve_poisson_negf_nonlinear; kwargs passed through.
        """
        return solve_poisson_negf_nonlinear(gf, ham, **kwargs)


def _example_run(mode: str = "boltzmann"):
    m = mode.lower()
    if m not in {"boltzmann", "negf", "negf_nl"}:
        raise ValueError("mode must be 'boltzmann', 'negf', or 'negf_nl'")
    if m == "boltzmann":
        solve_poisson_boltzmann()
        return
    ham = Hamiltonian("one_d_wire")
    gf = GreensFunction(ham)
    if m == "negf":
        solve_poisson_negf(gf, ham)
    else:  # negf_nl
        solve_poisson_negf_nonlinear(gf, ham)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="1D Poisson solver (Boltzmann or NEGF self-consistent).")
    parser.add_argument("--mode", choices=["boltzmann", "negf", "negf_nl"], default="boltzmann")
    parser.add_argument("--mesh", type=int, default=200)
    parser.add_argument("--length", type=float, default=100e-9, help="Device length (m)")
    parser.add_argument("--V_left", type=float, default=0.25)
    parser.add_argument("--V_right", type=float, default=-0.25)
    parser.add_argument("--max_scf", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--mix", type=float, default=0.3)
    args = parser.parse_args()
    if args.mode == "boltzmann":
        solve_poisson_boltzmann(mesh_size=args.mesh, device_length=args.length,
                                 V_left=args.V_left, V_right=args.V_right)
    elif args.mode == "negf":
        ham = Hamiltonian("one_d_wire")
        gf = GreensFunction(ham)
        solve_poisson_negf(gf, ham, mesh_size=args.mesh, device_length=args.length,
                           V_left=args.V_left, V_right=args.V_right, max_scf=args.max_scf,
                           scf_tol=args.tol, mix_alpha=args.mix)
    else:  # negf_nl
        ham = Hamiltonian("one_d_wire")
        gf = GreensFunction(ham)
        solve_poisson_negf_nonlinear(gf, ham, mesh_size=args.mesh, device_length=args.length,
                                     V_left=args.V_left, V_right=args.V_right, scf_max=args.max_scf,
                                     scf_tol=args.tol, mix_alpha=args.mix)