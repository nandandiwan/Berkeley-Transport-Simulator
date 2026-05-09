import numpy as np
from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from scipy.spatial import Delaunay
import scipy.constants as spc
from negf_functions import KT, V0

# ====================================================================== #
#  MESH + DOF MAPS                                                       #
# ====================================================================== #
def build_mesh_from_pos(pos):
    tri = Delaunay(pos)
    cells = tri.simplices.astype(np.int64)
    element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = ufl.Mesh(element)
    return dmesh.create_mesh(MPI.COMM_WORLD, cells, domain, pos)


def build_dof_maps(V_space, pos):
    dof_coords = V_space.tabulate_dof_coordinates()[:, :2]
    n_atoms = len(pos)
    n_dofs = len(dof_coords)
    site_to_dof = np.zeros(n_atoms, dtype=int)
    dof_to_site = np.zeros(n_dofs, dtype=int)
    for i, p in enumerate(pos):
        d = np.argmin(np.sum((dof_coords - p)**2, axis=1))
        site_to_dof[i] = d
        dof_to_site[d] = i
    return site_to_dof, dof_to_site


# ====================================================================== #
#  POISSON FE                                                            #
# ====================================================================== #
class PoissonFE:
    def __init__(self, msh, V_space, site_to_dof, dof_to_site, pos, eps_x):
        self.msh = msh
        self.V = V_space
        self.s2d = site_to_dof
        self.d2s = dof_to_site

        self.phi          = fem.Function(V_space)
        self.phi_cand     = fem.Function(V_space)
        self.n_fn         = fem.Function(V_space)
        self.p_fn         = fem.Function(V_space)
        self.J_fn         = fem.Function(V_space)
        self.NdNa_fn      = fem.Function(V_space)

        x_min = pos[:, 0].min(); x_max = pos[:, 0].max()
        def left_edge(x):  return x[0] < x_min + eps_x
        def right_edge(x): return x[0] > x_max - eps_x
        self.left_dofs  = fem.locate_dofs_geometrical(V_space, left_edge)
        self.right_dofs = fem.locate_dofs_geometrical(V_space, right_edge)

    def make_problem(self, alpha, eps_r, BC_L, BC_R, prefix="poisson_"):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        a = (eps_r * ufl.inner(ufl.grad(u), ufl.grad(v))
             + alpha * self.J_fn * u * v) * ufl.dx
        L = (alpha * (self.NdNa_fn - self.n_fn + self.p_fn
                      + self.J_fn * self.phi) * v) * ufl.dx
        bc_L = fem.dirichletbc(PETSc.ScalarType(BC_L), self.left_dofs, self.V)
        bc_R = fem.dirichletbc(PETSc.ScalarType(BC_R), self.right_dofs, self.V)
        self.bcs = [bc_L, bc_R]
        self.problem = LinearProblem(
            a, L, u=self.phi_cand, bcs=self.bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix=prefix,
        )

    def set_inputs(self, n_site, p_site, J_site, NdNa_site):
        self.n_fn.x.array[:]    = n_site[self.d2s]
        self.p_fn.x.array[:]    = p_site[self.d2s]
        self.J_fn.x.array[:]    = J_site[self.d2s]
        self.NdNa_fn.x.array[:] = NdNa_site[self.d2s]

    def solve_step(self):
        self.problem.solve()
        return self.phi_cand.x.array.copy()

    def get_phi_site(self):
        return self.phi.x.array[self.s2d]

    def set_phi_site(self, phi_site):
        self.phi.x.array[:] = phi_site[self.d2s]
        fem.set_bc(self.phi.x.array, self.bcs)


# ====================================================================== #
#  ITERATION HELPERS                                                     #
# ====================================================================== #
def gummel_J(n_site, p_site):
    return (V0 / KT) * (n_site + p_site)


def anderson_step(phi_in, residual_F, history, mix):
    if len(history) >= 2:
        phi_hist = np.array([h[0] for h in history])
        res_hist = np.array([h[1] for h in history])
        dF = res_hist[1:] - res_hist[:-1]
        dphi = phi_hist[1:] - phi_hist[:-1]
        try:
            gamma, *_ = np.linalg.lstsq(dF.T, residual_F, rcond=None)
            return phi_in + mix * residual_F - (dphi.T + dF.T) @ gamma * mix
        except np.linalg.LinAlgError:
            pass
    return phi_in + mix * residual_F


def update_history(history, phi_in, residual_F, depth):
    history.append((phi_in.copy(), residual_F.copy()))
    if len(history) > depth:
        history.pop(0)


# ====================================================================== #
#  PER-SLICE HELPERS                                                     #
# ====================================================================== #
def to_slice(arr, slice_idx, L):
    return np.array([arr[slice_idx == s].mean() for s in range(L)])


def from_slice(arr_s, slice_idx):
    return arr_s[slice_idx]

