
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from scipy.spatial import Delaunay
import scipy.constants as spc


E_CHARGE = 1.602176634e-19
EPS_0    = 8.854187817e-12
HBAR     = 1.054571817e-34
KB = spc.Boltzmann
T_KELVIN = 300.0
KT       = KB * T_KELVIN / E_CHARGE
V0       = KT


def alpha_from_lattice(a_lat_nm):
    a_m = a_lat_nm * 1e-9
    return E_CHARGE / (EPS_0 * KT * a_m)
 
 
def doping_per_site(N_cm3, a_lat_nm):
    a_cm = a_lat_nm * 1e-7
    return N_cm3 * a_cm**3


# ====================================================================== #
#  GEOMETRY                                                              #
# ====================================================================== #
def make_geometry(W_rows, L_slices, a_lat=0.5):
    n_per = W_rows
    n_atoms = L_slices * n_per
    pos = np.zeros((n_atoms, 2))
    slice_idx = np.zeros(n_atoms, dtype=int)
    for s in range(L_slices):
        i0 = s * n_per
        for r in range(n_per):
            pos[i0 + r] = (s * a_lat, r * a_lat)
        slice_idx[i0:i0 + n_per] = s
    return pos, slice_idx, n_per, a_lat


def make_doping(slice_idx, L_p, L_i, Nd_value, Na_value):
    n_atoms = len(slice_idx)
    Nd = np.zeros(n_atoms); Na = np.zeros(n_atoms)
    Na[slice_idx < L_p] = Na_value
    Nd[slice_idx >= L_p + L_i] = Nd_value
    return Nd, Na


