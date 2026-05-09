from __future__ import annotations

import os
import sys
from pathlib import Path

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import scipy.constants as spc


for _cand in (
    "/usr/include/eigen3",
    "/usr/local/include/eigen3",
    str(Path.home() / ".local" / "eigen-src" / "eigen-3.4.0"),
):
    if os.path.isfile(f"{_cand}/Eigen/Core"):
        print('here')
        os.environ.setdefault("EIGEN_INCLUDE", _cand)
        break

_here = Path().resolve()
for base in [_here, *_here.parents]:
    if (base / "negf").is_dir():
        sys.path.insert(0, str(base))
        break
    if (base / "src" / "negf").is_dir():
        sys.path.insert(0, str(base / "src"))
        break
else:
    raise ModuleNotFoundError("Cannot locate the negf package.")

from negf.gf.recursive_greens_functions import _recursive_inverse as rgf

E_CHARGE = 1.602176634e-19
EPS_0    = 8.854187817e-12
HBAR     = 1.054571817e-34
KB       = spc.Boltzmann
T_KELVIN = 300.0
KT       = KB * T_KELVIN / E_CHARGE
V0       = KT
N_WORKERS = 32


# ====================================================================== #
#  LEAD SELF-ENERGY                                                      #
# ====================================================================== #
def sancho_rubio(z, H_slice, V_slice, max_iter=120, tol=1e-13):
    n = H_slice.shape[0]; I = np.eye(n, dtype=complex)
    H_eff = H_slice.astype(complex).copy()
    H_s = H_slice.astype(complex).copy()
    alpha = V_slice.astype(complex).copy()
    beta = V_slice.conj().T.astype(complex).copy()
    for _ in range(max_iter):
        g = np.linalg.solve(z*I - H_eff, I)
        ag = alpha @ g; bg = beta @ g
        H_s = H_s + ag @ beta
        H_eff = H_eff + ag @ beta + bg @ alpha
        alpha = ag @ alpha; beta = bg @ beta
        if max(np.max(np.abs(alpha)), np.max(np.abs(beta))) < tol:
            break
    return np.linalg.solve(z*I - H_s, I)


def sigma_LR(E, V_L, V_R, phi_p, phi_n, H_slice, V_slice, eta):
    n = H_slice.shape[0]
    H_lp = H_slice - phi_p * np.eye(n)
    H_ln = H_slice - phi_n * np.eye(n)
    g_L = sancho_rubio(E + V_L + 1j*eta, H_lp, V_slice.conj().T)  # was V_slice
    g_R = sancho_rubio(E + V_R + 1j*eta, H_ln, V_slice)            # was V_slice.conj().T
    Sig_L = V_slice.conj().T @ g_L @ V_slice
    Sig_R = V_slice @ g_R @ V_slice.conj().T
    return Sig_L, Sig_R

def cache_self_energies(E_grid, V_L, V_R, phi_p, phi_n,
                          H_slice, V_slice, eta, verbose=True):
    n_E = len(E_grid)
    n_per = H_slice.shape[0]
    SL = np.zeros((n_E, n_per, n_per), dtype=complex)
    SR = np.zeros((n_E, n_per, n_per), dtype=complex)
    t0 = time.time()
    for k, E in enumerate(E_grid):
        SL[k], SR[k] = sigma_LR(E, V_L, V_R, phi_p, phi_n,
                                  H_slice, V_slice, eta)
    if verbose:
        print(f"  cached self-energies: {n_E} pts in {time.time()-t0:.1f}s")
    return SL, SR


# ====================================================================== #
#  DENSITY AT ENERGY (eq + neq split)                                    #
# ====================================================================== #
def density_at_E(E, H_eff, sigL, sigR, fL, fR, mu_eq, eta):
    GR_diag, _, _, _, _ = rgf(
        E, H_eff, sigL, sigR,
        compute_lesser=False, eta=eta,
        return_diag=True, return_gamma=True,
    )
    _, Glt_L, _, _, _ = rgf(
        E, H_eff, sigL, sigR,
        compute_lesser=True, occ_left=1.0, occ_right=0.0,
        eta=eta, return_diag=True, return_gamma=True,
    )
    _, Glt_R, _, _, _ = rgf(
        E, H_eff, sigL, sigR,
        compute_lesser=True, occ_left=0.0, occ_right=1.0,
        eta=eta, return_diag=True, return_gamma=True,
    )

    A   = -GR_diag.imag / np.pi
    A_L = +Glt_L.imag / (2*np.pi)
    A_R = +Glt_R.imag / (2*np.pi)

    f_eq = 1.0 / (1.0 + np.exp(np.clip((E - mu_eq)/KT, -200, 200)))
    dn = A_L * (fL - f_eq) + A_R * (fR - f_eq)
    n_E = A * f_eq + dn
    p_E = A * (1 - f_eq) - dn
    return A, n_E, p_E, GR_diag


_W = {}


def _init_worker(H_eff, eta, SL_cache, SR_cache):
    _W["H_eff"] = H_eff
    _W["eta"] = eta
    _W["SL"] = SL_cache
    _W["SR"] = SR_cache


def _compute_one(args):
    k, E, mu_L, mu_R, mu_eq = args
    sigL = _W["SL"][k]; sigR = _W["SR"][k]
    fL = 1.0 / (1.0 + np.exp(np.clip((E - mu_L)/KT, -200, 200)))
    fR = 1.0 / (1.0 + np.exp(np.clip((E - mu_R)/KT, -200, 200)))
    A, n_E, p_E, GR = density_at_E(E, _W["H_eff"], sigL, sigR,
                                     fL, fR, mu_eq, _W["eta"])
    # LDOS-based Jacobian: dn/dphi - dp/dphi = -2 A(E) * df/dE  (in 1/V0 units)
    # df/dE = -1/(4 KT) sech^2((E-mu)/(2 KT))
    x = np.clip((E - mu_eq) / (2*KT), -100, 100)
    minus_dfdE = 1.0 / (4*KT) / np.cosh(x)**2     # > 0, peaks at E = mu_eq
    J_E = 2 * A * minus_dfdE * V0
    return n_E, p_E, A, J_E


def negf_density(phi_site, mu_L, mu_R, SL_cache, SR_cache,
                  H_dev, E_grid, eta,
                  n_workers=N_WORKERS, verbose=True):
    """NEGF density on a block-tridiagonal Hamiltonian.

    phi_site: per-orbital potential array of length N = H_dev.shape[0], in V0 units.
    Returns per-orbital n, p, J, A arrays of length N.
    """
    N = H_dev.shape[0]
    mu_eq = min(mu_L, mu_R)
    H_eff = H_dev - V0 * np.diag(phi_site).astype(complex)
    args = [(k, E_grid[k], mu_L, mu_R, mu_eq) for k in range(len(E_grid))]
    dE = np.gradient(E_grid)
    n_t = np.zeros(N); p_t = np.zeros(N); A_t = np.zeros(N); J_t = np.zeros(N)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_init_worker,
                              initargs=(H_eff, eta, SL_cache, SR_cache)) as ex:
        for k, (n_E, p_E, A_E, J_E) in enumerate(ex.map(_compute_one, args)):
            n_t += n_E * dE[k]; p_t += p_E * dE[k]
            A_t += A_E * dE[k]; J_t += J_E * dE[k]
    if verbose:
        print(f"  NEGF: {len(E_grid)} E-pts in {time.time()-t0:.1f}s   "
              f"<A>={A_t.mean():.3f} <n>={n_t.mean():.3f} <p>={p_t.mean():.3f}")
    return n_t, p_t, J_t, A_t


# ====================================================================== #
#  CURRENT (Landauer-Caroli)                                             #
# ====================================================================== #
def current_at_E(E, H_eff, sigL, sigR, V_slice, fL, fR, eta, slice_pair=None):
    N = H_eff.shape[0]
    nb = sigL.shape[0]
    Gamma_L = 1j * (sigL - sigL.conj().T)
    Gamma_R = 1j * (sigR - sigR.conj().T)
    M = (E + 1j*eta) * np.eye(N, dtype=complex) - H_eff
    M[:nb, :nb] -= sigL
    M[-nb:, -nb:] -= sigR
    B = np.zeros((N, nb), dtype=complex)
    B[-nb:, :] = np.eye(nb, dtype=complex)
    X = np.linalg.solve(M, B)
    G_1N = X[:nb, :]
    T_E = np.real(np.trace(Gamma_L @ G_1N @ Gamma_R @ G_1N.conj().T))
    return T_E * (fL - fR)


def _compute_current_one(args):
    k, E, V_L, V_R, mu_L, mu_R = args
    sigL = _W["SL"][k]; sigR = _W["SR"][k]
    fL = 1.0 / (1.0 + np.exp(np.clip((E - mu_L)/KT, -200, 200)))
    fR = 1.0 / (1.0 + np.exp(np.clip((E - mu_R)/KT, -200, 200)))
    return current_at_E(E, _W["H_eff"], sigL, sigR, _W["V_slice"],
                          fL, fR, _W["eta"])


def _init_worker_with_V(H_eff, eta, SL_cache, SR_cache, V_slice):
    _W["H_eff"] = H_eff
    _W["eta"] = eta
    _W["SL"] = SL_cache
    _W["SR"] = SR_cache
    _W["V_slice"] = V_slice


def compute_current(phi_site, mu_L, mu_R,
                      SL_cache, SR_cache, H_dev, V_slice, E_grid, eta,
                      n_workers=N_WORKERS, verbose=True):
    H_eff = H_dev - V0 * np.diag(phi_site).astype(complex)
    args = [(k, E_grid[k], 0.0, 0.0, mu_L, mu_R) for k in range(len(E_grid))]
    dE = np.gradient(E_grid)
    I_E = np.zeros(len(E_grid))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_init_worker_with_V,
                              initargs=(H_eff, eta, SL_cache, SR_cache, V_slice)) as ex:
        for k, val in enumerate(ex.map(_compute_current_one, args)):
            I_E[k] = val
    I_internal = float(np.sum(I_E * dE))
    PREFACTOR = E_CHARGE**2 / (np.pi * HBAR)
    I_amps = PREFACTOR * I_internal
    if verbose:
        print(f"  current eval in {time.time()-t0:.1f}s   "
              f"I = {I_amps:.4e} A")
    return I_amps, I_E


# ====================================================================== #
#  k_perp PERIODIC VARIANTS                                              #
#                                                                        #
#  Same API as the non-periodic functions, but inputs are LISTS — one    #
#  entry per k_perp point, plus a k_weights array. Caller assembles the  #
#  per-k Hamiltonians outside the NEGF module.                           #
# ====================================================================== #
def make_kperp_grid(N_k, k_range=(-np.pi, np.pi)):
    """Uniform k_perp grid over k_range, excluding the right endpoint
    (BZ is periodic). Returns (kperp_pts, weights) where weights sum to 1
    so Σ w_k f(k) is the BZ AVERAGE."""
    k_min, k_max = k_range
    kperp_pts = np.linspace(k_min, k_max, N_k, endpoint=False)
    weights = np.full(N_k, 1.0 / N_k)
    return kperp_pts, weights


def cache_self_energies_kperp(
    E_grid, V_L, V_R, phi_p, phi_n,
    H_slice_per_k, V_slice_per_k, eta, verbose=True,
):
    """Cache lead self-energies for each k_perp.

    H_slice_per_k, V_slice_per_k: lists of slice/hop matrices, one per k_perp.
    Returns SL_per_k, SR_per_k: lists of (n_E, n_per, n_per) arrays.
    """
    SL_per_k, SR_per_k = [], []
    t0 = time.time()
    N_k = len(H_slice_per_k)
    for ik in range(N_k):
        SL, SR = cache_self_energies(
            E_grid, V_L, V_R, phi_p, phi_n,
            H_slice_per_k[ik], V_slice_per_k[ik], eta, verbose=False)
        SL_per_k.append(SL)
        SR_per_k.append(SR)
    if verbose:
        n_per = H_slice_per_k[0].shape[0]
        print(f"  cached self-energies (kperp): {N_k} k-pts × {len(E_grid)} E "
               f"(n_per={n_per}) in {time.time()-t0:.1f}s")
    return SL_per_k, SR_per_k


def negf_density_kperp(
    phi_site, mu_L, mu_R, SL_per_k, SR_per_k,
    H_dev_per_k, E_grid, eta, k_weights,
    n_workers=N_WORKERS, verbose=True,
):
    """k_perp-averaged NEGF density.

    Same call signature as negf_density, with SL/SR/H_dev replaced by
    LISTS (one per k) plus a k_weights array.

    All k-points must use Hamiltonians of the same size, so phi_site has
    a single shape — length N = H_dev_per_k[0].shape[0].

    Returns BZ-averaged per-orbital n, p, J, A arrays of length N.
    """
    N = H_dev_per_k[0].shape[0]
    N_k = len(k_weights)
    n_total = np.zeros(N); p_total = np.zeros(N)
    J_total = np.zeros(N); A_total = np.zeros(N)

    t0 = time.time()
    for ik in range(N_k):
        n_k, p_k, J_k, A_k = negf_density(
            phi_site, mu_L, mu_R,
            SL_per_k[ik], SR_per_k[ik],
            H_dev_per_k[ik], E_grid, eta,
            n_workers=n_workers, verbose=False)
        w = k_weights[ik]
        n_total += w * n_k; p_total += w * p_k
        J_total += w * J_k; A_total += w * A_k

    if verbose:
        print(f"  NEGF (kperp): {N_k} k × {len(E_grid)} E in {time.time()-t0:.1f}s   "
              f"<A>={A_total.mean():.3f} <n>={n_total.mean():.3f} "
              f"<p>={p_total.mean():.3f}")
    return n_total, p_total, J_total, A_total


def compute_current_kperp(
    phi_site, mu_L, mu_R, SL_per_k, SR_per_k,
    H_dev_per_k, V_slice_per_k, E_grid, eta, k_weights,
    n_workers=N_WORKERS, verbose=True,
):
    """k_perp-averaged Landauer current.

    Same call signature as compute_current, with SL/SR/H_dev/V_slice
    replaced by LISTS (one per k) plus a k_weights array.
    """
    N_k = len(k_weights)
    I_E_total = np.zeros(len(E_grid))
    t0 = time.time()
    for ik in range(N_k):
        _, I_E_k = compute_current(
            phi_site, mu_L, mu_R,
            SL_per_k[ik], SR_per_k[ik],
            H_dev_per_k[ik], V_slice_per_k[ik], E_grid, eta,
            n_workers=n_workers, verbose=False)
        I_E_total += k_weights[ik] * I_E_k

    dE = np.gradient(E_grid)
    I_internal = float(np.sum(I_E_total * dE))
    PREFACTOR = E_CHARGE**2 / (np.pi * HBAR)
    I_amps = PREFACTOR * I_internal
    if verbose:
        print(f"  current (kperp): {N_k} k × {len(E_grid)} E in {time.time()-t0:.1f}s   "
              f"I = {I_amps:.4e} A")
    return I_amps, I_E_total


# ====================================================================== #
#  TRANSMISSION T(E) — Landauer-Caroli, no Fermi window                  #
# ====================================================================== #
def _compute_transmission_one(args):
    k, E = args
    sigL = _W["SL"][k]; sigR = _W["SR"][k]
    H_eff = _W["H_eff"]; eta = _W["eta"]
    N = H_eff.shape[0]
    nb = sigL.shape[0]
    Gamma_L = 1j * (sigL - sigL.conj().T)
    Gamma_R = 1j * (sigR - sigR.conj().T)
    M = (E + 1j*eta) * np.eye(N, dtype=complex) - H_eff
    M[:nb, :nb]   -= sigL
    M[-nb:, -nb:] -= sigR
    B = np.zeros((N, nb), dtype=complex)
    B[-nb:, :] = np.eye(nb, dtype=complex)
    X = np.linalg.solve(M, B)
    G_1N = X[:nb, :]
    return float(np.real(np.trace(Gamma_L @ G_1N @ Gamma_R @ G_1N.conj().T)))


def transmission(phi_site, SL_cache, SR_cache, H_dev, E_grid, eta,
                  n_workers=N_WORKERS, verbose=True):
    """Compute Landauer transmission T(E) on an energy grid.

    Returns T_E array of length len(E_grid). Conductance is G(E) = (e^2/h) * T(E).
    For a clean ballistic device pass phi_site = np.zeros(H_dev.shape[0]).
    """
    H_eff = H_dev - V0 * np.diag(phi_site).astype(complex)
    args = [(k, E_grid[k]) for k in range(len(E_grid))]
    T_E = np.zeros(len(E_grid))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_init_worker,
                              initargs=(H_eff, eta, SL_cache, SR_cache)) as ex:
        for k, val in enumerate(ex.map(_compute_transmission_one, args)):
            T_E[k] = val
    if verbose:
        print(f"  T(E): {len(E_grid)} pts in {time.time()-t0:.1f}s   "
              f"<T>={T_E.mean():.3f}  max={T_E.max():.2f}")
    return T_E