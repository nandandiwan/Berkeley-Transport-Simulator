from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from negf_functions import cache_self_energies, transmission, N_WORKERS
from hamiltonian_functions import build_device_H


# ---------------------------------------------------------------------- #
#  Pauli matrices                                                         #
# ---------------------------------------------------------------------- #
sigma_x = np.array([[0,  1 ], [ 1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1,  0 ], [ 0,-1]], dtype=complex)


def build_slice_ti2d(N_perim, W=0.3, hbar_v=1.0, a_lat=1.0, periodic=True):
    """Slice + inter-slice hop matrices for the 2D TI lattice.

    Parameters
    ----------
    N_perim  : sites around the wire perimeter
    W        : Wilson coefficient (in units of hbar*v); 0.3 is the paper's choice
    hbar_v   : Dirac velocity scale (set to 1 for natural units)
    a_lat    : lattice spacing (set to 1 for natural units)
    periodic : close the perimeter ring (True for a nanowire surface)

    Returns
    -------
    H_slice : (2 N_perim, 2 N_perim) on-site T_0 + n-direction T_x hopping
    V_slice : (2 N_perim, 2 N_perim) block-diagonal T_y; couples slice m -> m+1
    """
    a2 = a_lat * a_lat
    T0 =  (2.0 * W * hbar_v / a2)         * sigma_z
    Tx = -(W * hbar_v / (2.0 * a2))       * sigma_z \
         - (1j * hbar_v / (2.0 * a_lat))  * sigma_y
    Ty = -(W * hbar_v / (2.0 * a2))       * sigma_z \
         + (1j * hbar_v / (2.0 * a_lat))  * sigma_x

    dim = 2 * N_perim
    H_slice = np.zeros((dim, dim), dtype=complex)
    V_slice = np.zeros((dim, dim), dtype=complex)

    for n in range(N_perim):
        i = 2 * n
        H_slice[i:i+2, i:i+2] = T0
        V_slice[i:i+2, i:i+2] = Ty       # transport-direction hop
        if n + 1 < N_perim:
            j = 2 * (n + 1)
            H_slice[i:i+2, j:j+2] = Tx
            H_slice[j:j+2, i:i+2] = Tx.conj().T

    if periodic and N_perim > 1:         # close the ring
        i = 2 * (N_perim - 1)
        j = 0
        H_slice[i:i+2, j:j+2] = Tx
        H_slice[j:j+2, i:i+2] = Tx.conj().T

    return H_slice, V_slice


# ====================================================================== #
#  CONDUCTANCE                                                           #
# ====================================================================== #
def conductance(
    N_perim=48, L_slices=12,
    W=0.3, hbar_v=1.0, a_lat=1.0, periodic=True,
    disorder=0.0, seed=None,
    E_grid=None, eta=5e-4,
    n_workers=N_WORKERS, verbose=True,
):
    """Zero-bias conductance G(E) = (e^2/h) * T(E) for the TI nanowire.

    Parameters
    ----------
    disorder : Anderson on-site disorder strength U (uniform in [-U/2, U/2]),
               in units of hbar*v/a.  Set to 0 for the clean limit.
    seed     : RNG seed for disorder (None for fresh).
    eta      : small imaginary part for retarded GFs.  Use ~1e-4..1e-3 to
               see clean step structure; too small -> ill-conditioning.
    """
    if E_grid is None:
        E_grid = np.linspace(-1.0, 1.0, 401)

    H_slice, V_slice = build_slice_ti2d(N_perim, W, hbar_v, a_lat, periodic)
    H_dev = build_device_H(H_slice, V_slice, L_slices)
    n_orb = H_dev.shape[0]

    if disorder > 0.0:
        rng = np.random.default_rng(seed)
        U = (rng.random(n_orb) - 0.5) * disorder
        H_dev = H_dev + np.diag(U).astype(complex)

    # No electrostatic potential and zero bias: clean Landauer setup.
    phi_site = np.zeros(n_orb)

    SL, SR = cache_self_energies(
        E_grid, V_L=0.0, V_R=0.0, phi_p=0.0, phi_n=0.0,
        H_slice=H_slice, V_slice=V_slice, eta=eta, verbose=verbose)

    T_E = transmission(phi_site, SL, SR, H_dev, E_grid, eta,
                        n_workers=n_workers, verbose=verbose)
    return E_grid, T_E


def conductance_disorder_avg(
    N_perim, L_slices, W, U_strength, n_configs,
    E_grid, eta=5e-4, seed_base=0,
    hbar_v=1.0, a_lat=1.0, periodic=True,
    n_workers=N_WORKERS, verbose=True,
):
    """Average G(E) over n_configs disorder realizations (paper Fig. 4 a,b)."""
    G_sum = np.zeros_like(E_grid, dtype=float)
    for k in range(n_configs):
        if verbose:
            print(f"--- disorder config {k+1}/{n_configs}  U={U_strength}")
        _, G_k = conductance(
            N_perim=N_perim, L_slices=L_slices,
            W=W, hbar_v=hbar_v, a_lat=a_lat, periodic=periodic,
            disorder=U_strength, seed=seed_base + k,
            E_grid=E_grid, eta=eta,
            n_workers=n_workers, verbose=False)
        G_sum += G_k
    return G_sum / n_configs


# ====================================================================== #
#  PLOTTING                                                              #
# ====================================================================== #
def plot_conductance(curves, save_path=None, title=None, xlabel=None):
    """curves : list of (label, E_grid, G_E)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, E, G in curves:
        ax.plot(E, G, label=label, lw=1.4)
    ax.set_xlabel(xlabel or r"$E_F$  (units of $\hbar v / a$)")
    ax.set_ylabel(r"$G$  $(e^2/h)$")
    if title:
        ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    return fig


# ====================================================================== #
#  MAIN                                                                  #
# ====================================================================== #
if __name__ == "__main__":

    # ----- arXiv:1612.08248 reference geometry ------------------------ #
    # Cross section (L_x, L_z) = (23.4 nm, 24.6 nm) -> perimeter ~ 96 nm
    # Wire length L_y = 24 nm
    # Pick a_lat = 2 nm  =>  N_perim = 48,  L_slices = 12
    # Working in natural units (hbar*v = 1, a = 1); to convert to eV
    # multiply E by hbar*v/a  (~ 0.2 eV for hbar*v = 0.4 eV.nm, a = 2 nm).

    N_perim, L_slices = 48, 12
    E_grid = np.linspace(-1.0, 1.0, 401)
    eta    = 5e-4

    # ---- Fig. 4(c)-style: clean wire, with vs without Wilson term ---- #
    print("=== W = 0.3 hbar*v   (paper's Wilson choice, Fig. 4c) ===")
    E1, G1 = conductance(N_perim=N_perim, L_slices=L_slices,
                          W=0.3, E_grid=E_grid, eta=eta)

    print("=== W = 0            (no Wilson term: doubling visible) ===")
    E2, G2 = conductance(N_perim=N_perim, L_slices=L_slices,
                          W=0.0, E_grid=E_grid, eta=eta)

    plot_conductance(
        [(r"$W = 0.3\,\hbar v$", E1, G1),
         (r"$W = 0$",            E2, G2)],
        save_path="ti_conductance_clean.png",
        title=f"TI nanowire conductance, clean   (N_perim={N_perim}, L={L_slices})",
    )

    # ---- Optional: disorder average like Fig. 4(b) ------------------- #
    # n_configs is small here so it runs in reasonable time; bump up for
    # smoother curves.
    RUN_DISORDER = False
    if RUN_DISORDER:
        n_cfg = 10
        E_d = np.linspace(-1.0, 1.0, 201)
        curves = [(r"$U=0$", *conductance(N_perim, L_slices, W=0.3,
                                            E_grid=E_d, eta=eta))]
        for U in (0.25, 0.50, 0.75, 1.00):
            G_avg = conductance_disorder_avg(
                N_perim, L_slices, W=0.3,
                U_strength=U, n_configs=n_cfg,
                E_grid=E_d, eta=eta)
            curves.append((rf"$U={U:.2f}$", E_d, G_avg))
        plot_conductance(
            curves, save_path="ti_conductance_disorder.png",
            title=(f"TI nanowire conductance, W=0.3  "
                   f"(N_perim={N_perim}, L={L_slices}, n_cfg={n_cfg})"))

    plt.show()