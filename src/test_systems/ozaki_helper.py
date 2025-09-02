import numpy as np
from functools import lru_cache

@lru_cache(maxsize=None)
def get_ozaki_poles_residues(cutoff:int, kT:float, T_key:float|None=None):
    """Return (poles,residues) for Ozaki continued-fraction Fermi approximation.
    Cached globally by (cutoff, kT, T_key). T_key can be an external temperature marker
    (e.g. physical temperature in K) to force regeneration when temperature changes even
    if kT matches numerically.
    """
    j = np.arange(cutoff - 1)
    b = 1.0 / (2.0 * np.sqrt((2 * (j + 1) - 1) * (2 * (j + 1) + 1)))
    J = np.zeros((cutoff, cutoff), dtype=float)
    J[j, j + 1] = b
    J[j + 1, j] = b
    vals, vecs = np.linalg.eigh(J)
    mask = vals > 0
    poles = vals[mask]
    v0 = vecs[0, mask]
    residues = 0.25 * (np.abs(v0) ** 2) / (poles ** 2)
    return poles, residues

def fermi_cfr(E: np.ndarray, mu: float | None, poles: np.ndarray, residues: np.ndarray, kT: float,
              V_vec: np.ndarray | None = None, Efn_vec: np.ndarray | None = None,
              tail_clip: float = 30.0, tail_blend_start: float = 20.0) -> np.ndarray:
        """Ozaki CFR approximation to Fermi-Dirac distribution.
        Two modes:
            - scalar mu (float): returns shape (len(E),) with f(E,mu)
            - V_vec and Efn_vec provided: returns shape (len(E), n_sites) computing
                f(E - V_i - Efn_i) for each site i.
        E, mu in eV
        """
        E = np.atleast_1d(E)
        # per-site vectorized path
        if V_vec is not None or Efn_vec is not None:
                V_vec = np.atleast_1d(np.zeros(1) if V_vec is None else V_vec)
                Efn_vec = np.atleast_1d(np.zeros_like(V_vec) if Efn_vec is None else Efn_vec)
                # x has shape (nE, n_sites)
                x = (E[:, None] - V_vec[None, :] - Efn_vec[None, :]) / kT
                x2 = x * x
                aj = poles[None, None, :]
                rj = residues[None, None, :]
                inv_a2 = (1.0 / aj) ** 2
                denom = x2[:, :, None] + inv_a2
                s = 2.0 * x[:, :, None] * rj / denom
                f = 0.5 - np.sum(s, axis=2)
                # Smoothly blend to exact logistic in extreme tails to avoid plateau bias.
                with np.errstate(over='ignore'):
                    x_abs = np.abs(x)
                    # exact logistic per-site
                    f_exact = 1.0 / (1.0 + np.exp(x))
                    # regions
                    far_mask = x_abs >= tail_clip
                    blend_mask = (x_abs >= tail_blend_start) & (x_abs < tail_clip)
                    # Hard set in far region
                    f[far_mask] = f_exact[far_mask]
                    # Linear blend (could use smoother; linear suffices)
                    if np.any(blend_mask):
                        w = (x_abs[blend_mask] - tail_blend_start) / (tail_clip - tail_blend_start)
                        f[blend_mask] = (1 - w) * f[blend_mask] + w * f_exact[blend_mask]
                return f.real

        # scalar-mu path (backwards compatible)
        x = (E - mu) / kT
        x_col = x[:, None]
        aj = poles[None, :]
        rj = residues[None, :]
        # (1/(x - i/a) + 1/(x + i/a)) simplifies to 2x/(x^2 + 1/a^2)
        inv_a2 = (1.0 / aj) ** 2
        denom = x_col**2 + inv_a2
        s = 2.0 * x_col * rj / denom
        f = 0.5 - np.sum(s, axis=1)
        with np.errstate(over='ignore'):
            x_abs = np.abs(x)
            f_exact = 1.0 / (1.0 + np.exp(x))
            far_mask = x_abs >= tail_clip
            blend_mask = (x_abs >= tail_blend_start) & (x_abs < tail_clip)
            f = f.real
            f[far_mask] = f_exact[far_mask]
            if np.any(blend_mask):
                w = (x_abs[blend_mask] - tail_blend_start) / (tail_clip - tail_blend_start)
                f[blend_mask] = (1 - w) * f[blend_mask] + w * f_exact[blend_mask]
        return f

def fermi_derivative_cfr_abs(E: np.ndarray, V_vec: np.ndarray, Efn_vec: np.ndarray,
                             poles: np.ndarray, residues: np.ndarray, kT: float) -> np.ndarray:
    """Return positive quantity corresponding to -df/dx (the usual |df/dx| of Fermi) for each site.
    We evaluate for each energy E a vector over sites: x_i = (E - V_i - Efn_i)/kT.
    Output shape: (n_sites,) per energy call.
    which is positive near x=0; if numerical negatives arise (far tails) we clamp to >=0.
    """
    x = (E - V_vec - Efn_vec) / kT  # shape (n_sites,)
    x2 = x * x
    beta = (1.0 / poles) ** 2  # shape (m,)

    x2_col = x2[:, None]
    beta_row = beta[None, :]
    num = beta_row - x2_col
    denom = (x2_col + beta_row) ** 2
    contrib = 2.0 * residues[None, :] * num / denom
    val = np.sum(contrib, axis=1)

    val[val < 0] = 0.0
    return val

def fermi_derivative_cfr_abs_batch(E_array: np.ndarray, V_vec: np.ndarray, Efn_vec: np.ndarray,
                                   poles: np.ndarray, residues: np.ndarray, kT: float) -> np.ndarray:
    """Vectorized batch version: return |df/dx| for each energy and site.
    E_array shape (nE,), V_vec/Efn_vec shape (n_sites,). Output shape (nE, n_sites).
    """
    E_array = np.atleast_1d(E_array)
    V_vec = np.atleast_1d(V_vec)
    Efn_vec = np.atleast_1d(Efn_vec)
    x = (E_array[:, None] - V_vec[None, :] - Efn_vec[None, :]) / kT  # (nE, n_sites)
    x2 = x * x
    beta = (1.0 / poles) ** 2  # (m,)
    # Broadcast to (nE, n_sites, m)
    num = beta[None, None, :] - x2[:, :, None]
    denom = (x2[:, :, None] + beta[None, None, :]) ** 2
    contrib = 2.0 * residues[None, None, :] * num / denom
    val = np.sum(contrib, axis=2)
    val[val < 0] = 0.0
    return val
