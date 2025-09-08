import numpy as np
from functools import lru_cache


@lru_cache(maxsize=None)
def get_ozaki_poles_residues(cutoff: int, kT: float, T_key: float | None = None):
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
    E = np.atleast_1d(E)
    if V_vec is not None or Efn_vec is not None:
        V_vec = np.atleast_1d(np.zeros(1) if V_vec is None else V_vec)
        Efn_vec = np.atleast_1d(np.zeros_like(V_vec) if Efn_vec is None else Efn_vec)
        x = (E[:, None] - V_vec[None, :] - Efn_vec[None, :]) / kT
        x2 = x * x
        aj = poles[None, None, :]
        rj = residues[None, None, :]
        inv_a2 = (1.0 / aj) ** 2
        denom = x2[:, :, None] + inv_a2
        s = 2.0 * x[:, :, None] * rj / denom
        f = 0.5 - np.sum(s, axis=2)
        with np.errstate(over='ignore'):
            x_abs = np.abs(x)
            f_exact = 1.0 / (1.0 + np.exp(x))
            far_mask = x_abs >= tail_clip
            blend_mask = (x_abs >= tail_blend_start) & (x_abs < tail_clip)
            f[far_mask] = f_exact[far_mask]
            if np.any(blend_mask):
                w = (x_abs[blend_mask] - tail_blend_start) / (tail_clip - tail_blend_start)
                f[blend_mask] = (1 - w) * f[blend_mask] + w * f_exact[blend_mask]
        return f.real
    x = (E - mu) / kT
    x_col = x[:, None]
    aj = poles[None, :]
    rj = residues[None, :]
    inv_a2 = (1.0 / aj) ** 2
    denom = x_col ** 2 + inv_a2
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
    x = (E - V_vec - Efn_vec) / kT
    x2 = x * x
    beta = (1.0 / poles) ** 2
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
    E_array = np.atleast_1d(E_array)
    V_vec = np.atleast_1d(V_vec)
    Efn_vec = np.atleast_1d(Efn_vec)
    x = (E_array[:, None] - V_vec[None, :] - Efn_vec[None, :]) / kT
    x2 = x * x
    beta = (1.0 / poles) ** 2
    num = beta[None, None, :] - x2[:, :, None]
    denom = (x2[:, :, None] + beta[None, None, :]) ** 2
    contrib = 2.0 * residues[None, None, :] * num / denom
    val = np.sum(contrib, axis=2)
    val[val < 0] = 0.0
    return val
