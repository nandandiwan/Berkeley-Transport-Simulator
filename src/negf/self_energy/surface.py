import numpy as np
from scipy import linalg
import scipy.sparse as sp


"""
Lead self-energy calculation using surface Green's functions.
Methods: Sancho-Rubio (default), iterative (disabled), transfer, and a mixed recursive fallback.
"""

eta = 1e-9j

def _add_eta(E):
    if np.imag(E) == 0:
        return E + 1j * eta
    return E

def surface_greens_function(E, H00, H01, method="sancho_rubio",
                            iteration_max=1000, tolerance=1e-6):
    E = _add_eta(E)
    if method == "sancho_rubio":
        return _sancho_rubio_surface_gf(E, H00, H01, None, 100, 1e-10)
    elif method == "iterative":
        raise Exception("iterative surface GF disabled (unstable)")
    elif method == "transfer":
        return _transfer_surface_gf(E, H00, H01, tolerance, iteration_max)
    elif method == "recursive":
        return _recursive_self_energy_mixed(E, H00, H01, max_iter=iteration_max, tol=tolerance)
    else:
        raise ValueError(f"Unknown method: {method}")
def _sancho_rubio_surface_gf(E, H00, H01, S00=None, iter_max=200, TOL=1e-12):
    """
    Jiezi surface_gf algorithm translated to use numpy arrays.
    Returns surface Green's function G00 for a semi-infinite lead with onsite H00 and coupling H01 (to the right neighbor).
    """
    E = E + 1e-3j
    n = H00.shape[0]
    I = np.eye(n, dtype=complex)
    S00 = I if S00 is None else S00
    # Convert to dense if needed
    if hasattr(H00, 'toarray'): H00 = H00.toarray()
    if hasattr(H01, 'toarray'): H01 = H01.toarray()
    if hasattr(S00, 'toarray'): S00 = S00.toarray()

    iter_c = 0
    H10 = H01.conj().T
    alpha = H10.copy()   # coupling to the left
    beta  = H01.copy()   # coupling to the right
    epsilon = H00.copy()
    epsilon_s = H00.copy()
    Eeye = I * E

    while iter_c < iter_max:
        iter_c += 1
        inv_term = np.linalg.solve(Eeye - epsilon, I)
        alpha_new = alpha @ inv_term @ alpha
        beta_new  = beta  @ inv_term @ beta
        epsilon_new   = epsilon   + alpha @ inv_term @ beta + beta @ inv_term @ alpha
        epsilon_s_new = epsilon_s + alpha @ inv_term @ beta
        if np.linalg.norm(alpha_new, ord='fro') < TOL and np.linalg.norm(beta_new, ord='fro') < TOL:
            G00 = np.linalg.solve(Eeye - epsilon_s_new, I)
            break
        alpha, beta = alpha_new, beta_new
        epsilon, epsilon_s = epsilon_new, epsilon_s_new
    else:
        print(f"Warning: Surface GF did not converge after {iter_max} iterations")
        G00 = np.linalg.solve(Eeye - epsilon_s, I)
    return G00
def _recursive_self_energy_mixed(E, H00, H01, max_iter=500, tol=1e-8, mixing_beta=0.1):
    if sp.issparse(H00):
        H00 = H00.toarray()
    if sp.issparse(H01):
        H01 = H01.toarray()
    H10 = H01.conj().T
    w = E
    identity = np.eye(H00.shape[0])
    try:
        g_s = np.linalg.solve(w * identity - H00, identity)
    except np.linalg.LinAlgError:
        return None
    for _ in range(max_iter):
        g_s_old = g_s.copy()
        sigma = H10 @ g_s @ H01
        mat_to_invert = w * identity - H00 - sigma
        try:
            g_s_new = np.linalg.solve(mat_to_invert, identity)
        except np.linalg.LinAlgError:
            return None
        g_s = (1 - mixing_beta) * g_s_old + mixing_beta * g_s_new
        diff = np.linalg.norm(g_s - g_s_old) / max(np.linalg.norm(g_s), 1e-30)
        if diff < tol:
            final_sigma = H10 @ g_s @ H01
            return final_sigma
    return g_s

def _transfer_surface_gf(E, H00, H01, tolerance=1e-6, iteration_max=1000):
    n = H00.shape[0]
    I = np.eye(n, dtype=complex)
    if hasattr(H00, 'toarray'):
        H00 = H00.toarray()
    if hasattr(H01, 'toarray'):
        H01 = H01.toarray()
    H10 = H01.conj().T
    try:
        gr00_inv = linalg.solve(E * I - H00, I)
    except linalg.LinAlgError:
        gr00_inv = linalg.pinv(E * I - H00)
    t_i = gr00_inv @ H10
    bar_t_i = gr00_inv @ H01
    T_i = t_i.copy()
    bar_T_i = bar_t_i.copy()
    T_i_old = T_i.copy()
    for iteration in range(1, iteration_max):
        temp1 = t_i @ bar_t_i
        temp2 = bar_t_i @ t_i
        denominator = I - temp1 - temp2
        try:
            inv_denom = linalg.solve(denominator, I)
        except linalg.LinAlgError:
            inv_denom = linalg.pinv(denominator)
        t_i_new = inv_denom @ (t_i @ t_i)
        bar_t_i_new = inv_denom @ (bar_t_i @ bar_t_i)
        bar_T_i_new = bar_T_i @ bar_t_i_new
        T_i_new = T_i + bar_T_i @ t_i_new
        diff = T_i_new - T_i_old
        rms = np.sqrt(np.max(np.abs(diff) ** 2))
        if rms < tolerance:
            break
        t_i = t_i_new
        bar_t_i = bar_t_i_new
        T_i_old = T_i.copy()
        T_i = T_i_new
        bar_T_i = bar_T_i_new
    final_matrix = E * I - H00 - H01 @ T_i
    try:
        return linalg.solve(final_matrix, I)
    except linalg.LinAlgError:
        return linalg.pinv(final_matrix)
