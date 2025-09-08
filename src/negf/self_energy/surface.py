import numpy as np
from scipy import linalg
import scipy.sparse as sp


class LeadSelfEnergy:
    """
    Lead self-energy calculation using surface Green's functions.
    Methods: Sancho-Rubio (default), iterative (disabled), transfer, and a mixed recursive fallback.
    """

    def __init__(self, hamiltonian=None):
        """Lead self-energy helper.

        The class no longer relies on a specific Hamiltonian interface. Instead,
        provide the principal-layer lead matrices via set_lead_matrices(...).
        Passing a hamiltonian object is optional and kept for compatibility; it
        is not used by default.
        """
        self.ham = hamiltonian
        self.eta = 1e-12
        # Optional cached principal-layer blocks for each side
        self._H00 = {"left": None, "right": None}
        self._H01 = {"left": None, "right": None}

    def _add_eta(self, E):
        if np.imag(E) == 0:
            return E + 1j * self.eta
        return E

    def surface_greens_function(self, E, H00, H01, method="sancho_rubio",
                                iteration_max=1000, tolerance=1e-6):
        E = self._add_eta(E)
        if method == "sancho_rubio":
            return self._sancho_rubio_surface_gf(E, H00, H01, None, 100, 1e-10)
        elif method == "iterative":
            raise Exception("iterative surface GF disabled (unstable)")
        elif method == "transfer":
            return self._transfer_surface_gf(E, H00, H01, tolerance, iteration_max)
        elif method == "recursive":
            return self._recursive_self_energy_mixed(E, H00, H01, max_iter=iteration_max, tol=tolerance)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _sancho_rubio_surface_gf(self, E, H00, H01, S00=None, iter_max=100, TOL=1e-10):
        n = H00.shape[0]
        I = np.eye(n, dtype=complex)
        if S00 is None:
            S00 = I
        if hasattr(H00, 'toarray'):
            H00 = H00.toarray()
        if hasattr(H01, 'toarray'):
            H01 = H01.toarray()
        if hasattr(S00, 'toarray'):
            S00 = S00.toarray()
        iter_c = 0
        H10 = H01.conj().T
        alpha = H10.copy()
        beta = H10.conj().T
        epsilon = H00.copy()
        epsilon_s = H00.copy()
        wI = I * E
        while iter_c < iter_max:
            iter_c += 1
            inv_term = np.linalg.solve(wI - epsilon, I)
            alpha_new = alpha @ inv_term @ alpha
            beta_new = beta @ inv_term @ beta
            epsilon_new = epsilon + alpha @ inv_term @ beta + beta @ inv_term @ alpha
            epsilon_s_new = epsilon_s + alpha @ inv_term @ beta
            convergence_check = np.linalg.norm(alpha_new, ord='fro')
            if convergence_check < TOL:
                G00 = np.linalg.solve(wI - epsilon_s_new, I)
                return G00
            alpha = alpha_new.copy()
            beta = beta_new.copy()
            epsilon = epsilon_new.copy()
            epsilon_s = epsilon_s_new.copy()
        # fallback
        return self._recursive_self_energy_mixed(E, H00, H01)

    def _recursive_self_energy_mixed(self, E, H00, H01, max_iter=500, tol=1e-8, mixing_beta=0.1):
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

    def _transfer_surface_gf(self, E, H00, H01, tolerance=1e-6, iteration_max=1000):
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

    def set_lead_matrices(self, side: str, H00, H01):
        """Inject principal-layer matrices for a lead side.

        side: "left" or "right"
        H00: onsite block of a principal layer
        H01: coupling from current layer to the next (H10 inferred as H01^H)
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self._H00[side] = H00
        self._H01[side] = H01

    def get_lead_matrices(self, side: str):
        H00 = self._H00.get(side)
        H01 = self._H01.get(side)
        if H00 is None or H01 is None:
            raise RuntimeError(f"Lead matrices not set for side='{side}'. Call set_lead_matrices().")
        H10 = H01.conj().T
        return H00, H01, H10

    def self_energy(self, side, E, ky=0, method="sancho_rubio"):
        # Prefer injected matrices; fall back to legacy ham API if available
        if self._H00.get(side) is not None:
            H00, H01, H10 = self.get_lead_matrices(side)
        elif self.ham is not None and hasattr(self.ham, 'get_H00_H01_H10'):
            H00, H01, H10 = self.ham.get_H00_H01_H10(ky=ky, side=side)
        else:
            raise RuntimeError("LeadSelfEnergy requires lead matrices via set_lead_matrices or a hamiltonian with get_H00_H01_H10().")
        if np.abs(E) > 5e5:
            return np.zeros((H00.shape[0], H00.shape[0]), dtype=complex)
        try:
            G_surface = self.surface_greens_function(E, H00, H01, method=method)
        except Exception:
            n = H00.shape[0]
            G_surface = linalg.pinv(self._add_eta(E) * np.eye(n) - H00)
        if side == "left":
            return H10 @ G_surface @ H01
        else:
            return H01 @ G_surface @ H10
