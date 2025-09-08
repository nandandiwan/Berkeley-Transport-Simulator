# greens_function.py

import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv # Keep for reference, but smart_inverse is used
import warnings

# Local application imports

from device import Device
from device_hamiltonian import Hamiltonian
from lead_self_energy import LeadSelfEnergy
from utils import smart_inverse  # <-- Import your utility function

class GreensFunction:
    """
    Recursive Green's Function (RGF) implementation optimized with a smart inversion utility.
    """

    def __init__(self, dev : Device, hamiltonian: Hamiltonian, self_energy_method="sancho_rubio"):
        """
        Initialize the Green's function calculator.
        """
        self.dev = dev
        
        self.ham = hamiltonian
        self.lead_self_energy = LeadSelfEnergy(dev,hamiltonian)
        self.self_energy_method = self_energy_method
        self.eta = 1e-6
        # The sparse_threshold is now handled by the smart_inverse function
        # self.sparse_threshold = 0.1

    def fermi_distribution(self, E, mu, kT):
        """Fermi-Dirac distribution function."""
        x = (E - mu) / kT
        with np.errstate(over='ignore'):
            return 1.0 / (1.0 + np.exp(x))

    def add_eta(self, E):
        """Add small imaginary part for numerical stability."""
        if np.iscomplexobj(E):
            return E
        return E + 1j * self.eta

    def compute_central_greens_function(self, E, compute_lesser=True,
                                        use_rgf=True, self_energy_method=None, equilibrium=False):
        """
        Compute central region Green's function.
        """
        E = self.add_eta(E)

        if self_energy_method is None:
            self_energy_method = self.self_energy_method

        if use_rgf:
            return self._compute_rgf_greens_function(E, compute_lesser, self_energy_method, equilibrium=equilibrium)
        else:
            return self._compute_direct_greens_function(E, compute_lesser, self_energy_method, equilibrium=equilibrium)

    def _compute_direct_greens_function(self, E, compute_lesser, self_energy_method, equilibrium=False):
        """
        Direct matrix inversion for smaller systems using the smart_inverse utility.
        Uses general Hamiltonian.create_hamiltonian interface.
        """
        if equilibrium and compute_lesser:
            raise ValueError("Cannot compute lesser Green's function in equilibrium.")

        # Get full device Hamiltonian for current type
        H = self.ham.create_sparse_channel_hamlitonian(blocks=False)
        n = H.shape[0]

        sigma_L = self.lead_self_energy.self_energy("left", E, self_energy_method)
        sigma_R = self.lead_self_energy.self_energy("right", E, self_energy_method)

        if equilibrium:
            sigma_L *= 0
            sigma_R *= 0

        block_size = sigma_L.shape[0]

        Sigma_L_full = sp.lil_matrix((n, n), dtype=complex)
        Sigma_R_full = sp.lil_matrix((n, n), dtype=complex)
        Sigma_L_full[:block_size, :block_size] = sigma_L
      
        Sigma_R_full[-block_size:, -block_size:] = sigma_R
        Sigma_L_full = Sigma_L_full.tocsc()
        Sigma_R_full = Sigma_R_full.tocsc()

        H_eff = H + Sigma_L_full + Sigma_R_full
        A = E * sp.identity(n, dtype=complex, format='csc') - H_eff

        # Use smart_inverse for efficient computation
        G_R = smart_inverse(A)

        if not sp.issparse(G_R):
            G_R = sp.csc_matrix(G_R)

        Gamma_L = 1j * (Sigma_L_full - Sigma_L_full.conj().T)
        Gamma_R = 1j * (Sigma_R_full - Sigma_R_full.conj().T)

        if not compute_lesser:
            return G_R, Gamma_L, Gamma_R

        f_L = self.fermi_distribution(np.real(E), self.dev.Vs, self.dev.kbT_eV)
        f_R = self.fermi_distribution(np.real(E), self.dev.Vd, self.dev.kbT_eV)

        Sigma_lesser = sp.lil_matrix((n, n), dtype=complex)
        Sigma_lesser +=  Gamma_L * f_L
        Sigma_lesser += Gamma_R * f_R
        Sigma_lesser = Sigma_lesser.tocsc()

        G_A = G_R.conj().T
        G_lesser = G_R @ Sigma_lesser @ G_A
        G_lesser_diag = G_lesser.diagonal()

        return G_R, G_lesser_diag, Gamma_L, Gamma_R

    def _compute_rgf_greens_function(self, E, compute_lesser, self_energy_method, equilibrium=False):
        """
        Generalized RGF computation using smart_inverse for all block inversions.
        Uses Hamiltonian.create_hamiltonian for block construction.
        """
        if equilibrium and compute_lesser:
            raise ValueError("Cannot compute lesser Green's function in equilibrium.")

        dagger = lambda A: np.conjugate(A.T)

        # Get blocks for current device type
        H_ii, H_ij = self.ham.create_sparse_channel_hamlitonian(blocks=True)
        num_blocks = len(H_ii)
        block_size = H_ii[0].shape[0]

        H_ii = [block.toarray() if sp.issparse(block) else block for block in H_ii]
        H_ij = [block.toarray() if sp.issparse(block) else block for block in H_ij]

        sigma_L = self.lead_self_energy.self_energy("left", E, self_energy_method)
        sigma_R = self.lead_self_energy.self_energy("right", E, self_energy_method)
        
        sigma_L = sigma_R
        if equilibrium:
            sigma_L *= 0
            sigma_R *= 0

        Gamma_L = 1j * (sigma_L - dagger(sigma_L))
        Gamma_R = 1j * (sigma_R - dagger(sigma_R))

        f_L = self.fermi_distribution(np.real(E), self.dev.Vs, self.dev.kbT_eV)
        f_R = self.fermi_distribution(np.real(E), self.dev.Vd, self.dev.kbT_eV)

        sigma_L_lesser = Gamma_L * f_L
        sigma_R_lesser = Gamma_R * f_R

        g_R = []
        g_lesser = []

        # Forward sweep - using smart_inverse
        H00_eff = H_ii[0] + sigma_L
        g0_R = smart_inverse(E * np.eye(block_size) - H00_eff)
        g_R.append(g0_R)

        if compute_lesser:
            g0_lesser = g0_R @ sigma_L_lesser @ dagger(g0_R)
            g_lesser.append(g0_lesser)

        for i in range(1, num_blocks):
            H_i_im1 = dagger(H_ij[i-1])
            sigma_recursive_R = H_i_im1 @ g_R[i - 1] @ H_ij[i-1]
            g_i_R = smart_inverse(E * np.eye(block_size) - H_ii[i] - sigma_recursive_R)
            g_R.append(g_i_R)
            if compute_lesser:
                sigma_recursive_lesser = H_i_im1 @ g_lesser[i-1] @ H_ij[i-1]
                g_i_lesser = g_R[i] @ sigma_recursive_lesser @ dagger(g_R[i])
                g_lesser.append(g_i_lesser)

        G_R = [None] * num_blocks
        G_lesser = [None] * num_blocks

        # Backward sweep - using smart_inverse
        H_N_Nm1 = dagger(H_ij[-1])
        sigma_eff_R = H_N_Nm1 @ g_R[-2] @ H_ij[-1]
        GN_R = smart_inverse(E * np.eye(block_size) - H_ii[-1] - sigma_R - sigma_eff_R)
        G_R[-1] = GN_R

        if compute_lesser:
            sigma_eff_lesser = H_N_Nm1 @ g_lesser[-2] @ H_ij[-1]
            total_sigma_lesser = sigma_R_lesser + sigma_eff_lesser
            GN_lesser = G_R[-1] @ total_sigma_lesser @ dagger(G_R[-1])
            G_lesser[-1] = GN_lesser

        for i in range(num_blocks - 2, -1, -1):
            H_i_ip1 = H_ij[i]
            H_ip1_i = dagger(H_i_ip1)
            propagator = g_R[i] @ H_i_ip1 @ G_R[i + 1] @ H_ip1_i
            G_R[i] = g_R[i] + propagator @ g_R[i]
            if compute_lesser:
                g_R_dag = dagger(g_R[i])
                term1 = g_lesser[i]
                term2 = (propagator @ g_lesser[i])
                term3 = (g_lesser[i] @ dagger(propagator))
                term4 = (g_R[i] @ H_i_ip1 @ G_lesser[i + 1] @ H_ip1_i @ g_R_dag)
                G_lesser[i] = term1 + term2 + term3 + term4

        G_R_diag = np.concatenate([np.diag(block) for block in G_R])

        if not compute_lesser:
            return G_R_diag, Gamma_L, Gamma_R

        G_lesser_diag = np.concatenate([np.diag(block) for block in G_lesser])
        return G_R_diag, G_lesser_diag, Gamma_L, Gamma_R
    


    def compute_density_of_states(self, E, self_energy_method=None, use_rgf=True, equilibrium=False):
        G_R_diag, _, _ = self.compute_central_greens_function(
            E, compute_lesser=False, use_rgf=use_rgf,
            self_energy_method=self_energy_method, equilibrium=equilibrium
        )
        dos = -np.imag(G_R_diag) / np.pi
        return np.maximum(dos, 0.0)

    def compute_electron_density(self, E, self_energy_method=None, use_rgf=True):
        _, G_lesser_diag, _, _ = self.compute_central_greens_function(
            E, compute_lesser=True, use_rgf=use_rgf,
            self_energy_method=self_energy_method
        )
        density = -np.imag(G_lesser_diag) / (2 * np.pi)
        return np.maximum(density, 0.0)
    
    def compute_transmission(self, E, self_energy_method=None):
        """
        Compute the transmission coefficient T(E) using the Caroli formula.
        T(E) = Tr[Γ_L * G_R * Γ_R * G_A]
        """
        # Get Green's function and broadening matrices
        G_R, Gamma_L, Gamma_R = self.compute_central_greens_function(
            E, use_rgf=False, self_energy_method=self_energy_method, compute_lesser=False
        )

        # Convert to dense arrays for matrix multiplication
        G_R_dense = G_R.toarray()
        G_A_dense = G_R_dense.conj().T
        Gamma_L_dense = Gamma_L.toarray()
        Gamma_R_dense = Gamma_R.toarray()

        # Calculate the transmission matrix product
        T_matrix = Gamma_L_dense @ G_R_dense @ Gamma_R_dense @ G_A_dense
        
        # The transmission is the trace of this matrix.
        # It should be real, but we take np.real to discard numerical noise.
        T = np.real(np.trace(T_matrix))
        
        return max(T, 0) # Ensure non-negative due to any remaining noise

    def compute_conductance(self, E_F=0.0, self_energy_method=None):
        """
        Compute the zero-temperature conductance G = (2e^2/h) * T(E_F).
        """
        # Conductance in units of G0 = 2e^2/h is simply the transmission at the Fermi energy.
        T = self.compute_transmission(E=E_F, self_energy_method=self_energy_method)
        return T