"""
Recursive Green's Function (RGF) implementation based on OpenMX TRAN approach.

This implementation follows the TRAN_Calc_CentGreen approach for computing 
Green's functions efficiently using sparse matrices and block-wise operations.

The core computation is:
G_R(E) = (E*S - H - Σ_L - Σ_R)^(-1)
G_<(E) = G_R @ Σ_< @ G_A

where:
- E is energy (complex with small imaginary part)
- S is overlap matrix (identity for tight-binding)
- H is Hamiltonian matrix  
- Σ_L, Σ_R are left/right lead self-energies
- Σ_< = Γ_L*f_L + Γ_R*f_R (lesser self-energy)
- Γ = i(Σ - Σ†) (broadening functions)
- f_L, f_R are Fermi distributions in the leads

Author: Based on OpenMX TRAN implementation
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve, inv
from device import Device
from hamiltonian import Hamiltonian
from lead_self_energy import LeadSelfEnergy
import warnings


class GreensFunction:

    
    def __init__(self, device: Device, hamiltonian: Hamiltonian, self_energy_method="sancho_rubio"):
        """
        Initialize the Green's function calculator.
        
        Args:
            device: Device object containing physical parameters
            hamiltonian: Hamiltonian object for matrix construction
        """
        self.device = device
        self.ham = hamiltonian
        self.lead_self_energy = LeadSelfEnergy(device, hamiltonian)
        self.self_energy_method = self_energy_method
        # Numerical parameters
        self.eta = 1e-5  # Small imaginary part for numerical stability
        self.sparse_threshold = 0.1  # Use sparse solver if density < threshold
        
    def fermi_distribution(self, E, mu, kT):
        """Fermi-Dirac distribution function."""
        x = (E - mu) / kT
        # Avoid overflow in exponential
        return 1.0 / (1.0 + np.exp(x))
    
    def add_eta(self, E):
        """Add small imaginary part for numerical stability."""
        if np.imag(E) == 0:
            return E + 1j * self.eta
        return E
    
    def compute_central_greens_function(self, E, ky=0.0, compute_lesser=True, 
                                      use_rgf=True, self_energy_method=None):
        """
        Compute central region Green's function
        
        This is the main function that computes G_R and optionally G_< for 
        the central device region.
        
        Args:
            E: Energy (real or complex)
            ky: Transverse momentum
            compute_lesser: Whether to compute lesser Green's function
            use_rgf: Whether to use RGF for large systems
            self_energy_method: Method for surface Green's function calculation
            
        Returns:
            If compute_lesser=True: (G_R_diag, G_lesser_diag, Gamma_L, Gamma_R)
            If compute_lesser=False: (G_R_diag, Gamma_L, Gamma_R)
            where *_diag are 1D arrays of diagonal elements
        """
        E = self.add_eta(E)
        
        # Use instance method if not specified
        if self_energy_method is None:
            self_energy_method = self.self_energy_method
        
        # Get Hamiltonian and self-energies
        if use_rgf:
            # Use block-wise RGF for large systems
            return self._compute_rgf_greens_function(E, ky, compute_lesser, self_energy_method)
        else:
            # Direct matrix inversion for smaller systems
            return self._compute_direct_greens_function(E, ky, compute_lesser, self_energy_method)
    
    def _compute_direct_greens_function(self, E, ky, compute_lesser, self_energy_method):

        # Get channel Hamiltonian
        H = self.ham.create_sparse_channel_hamlitonian(ky, blocks=False)
        
        # Get self-energies using existing implementation
        sigma_L = self.lead_self_energy.self_energy("left", E, ky, self_energy_method)
        sigma_R = self.lead_self_energy.self_energy("right", E, ky, self_energy_method)
        
        # Convert to sparse matrices
        if not sp.issparse(H):
            H = sp.csc_matrix(H)
        
        n = H.shape[0]
        block_size = sigma_L.shape[0]
        
        # Create identity matrix (overlap matrix S = I for tight-binding)
        I = sp.identity(n, dtype=complex, format='csc')
        
        # Add self-energies to appropriate blocks efficiently
        # Create full self-energy matrices 
        Sigma_L_full = sp.csc_matrix((n, n), dtype=complex)
        Sigma_R_full = sp.csc_matrix((n, n), dtype=complex)
        
        # Use sparse matrix construction to avoid efficiency warnings
        from scipy.sparse import lil_matrix
        Sigma_L_lil = lil_matrix((n, n), dtype=complex)
        Sigma_R_lil = lil_matrix((n, n), dtype=complex)
        
        # Left self-energy goes to top-left block
        Sigma_L_lil[:block_size, :block_size] = sigma_L
        # Right self-energy goes to bottom-right block
        Sigma_R_lil[-block_size:, -block_size:] = sigma_R
        
        # Convert back to CSC for efficient arithmetic
        Sigma_L_full = Sigma_L_lil.tocsc()
        Sigma_R_full = Sigma_R_lil.tocsc()
        
        
        
        # Effective Hamiltonian
        H_eff = H + Sigma_L_full + Sigma_R_full
        
        # Compute central Green's function: G_R = (E*I - H_eff)^(-1)
        A = E * I - H_eff
        
        # Choose solver based on matrix density
        density = A.nnz / (A.shape[0] * A.shape[1])
        
        if density > self.sparse_threshold:
            # Use dense solver for dense matrices
            G_R = inv(A.toarray())
            G_R = sp.csc_matrix(G_R)
        else:
            # Use sparse solver
            G_R = self._sparse_matrix_inverse(A)
        
        # Extract diagonal elements
        G_R_diag = np.array([G_R[i, i] for i in range(n)])
        
        # Compute broadening functions
        Gamma_L = 1j * (sigma_L - sigma_L.conj().T)
        Gamma_R = 1j * (sigma_R - sigma_R.conj().T)
        
        if not compute_lesser:
            return G_R_diag, Gamma_L, Gamma_R
        
        # Compute lesser Green's function
        f_L = self.fermi_distribution(E, self.device.Vs, self.device.kbT_eV)
        f_R = self.fermi_distribution(E, self.device.Vd, self.device.kbT_eV)
        
        # Use LIL format for efficient construction
        Sigma_lesser_lil = lil_matrix((n, n), dtype=complex)
        Sigma_lesser_lil[:block_size, :block_size] = Gamma_L * f_L
        Sigma_lesser_lil[-block_size:, -block_size:] = Gamma_R * f_R
        Sigma_lesser = Sigma_lesser_lil.tocsc()
        
        G_A = G_R.conj().T  # Advanced Green's function
        G_lesser = G_R @ Sigma_lesser @ G_A
        
        G_lesser_diag = np.array([G_lesser[i, i] for i in range(n)])
        
        return G_R_diag, G_lesser_diag, Gamma_L, Gamma_R
    
    def _compute_rgf_greens_function(self, E, ky, compute_lesser, self_energy_method):
        """
        RGF computation, lesser greens function is wrong!!!
        """
        try:
            # Get full Hamiltonian matrix instead of blocks
            H = self.ham.create_sparse_channel_hamlitonian(ky, blocks=False)
            
            # Convert to dense for compatibility with the reference implementation
            if sp.issparse(H):
                H = H.toarray()
            
            # Get self-energies and construct full matrices
            sigma_L_block = self.lead_self_energy.self_energy("left", E, ky, self_energy_method)
            sigma_R_block = self.lead_self_energy.self_energy("right", E, ky, self_energy_method)
            
            N = H.shape[0]
            
            # Create full self-energy matrices
            self_energy_left = np.zeros_like(H, dtype=complex)
            self_energy_right = np.zeros_like(H, dtype=complex)
            dagger = lambda A: np.conjugate(A.T)
            # Apply self-energies to boundary blocks
            self_energy_size = sigma_L_block.shape[0]
            self_energy_left[:self_energy_size, :self_energy_size] = sigma_L_block
            self_energy_right[-self_energy_size:, -self_energy_size:] = sigma_R_block
            Gamma_L = 1j * (self_energy_left - self_energy_left.conj().T)
            Gamma_R = 1j * (self_energy_right - self_energy_right.conj().T)
  
            # Compute lesser Green's function
            f_L = self.fermi_distribution(np.real(E), self.device.Vs, self.device.kbT_eV)
            f_R = self.fermi_distribution(np.real(E), self.device.Vd, self.device.kbT_eV)
            self_energy_lesser = Gamma_L * f_L + Gamma_R * f_R
            # Determine block size from device structure
            # This should match the orbital structure
            block_size = self.ham.Nz * 2 * 10  # Use self-energy size as block size
            num_blocks = N // block_size
            
            if num_blocks * block_size != N:
                raise ValueError(f"Matrix size {N} not divisible by block size {block_size}")
            
            # Construct A matrix: A = E*I - H - Σ_L - Σ_R  
            E_matrix = np.eye(N, dtype=complex) * E
            A = E_matrix - H - self_energy_left - self_energy_right
            I_blk = np.eye(block_size, dtype=complex)
            
            # Initialize arrays (this style of storing data was taken from jiezi program)
            g_R_blocks = []
            g_lesser_blocks = []

            
            G_R = [None] * num_blocks
            G_R_1 = [None] * (num_blocks - 1)
            G_lesser = [None] * num_blocks
            G_lesser_1 = [None] * (num_blocks - 1)
            
            # Forward recursion: Calculate diagonal blocks of g_R
            for i in range(num_blocks):
                start = i * block_size
                end = (i+1) * block_size
                prev = (i - 1) * block_size
                if i == 0:
                    # First block
                    g_0_r = np.linalg.inv(A[start:end, start:end])
                    g_R_blocks.append(g_0_r)
                    #g_lesser
                    g_0_lesser = g_0_r @ self_energy_lesser[start:end, start:end] @ dagger(g_0_r)
                    g_lesser_blocks.append(g_0_lesser)
                else:          
                    
                    H_eff = A[start:end, start:end] - A[start:end, prev:start] @ g_R_blocks[i-1] @ A[prev:start, start:end]
                    g_R_blocks.append(np.linalg.inv(H_eff))

                    #g_i_lesser calculation
                    sigma_lesser = A[start:end, prev:start] @ g_lesser_blocks[i - 1] @ dagger(A[prev:start, start:end])
                    g_i_lesser = g_R_blocks[i] @ (self_energy_lesser[start: end, start: end] + sigma_lesser - \
                        self_energy_lesser[start:end, prev:start] @ dagger(g_R_blocks[i - 1]) @ dagger(A[prev:start, start:end]) - \
                            A[start:end, prev:start] @ g_R_blocks[i-1] @ self_energy_lesser[prev:start, start:end]) @ dagger(g_R_blocks[i])
                    g_lesser_blocks.append(g_i_lesser)        

            G_R[-1] = g_R_blocks[-1]
            G_lesser[-1] = g_lesser_blocks[-1]

            for i in reversed(range(num_blocks - 1)):
                start = i * block_size
                end = (i+1)*block_size
                after = (i+2)*block_size

                
                # Dyson equation for current block
                G_R[i] = g_R_blocks[i] @ (np.eye(block_size) + 
                A[start:end, end:after]@G_R[i+1]@A[end:after, start:end]@g_R_blocks[i])
                
                G_R_1[i] = -G_R[i + 1] @ A[end:after, start:end] @ g_R_blocks[i]
                
            
                #lesser function
                
                gr0 = np.linalg.inv(E * np.eye(block_size) - H[start:end, start:end]) 
                ga0 = dagger(gr0)
                gr1 = np.linalg.inv(E * np.eye(block_size) - H[end:after, end:after])
                ga1 = dagger(gr1)
                gqq1 = gr0 @ self_energy_lesser[start:end, end:after] @ ga1
                gq1q = gr1 @ self_energy_lesser[end:after, start:end] @ ga0 
                
                G_i_lesser = g_lesser_blocks[i] + g_R_blocks[i] @ (A[start:end, end:after] @ G_lesser[i + 1] @ dagger(A[end:after, start:end])) @ dagger(g_R_blocks[i]) - \
                    (g_lesser_blocks[i] @ A[end:after, start:end] @ dagger(G_R_1[i].T) + G_R_1[i].T @ A[end:after, start:end] @ g_lesser_blocks[i]) - \
                        (gqq1 @ dagger(A[end:after, start:end]) @ dagger(G_R[i]) + G_R[i] @ A[start:end, end:after] @ gq1q)
                
                G_lesser[i] = G_i_lesser
                
                G_i_lesser_1 = gq1q - G_R_1[i] @ A[start:end, end:after] @ gq1q - G_R[i+1] @ A[end:after,start:end] @ g_lesser_blocks[i] - \
                    G_lesser[i+1] @ dagger(A[end:after,start:end]) @ dagger(g_R_blocks[i])
                
                G_lesser_1[i] = G_i_lesser_1[0]
            
            # Extract diagonal elements and return results
            G_R_diag = np.concatenate([np.diag(block) for block in G_R])
            
            # Compute broadening functions for return
            Gamma_L = 1j * (sigma_L_block - sigma_L_block.conj().T)
            Gamma_R = 1j * (sigma_R_block - sigma_R_block.conj().T)
            
            if not compute_lesser:
                return G_R_diag, Gamma_L, Gamma_R
            
            # Extract diagonal elements from lesser Green's function
            G_lesser_diag = np.concatenate([np.diag(block) for block in G_lesser])
            
            return G_R_diag, G_lesser_diag, Gamma_L, Gamma_R
            
        except Exception as e:
            raise RuntimeError(f"RGF method failed: {str(e)}")
    def _solve_block_matrix(self, A):
        """
        Solve block matrix equation A @ X = I efficiently.
        
        Use the same method as the direct approach for consistency.
        """
        try:
            # Always use dense solver for consistency with direct method
            A_dense = A.toarray() if hasattr(A, 'toarray') else A
            X = inv(A_dense)
            return sp.csc_matrix(X)
        except:
            # Fallback: try scipy.linalg.solve which is more numerically stable
            try:
                A_dense = A.toarray() if hasattr(A, 'toarray') else A
                n = A_dense.shape[0]
                I_dense = np.eye(n, dtype=A_dense.dtype)
                X = solve(A_dense, I_dense)
                return sp.csc_matrix(X)
            except:
                # Last resort: pseudo-inverse
                A_dense = A.toarray() if hasattr(A, 'toarray') else A
                X = np.linalg.pinv(A_dense)
                return sp.csc_matrix(X)
    
    def _sparse_matrix_inverse(self, A):
        """
        Compute sparse matrix inverse efficiently using spsolve(A, I).
        
        Args:
            A: Sparse matrix to invert
            
        Returns:
            Sparse matrix inverse
        """
        n = A.shape[0]
        I = sp.identity(n, dtype=A.dtype, format='csc')
        
        try:
            # Use spsolve to solve A @ X = I directly
            A_inv = spsolve(A, I)
            
            # spsolve returns dense array when solving with identity matrix
            # Convert back to sparse for consistency
            return sp.csc_matrix(A_inv)
            
        except:
            # Fallback to dense computation
            try:
                A_inv_dense = inv(A.toarray())
                return sp.csc_matrix(A_inv_dense)
            except:
                # Last resort: pseudo-inverse
                warnings.warn("Using pseudo-inverse for matrix inversion - numerical instability possible")
                A_inv_dense = np.linalg.pinv(A.toarray())
                return sp.csc_matrix(A_inv_dense)
    
    def _extract_diagonal(self, sparse_matrix):
        """Extract diagonal elements from sparse matrix."""
        if hasattr(sparse_matrix, 'diagonal'):
            return sparse_matrix.diagonal()
        else:
            # For dense matrices
            return np.diag(sparse_matrix)
    
    def approx_compute_transmission(self, E, ky=0.0, self_energy_method=None):
        """
        Compute approximate transmission coefficient T(E) = Tr[Γ_L @ G_R @ Γ_R @ G_A].
        This method is for testing purposes
        Args:
            E: Energy
            ky: Transverse momentum  
            self_energy_method: Method for self-energy calculation
            
        Returns:
            Transmission coefficient (real number)
        """
        # Get Green's function and broadening functions
        G_R_diag, Gamma_L, Gamma_R = self.compute_central_greens_function(
            E, ky, compute_lesser=False, use_rgf=True, 
            self_energy_method=self_energy_method
        )
        G_R_trace = np.sum(G_R_diag)
        G_A_trace = np.sum(G_R_diag.conj())
        
        Gamma_L_trace = np.trace(Gamma_L)
        Gamma_R_trace = np.trace(Gamma_R)
        
        # Approximate transmission (exact calculation would need full matrices)
        T = np.real(Gamma_L_trace * G_R_trace * Gamma_R_trace * G_A_trace)
        
        return max(0.0, T)  # Ensure non-negative
    
    def compute_density_of_states(self, E, ky=0.0, self_energy_method=None):
        """
        Compute local density of states: DOS(E) = -Im[G_R]/pi.
        
        Args:
            E: Energy
            ky: Transverse momentum
            self_energy_method: Method for self-energy calculation
            
        Returns:
            1D array of local DOS values
        """
        G_R_diag, _, _ = self.compute_central_greens_function(
            E, ky, compute_lesser=False, use_rgf=True,
            self_energy_method=self_energy_method
        )
        
        dos = -np.imag(G_R_diag) / np.pi
        
        return np.maximum(dos, 0.0)  # Ensure non-negative
    
    def compute_electron_density(self, E, ky=0.0, self_energy_method=None, use_rgf=False):
        """
        Compute local electron density: n(r) = -Im[G_<]/π.
        
        Args:
            E: Energy
            ky: Transverse momentum
            self_energy_method: Method for self-energy calculation
            
        Returns:
            1D array of local electron density values
        """
        _, G_lesser_diag, _, _ = self.compute_central_greens_function(
            E, ky, compute_lesser=True, use_rgf=use_rgf,
            self_energy_method=self_energy_method
        )
        
        # Electron density = -Im[G_<]/π
        density = -np.imag(G_lesser_diag) / np.pi
        
        return np.maximum(density, 0.0)  # Ensure non-negative


# Convenience functions for backward compatibility
def rgf(device, hamiltonian, E, ky=0.0, compute_lesser=True, **kwargs):
    """
    Convenience function for computing Green's functions using TRAN approach.
    
    Args:
        device: Device object
        hamiltonian: Hamiltonian object
        E: Energy
        ky: Transverse momentum
        compute_lesser: Whether to compute lesser Green's function
        **kwargs: Additional arguments passed to compute_central_greens_function
        
    Returns:
        Green's function results
    """
    gf = GreensFunction(device, hamiltonian)
    return gf.compute_central_greens_function(E, ky, compute_lesser, **kwargs)


def transmission(device, hamiltonian, E, ky=0.0, **kwargs):
    """
    Convenience function for computing transmission coefficient.
    
    Args:
        device: Device object
        hamiltonian: Hamiltonian object  
        E: Energy
        ky: Transverse momentum
        **kwargs: Additional arguments
        
    Returns:
        Transmission coefficient
    """
    gf = GreensFunction(device, hamiltonian)
    return gf.approx_compute_transmission(E, ky, **kwargs)


def dos(device, hamiltonian, E, ky=0.0, **kwargs):

    gf = GreensFunction(device, hamiltonian)
    return gf.compute_density_of_states(E, ky, **kwargs)