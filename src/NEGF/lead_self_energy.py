

from device import Device
import numpy as np
from scipy import linalg
import scipy.sparse as spa
from hamiltonian import Hamiltonian

class LeadSelfEnergy():
    """
    Lead self-energy calculation using surface Green's functions.
    Based on the robust implementations from OpenMX TRAN_Calc_SurfGreen.c
    """
    
    def __init__(self, device: Device, hamiltonian: Hamiltonian):
        self.ds = device
        self.ham = hamiltonian
        self.eta = 1e-6  # Small imaginary part for numerical stability
        
    def _add_eta(self, E):
        """Add small imaginary part if energy is real for numerical stability"""
        if np.imag(E) == 0:
            return E + 1j * self.eta
        return E
        
    def surface_greens_function(self, E, H00, H01, method="sancho_rubio", 
                               iteration_max=1000, tolerance=1e-6):
        """
        Calculate surface Green's function using specified method.
        
        Args:
            E: Energy (complex)
            H00: On-site Hamiltonian matrix
            H01: Hopping matrix (coupling to next layer)
            method: "sancho_rubio", "iterative", or "transfer"
            iteration_max: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Surface Green's function matrix
        """
        E = self._add_eta(E)
        
        if method == "sancho_rubio":
            return self._sancho_rubio_surface_gf(E, H00, H01, tolerance, iteration_max)
        elif method == "iterative":
            return self._iterative_surface_gf(E, H00, H01, tolerance, iteration_max)
        elif method == "transfer":
            return self._transfer_surface_gf(E, H00, H01, tolerance, iteration_max)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _sancho_rubio_surface_gf(self, E, H00, H01, tolerance=1e-6, iteration_max=1000):
        """
        Standard Sancho-Rubio algorithm for surface Green's function.
        Based on TRAN_Calc_SurfGreen_Normal from OpenMX.
        """
        n = H00.shape[0]
        I = np.eye(n, dtype=complex)
        
        # Convert to dense arrays for stability
        if hasattr(H00, 'toarray'):
            H00 = H00.toarray()
        if hasattr(H01, 'toarray'):
            H01 = H01.toarray()
            
        H10 = H01.conj().T
        
        # Initialize
        es0 = E * I - H00  # Surface term
        e00 = E * I - H00  # Bulk term
        alp = H01.copy()   # Forward coupling
        bet = H10.copy()   # Backward coupling
        
        # Initial surface Green's function
        try:
            gr = linalg.solve(es0, I)
        except linalg.LinAlgError:
            gr = linalg.pinv(es0)
        
        gr_old = gr.copy()
        
        for iteration in range(1, iteration_max):
            try:
                # Invert (E*I - e00)
                inv_e00 = linalg.solve(e00, I)
            except linalg.LinAlgError:
                inv_e00 = linalg.pinv(e00)
            
            # Update surface term
            temp1 = inv_e00 @ bet
            temp2 = alp @ temp1
            es0 = es0 - temp2
            
            # Update bulk term  
            temp3 = inv_e00 @ alp
            temp4 = bet @ temp3
            temp5 = alp @ temp1
            e00 = e00 - temp4 - temp5
            
            # Update coupling terms
            alp = alp @ temp3
            bet = bet @ temp1
            
            # Calculate new surface Green's function
            try:
                gr = linalg.solve(es0, I)
            except linalg.LinAlgError:
                gr = linalg.pinv(es0)
            
            # Check convergence
            diff = gr - gr_old
            rms = np.sqrt(np.max(np.abs(diff)**2))
            
            if rms < tolerance:
                break
                
            gr_old = gr.copy()
        
        if iteration >= iteration_max - 1:
            print(f"Warning: Surface GF did not converge after {iteration_max} iterations, rms={rms}")
            
        return gr
    
    def _iterative_surface_gf(self, E, H00, H01, tolerance=1e-6, iteration_max=1000):
        """
        Alternative iterative method for surface Green's function.
        Based on TRAN_Calc_SurfGreen_Multiple_Inverse from OpenMX.
        """
        n = H00.shape[0]
        I = np.eye(n, dtype=complex)
        
        # Convert to dense arrays
        if hasattr(H00, 'toarray'):
            H00 = H00.toarray()
        if hasattr(H01, 'toarray'):
            H01 = H01.toarray()
        
        # h0 = E*I - H00
        h0 = E * I - H00
        
        # hl = H01, hr = H01^dagger
        hl = H01.copy()
        hr = H01.conj().T
        
        # Initial Green's function
        try:
            g0 = linalg.solve(h0, I)
        except linalg.LinAlgError:
            g0 = linalg.pinv(h0)
        
        for iteration in range(1, iteration_max):
            # Calculate hl*g0*hr
            temp1 = hl @ g0
            temp2 = temp1 @ hr
            
            # New denominator: h0 - hl*g0*hr
            h_new = h0 - temp2
            
            try:
                g_new = linalg.solve(h_new, I)
            except linalg.LinAlgError:
                g_new = linalg.pinv(h_new)
            
            # Check convergence
            diff = g_new - g0
            rms = np.sqrt(np.max(np.abs(diff)**2))
            
            if rms < tolerance:
                break
                
            g0 = g_new.copy()
        
        if iteration >= iteration_max - 1:
            print(f"Warning: Iterative Surface GF did not converge after {iteration_max} iterations")
            
        return g0
    
    def _transfer_surface_gf(self, E, H00, H01, tolerance=1e-6, iteration_max=1000):
        """
        Transfer matrix method for surface Green's function.
        Based on TRAN_Calc_SurfGreen_transfer from OpenMX.
        """
        n = H00.shape[0]
        I = np.eye(n, dtype=complex)
        
        # Convert to dense arrays
        if hasattr(H00, 'toarray'):
            H00 = H00.toarray()
        if hasattr(H01, 'toarray'):
            H01 = H01.toarray()
        
        H10 = H01.conj().T
        
        # Initial inverse: (E*I - H00)^-1
        try:
            gr00_inv = linalg.solve(E * I - H00, I)
        except linalg.LinAlgError:
            gr00_inv = linalg.pinv(E * I - H00)
        
        # Initial transfer matrices
        t_i = gr00_inv @ H10
        bar_t_i = gr00_inv @ H01
        
        T_i = t_i.copy()
        bar_T_i = bar_t_i.copy()
        
        T_i_old = T_i.copy()
        
        for iteration in range(1, iteration_max):
            # Calculate (I - t*bar_t - bar_t*t)^-1
            temp1 = t_i @ bar_t_i
            temp2 = bar_t_i @ t_i
            denominator = I - temp1 - temp2
            
            try:
                inv_denom = linalg.solve(denominator, I)
            except linalg.LinAlgError:
                inv_denom = linalg.pinv(denominator)
            
            # Update transfer matrices
            t_i_new = inv_denom @ (t_i @ t_i)
            bar_t_i_new = inv_denom @ (bar_t_i @ bar_t_i)
            
            # Update accumulated transfer matrices
            bar_T_i_new = bar_T_i @ bar_t_i_new
            T_i_new = T_i + bar_T_i @ t_i_new
            
            # Check convergence
            diff = T_i_new - T_i_old
            rms = np.sqrt(np.max(np.abs(diff)**2))
            
            if rms < tolerance:
                break
            
            # Update for next iteration
            t_i = t_i_new
            bar_t_i = bar_t_i_new
            T_i_old = T_i.copy()
            T_i = T_i_new
            bar_T_i = bar_T_i_new
        
        if iteration >= iteration_max - 1:
            print(f"Warning: Transfer Surface GF did not converge after {iteration_max} iterations")
        
        # Final surface Green's function
        final_matrix = E * I - H00 - H01 @ T_i
        try:
            return linalg.solve(final_matrix, I)
        except linalg.LinAlgError:
            return linalg.pinv(final_matrix)
    
    def self_energy(self, side, E, ky, method="sancho_rubio"):
        """
        Calculate lead self-energy using surface Green's functions.
        
        Args:
            side: "left" or "right" lead
            E: Energy
            ky: Transverse momentum
            method: Surface GF method ("sancho_rubio", "iterative", "transfer")
            
        Returns:
            Self-energy matrix
        """
        # Get lead Hamiltonian matrices
        H00, H01, H10 = self.ham.get_H00_H01_H10(ky, side=side, sparse=False)
        
        # Handle large energies
        if np.abs(E) > 5e5:
            return np.zeros((H00.shape[0], H00.shape[0]), dtype=complex)
        
        # Apply bias voltage
        if side == "left":
            E_lead = E - self.ds.Vs
        else:  # right
            E_lead = E - self.ds.Vd
        
        # Calculate surface Green's function
        try:
            G_surface = self.surface_greens_function(E_lead, H00, H01, method=method)
        except Exception as e:
            print(f"Warning: Surface GF calculation failed for {side} lead: {e}")
            # Fallback to simple approximation
            n = H00.shape[0]
            eta = 1e-3  # Larger eta for stability
            G_surface = linalg.pinv(self._add_eta(E_lead) * np.eye(n) - H00)
        
        # Calculate self-energy - use same formula for both leads at zero bias
        # For symmetric leads, both should use the same computation
        if side == "left":
            # Self-energy: Σ_L = H10 @ G_surface @ H01
            self_energy = H10 @ G_surface @ H01
            # Extract top-left block (coupling to first supercell)
            device_size = 4 * self.ham.Nz * 2 * 10  # 4 layers × Nz × 2 atoms × 10 orbitals
            return self_energy[:device_size, :device_size]
        else:  # right
            # For symmetric leads at zero bias, use same formula but extract different block
            # This ensures the physics is symmetric
            self_energy = H10 @ G_surface @ H01  # Same as left!
            # Extract bottom-right block (coupling to second supercell)
            device_size = 4 * self.ham.Nz * 2 * 10
            return self_energy[-device_size:, -device_size:]
