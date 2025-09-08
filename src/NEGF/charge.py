import os

# --- Add these lines at the very top of your script ---
# This must be done BEFORE importing numpy or other scientific libraries.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import multiprocessing
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import scipy as sp
from scipy import linalg
import scipy.sparse as spa
import scipy.sparse.linalg as spla
from lead_self_energy import LeadSelfEnergy
from device import Device
from poisson import PoissonSolver
from rgf import GreensFunction
from hamiltonian import Hamiltonian
import time

class Charge():
    def __init__(self, device : Device):
        self.device = device
        
        self.poisson = PoissonSolver(device)
        self.ham = Hamiltonian(device)
        self.GF = GreensFunction(device, self.ham)
        self.lse = LeadSelfEnergy(self.device, self.ham)
        self.num_energy = 200
        self.energy_range = np.linspace(-5,5, 100)
        self.k_space = np.linspace(0,1,32) # exact number of processes 
        
        self.atoms = self.ham.unitCell.ATOM_POSITIONS # the atoms         
        
        # These are lists that give quasi fermi energy and potential at location of every atom, needs to be initialized every time 
        self.smearedEFN = np.zeros((self.device.nx * self.device.nz)) # EFN has same dimension as poisson solver 
        self.smearedPhi = self.device.potential
        self.smearedLDOS = np.zeros((self.device.nx, self.device.nz, self.num_energy))
    
        self.weights = {}
        
        
    def unsmear_to_smear(self, A : dict):
        """performs interpolation to get values that match up with poisson matrix (DO THIS AT END OF MP FOR ALL CALCULATIONS)"""
        max_X = self.device.unitX / 4
        max_Z = self.device.unitZ 
        nx, nz = self.device.nx, self.device.nz
        
        # Create regular grid coordinates
        x_grid = np.linspace(0, max_X, nx)
        z_grid = np.linspace(0, max_Z, nz)
        X_grid, Z_grid = np.meshgrid(x_grid, z_grid, indexing='ij')
        
        # Extract coordinates and values from dictionary (ignoring y-component)
        points = []
        values = []
        for coord, value in A.items():
            x, y, z = coord  
            points.append([x, z])
            values.append(value)
        
        points = np.array(points)
        values = np.array(values)
        
        # Create target grid points for interpolation
        grid_points = np.column_stack([X_grid.ravel(), Z_grid.ravel()])
        
        from scipy.interpolate import griddata
        interpolated_values = griddata(points, values, grid_points, method='nearest', fill_value=0.0)
        smeared_array = interpolated_values.reshape((nx, nz))
        
        return smeared_array
        
        
        
    
    def calculate_real_GR(self, E):
        G_R_k_list = self.gf_calculations_k_space(E)
        
        # 2. Check for any inconsistencies.
        if not G_R_k_list:
            print("Warning: No Green's functions were calculated.")
            return None
        if len(G_R_k_list) != len(self.k_space):
            raise ValueError("Mismatch between the number of calculated GFs and k-points.")

        # 3. Initialize the real-space Green's function matrix with complex numbers.
        G_R_REAL = np.zeros_like(G_R_k_list[0], dtype=np.complex128)

        # 4. Perform the discrete Fourier transform.
        # We iterate through both the k-space GFs and the normalized k-points simultaneously.
        for G_R_k, k_norm in zip(G_R_k_list, self.k_space):
            G_R_REAL += G_R_k 

        dk = (self.k_space[1] - self.k_space[0]) * 2*np.pi / (5.431e-10)
        
        G_R_REAL = 1 / (2 * np.pi) *  G_R_REAL * dk
        
        return G_R_REAL
        

    def gf_calculations_k_space(self, E) -> list:
        """Uses multiprocessing to cache GF for E,ky """
        
        # Handle both single energy values and lists
        if np.isscalar(E):
            E_list = [E]
        else:
            E_list = E
            
        param_grid = list(product(E_list, self.k_space))

        print(f"Starting DOS calculations for {len(param_grid)} (E, ky) pairs...")
        print(f"ky range: {self.k_space[0]:.2f} to {self.k_space[-1]:.2f}")

        start_time = time.time()

        with multiprocessing.Pool(processes=32) as pool:
            results = pool.map(self._calculate_gf_simple, param_grid)
        end_time = time.time()
        return results
    
    def _calculate_gf_simple(self, param):
        """Simple Green's function calculation"""
        energy, ky = param
        G_R_diag, Gamma_L, Gamma_R = self.GF.compute_central_greens_function(energy, ky, compute_lesser=False)
        
        return G_R_diag
    
    def calculate_LDOS(self, E) -> dict:
        """
        Calculate Local Density of States (LDOS) at each atom position.
        
        Args:
            E: Energy point
            
        Returns:
            dict: Dictionary mapping atom positions to LDOS values
        """
        print(f"Calculating LDOS at E={E:.3f} eV")
        
        # Use multiprocessing for k-space integration
        param_grid = [(E, ky) for ky in self.k_space]
        
        with multiprocessing.Pool(processes=32) as pool:
            ldos_results = pool.map(self._calculate_ldos_point_mp, param_grid)
        
        # Average over k-space and create dictionary
        num_atoms = len(self.atoms)
        total_ldos = np.zeros(num_atoms)
        
        for ldos_array in ldos_results:
            total_ldos += ldos_array
        
        # Normalize by k-space integration
        dk = (self.k_space[1] - self.k_space[0]) * 2*np.pi / (5.431e-10) if len(self.k_space) > 1 else 1.0
        total_ldos = total_ldos / (2 * np.pi) * dk
        
        # Create dictionary mapping atom positions to LDOS
        LDOS_points = {}
        for atom_idx, atom in enumerate(self.atoms):
            LDOS_points[atom.getPos()] = total_ldos[atom_idx]
        
        return LDOS_points
    
    def _calculate_ldos_point_mp(self, params):
        """Multiprocessing helper for LDOS calculation"""
        energy, ky = params
        return self._calculate_ldos_at_atoms(energy, ky)
    
    def _calculate_ldos_at_atoms(self, energy, ky):
        """
        Calculate LDOS at each atom for a single (E, ky) point.
        
        Args:
            energy: Energy point
            ky: Transverse momentum
            
        Returns:
            np.array: LDOS at each atom position
        """
        try:
            # Get DOS array from RGF
            dos_array = self.GF.compute_density_of_states(energy, ky)
            
            # Sum over 10 orbitals for each atom
            num_atoms = len(self.atoms)
            num_orbitals_per_atom = 10
            
            if len(dos_array) != num_atoms * num_orbitals_per_atom:
                raise ValueError(f"DOS array size {len(dos_array)} doesn't match "
                               f"expected size {num_atoms * num_orbitals_per_atom}")
            
            # Reshape and sum over orbitals for each atom
            dos_per_atom = dos_array.reshape(num_atoms, num_orbitals_per_atom)
            atom_ldos = np.sum(dos_per_atom, axis=1)
            
            return atom_ldos
            
        except Exception as e:
            print(f"Error in LDOS calculation at (E={energy:.3f}, ky={ky:.3f}): {e}")
            return np.zeros(len(self.atoms))
    
    def calculate_smeared_LDOS(self, E):
        LDOS_points = self.calculate_LDOS(E)
        return self.unsmear_to_smear(LDOS_points)
        
    
    def fermi(self, E, mod="False"):
        EFN, Phi = self.smearedEFN, self.smearedPhi
        if mod:
            raise NotImplemented
        else:
            return 1 /(1 + np.exp((E*np.ones_like(EFN) - Phi - EFN) / self.device.kbT))
    
    def compute_EFN_helper(self, E):
        """This will be part for the bracket bisect way of finding the EFN
        , we take EC as 1.2 for now (note even if 'EC' is actually in the bandgap it 
        makes no diff since LDOS is zero in bandgap)
        
        this function also uses vectorization to find EFN in entire device at the same time 
        """
        LDOS_points = self.calculate_LDOS(E)
        
        # all are numpy arrays
        return LDOS_points * self.fermi(E)
    
    def calculate_EC(self):
        """Find conduction band minimum from DOS calculation"""
        energy_range = self.energy_range
        _, dos_values = self.calculate_DOS(energy_range=energy_range, save_data=False)
        
        # Find first significant DOS peak (conduction band minimum)
        dos_threshold = 0.01 * np.max(dos_values)
        significant_indices = np.where(dos_values > dos_threshold)[0]
        
        if len(significant_indices) > 0:
            # Look for the first peak above zero energy (conduction band)
            positive_energy_indices = significant_indices[energy_range[significant_indices] > 0]
            if len(positive_energy_indices) > 0:
                ec_index = positive_energy_indices[0]
                return energy_range[ec_index]
        
        # Default fallback
        return 1.2
    
    def calculate_DOS(self, energy_range=None, ky_range=None, method="sancho_rubio", 
                     eta=1e-6, save_data=True, filename="dos_data.txt"):
        """
        Calculate Density of States (DOS) using RGF with multiprocessing.
        
        Args:
            energy_range: Energy points for DOS calculation
            ky_range: Transverse momentum points  
            method: Surface GF method ("sancho_rubio", "iterative", "transfer")
            eta: Small imaginary part for broadening
            save_data: Whether to save DOS data to file
            filename: Output filename for DOS data
            
        Returns:
            energies, total_dos: Energy points and corresponding DOS values
        """
        if energy_range is None:
            energy_range = np.linspace(-2.0, 2.0, 200)
        if ky_range is None:
            ky_range = np.linspace(0, 1, 32)
            
        print(f"Calculating DOS with {len(energy_range)} energy points and {len(ky_range)} ky points")
        print(f"Using {method} method for surface Green's functions")
        
        # Create parameter grid for multiprocessing
        param_grid = list(product(energy_range, ky_range))
        
        print(f"Starting DOS calculations for {len(param_grid)} (E, ky) pairs...")
        start_time = time.time()
        
        # Use multiprocessing to calculate DOS points
        with multiprocessing.Pool(processes=32) as pool:
            dos_results = pool.map(self._calculate_dos_point_mp, 
                                 [(e, ky, method, eta) for e, ky in param_grid])
        
        # Reshape results and average over ky for each energy
        dos_results = np.array(dos_results).reshape(len(energy_range), len(ky_range))
        total_dos = np.mean(dos_results, axis=1)
        
        end_time = time.time()
        print(f"DOS calculation completed in {end_time - start_time:.2f} seconds")
        
        if save_data:
            # Save DOS data
            dos_data = np.column_stack((energy_range, total_dos))
            np.savetxt(filename, dos_data, header="Energy(eV)  DOS(states/eV)", 
                      fmt='%.6e', delimiter='\t')
            print(f"DOS data saved to {filename}")
        
        return energy_range, total_dos
    
    def _calculate_dos_point(self, energy, ky, method="sancho_rubio", eta=1e-6):
        """
        Calculate DOS at a single (E, ky) point using RGF Green's functions.
        
        Args:
            energy: Energy point
            ky: Transverse momentum
            method: Self-energy calculation method
            eta: Small imaginary part for broadening
            
        Returns:
            float: DOS value at this energy point
        """
        try:
            # Use RGF to compute DOS directly
            dos_array = self.GF.compute_density_of_states(energy, ky, self_energy_method=method)
            
            # Sum over all orbitals to get total DOS
            total_dos = np.sum(dos_array)
            
            return float(total_dos) if np.isfinite(total_dos) else 0.0
            
        except Exception as e:
            print(f"Error in DOS calculation at (E={energy:.3f}, ky={ky:.3f}): {e}")
            return 0.0
    
    def _calculate_dos_point_mp(self, params):
        """Multiprocessing helper for DOS calculation"""
        energy, ky, method, eta = params
        return self._calculate_dos_point(energy, ky, method, eta)
    
    def calculate_electron_density_at_energy(self, energy, ky_range=None, method="sancho_rubio"):
        """
        Calculate electron density at each atom position for a given energy.
        
        Args:
            energy: Energy point
            ky_range: Transverse momentum points for k-space integration
            method: Self-energy calculation method
            
        Returns:
            dict: Dictionary mapping atom positions to electron density values
        """
        if ky_range is None:
            ky_range = self.k_space
            
        print(f"Calculating electron density at E={energy:.3f} eV")
        
        # Calculate electron density for each ky point using multiprocessing
        param_grid = [(energy, ky, method) for ky in ky_range]
        
        with multiprocessing.Pool(processes=32) as pool:
            density_results = pool.map(self._calculate_density_point_mp, param_grid)
        
        # Average over ky points to get final density at each atom
        num_atoms = len(self.atoms)
        total_density = np.zeros(num_atoms)
        
        for density_array in density_results:
            total_density += density_array
        
        # Normalize by number of k-points and k-space integration factor
        dk = (ky_range[1] - ky_range[0]) * 2*np.pi / (5.431e-10) if len(ky_range) > 1 else 1.0
        total_density = total_density / (2 * np.pi) * dk
        
        # Create dictionary mapping atom positions to densities
        density_dict = {}
        for atom_idx, atom in enumerate(self.atoms):
            density_dict[atom.getPos()] = total_density[atom_idx]
        
        return density_dict
    
    def _calculate_density_point_mp(self, params):
        """Multiprocessing helper for electron density calculation"""
        energy, ky, method = params
        return self._calculate_density_at_atoms(energy, ky, method)
    
    def _calculate_density_at_atoms(self, energy, ky, method="sancho_rubio"):
        """
        Calculate electron density at each atom for a single (E, ky) point.
        
        The Hamiltonian structure is:
        atom_1_orb_1, atom_1_orb_2, ..., atom_1_orb_10, atom_2_orb_1, ...
        
        Args:
            energy: Energy point
            ky: Transverse momentum
            method: Self-energy calculation method
            
        Returns:
            np.array: Electron density at each atom position
        """
        try:
            # Get electron density array from RGF
            density_array = self.GF.compute_electron_density(energy, ky, self_energy_method=method)
            
            # Sum over 10 orbitals for each atom
            num_atoms = len(self.atoms)
            num_orbitals_per_atom = 10
            
            if len(density_array) != num_atoms * num_orbitals_per_atom:
                raise ValueError(f"Density array size {len(density_array)} doesn't match "
                               f"expected size {num_atoms * num_orbitals_per_atom}")
            
            # Reshape and sum over orbitals for each atom
            density_per_atom = density_array.reshape(num_atoms, num_orbitals_per_atom)
            atom_densities = np.sum(density_per_atom, axis=1)
            
            return atom_densities
            
        except Exception as e:
            print(f"Error in density calculation at (E={energy:.3f}, ky={ky:.3f}): {e}")
            return np.zeros(len(self.atoms))
    
    def calculate_total_electron_density(self, energy_range=None, ky_range=None, method="sancho_rubio"):
        """
        Calculate total electron density by integrating over energy and k-space.
        
        Args:
            energy_range: Energy points for integration
            ky_range: Transverse momentum points
            method: Self-energy calculation method
            
        Returns:
            dict: Dictionary mapping atom positions to total electron density
        """
        if energy_range is None:
            energy_range = self.energy_range
        if ky_range is None:
            ky_range = self.k_space
            
        print(f"Calculating total electron density with {len(energy_range)} energy points")
        
        # Create parameter grid for all (E, ky) combinations
        param_grid = list(product(energy_range, ky_range))
        
        print(f"Starting density calculations for {len(param_grid)} (E, ky) pairs...")
        start_time = time.time()
        
        # Use multiprocessing to calculate density for all points
        with multiprocessing.Pool(processes=32) as pool:
            density_results = pool.map(self._calculate_density_point_mp, 
                                     [(e, ky, method) for e, ky in param_grid])
        
        # Reshape results and integrate over energy and k-space
        num_energies = len(energy_range)
        num_ky = len(ky_range)
        num_atoms = len(self.atoms)
        
        density_results = np.array(density_results).reshape(num_energies, num_ky, num_atoms)
        
        # Integration weights
        dE = energy_range[1] - energy_range[0] if len(energy_range) > 1 else 1.0
        # For now, simplify k-space integration - treat as normalized integration over [0,1]
        dk = (ky_range[1] - ky_range[0]) if len(ky_range) > 1 else 1.0
        
        # Integrate over energy and k-space
        # First average over ky, then integrate over energy
        density_vs_energy = np.mean(density_results, axis=1)  # Average over ky
        total_density = np.trapz(density_vs_energy, dx=dE, axis=0) * dk
        
        end_time = time.time()
        print(f"Total density calculation completed in {end_time - start_time:.2f} seconds")
        
        # Create dictionary mapping atom positions to total densities
        density_dict = {}
        for atom_idx, atom in enumerate(self.atoms):
            density_dict[atom.getPos()] = total_density[atom_idx]
        
        return density_dict
    
    def calculate_smeared_electron_density(self, energy_range=None, ky_range=None, method="sancho_rubio"):
        """
        Calculate electron density and interpolate to device grid for Poisson solver.
        
        Args:
            energy_range: Energy points for integration
            ky_range: Transverse momentum points
            method: Self-energy calculation method
            
        Returns:
            np.array: Electron density on device grid (nx, nz)
        """
        # Get unsmeared density at atom positions
        density_dict = self.calculate_total_electron_density(energy_range, ky_range, method)
        
        # Interpolate to device grid
        smeared_density = self.unsmear_to_smear(density_dict)
        
        return smeared_density
    
    def calculate_density_vs_energy(self, energy_range=None, ky_range=None, method="sancho_rubio", 
                                   save_data=True, filename="density_vs_energy.txt"):
        """
        Calculate electron density vs energy at each atom position.
        Useful for analyzing energy-resolved charge distribution.
        
        Args:
            energy_range: Energy points for calculation
            ky_range: Transverse momentum points
            method: Self-energy calculation method
            save_data: Whether to save data to file
            filename: Output filename
            
        Returns:
            energies, density_vs_energy: Energy points and density array (num_energies, num_atoms)
        """
        if energy_range is None:
            energy_range = np.linspace(-2.0, 2.0, 100)
        if ky_range is None:
            ky_range = self.k_space
            
        print(f"Calculating density vs energy with {len(energy_range)} energy points")
        
        num_atoms = len(self.atoms)
        density_vs_energy = np.zeros((len(energy_range), num_atoms))
        
        for i, energy in enumerate(energy_range):
            density_dict = self.calculate_electron_density_at_energy(energy, ky_range, method)
            
            # Convert dictionary to array maintaining atom order
            for atom_idx, atom in enumerate(self.atoms):
                density_vs_energy[i, atom_idx] = density_dict[atom.getPos()]
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(energy_range)} ({100*(i+1)/len(energy_range):.1f}%)")
        
        if save_data:
            # Save density vs energy data
            header = "Energy(eV)"
            for atom_idx in range(num_atoms):
                header += f"\tAtom_{atom_idx}_Density"
            
            data = np.column_stack((energy_range, density_vs_energy))
            np.savetxt(filename, data, header=header, fmt='%.6e', delimiter='\t')
            print(f"Density vs energy data saved to {filename}")
        
        return energy_range, density_vs_energy
    
    def solve_efn(self, energy_range=None, method="sancho_rubio", kT=None, 
                  tolerance=1e-11, efn_bounds=None, save_data=True, 
                  filename="efn_data.txt"):
        """
        Solve for electron quasi-fermi energy (EFN) across the device.
        
        This method integrates the EFN solver with the existing charge calculation
        framework to find the EFN distribution that satisfies charge balance.
        
        Args:
            energy_range: Energy points for LDOS calculation
            method: Self-energy calculation method for LDOS
            kT: Thermal energy in eV (uses device value if None)
            tolerance: Convergence tolerance for EFN solver
            efn_bounds: Optional bounds for EFN search
            save_data: Whether to save EFN data to file
            filename: Output filename for EFN data
            
        Returns:
            efn_grid: EFN distribution (2D array: nx × nz)
        """
        from efn_solver import EFNSolver
        
        if energy_range is None:
            energy_range = self.energy_range
        if kT is None:
            kT = self.device.kbT  # Convert from J to eV
            
        print(f"Solving EFN with {len(energy_range)} energy points")
        print(f"Method: {method}, kT: {kT:.4f} eV, Tolerance: {tolerance:.0e}")
        
        # Initialize EFN solver
        solver = EFNSolver(kT=kT, tolerance=tolerance)
        
        # Calculate LDOS grid
        print("Calculating LDOS grid...")
        N_E = len(energy_range)
        N_x, N_z = self.device.nx, self.device.nz
        dos_grid = np.zeros((N_E, N_x, N_z))
        
        start_time = time.time()
        for i, energy in enumerate(energy_range):
            ldos_smeared = self.calculate_smeared_LDOS(energy)
            dos_grid[i, :, :] = ldos_smeared
            
            if (i + 1) % max(1, N_E // 10) == 0:
                elapsed = time.time() - start_time
                eta = elapsed * (N_E - i - 1) / (i + 1)
                print(f"LDOS progress: {i+1}/{N_E} ({100*(i+1)/N_E:.1f}%), ETA: {eta:.1f}s")
        
        # Get potential grid (from Poisson solver)
        potential_grid = self.smearedPhi.reshape(N_x, N_z)
        
        # Calculate electron density grid
        print("Calculating electron density grid...")
        density_dict = self.calculate_total_electron_density(energy_range, method=method)
        density_grid = self.unsmear_to_smear(density_dict)
        
        # Solve for EFN
        print("Solving EFN across device grid...")
        efn_grid = solver.solve_efn_grid(
            energy_range, dos_grid, potential_grid, density_grid, 
            efn_bounds=efn_bounds, show_progress=True
        )
        
        # Update the charge object
        self.smearedEFN = efn_grid.flatten()
        
        if save_data:
            # Save EFN data
            x_coords = np.arange(N_x)
            z_coords = np.arange(N_z)
            X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')
            
            # Create data array: x, z, potential, density, efn
            data = np.column_stack([
                X.flatten(), Z.flatten(), 
                potential_grid.flatten(), density_grid.flatten(), 
                efn_grid.flatten()
            ])
            
            header = "x_index\tz_index\tPotential(eV)\tDensity(cm^-3)\tEFN(eV)"
            np.savetxt(filename, data, header=header, fmt='%.6e', delimiter='\t')
            print(f"EFN data saved to {filename}")
        
        print(f"EFN solving completed!")
        print(f"EFN range: {efn_grid.min():.3f} to {efn_grid.max():.3f} eV")
        
        return efn_grid
    
    def plot_efn_analysis(self, efn_grid=None, save_plots=True):
        """
        Create analysis plots for EFN distribution.
        
        Args:
            efn_grid: EFN grid (if None, uses self.smearedEFN)
            save_plots: Whether to save plots to files
        """
        import matplotlib.pyplot as plt
        
        if efn_grid is None:
            efn_grid = self.smearedEFN.reshape(self.device.nx, self.device.nz)
        
        potential_grid = self.smearedPhi.reshape(self.device.nx, self.device.nz)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot potential
        im1 = ax1.imshow(potential_grid.T, origin='lower', cmap='viridis', aspect='auto')
        ax1.set_title('Electrostatic Potential (eV)')
        ax1.set_xlabel('x index')
        ax1.set_ylabel('z index')
        plt.colorbar(im1, ax=ax1)
        
        # Plot EFN
        im2 = ax2.imshow(efn_grid.T, origin='lower', cmap='coolwarm', aspect='auto')
        ax2.set_title('Electron Quasi-Fermi Energy (eV)')
        ax2.set_xlabel('x index')
        ax2.set_ylabel('z index')
        plt.colorbar(im2, ax=ax2)
        
        # Plot EFN - potential (relative to band edge)
        efn_relative = efn_grid - potential_grid
        im3 = ax3.imshow(efn_relative.T, origin='lower', cmap='plasma', aspect='auto')
        ax3.set_title('EFN - Potential (eV)')
        ax3.set_xlabel('x index')
        ax3.set_ylabel('z index')
        plt.colorbar(im3, ax=ax3)
        
        # Correlation plot
        ax4.scatter(potential_grid.flatten(), efn_grid.flatten(), alpha=0.6, s=20)
        ax4.set_xlabel('Potential (eV)')
        ax4.set_ylabel('EFN (eV)')
        ax4.set_title('EFN vs Potential')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('efn_analysis.png', dpi=300, bbox_inches='tight')
            print("EFN analysis plot saved as 'efn_analysis.png'")
        
        plt.show()
        
        # Print statistics
        print("\nEFN Analysis Statistics:")
        print(f"Potential range: {potential_grid.min():.3f} to {potential_grid.max():.3f} eV")
        print(f"EFN range: {efn_grid.min():.3f} to {efn_grid.max():.3f} eV")
        print(f"EFN-Potential range: {efn_relative.min():.3f} to {efn_relative.max():.3f} eV")
        print(f"Mean EFN: {efn_grid.mean():.3f} eV")
        print(f"EFN std dev: {efn_grid.std():.3f} eV")



