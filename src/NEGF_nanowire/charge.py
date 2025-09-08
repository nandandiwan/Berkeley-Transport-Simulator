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
from rgf import GreensFunction
from device_hamiltonian import Hamiltonian
import time

class Charge():
    def __init__(self, device : Device):
        self.device = device

        self.ham = Hamiltonian(device)
        self.GF = GreensFunction(device, self.ham)
        self.lse = LeadSelfEnergy(self.device, self.ham)
        self.num_energy = 200
        self.energy_range = np.linspace(-5,5, 100) 
        
        self.atoms = self.ham.unitCell.ATOM_POSITIONS # the atoms         
        # These are lists that give quasi fermi energy and potential at location of every atom, needs to be initialized every time 
        self.smearedEFN = np.zeros((self.device.nx,self.device.ny, self.device.nz)) # EFN has same dimension as poisson solver 
        self.smearedPhi = self.device.potential
        self.smearedLDOS = np.zeros((self.device.nx, self.device.ny,self.device.nz, self.num_energy))
    
        self.weights = {}
        
        
    def unsmear_to_smear(self, A : dict):
        """performs interpolation to get values that match up with poisson matrix (DO THIS AT END OF MP FOR ALL CALCULATIONS)"""
        max_X = self.device.unitX 
        max_Z = self.device.unitZ 
        max_Y = self.device.unitY
        nx, nz,ny = self.device.nx, self.device.nz,self.device.ny
        
        # Create regular grid coordinates
        x_grid = np.linspace(0, max_X, nx)
        y_grid= np.linspace(0, max_Y, ny)
        z_grid = np.linspace(0, max_Z, nz)
        X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        
        # Extract coordinates and values from dictionary (ignoring y-component)
        points = []
        values = []
        for coord, value in A.items():
            x, y, z = coord  
            points.append([x, y,z])
            values.append(value)
        
        points = np.array(points)
        values = np.array(values)
        
        # Create target grid points for interpolation
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(),Z_grid.ravel()])
        
        from scipy.interpolate import griddata
        interpolated_values = griddata(points, values, grid_points, method='nearest', fill_value=0.0)
        smeared_array = interpolated_values.reshape((nx, ny,nz))
        
        return smeared_array
        
        
        
    
    def calculate_real_GR(self, E):
        G_R_REAL= self._calculate_gf_simple(E)
        return G_R_REAL
        

    def _calculate_gf_simple(self, param):
        """Simple Green's function calculation"""
        energy = param
        G_R_diag, Gamma_L, Gamma_R = self.GF.compute_central_greens_function(energy, compute_lesser=False)
        
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
        
        total_ldos = self._calculate_ldos_at_atoms(E)

        LDOS_points = {}
        for atom_idx, atom in enumerate(self.atoms):
            LDOS_points[atom.getPos()] = total_ldos[atom_idx]
        
        return LDOS_points

    
    def _calculate_ldos_at_atoms(self, energy):
        """
        Calculate LDOS at each atom for a single (E) point.
        
        Args:
            energy: Energy point
            
        Returns:
            np.array: LDOS at each atom position
        """
        try:
            # Get DOS array from RGF
            dos_array = self.GF.compute_density_of_states(energy)
            
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
            print(f"Error in LDOS calculation at (E={energy:.3f}: {e}")
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

        return LDOS_points * self.fermi(E)
    
    def calculate_EC(self):
        """Find conduction band minimum from DOS calculation

           Rewrite this method
        
        """

        return 1.2
    
    def calculate_DOS(self, energy_range=None, method="sancho_rubio", 
                     eta=1e-6, save_data=True, filename="dos_data.txt", equilibrium=False):
        """
        Calculate Density of States (DOS) using RGF with multiprocessing.
        
        Args:
            energy_range: Energy points for DOS calculation
            method: Surface GF method ("sancho_rubio", "iterative", "transfer")
            eta: Small imaginary part for broadening
            save_data: Whether to save DOS data to file
            filename: Output filename for DOS data
            
        Returns:
            energies, total_dos: Energy points and corresponding DOS values
        """
        if energy_range is None:
            energy_range = np.linspace(-2.0, 2.0, 200)

            
        print(f"Calculating DOS with {len(energy_range)} ")
        print(f"Using {method} method for surface Green's functions")
        
        # Create parameter grid for multiprocessing
        param_grid = list(energy_range)
        
        print(f"Starting DOS calculations for {len(param_grid)} energy pts...")
        start_time = time.time()
        
        # Use multiprocessing to calculate DOS points
        with multiprocessing.Pool(processes=32) as pool:
            total_dos = pool.map(self._calculate_dos_point_mp, 
                                 [(e, method, eta, equilibrium) for e in param_grid])
        
        end_time = time.time()
        print(f"DOS calculation completed in {end_time - start_time:.2f} seconds")
        
        if save_data:
            # Save DOS data
            dos_data = np.column_stack((energy_range, total_dos))
            np.savetxt(filename, dos_data, header="Energy(eV)  DOS(states/eV)", 
                      fmt='%.6e', delimiter='\t')
            print(f"DOS data saved to {filename}")
        
        return energy_range, total_dos
    
    def _calculate_dos_point(self, energy, method="sancho_rubio", eta=1e-6, equilibrium=False):
        """
        Calculate DOS at a single E point using RGF Green's functions.
        
        Args:
            energy: Energy point
            method: Self-energy calculation method
            eta: Small imaginary part for broadening
            
        Returns:
            float: DOS value at this energy point
        """
        try:
            # Use RGF to compute DOS directly
            dos_array = self.GF.compute_density_of_states(energy, self_energy_method=method, equilibrium=equilibrium)
            
            # Sum over all orbitals to get total DOS
            total_dos = np.sum(dos_array)
            
            return float(total_dos) if np.isfinite(total_dos) else 0.0
            
        except Exception as e:
            print(f"Error in DOS calculation at (E={energy:.3f}: {e}")
            return 0.0
    
    def _calculate_dos_point_mp(self, params):
        """Multiprocessing helper for DOS calculation"""
        energy, method, eta, equilibrium = params
        return self._calculate_dos_point(energy,method, eta,equilibrium)
    
    def calculate_electron_density_at_energy(self, energy, method="sancho_rubio"):
        """
        Calculate electron density at each atom position for a given energy.
        
        Args:
            energy: Energy point

            method: Self-energy calculation method
            
        Returns:
            dict: Dictionary mapping atom positions to electron density values
        """
  
            
        print(f"Calculating electron density at E={energy:.3f} eV")

        density_array = self._calculate_density_at_atoms(energy=energy)
        # Create dictionary mapping atom positions to densities
        density_dict = {}
        for atom_idx, atom in enumerate(self.atoms):
            density_dict[atom.getPos()] = density_array[atom_idx]
        
        return density_dict
    

    
    def _calculate_density_at_atoms(self, energy, method="sancho_rubio"):
        """
        Calculate electron density at each atom for a single E point.
        
        The Hamiltonian structure is:
        atom_1_orb_1, atom_1_orb_2, ..., atom_1_orb_10, atom_2_orb_1, ...
        
        Args:
            energy: Energy point
            method: Self-energy calculation method
            
        Returns:
            np.array: Electron density at each atom position
        """
        try:
            # Get electron density array from RGF
            density_array = self.GF.compute_electron_density(energy, self_energy_method=method)
            
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
            print(f"Error in density calculation at (E={energy:.3f}): {e}")
            return np.zeros(len(self.atoms))
    
    def calculate_total_electron_density(self, energy_range=None,  method="sancho_rubio"):
        """
        Calculate total electron density by integrating over energy 
        
        Args:
            energy_range: Energy points for integration
            method: Self-energy calculation method
            
        Returns:
            dict: Dictionary mapping atom positions to total electron density
        """
        if energy_range is None:
            energy_range = self.energy_range

            
        print(f"Calculating total electron density with {len(energy_range)} energy points")
        
        # Create parameter grid for E
        param_grid = list(energy_range)
        
        print(f"Starting density calculations for {len(param_grid)} E pts...")
        start_time = time.time()
        
        # Use multiprocessing to calculate density for all points
        with multiprocessing.Pool(processes=32) as pool:
            density_vs_energy = pool.map(self._calculate_density_at_atoms, 
                                     [(e, method) for e in param_grid])

        dE = energy_range[1] - energy_range[0]
        total_density = np.trapz(density_vs_energy, dx=dE, axis=0) 
        
        end_time = time.time()
        print(f"Total density calculation completed in {end_time - start_time:.2f} seconds")
        
        # Create dictionary mapping atom positions to total densities
        density_dict = {}
        for atom_idx, atom in enumerate(self.atoms):
            density_dict[atom.getPos()] = total_density[atom_idx]
        
        return density_dict
    
    def calculate_smeared_electron_density(self, energy_range=None, method="sancho_rubio"):
        """
        Calculate electron density and interpolate to device grid for Poisson solver.
        
        Args:
            energy_range: Energy points for integration

            method: Self-energy calculation method
            
        Returns:
            np.array: Electron density on device grid (nx, nz)
        """
        # Get unsmeared density at atom positions
        density_dict = self.calculate_total_electron_density(energy_range, method)
        
        # Interpolate to device grid
        smeared_density = self.unsmear_to_smear(density_dict)
        
        return smeared_density
    

