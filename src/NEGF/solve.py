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
from lead_self_energy import LeadSelfEnergy
from device import Device
import numpy as np
import scipy as sp
import scipy.sparse as spa
import scipy.sparse.linalg as spla
from poisson import PoissonSolver
from NEGF_sim_git.src.archive.rgf import GreensFunction
from hamiltonian import Hamiltonian
import time
class Solve:
    """wrapper class to self consistently solve poisson equation and NEGF equations"""
    
    def __init__(self, device : Device):
        self.device = device
        
        self.poisson = PoissonSolver(device)
        self.ham = Hamiltonian(device)
        self.GF = GreensFunction(device, self.ham)
        
        self.energy_range = np.linspace(-5,5, 200)
        self.k_space = np.linspace(0,1,40)
        self.Ek_to_GR = {} #use E *200 + k 
    

    
    def calculate_greens_functions(self, energy_range=None, k_range=None, eta=1e-6, num_processes=32):
        """
        Calculate Green's functions for multiple (E, ky, eta) combinations using multiprocessing.
        
        Args:
            energy_range: Array of energy values in eV (default: self.energy_range)
            k_range: Array of k-space values (default: self.k_space)  
            eta: Small imaginary part for G^R = [E + i*eta - H]^(-1)
            num_processes: Number of parallel processes to use
            
        Returns:
            dict: Dictionary with keys (E, ky) and values G_R (diagonal elements)
        """
        if energy_range is None:
            energy_range = self.energy_range
        if k_range is None:
            k_range = self.k_space
            
        print(f"Calculating Green's functions...")
        print(f"Energy range: {energy_range[0]:.2f} to {energy_range[-1]:.2f} eV ({len(energy_range)} points)")
        print(f"K-space range: {k_range[0]:.2f} to {k_range[-1]:.2f} ({len(k_range)} points)")
        print(f"eta = {eta:.2e}")
        
        # Create parameter grid for multiprocessing  
        param_grid = [(E, ky, eta) for E in energy_range for ky in k_range]
        total_points = len(param_grid)
        
        print(f"Starting calculation for {total_points} (E, ky, eta) combinations...")
        print(f"Using {num_processes} parallel processes...")
        
        start_time = time.time()
        
        # Calculate Green's functions using multiprocessing
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(self._calculate_gf_with_eta, param_grid)
        
        end_time = time.time()
        
        # Organize results into dictionary
        gf_dict = {}
        successful = 0
        for i, result in enumerate(results):
            if result is not None:
                E, ky, eta_val = param_grid[i]
                gf_dict[(E, ky, eta_val)] = result
                successful += 1
        
        print(f"Calculation completed in {end_time - start_time:.2f} seconds")
        print(f"Successfully calculated {successful}/{total_points} Green's functions")
        
        return gf_dict
    
    def _calculate_gf_with_eta(self, param):
        """
        Calculate Green's function for a single (E, ky, eta) point.
        
        Args:
            param: Tuple of (energy, ky, eta)
            
        Returns:
            numpy.ndarray: Diagonal elements of G_R, or None if calculation failed
        """
        energy, ky, eta = param
        
        try:
            # Create Hamiltonian for this ky
            H = self.ham.create_sparse_channel_hamlitonian(ky, blocks=False)
            
            # Add self-energies at contacts
            lse = LeadSelfEnergy(self.device, self.ham)
            sl = lse.self_energy(side="left", E=energy, ky=ky)
            sr = lse.self_energy(side="right", E=energy, ky=ky)
            
            # Add to boundary blocks
            H[:sl.shape[0], :sl.shape[0]] += sl
            H[-sr.shape[0]:, -sr.shape[0]:] += sr
            
            # Construct Green's function: G^R = [E + i*eta - H]^(-1)
            E_complex = energy + 1j * eta
            H_gf = spa.csc_matrix(np.eye(H.shape[0], dtype=complex) * E_complex) - H
            I = spa.csc_matrix(np.eye(H.shape[0], dtype=complex))
            
            # Solve for diagonal elements only (more memory efficient)
            G_R = spla.spsolve(H_gf, I)
            if spa.issparse(G_R):
                G_R = G_R.diagonal()
            else:
                G_R = np.diagonal(G_R)
            
            return G_R
            
        except Exception as e:
            # Return None for failed calculations
            return None
    
