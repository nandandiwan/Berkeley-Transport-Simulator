"""
Quasi-Fermi Energy (EFN) Solver for NEGF Simulations

This module implements a robust quasi-fermi energy solver based on the Jiezi approach,
adapted for NEGF device simulations. It solves for the electron quasi-fermi energy (EFN)
at each grid point given the potential, local density of states (LDOS), and electron density.

The core equation being solved is:
n(r) = ∫ DOS(E,r) * f(E - φ(r) - EFN(r)) dE

where:
- n(r) is the local electron density
- DOS(E,r) is the local density of states at energy E and position r
- f(E) is the Fermi-Dirac distribution function
- φ(r) is the electrostatic potential
- EFN(r) is the electron quasi-fermi energy

The solver uses Brent's method for robust root finding with proper bracketing.

Author: Adapted from Jiezi implementation for NEGF simulations
"""

import numpy as np
import copy
from scipy.integrate import trapezoid
import time
from typing import Tuple, Optional, Union
import warnings


class EFNSolver:
    """
    Electron Quasi-Fermi Energy (EFN) Solver for NEGF Device Simulations.
    
    This class provides methods to solve for the electron quasi-fermi energy
    distribution in a device given the potential, LDOS, and electron density.
    """
    
    def __init__(self, kT: float = 0.026, tolerance: float = 1e-11):
        """
        Initialize the EFN solver.
        
        Args:
            kT: Thermal energy in eV (default: 26 meV at 300K)
            tolerance: Convergence tolerance for root finding
        """
        self.kT = kT
        self.tolerance = tolerance
        
    def fermi_dirac(self, E: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Fermi-Dirac distribution function.
        
        Args:
            E: Energy argument (E - μ)/kT
            
        Returns:
            Fermi-Dirac distribution value
        """
        # Clip extreme values to avoid overflow
        E_clipped = np.clip(E / self.kT, -700, 700)
        return 1.0 / (1.0 + np.exp(E_clipped))
    
    def charge_balance_function(self, efn: float, energy_list: np.ndarray, 
                               dos_at_point: np.ndarray, potential: float, 
                               target_density: float) -> float:
        """
        Charge balance function for root finding: f(EFN) = n_calculated - n_target.
        
        This function calculates the electron density for a given EFN and compares
        it to the target density. The root of this function gives the correct EFN.
        
        Args:
            efn: Trial electron quasi-fermi energy
            energy_list: Energy grid points (1D array)
            dos_at_point: DOS values at each energy for this spatial point
            potential: Electrostatic potential at this point
            target_density: Target electron density at this point
            
        Returns:
            Difference between calculated and target density
        """
        # Calculate n = ∫ DOS(E) * f(E - φ - EFN) dE
        # Only integrate over conduction band (positive energies)
        zero_index = np.searchsorted(energy_list, 0.0)
        E_conduction = energy_list[zero_index:]
        dos_conduction = dos_at_point[zero_index:]
        
        if len(E_conduction) == 0:
            return -target_density
        
        # Calculate Fermi function arguments
        fermi_args = E_conduction - potential - efn
        fermi_values = self.fermi_dirac(fermi_args)
        
        # Integrate using trapezoidal rule
        calculated_density = trapezoid(dos_conduction * fermi_values, E_conduction)
        
        return calculated_density - target_density
    
    def solve_efn_point(self, energy_list: np.ndarray, dos_at_point: np.ndarray,
                       potential: float, target_density: float,
                       efn_bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """
        Solve for EFN at a single spatial point using Brent's method.
        
        Args:
            energy_list: Energy grid points (1D array)
            dos_at_point: DOS values at each energy for this spatial point
            potential: Electrostatic potential at this point
            target_density: Target electron density at this point
            efn_bounds: Optional bounds for EFN search (min, max)
            
        Returns:
            Electron quasi-fermi energy at this point, or None if no solution found
        """
        # Set default bounds if not provided
        if efn_bounds is None:
            E_min, E_max = energy_list[0], energy_list[-1]
            efn_bounds = (E_min - 10.0, E_max + 10.0)
        
        a, b = efn_bounds
        
        # Check if target density is too small (numerical limit)
        if target_density < 1e-22:
            # Return a very negative EFN (empty state)
            return efn_bounds[0]
        
        # Evaluate function at bounds
        f_a = self.charge_balance_function(a, energy_list, dos_at_point, potential, target_density)
        f_b = self.charge_balance_function(b, energy_list, dos_at_point, potential, target_density)
        
        # Check if root exists in interval
        if f_a * f_b > 0:
            # Try to find better bounds
            if f_a > 0 and f_b > 0:
                # Both positive - EFN should be lower
                return self._find_efn_with_extended_bounds(
                    energy_list, dos_at_point, potential, target_density, a, "lower"
                )
            elif f_a < 0 and f_b < 0:
                # Both negative - EFN should be higher
                return self._find_efn_with_extended_bounds(
                    energy_list, dos_at_point, potential, target_density, b, "higher"
                )
        
        # Use Brent's method for root finding
        return self._brent_method(energy_list, dos_at_point, potential, target_density, a, b)
    
    def _find_efn_with_extended_bounds(self, energy_list: np.ndarray, dos_at_point: np.ndarray,
                                     potential: float, target_density: float, 
                                     start_point: float, direction: str) -> Optional[float]:
        """
        Find EFN with extended bounds when initial bounds don't bracket the root.
        
        Args:
            energy_list: Energy grid points
            dos_at_point: DOS values at this spatial point
            potential: Electrostatic potential
            target_density: Target electron density
            start_point: Starting point for bound extension
            direction: "lower" or "higher" to extend bounds
            
        Returns:
            EFN value or None if no solution found
        """
        step = 1.0
        max_iterations = 50
        
        for i in range(max_iterations):
            if direction == "lower":
                new_bound = start_point - step * (i + 1)
                f_new = self.charge_balance_function(new_bound, energy_list, dos_at_point, potential, target_density)
                f_old = self.charge_balance_function(start_point, energy_list, dos_at_point, potential, target_density)
                if f_new * f_old < 0:
                    return self._brent_method(energy_list, dos_at_point, potential, target_density, new_bound, start_point)
            else:  # direction == "higher"
                new_bound = start_point + step * (i + 1)
                f_new = self.charge_balance_function(new_bound, energy_list, dos_at_point, potential, target_density)
                f_old = self.charge_balance_function(start_point, energy_list, dos_at_point, potential, target_density)
                if f_new * f_old < 0:
                    return self._brent_method(energy_list, dos_at_point, potential, target_density, start_point, new_bound)
        
        warnings.warn(f"Could not find bracketing bounds for EFN at density {target_density}")
        return None
    
    def _brent_method(self, energy_list: np.ndarray, dos_at_point: np.ndarray,
                     potential: float, target_density: float, a: float, b: float) -> Optional[float]:
        """
        Brent's method for robust root finding (adapted from Jiezi implementation).
        
        Args:
            energy_list: Energy grid points
            dos_at_point: DOS values at this spatial point
            potential: Electrostatic potential
            target_density: Target electron density
            a, b: Initial brackets for the root
            
        Returns:
            Root (EFN value) or None if convergence fails
        """
        def func_F(efn):
            return self.charge_balance_function(efn, energy_list, dos_at_point, potential, target_density)
        
        f_a = func_F(a)
        f_b = func_F(b)
        
        # Ensure f(b) is closer to zero
        if abs(f_a) < abs(f_b):
            a, b = b, a
            f_a, f_b = f_b, f_a
        
        c = a
        d = a
        flag_bisection = False
        max_iterations = 100
        
        for iteration in range(max_iterations):
            # Check convergence
            if abs(a - b) < self.tolerance or abs(func_F(b)) < 1e-15:
                return b
            
            # Ensure f(b) is closer to zero
            f_a = func_F(a)
            f_b = func_F(b)
            if abs(f_a) < abs(f_b):
                a, b = b, a
                f_a, f_b = f_b, f_a
            
            f_c = func_F(c)
            m = (a + b) / 2
            k = (3 * a + b) / 4
            
            # Choose interpolation method
            if f_a != f_c and f_b != f_c:
                # Inverse quadratic interpolation
                s = (a * f_b * f_c / ((f_a - f_b) * (f_a - f_c)) +
                     b * f_a * f_c / ((f_b - f_a) * (f_b - f_c)) +
                     c * f_a * f_b / ((f_c - f_a) * (f_c - f_b)))
            elif b != c and f_b != f_c:
                # Secant method
                s = b - f_b * (b - c) / (f_b - f_c)
            else:
                # Bisection
                s = m
            
            # Check conditions for using interpolation
            condition1 = (k <= s <= b) or (b <= s <= k)
            if flag_bisection:
                condition2 = abs(s - b) < abs(b - c) / 2
            else:
                condition2 = (c == d) or (abs(s - b) < abs(c - d) / 2)
            
            # Decide whether to use interpolation or bisection
            if condition1 and condition2:
                # Use interpolation
                flag_bisection = False
                d = copy.deepcopy(c)
                c = copy.deepcopy(b)
                
                f_s = func_F(s)
                if f_s * f_a < 0:
                    b = s
                else:
                    a = b
                    b = s
            else:
                # Use bisection
                flag_bisection = True
                d = copy.deepcopy(c)
                c = copy.deepcopy(b)
                
                f_m = func_F(m)
                if f_m * f_a < 0:
                    b = m
                else:
                    a = b
                    b = m
        
        warnings.warn(f"Brent method did not converge for density {target_density}")
        return None
    
    def solve_efn_grid(self, energy_list: np.ndarray, dos_grid: np.ndarray,
                      potential_grid: np.ndarray, density_grid: np.ndarray,
                      efn_bounds: Optional[Tuple[float, float]] = None,
                      show_progress: bool = True) -> np.ndarray:
        """
        Solve for EFN across the entire device grid.
        
        Args:
            energy_list: Energy grid points (1D array, length N_E)
            dos_grid: Local density of states (3D array: N_E × N_x × N_z)
            potential_grid: Electrostatic potential (2D array: N_x × N_z)
            density_grid: Target electron density (2D array: N_x × N_z)
            efn_bounds: Optional bounds for EFN search
            show_progress: Whether to show progress information
            
        Returns:
            EFN grid (2D array: N_x × N_z)
        """
        N_E, N_x, N_z = dos_grid.shape
        efn_grid = np.zeros((N_x, N_z))
        
        total_points = N_x * N_z
        failed_points = 0
        
        if show_progress:
            print(f"Solving EFN for {total_points} grid points...")
            start_time = time.time()
        
        for i in range(N_x):
            for j in range(N_z):
                dos_at_point = dos_grid[:, i, j]
                potential = potential_grid[i, j]
                target_density = density_grid[i, j]
                
                # Solve for EFN at this point
                efn = self.solve_efn_point(energy_list, dos_at_point, potential, 
                                         target_density, efn_bounds)
                
                if efn is not None:
                    efn_grid[i, j] = efn
                else:
                    efn_grid[i, j] = efn_bounds[0] if efn_bounds else energy_list[0] - 10.0
                    failed_points += 1
            
            # Progress update
            if show_progress and (i + 1) % max(1, N_x // 10) == 0:
                progress = (i + 1) * N_z / total_points * 100
                elapsed = time.time() - start_time
                eta = elapsed * (total_points - (i + 1) * N_z) / ((i + 1) * N_z)
                print(f"Progress: {progress:.1f}% ({(i+1)*N_z}/{total_points}), "
                      f"ETA: {eta:.1f}s, Failed: {failed_points}")
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"EFN solving completed in {total_time:.2f} seconds")
            print(f"Failed convergence at {failed_points}/{total_points} points "
                  f"({100*failed_points/total_points:.2f}%)")
        
        return efn_grid
    
    def solve_efn_from_charge_data(self, charge_calc, energy_range: Optional[np.ndarray] = None,
                                  efn_bounds: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Solve for EFN using data from a Charge calculation object.
        
        This is a convenience method that extracts the necessary data from
        your Charge class and solves for EFN.
        
        Args:
            charge_calc: Charge calculation object with device and calculated data
            energy_range: Energy points for calculation (if None, uses charge_calc.energy_range)
            efn_bounds: Optional bounds for EFN search
            
        Returns:
            EFN grid (2D array: N_x × N_z)
        """
        if energy_range is None:
            energy_range = charge_calc.energy_range
        
        # Calculate LDOS for each energy point across the grid
        print("Calculating LDOS grid for EFN solver...")
        N_E = len(energy_range)
        N_x, N_z = charge_calc.device.nx, charge_calc.device.nz
        dos_grid = np.zeros((N_E, N_x, N_z))
        
        for i, energy in enumerate(energy_range):
            ldos_smeared = charge_calc.calculate_smeared_LDOS(energy)
            dos_grid[i, :, :] = ldos_smeared
            
            if (i + 1) % max(1, N_E // 10) == 0:
                print(f"LDOS calculation progress: {i+1}/{N_E} ({100*(i+1)/N_E:.1f}%)")
        
        # Get potential and density grids
        potential_grid = charge_calc.smearedPhi
        
        # Calculate total electron density
        print("Calculating electron density grid...")
        density_dict = charge_calc.calculate_total_electron_density(energy_range)
        density_grid = charge_calc.unsmear_to_smear(density_dict)
        
        # Solve for EFN
        return self.solve_efn_grid(energy_range, dos_grid, potential_grid, 
                                 density_grid, efn_bounds)


# Convenience functions for backward compatibility and easy usage
def solve_efn_point(energy_list: np.ndarray, dos_at_point: np.ndarray,
                   potential: float, target_density: float,
                   kT: float = 0.026, tolerance: float = 1e-11,
                   efn_bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
    """
    Convenience function to solve EFN at a single point.
    
    Args:
        energy_list: Energy grid points
        dos_at_point: DOS values at this spatial point
        potential: Electrostatic potential at this point
        target_density: Target electron density at this point
        kT: Thermal energy in eV
        tolerance: Convergence tolerance
        efn_bounds: Optional bounds for EFN search
        
    Returns:
        Electron quasi-fermi energy or None if no solution found
    """
    solver = EFNSolver(kT=kT, tolerance=tolerance)
    return solver.solve_efn_point(energy_list, dos_at_point, potential, 
                                target_density, efn_bounds)


def solve_efn_grid(energy_list: np.ndarray, dos_grid: np.ndarray,
                  potential_grid: np.ndarray, density_grid: np.ndarray,
                  kT: float = 0.026, tolerance: float = 1e-11,
                  efn_bounds: Optional[Tuple[float, float]] = None,
                  show_progress: bool = True) -> np.ndarray:
    """
    Convenience function to solve EFN across a device grid.
    
    Args:
        energy_list: Energy grid points (1D array)
        dos_grid: Local density of states (3D array: N_E × N_x × N_z)
        potential_grid: Electrostatic potential (2D array: N_x × N_z)
        density_grid: Target electron density (2D array: N_x × N_z)
        kT: Thermal energy in eV
        tolerance: Convergence tolerance
        efn_bounds: Optional bounds for EFN search
        show_progress: Whether to show progress
        
    Returns:
        EFN grid (2D array: N_x × N_z)
    """
    solver = EFNSolver()
    return solver.solve_efn_grid(energy_list, dos_grid, potential_grid, 
                               density_grid, efn_bounds, show_progress)
