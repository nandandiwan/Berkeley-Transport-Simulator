#!/usr/bin/env python3
"""
Test script for the new charge calculation methods.
Demonstrates usage of DOS and electron density calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from charge import Charge
from device import Device

def test_dos_calculation():
    """Test DOS calculation with RGF"""
    print("=" * 50)
    print("Testing DOS Calculation")
    print("=" * 50)
    
    # Create device (you'll need to replace this with your actual device setup)
    # device = Device(...)  # Your device initialization
    # charge = Charge(device)
    
    # Example usage:
    # energies, dos_values = charge.calculate_DOS(
    #     energy_range=np.linspace(-2.0, 2.0, 100),
    #     ky_range=np.linspace(0, 1, 16),
    #     method="sancho_rubio",
    #     save_data=True,
    #     filename="test_dos.txt"
    # )
    
    # Plot DOS
    # plt.figure(figsize=(10, 6))
    # plt.plot(energies, dos_values, 'b-', linewidth=2)
    # plt.xlabel('Energy (eV)')
    # plt.ylabel('DOS (states/eV)')
    # plt.title('Density of States')
    # plt.grid(True, alpha=0.3)
    # plt.savefig('dos_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("DOS calculation test completed (code template)")

def test_electron_density():
    """Test electron density calculation"""
    print("=" * 50)
    print("Testing Electron Density Calculation")
    print("=" * 50)
    
    # Example usage:
    # device = Device(...)  # Your device initialization
    # charge = Charge(device)
    
    # Single energy point density
    # density_dict = charge.calculate_electron_density_at_energy(
    #     energy=0.0,
    #     ky_range=np.linspace(0, 1, 16),
    #     method="sancho_rubio"
    # )
    # print("Electron density at E=0.0 eV:")
    # for pos, density in density_dict.items():
    #     print(f"  Position {pos}: {density:.6e} electrons")
    
    # Total density (integrated over energy)
    # total_density = charge.calculate_total_electron_density(
    #     energy_range=np.linspace(-2.0, 2.0, 50),
    #     ky_range=np.linspace(0, 1, 16),
    #     method="sancho_rubio"
    # )
    # print("\nTotal electron density:")
    # for pos, density in total_density.items():
    #     print(f"  Position {pos}: {density:.6e} electrons")
    
    # Smeared density for Poisson solver
    # smeared_density = charge.calculate_smeared_electron_density(
    #     energy_range=np.linspace(-2.0, 2.0, 50),
    #     ky_range=np.linspace(0, 1, 16)
    # )
    # print(f"\nSmeared density grid shape: {smeared_density.shape}")
    
    print("Electron density calculation test completed (code template)")

def test_ldos_calculation():
    """Test LDOS calculation"""
    print("=" * 50)
    print("Testing LDOS Calculation")
    print("=" * 50)
    
    dev = Device(2e-9, 1e-9)
   
    charge = Charge(dev)
    
    # LDOS at specific energy
    ldos_dict = charge.calculate_LDOS(E=0.0)
    print("LDOS at E=0.0 eV:")
    for pos, ldos in ldos_dict.items():
        print(f"  Position {pos}: {ldos:.6e} states/eV")
    
    # Smeared LDOS for visualization
    smeared_ldos = charge.calculate_smeared_LDOS(E=0.0)
    print(f"Smeared LDOS grid shape: {smeared_ldos.shape}")
    
    print("LDOS calculation test completed (code template)")

def test_density_vs_energy():
    """Test density vs energy calculation"""
    print("=" * 50)
    print("Testing Density vs Energy")
    print("=" * 50)
    
    # Example usage:
    # device = Device(...)  # Your device initialization
    # charge = Charge(device)
    
    # Density vs energy
    # energies, density_array = charge.calculate_density_vs_energy(
    #     energy_range=np.linspace(-2.0, 2.0, 50),
    #     ky_range=np.linspace(0, 1, 16),
    #     save_data=True,
    #     filename="density_vs_energy.txt"
    # )
    
    # Plot density vs energy for first few atoms
    # plt.figure(figsize=(12, 8))
    # for i in range(min(5, density_array.shape[1])):
    #     plt.plot(energies, density_array[:, i], label=f'Atom {i}')
    # plt.xlabel('Energy (eV)')
    # plt.ylabel('Electron Density')
    # plt.title('Electron Density vs Energy')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.savefig('density_vs_energy.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("Density vs energy calculation test completed (code template)")

def main():
    """Run all tests"""
    print("Testing new charge calculation methods")
    print("Note: These are template functions - uncomment and modify for actual use")
    
    test_dos_calculation()
    test_electron_density()
    test_ldos_calculation()
    test_density_vs_energy()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
