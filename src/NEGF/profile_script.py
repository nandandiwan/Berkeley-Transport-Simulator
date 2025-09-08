import os

# --- Add these lines at the very top of your script ---
# This must be done BEFORE importing numpy or other scientific libraries.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
import time
from device import Device
from NEGF_sim_git.src.archive.rgf import GreensFunction
from hamiltonian import Hamiltonian
from device import Device
from NEGF_sim_git.src.archive.rgf import GreensFunction
import scipy as sp
from hamiltonian import Hamiltonian
from helper import Helper_functions
import scipy.sparse as spa
import numpy as np
import scipy.sparse as sp
from lead_self_energy import LeadSelfEnergy
from scipy.sparse import bmat, identity, random, csc_matrix
from scipy.sparse.linalg import eigsh, eigs, spsolve
import time

def DOS(param): 
    
    E, ky = param
    dev = Device(5e-9, 1e-9)
    
    ham = Hamiltonian(dev)

    gf = GreensFunction(dev, ham)
    lse = LeadSelfEnergy(dev, ham)

    H = ham.create_sparse_channel_hamlitonian(ky, blocks=False)

    sl = csc_matrix(lse.iterative_self_energy(E, ky, "left"))
    sr = csc_matrix(lse.iterative_self_energy(E, ky, "right"))
    block_size = sl.shape[0]
    H[:block_size, :block_size] += sl
    H[-block_size:,-block_size:] += sr

    H = csc_matrix(np.eye(H.shape[0], dtype=complex) * E) - H

    I = csc_matrix(np.eye(H.shape[0], dtype=complex))

    H.shape[0]

    A = spsolve(H,I)
    tr = -1/np.pi * np.imag(A.trace())

    return (E, ky, tr)


energy_values = np.linspace(-2, 3, 50)  # More energy points for better resolution
ky_values = np.linspace(0, 1.0, 20)     # Fewer ky points for faster computation
dk = ky_values[1] - ky_values[0]        # The differential k_y step

param_grid = list(product(energy_values, ky_values))

print(f"Starting DOS calculations for {len(param_grid)} (E, ky) pairs...")
print(f"Energy range: {energy_values[0]:.2f} to {energy_values[-1]:.2f} eV")
print(f"ky range: {ky_values[0]:.2f} to {ky_values[-1]:.2f}")

start_time = time.time()

# Use fewer processes for debugging - increase to 32 once working
with multiprocessing.Pool(processes=32) as pool:
    results = pool.map(DOS, param_grid)
    
end_time = time.time()
print(f"Calculations finished in {end_time - start_time:.2f} seconds.")

# Aggregate traces for each energy value (sum over all ky values)
energy_traces = defaultdict(complex)
valid_results = 0

for energy, ky, trace in results:
    if not (np.isnan(trace) or np.isinf(trace)):
        energy_traces[energy] += trace
        valid_results += 1
    else:
        print(f"Invalid result at E={energy:.3f}, ky={ky:.3f}: {trace}")

print(f"Valid results: {valid_results}/{len(results)}")

# Sort energies and calculate DOS
dos_energies = sorted(energy_traces.keys())

# DOS formula: D(E) = -(1/π) * Im[Tr(G_R(E))] * (dk/2π)

dos_values = []
for E in dos_energies:
    trace_val = energy_traces[E]
    dos_val = trace_val *  (dk / (2 * np.pi))
    dos_values.append(dos_val)

print(f"DOS calculation complete. Energy points: {len(dos_energies)}")
# Plotting the Density of States
plt.figure(figsize=(12, 8))

# Main DOS plot
plt.subplot(2, 1, 1)
plt.plot(dos_energies, dos_values, 'b-', linewidth=2, label='Density of States D(E)')
plt.xlabel('Energy (eV)')
plt.ylabel('DOS (states/eV)')
plt.title('Density of States from Green\'s Function')
plt.grid(True, alpha=0.3)
plt.legend()



plt.tight_layout()
plt.savefig("dos.png", dpi=300, bbox_inches='tight')
plt.show()

# Save data for further analysis
np.savetxt("dos_data.txt", np.column_stack([dos_energies, dos_values]), 
            header="Energy(eV)\tDOS(states/eV)", fmt="%.6e")