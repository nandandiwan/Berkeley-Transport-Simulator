"""
Transmission of a parametric Si nanowire using block-recursive NEGF.

Config:
- Unit cell from generator: nx=2, ny=2, nz=1
- Periodic along z; transport along z; passivate_x=True
- Periodic BCs are used only because transport dir == periodic dir (z)
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure local src/ on sys.path when run as a script
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent  # .../src

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hamiltonian import Hamiltonian
from hamiltonian.base.block_tridiagonalization import split_into_subblocks_optimized
from negf.self_energy.surface import LeadSelfEnergy
from negf.gf.recursive_greens_functions import recursive_gf
from negf.self_energy import greens_functions
from hamiltonian.tb.orbitals import Orbitals


def build_hamiltonian():
    # Set tight-binding orbital sets
    Orbitals.orbital_sets = {"Si": "SiliconSP3D5S", "H": "HydrogenS"}
    # Parametric nanowire generator parameters
    a_si = 5.50
    hamiltonian = Hamiltonian(nx=2, ny=2, nz=1, a=a_si, periodic_dirs='z', passivate_x=True, nn_distance=2.4)
    hamiltonian.initialize()
    
    
    a_si = 5.50
    primitive_cell = [[0, 0, a_si]]
    hamiltonian.set_periodic_bc(primitive_cell)
    hl, h0, hr = hamiltonian.get_hamiltonians()
    energy = np.linspace(2.1, 2.5, 50)
    damp = 0.001j
    tr = np.zeros(energy.shape)
    dos = np.zeros(energy.shape)
    for j, E in enumerate(energy):

        L, R = greens_functions.surface_greens_function(E, hl, h0, hr, iterate=True, damp=damp)
        g_trans, grd, grl, gru, gr_left = recursive_gf(E, [hl], [h0 + L + R], [hr], damp=damp)
        num_blocks = len(grd)
        for jj in range(num_blocks):
            dos[j] = dos[j] - np.trace(np.imag(grd[jj])) / num_blocks
        gamma_l = 1j * (L - L.conj().T)
        gamma_r = 1j * (R - R.conj().T)
        tr[j] = np.real(np.trace(gamma_l.dot(g_trans).dot(gamma_r).dot(g_trans.conj().T)))
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(energy, dos, 'k')
    ax[0].set_ylabel(r'DOS (a.u)')
    ax[0].set_xlabel(r'Energy (eV)')
    ax[1].plot(energy, tr, 'k')
    ax[1].set_ylabel(r'Transmission (a.u.)')
    ax[1].set_xlabel(r'Energy (eV)')
    fig.tight_layout()
    plt.savefig("si transmission")


build_hamiltonian()