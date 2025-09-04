"""Band structure example for the silicon nanowire structure SiNW2.xyz.
Samples k along the wire axis (z) from 0 to kmax, applies 1D periodic BC,
and saves JSON + PNG outputs.

If --xyz is not provided the script searches common locations for SiNW2.xyz.

Usage:
  PYTHONPATH=NEGF_sim_git/src python -m hamiltonian.examples.band_structure_silicon \
      --bands 40 --kpts 60 --kmax 0.5
"""
import os, json, argparse, numpy as np
from hamiltonian import Hamiltonian

DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), 'outputs', 'sinw2')

SINW2_CANDIDATES = [
    # relative to repo root (two levels up from this file)
    os.path.join('resources','Nanonet','NanoNet','examples','input_samples','SiNW2.xyz'),
    os.path.join('resources','Nanonet','NanoNet','nanonet','SiNW2.xyz'),
    os.path.join('Si_transmission','input_samples','SiNW2.xyz')
]

def locate_sinw2_xyz(explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    # repo root assumed two directories above this file's directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..','..'))
    for rel in SINW2_CANDIDATES:
        cand = os.path.join(repo_root, rel)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError('SiNW2.xyz not found; supply with --xyz')

def compute_nanowire_bands(xyz_path, kpts, kmax, bands, nn):
    H = Hamiltonian(xyz=xyz_path, nn_distance=nn).initialize()
    coords = np.array(list(H.atom_list.values()))
    z_min, z_max = coords[:,2].min(), coords[:,2].max()
    a = (z_max - z_min) if (z_max - z_min) > 0 else 1.0
    H.set_periodic_bc([[0,0,a]])
    ks = np.linspace(0, kmax, kpts)
    energies = []
    for kz in ks:
        vals,_ = H.diagonalize_k([0,0,kz])
        energies.append(vals[:bands])
    energies = np.array(energies)
    return ks, energies, a

def plot_and_save(ks, energies, out_png, ylim=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,6))
    for i in range(min(energies.shape[1], 80)):
        plt.plot(ks, energies[:,i], color='black', linewidth=0.6)
    plt.xlabel('k_z (1/Ang)')
    plt.ylabel('Energy (eV)')
    plt.title('SiNW2 Band Structure')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xyz', help='Path to SiNW2.xyz (optional if auto-detect succeeds)')
    ap.add_argument('--kpts', type=int, default=60)
    ap.add_argument('--kmax', type=float, default=3.141)
    ap.add_argument('--bands', type=int, default=60)
    ap.add_argument('--nn', type=float, default=2.39)
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR, help=f'Output directory (default: {DEFAULT_OUTDIR})')
    ap.add_argument('--center', choices=['average','min','max','none'], default='average', help='Energy centering mode (default: average)')
    ap.add_argument('--energy-window', type=float, nargs=2, default=[-3.0, 3.0], help='Y-axis energy window after centering (default: -3 3)')
    args = ap.parse_args()

    os.makedirs(DEFAULT_OUTDIR, exist_ok=True)
    xyz_path = locate_sinw2_xyz(args.xyz)
    ks, energies_raw, a = compute_nanowire_bands(xyz_path, args.kpts, args.kmax, args.bands, args.nn)

    png_path = os.path.join(args.outdir, 'SiNW2_band_structure.png')

    plot_and_save(ks, energies_raw, png_path, ylim=tuple(args.energy_window))
if __name__ == '__main__':
    main()
