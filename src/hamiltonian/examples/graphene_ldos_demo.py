"""Graphene lattice and LDOS demonstration script.

Usage example:
    python graphene_ldos_demo.py --nx 8 --ny 6 --orientation zigzag --outdir outputs/graphene
"""
import argparse
import json
import os
import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.abspath(os.path.dirname(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from hamiltonian import Hamiltonian
from hamiltonian.geometry.graphene_generator import GrapheneGenerator


def _normalize_periodic(value: str | None) -> str | None:
    if value is None:
        return None
    val = value.strip().lower()
    if val in {'', 'none', 'open'}:
        return None
    return val


def build_geometry(
    nx: int,
    ny: int,
    orientation: str,
    a_cc: float,
    periodic: str | None,
    passivate_edges: bool,
    passivate_x: bool,
):
    gen = GrapheneGenerator.generate(
        nx=nx,
        ny=ny,
        orientation=orientation,
        a_cc=a_cc,
        periodic_dirs=periodic,
        passivate_edges=passivate_edges,
        passivate_x=passivate_x,
    )
    coords = np.array(gen.c_positions)
    return gen, coords


def plot_geometry(gen, coords: np.ndarray, orientation: str, out_path: str) -> None:
    plt.figure(figsize=(5, 5))
    carbon = np.asarray(gen.c_positions)
    plt.scatter(carbon[:, 0], carbon[:, 1], s=50, facecolor='#1b9e77', edgecolor='k', linewidths=0.3, label='C')
    if gen.h_positions:
        hydrogens = np.asarray(gen.h_positions)
        plt.scatter(hydrogens[:, 0], hydrogens[:, 1], s=25, facecolor='#d95f02', edgecolor='k', linewidths=0.2, label='H')
        plt.legend(loc='upper right', frameon=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x (Å)')
    plt.ylabel('y (Å)')
    plt.title(f'Graphene {orientation} lattice')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def compute_ldos(
    nx: int,
    ny: int,
    orientation: str,
    a_cc: float,
    periodic: str | None,
    e_min: float,
    e_max: float,
    e_points: int,
    eta: float,
    passivate_edges: bool,
    passivate_x: bool,
):
    ham = Hamiltonian(
        structure='graphene',
        nx=nx,
        ny=ny,
        graphene_orientation=orientation,
        a_cc=a_cc,
        periodic_dirs=periodic,
        graphene_passivate=passivate_edges,
        passivate_x=passivate_x,
        comp_overlap=False,
    ).initialize()
    matrix = ham.h_matrix.toarray() if hasattr(ham.h_matrix, 'toarray') else ham.h_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    energies = np.linspace(e_min, e_max, e_points)
    prefactor = 1.0 / (eta * np.sqrt(np.pi))
    num_atoms = ham.num_of_nodes
    offsets = list(ham._offsets) + [ham.basis_size]
    ldos_atoms = np.zeros((num_atoms, e_points), dtype=float)
    vec_sq = np.abs(eigenvectors) ** 2
    for idx, eig_val in enumerate(eigenvalues):
        weight = np.exp(-((energies - eig_val) / eta) ** 2) * prefactor
        orb_density = vec_sq[:, idx]
        for atom_index in range(num_atoms):
            start = offsets[atom_index]
            end = offsets[atom_index + 1]
            contribution = orb_density[start:end].sum()
            ldos_atoms[atom_index] += contribution * weight
    total_dos = ldos_atoms.sum(axis=0)
    atom_coords = np.asarray(list(ham.atom_list.values()))
    atom_labels = list(ham.atom_list.keys())
    return {
        'energies': energies,
        'total_dos': total_dos,
        'eigenvalues': eigenvalues,
        'ldos_atoms': ldos_atoms,
        'atom_coords': atom_coords,
        'atom_labels': atom_labels,
    }


def plot_dos_curve(energies: np.ndarray, total_dos: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(energies, total_dos, color='#d95f02', linewidth=1.3)
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS (states / eV)')
    plt.title('Graphene DOS (Gaussian broadened)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_ldos_map(coords: np.ndarray, ldos_atoms: np.ndarray, energies: np.ndarray, target_energy: float, out_path: str) -> None:
    idx = int(np.argmin(np.abs(energies - target_energy)))
    values = ldos_atoms[:, idx]
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis', s=55, edgecolors='k', linewidths=0.25)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(sc, label=f'LDOS at {energies[idx]:.3f} eV (a.u.)')
    plt.xlabel('x (Å)')
    plt.ylabel('y (Å)')
    plt.title('Site-resolved LDOS')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate graphene lattice and LDOS plots.')
    parser.add_argument('--nx', type=int, default=8, help='Number of cells along transport (x) direction.')
    parser.add_argument('--ny', type=int, default=6, help='Number of cells along transverse (y) direction.')
    parser.add_argument('--orientation', choices=['zigzag', 'armchair'], default='zigzag', help='Edge orientation along x.')
    parser.add_argument('--a-cc', type=float, default=1.42, help='Carbon-carbon bond length in Å.')
    parser.add_argument('--periodic', default='xy', help="Periodic directions (subset of 'xy') or 'none'.")
    parser.add_argument('--eta', type=float, default=0.05, help='Gaussian broadening for LDOS (eV).')
    parser.add_argument('--emin', type=float, default=-5.0, help='Minimum energy for DOS/LDOS sampling (eV).')
    parser.add_argument('--emax', type=float, default=5.0, help='Maximum energy for DOS/LDOS sampling (eV).')
    parser.add_argument('--points', type=int, default=301, help='Number of energy samples for DOS/LDOS.')
    parser.add_argument('--ldos-energy', type=float, default=0.0, help='Energy at which to plot site-resolved LDOS map (eV).')
    parser.add_argument('--outdir', default=os.path.join(_HERE, 'outputs', 'graphene'), help='Output directory for artefacts.')
    parser.add_argument('--passivate-edges', action='store_true', help='Add hydrogen passivation to dangling bonds.')
    parser.add_argument('--include-x-passivation', action='store_true', help='When passivating, also add hydrogens to x edges.')
    args = parser.parse_args()

    periodic = _normalize_periodic(args.periodic)
    os.makedirs(args.outdir, exist_ok=True)

    passivate_x = not args.include_x_passivation
    gen, coords = build_geometry(
        args.nx,
        args.ny,
        args.orientation,
        args.a_cc,
        periodic,
        args.passivate_edges,
        passivate_x,
    )
    lattice_path = os.path.join(args.outdir, f'graphene_{args.orientation}_lattice.png')
    xyz_path = os.path.join(args.outdir, f'graphene_{args.orientation}_nx{args.nx}_ny{args.ny}.xyz')
    GrapheneGenerator.write_xyz(gen, xyz_path)
    plot_geometry(gen, coords, args.orientation, lattice_path)

    ldos_data = compute_ldos(
        nx=args.nx,
        ny=args.ny,
        orientation=args.orientation,
        a_cc=args.a_cc,
        periodic=periodic,
        e_min=args.emin,
        e_max=args.emax,
        e_points=args.points,
        eta=args.eta,
        passivate_edges=args.passivate_edges,
        passivate_x=passivate_x,
    )

    dos_path = os.path.join(args.outdir, f'graphene_{args.orientation}_dos.png')
    plot_dos_curve(ldos_data['energies'], ldos_data['total_dos'], dos_path)

    ldos_map_path = os.path.join(args.outdir, f'graphene_{args.orientation}_ldos_map.png')
    plot_ldos_map(ldos_data['atom_coords'], ldos_data['ldos_atoms'], ldos_data['energies'], args.ldos_energy, ldos_map_path)

    json_path = os.path.join(args.outdir, f'graphene_{args.orientation}_ldos.json')
    with open(json_path, 'w', encoding='ascii') as handle:
        json.dump(
            {
                'nx': args.nx,
                'ny': args.ny,
                'orientation': args.orientation,
                'a_cc': args.a_cc,
                'periodic': periodic,
                'energies': ldos_data['energies'].tolist(),
                'total_dos': ldos_data['total_dos'].tolist(),
            },
            handle,
            indent=2,
        )

    print('Saved lattice plot   :', lattice_path)
    print('Saved DOS plot       :', dos_path)
    print('Saved LDOS map       :', ldos_map_path)
    print('Saved XYZ structure  :', xyz_path)
    print('Saved DOS data JSON  :', json_path)


if __name__ == '__main__':
    main()
