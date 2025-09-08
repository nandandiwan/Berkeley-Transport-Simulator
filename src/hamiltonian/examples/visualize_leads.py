from __future__ import annotations
"""
Visualize lead/device partitioning for an open-system Si nanowire.

Config:
- transport axis: x
- periodic axes: y
- Si nanowire params: nx=3, ny=1, nz=2, a=5.50, periodic_dirs='y', passivate_x=False

Outputs (examples/outputs/open_system_leads/):
- partitions.json: atom list with L/D/R and coordinates
- edges.json: neighbour pairs grouped by LL/DD/RR/LD/DR/LR
- blocks.json: Hamiltonian block shapes and nnz stats
- positions.png: scatter of atoms colored by partition (x-z projection)
"""
import os, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from hamiltonian.geometry.si_nanowire_generator import generate_sinw_xyz
from hamiltonian.io.xyz import xyz2np
from hamiltonian.geometry.open_system_designer import OpenSystemDesigner
from hamiltonian import Hamiltonian


def ensure_outdir() -> Path:
    outdir = Path(__file__).resolve().parent / 'outputs' / 'open_system_leads'
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def build_system():
    a = 5.50
    params = dict(nx=3, ny=1, nz=2, a=a, periodic_dirs='y', passivate_x=False,
                  title='Open-system SiNW for lead visualization')
    xyz = generate_sinw_xyz(**params)
    labels, coords = xyz2np(xyz)
    return a, params, labels, np.array(coords)


def partitions_and_edges(labels, coords, nn_distance, transport_axis='x', periodic_axes=('y',)):
    osd = OpenSystemDesigner(labels=labels, coords=coords, nn_distance=nn_distance,
                             transport_axis=transport_axis, periodic_axes=periodic_axes,
                             lead_thickness=1.0)
    P = osd.build_partitions()
    edges = osd.classify_edges()
    return osd, P, edges


def atom_to_basis_slices(H: Hamiltonian):
    # For each atom index j, return (start, count) of orbital block in the basis.
    labels = list(H.atom_list.keys())
    slices = []
    for j, lab in enumerate(labels):
        norb = H.orbitals_dict[lab].num_of_orbitals  # MyDict strips digits
        start = H._offsets[j]
        slices.append((start, norb))
    return slices


def expand_indices(atom_indices, slices):
    idx = []
    for j in atom_indices:
        start, norb = slices[j]
        idx.extend(range(start, start + norb))
    return np.array(idx, dtype=int)


def plot_positions(coords, parts, outpng: Path):
    # x-z projection colored by partition
    colors = {'L':'tab:blue', 'D':'0.5', 'R':'tab:red'}
    plt.figure(figsize=(6,5))
    for cls, col in colors.items():
        mask = np.array(parts) == cls
        if np.any(mask):
            plt.scatter(coords[mask,0], coords[mask,2], s=30, c=col, label=cls, edgecolors='k', linewidths=0.3)
    plt.xlabel('x (Å)'); plt.ylabel('z (Å)'); plt.title('Partitions (x-z)')
    plt.legend(); plt.tight_layout(); plt.savefig(outpng, dpi=150); plt.close()


def main():
    outdir = ensure_outdir()
    a, params, labels, coords = build_system()

    # Open-system partitions (finite in x, periodic in y)
    osd, P, edges = partitions_and_edges(labels, coords, nn_distance=2.39, transport_axis='x', periodic_axes=('y',))

    # Persist partitions
    parts = ['L' if i in P.L else 'R' if i in P.R else 'D' for i in range(len(coords))]
    with open(outdir / 'partitions.json','w') as f:
        json.dump({
            'params': params,
            'counts': {'L': len(P.L), 'D': len(P.D), 'R': len(P.R)},
            'atoms': [
                {'i': i, 'label': labels[i], 'part': parts[i], 'x': float(coords[i,0]), 'y': float(coords[i,1]), 'z': float(coords[i,2])}
                for i in range(len(coords))
            ]
        }, f, indent=2)

    # Persist edges
    with open(outdir / 'edges.json','w') as f:
        json.dump({k: v for k,v in edges.items()}, f, indent=2)

    # Plot positions
    plot_positions(coords, parts, outdir / 'positions.png')

    # Build Hamiltonian with the same geometry (uses internal TB parameters)
    H = Hamiltonian(nx=params['nx'], ny=params['ny'], nz=params['nz'], a=params['a'],
                    periodic_dirs=params['periodic_dirs'], passivate_x=params['passivate_x'],
                    comp_overlap=False, comp_angular_dep=True)
    H.initialize()
    slices = atom_to_basis_slices(H)
    idxL = expand_indices(P.L, slices); idxD = expand_indices(P.D, slices); idxR = expand_indices(P.R, slices)

    # Extract block sizes and nnz counts above small threshold
    def nnz_block(A):
        return int(np.count_nonzero(np.abs(A) > 1e-12))

    Hmat = H.h_matrix
    blocks = {
        'H_LL_shape': [int(idxL.size), int(idxL.size)],
        'H_DD_shape': [int(idxD.size), int(idxD.size)],
        'H_RR_shape': [int(idxR.size), int(idxR.size)],
        'H_LD_nnz': nnz_block(Hmat[np.ix_(idxL, idxD)]),
        'H_DL_nnz': nnz_block(Hmat[np.ix_(idxD, idxL)]),
        'H_DR_nnz': nnz_block(Hmat[np.ix_(idxD, idxR)]),
        'H_RD_nnz': nnz_block(Hmat[np.ix_(idxR, idxD)]),
    }
    with open(outdir / 'blocks.json','w') as f:
        json.dump(blocks, f, indent=2)

    # Optional small diagnostic: write adjacency CSV for LD and DR couplings on atomic level
    def coupling_pairs(atom_set_a, atom_set_b):
        pairs=[]
        setb = set(atom_set_b)
        for i in atom_set_a:
            for j in osd.get_neighbours(i):
                if j in setb:
                    pairs.append((i,j))
        return pairs
    ld_pairs = coupling_pairs(P.L, P.D)
    dr_pairs = coupling_pairs(P.D, P.R)
    with open(outdir / 'atomic_couplings.json','w') as f:
        json.dump({'LD': ld_pairs, 'DR': dr_pairs}, f, indent=2)

    print(f"Wrote outputs to {outdir}")


if __name__ == '__main__':
    main()
