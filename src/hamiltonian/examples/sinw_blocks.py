"""Example: load Si nanowire XYZ and build placeholder Hamiltonian blocks.
Run: python -m hamiltonian.examples.sinw_blocks --xyz path/to/SiNW2.xyz
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from ..geometry.xyz_loader import load_geometry
from ..base.assemble import build_blocks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xyz', required=True, help='Path to SiNW2.xyz file')
    ap.add_argument('--out', help='Optional JSON output for summary metrics')
    args = ap.parse_args()
    geom = load_geometry(args.xyz)
    blocks = build_blocks(geom, layer_indices=[0])
    print(f"Loaded geometry: {geom.natoms} atoms -> dim {blocks.dim} orbitals")
    print("Sample diag(H0) first 10:", blocks.H0.diagonal()[:10])
    metrics = {
        'natoms': geom.natoms,
        'dim': blocks.dim,
        'H0_trace': float(blocks.H0.trace()),
        'Hl_norm': float(np.linalg.norm(blocks.Hl)),
        'Hr_norm': float(np.linalg.norm(blocks.Hr)),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
    else:
        print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
