"""Reconstruct Hamiltonian for Si nanowire and dump norms (parity check helper).
Usage:
  PYTHONPATH=NEGF_sim_git/src python -m hamiltonian.examples.reconstruct_si_nw --xyz sinw_params_test.xyz
"""
import argparse, json, numpy as np
from hamiltonian import Hamiltonian

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--nn', type=float, default=2.39)
    ap.add_argument('--out')
    args = ap.parse_args()
    H = Hamiltonian(xyz=args.xyz, nn_distance=args.nn).initialize()
    vals, _ = H.diagonalize()
    metrics = {
        'basis_size': H.basis_size,
        'trace': float(np.trace(H.h_matrix.real)),
        'min_eig': float(vals.min()),
        'max_eig': float(vals.max()),
        'eig_checksum': float(np.sum(vals**2)),
    }
    if args.out:
        open(args.out,'w').write(json.dumps(metrics, indent=2))
    else:
        print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
