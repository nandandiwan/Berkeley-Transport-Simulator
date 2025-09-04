"""Compute band structure along k-path for a periodic system (simple example).
Usage:
  PYTHONPATH=NEGF_sim_git/src python -m hamiltonian.examples.band_structure --xyz sinw_params_test.xyz \
      --kpts 50 --kmax 0.5
Outputs JSON with eigenvalues at sampled k (only energies list per k).
"""
import argparse, json, numpy as np
from hamiltonian import Hamiltonian

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--kpts', type=int, default=40)
    ap.add_argument('--kmax', type=float, default=0.5, help='max k (1/Ang) sampled from 0->kmax along z')
    ap.add_argument('--nn', type=float, default=2.39)
    ap.add_argument('--out')
    ap.add_argument('--bands', type=int, default=20, help='number of lowest bands to keep')
    args = ap.parse_args()
    H = Hamiltonian(xyz=args.xyz, nn_distance=args.nn).initialize()
    # Set a primitive cell vector approximated as z-span (simplistic)
    coords = np.array(list(H.atom_list.values()))
    z_min, z_max = coords[:,2].min(), coords[:,2].max()
    a = (z_max - z_min) if (z_max - z_min) > 0 else 1.0
    H.set_periodic_bc([[0,0,a]])
    ks = np.linspace(0, args.kmax, args.kpts)
    band_data = []
    for kz in ks:
        vals,_ = H.diagonalize_k([0,0,kz])
        band_data.append(vals[:args.bands].tolist())
    out = {'kz': ks.tolist(), 'bands': band_data}
    if args.out:
        open(args.out,'w').write(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
