import json, numpy as np, os
from hamiltonian import Hamiltonian

def infer_cell_vector(coords):
    z_min, z_max = coords[:,2].min(), coords[:,2].max()
    a = (z_max - z_min) if (z_max - z_min) > 0 else 1.0
    return [0,0,a]

def compute_band_structure(xyz_path, kpts=40, kmax=0.5, bands=20, nn=2.39):
    H = Hamiltonian(xyz=xyz_path, nn_distance=nn).initialize()
    coords = np.array(list(H.atom_list.values()))
    H.set_periodic_bc([infer_cell_vector(coords)])
    ks = np.linspace(0, kmax, kpts)
    band_data = []
    for kz in ks:
        vals,_ = H.diagonalize_k([0,0,kz])
        band_data.append(vals[:bands].tolist())
    return {'kz': ks.tolist(), 'bands': band_data}

def save_band_structure(data, out_json):
    with open(out_json,'w') as f:
        json.dump(data, f, indent=2)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--kpts', type=int, default=60)
    ap.add_argument('--kmax', type=float, default=0.5)
    ap.add_argument('--bands', type=int, default=40)
    ap.add_argument('--nn', type=float, default=2.39)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    data = compute_band_structure(args.xyz, kpts=args.kpts, kmax=args.kmax, bands=args.bands, nn=args.nn)
    save_band_structure(data, args.out)
    print(f"Saved band structure to {args.out}")

if __name__ == '__main__':
    main()
