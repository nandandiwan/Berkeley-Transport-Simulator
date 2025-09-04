"""Replica of NanoNet silicon nanowire band structure example using new package layout.

Mimics:
  - Orbitals auto from current Hamiltonian implementation
  - nn_distance=2.4 (vs 2.39 default) as in NanoNet
  - Primitive cell z-vector a_si=5.50 Ang
  - k-path 0 -> 0.57 1/Ang with num_points samples
  - Uses diagonalize_periodic_bc alias
  - Splits bands into valence (<0) and conduction (>0) for plotting

Outputs PNG + JSON in examples/outputs/nanonet_style by default.
"""
import os, json, numpy as np, argparse
from hamiltonian import Hamiltonian

DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), 'outputs', 'nanonet_style')
SINW2_CANDIDATES = [
    os.path.join('resources','Nanonet','NanoNet','examples','input_samples','SiNW2.xyz'),
    os.path.join('resources','Nanonet','NanoNet','nanonet','SiNW2.xyz'),
    os.path.join('Si_transmission','input_samples','SiNW2.xyz')
]

def locate_xyz(explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..','..'))
    for rel in SINW2_CANDIDATES:
        cand = os.path.join(repo_root, rel)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError('SiNW2.xyz not found; supply with --xyz')

def compute_bands(xyz, num_points, nn=2.4, a_si=5.50, kmax=0.57):
    H = Hamiltonian(xyz=xyz, nn_distance=nn).initialize()
    H.set_periodic_bc([[0,0,a_si]])
    kk = np.linspace(0, kmax, num_points, endpoint=True)
    bands = []
    for jj,kz in enumerate(kk):
        print(f"{jj}. Processing wave vector {[0,0,kz]}")
        vals,_ = H.diagonalize_periodic_bc([0,0,kz])
        bands.append(vals)
    return kk, np.array(bands)

def plot_split(kk, bands, out_png):
    import matplotlib.pyplot as plt
    vba = bands.copy()
    cba = bands.copy()
    cba[cba < 0] = np.inf
    vba[vba > 0] = -np.inf
    plt.figure(figsize=(5,6))
    # valence
    for row in vba.T:
        plt.plot(kk, row, color='tabblue' if 'tabblue' in plt.colormaps() else 'blue', linewidth=0.7)
    # conduction
    for row in cba.T:
        plt.plot(kk, row, color='tabred' if 'tabred' in plt.colormaps() else 'red', linewidth=0.7)
    plt.axhline(0.0, color='grey', linewidth=0.6, linestyle='--')
    plt.xlabel('k_z (1/Ang)')
    plt.ylabel('Energy (eV)')
    plt.title('SiNW2 Band Structure (NanoNet style)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xyz', help='Path to SiNW2.xyz (optional if auto-detect succeeds)')
    ap.add_argument('--points', type=int, default=20)
    ap.add_argument('--kmax', type=float, default=0.57)
    ap.add_argument('--a_si', type=float, default=5.50)
    ap.add_argument('--nn', type=float, default=2.4)
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    xyz = locate_xyz(args.xyz)
    kk, bands = compute_bands(xyz, args.points, nn=args.nn, a_si=args.a_si, kmax=args.kmax)
    json_path = os.path.join(args.outdir, 'SiNW2_band_structure_nanonet_style.json')
    png_path = os.path.join(args.outdir, 'SiNW2_band_structure_nanonet_style.png')
    with open(json_path,'w') as f:
        json.dump({'kz': kk.tolist(), 'bands': bands.tolist(), 'xyz': xyz, 'a_si': args.a_si, 'nn_distance': args.nn}, f, indent=2)
    plot_split(kk, bands, png_path)
    print('Saved JSON:', json_path)
    print('Saved PNG :', png_path)

if __name__ == '__main__':
    main()