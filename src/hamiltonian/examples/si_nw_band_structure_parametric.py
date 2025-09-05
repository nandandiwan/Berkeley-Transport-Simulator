"""Band structure of a parametric silicon nanowire generated on-the-fly.

Replicates nanoNet-style example but uses internal generator instead of an
external XYZ file.

Parameters match request: nx=2, ny=2, nz=1, periodic along z, passivate_x=True.
"""
import os, sys, json, argparse, numpy as np
_HERE = os.path.abspath(os.path.dirname(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))  # points to .../src
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from hamiltonian import Hamiltonian

DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), 'outputs', 'nanonet_style_param')

def compute(nx, ny, nz, kpts, kmax, nn, a, passivate_x, bands):
    H = Hamiltonian(nx=nx, ny=ny, nz=nz, a=a, periodic_dirs='z', passivate_x=passivate_x, nn_distance=nn).initialize()
    # periodic cell along z length = nz * a (generator uses that periodicity implicitly)
    H.set_periodic_bc([[0,0,nz*a]])
    kk = np.linspace(0, kmax, kpts)
    bands_out = []
    for i,kz in enumerate(kk):
        vals,_ = H.diagonalize_periodic_bc([0,0,kz])
        bands_out.append(vals[:bands])
    return kk, np.array(bands_out)

def plot(kk, bands, out_png):
    import matplotlib.pyplot as plt
    vba = bands.copy(); cba = bands.copy()
    cba[cba < 0] = np.inf; vba[vba > 0] = -np.inf
    plt.figure(figsize=(5,6))
    for row in vba.T: plt.plot(kk, row, color='blue', linewidth=0.7)
    for row in cba.T: plt.plot(kk, row, color='red', linewidth=0.7)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.6)
    plt.xlabel('k_z (1/Ang)'); plt.ylabel('Energy (eV)')
    plt.title('Parametric SiNW Band Structure (2x2x1)')
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=2)
    ap.add_argument('--ny', type=int, default=2)
    ap.add_argument('--nz', type=int, default=1)
    ap.add_argument('--a', type=float, default=5.50)
    ap.add_argument('--kpts', type=int, default=20)
    ap.add_argument('--kmax', type=float, default=0.57)
    ap.add_argument('--nn', type=float, default=2.4)
    ap.add_argument('--bands', type=int, default=120)
    ap.add_argument('--no-passivate-x', action='store_true')
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    kk, bands = compute(args.nx, args.ny, args.nz, args.kpts, args.kmax, args.nn, args.a, not args.no_passivate_x, args.bands)
    json_path = os.path.join(args.outdir, 'SiNW_param_band_structure.json')
    png_path = os.path.join(args.outdir, 'SiNW_param_band_structure.png')
    with open(json_path,'w') as f:
        json.dump({'kz': kk.tolist(), 'bands': bands.tolist(), 'nx': args.nx, 'ny': args.ny, 'nz': args.nz, 'a': args.a}, f, indent=2)
    plot(kk, bands, png_path)
    print('Saved JSON:', json_path)
    print('Saved PNG :', png_path)

if __name__ == '__main__':
    main()