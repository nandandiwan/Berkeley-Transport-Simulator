import os, json, numpy as np
from hamiltonian import Hamiltonian

# Simple regression-style test for k-dependent diagonalization.
# Uses a tiny k-grid and checks stability of first few bands.

def test_band_structure_small_kgrid():
    xyz_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'sinw_params_test.xyz'))
    H = Hamiltonian(xyz=xyz_path, nn_distance=2.39).initialize()
    # Primitive cell along z inferred from span
    coords = np.array(list(H.atom_list.values()))
    z_min, z_max = coords[:,2].min(), coords[:,2].max()
    a = (z_max - z_min) if (z_max - z_min) > 0 else 1.0
    H.set_periodic_bc([[0,0,a]])
    kpts = [0.0, 0.1]
    prev_vals = None
    for kz in kpts:
        vals, _ = H.diagonalize_k([0,0,kz])
        # basic sanity: eigenvalues sorted ascending
        assert np.all(np.diff(vals[:50]) >= -1e-8)
        if prev_vals is not None:
            # continuity: bands shouldn't jump wildly between nearby k points
            diff = np.abs(vals[:10] - prev_vals[:10])
            assert diff.max() < 1.0  # loose threshold
        prev_vals = vals
    # store snapshot of first 5 bands at last k for human inspection if needed
    snapshot = prev_vals[:5].round(6).tolist()
    print('band_snapshot', json.dumps(snapshot))
