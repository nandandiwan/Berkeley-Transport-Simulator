import os, json, numpy as np
from hamiltonian.bandcalc.compute_band_structure import compute_band_structure

# Test band structure generation for SiNW2 xyz structure.
# Basic validation plus a simple in-memory plot (not saved).

def test_sinw2_band_structure():
    # Choose first occurrence of SiNW2.xyz in repo resources
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    candidates = [
        'resources/Nanonet/NanoNet/examples/input_samples/SiNW2.xyz',
        'resources/Nanonet/NanoNet/nanonet/SiNW2.xyz',
        'Si_transmission/input_samples/SiNW2.xyz'
    ]
    xyz_path = None
    for rel in candidates:
        p = os.path.join(repo_root, rel)
        if os.path.exists(p):
            xyz_path = p
            break
    assert xyz_path is not None, 'SiNW2.xyz not found in expected locations'
    data = compute_band_structure(xyz_path, kpts=20, kmax=0.3, bands=20)
    # Basic shape checks
    assert 'kz' in data and 'bands' in data
    assert len(data['kz']) == 20
    assert len(data['bands']) == 20
    assert all(len(row) == 20 for row in data['bands'])
    # Monotonic k
    kz = np.array(data['kz'])
    assert np.all(np.diff(kz) > 0)
    # Energies real and sorted per k (first 30 considered)
    energies = np.array(data['bands'])
    assert np.isfinite(energies).all()
    assert all(np.all(np.diff(row) >= -1e-8) for row in energies)
    # Basic plotting (not saved)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,5))
        for band in energies.T[:10]:
            ax.plot(kz, band, color='black', linewidth=0.7)
        ax.set_xlabel('k_z (1/Ang)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('SiNW2 Bands (preview)')
        plt.close(fig)
    except Exception as e:
        # If plotting backend unavailable, skip silently per request of basic functionality.
        print('Plotting skipped:', e)
    # Print concise snapshot for debug visibility
    print('sinw2_first_band_head', np.array(data['bands'])[0][:5])
