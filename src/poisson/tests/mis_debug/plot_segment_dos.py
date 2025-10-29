from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent


def _find_src_root(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Unable to locate 'src' directory from {path}")


SRC_ROOT = _find_src_root(THIS_DIR)
sys.path.insert(0, str(SRC_ROOT))

from poisson.negf_coupling import build_tridiagonal_hamiltonian  # noqa: E402
from poisson.tests.mis_params import build_mis_segments  # noqa: E402

FERMI_LEVEL = 0.1
ENERGY_GRID = np.linspace(-6.0, 6.0, 1201)
BROADENING = 0.05


def gaussian_broadened_dos(eigenvalues: np.ndarray, energy: np.ndarray, sigma: float) -> np.ndarray:
    diff = (energy[:, None] - eigenvalues[None, :]) / sigma
    pref = 1.0 / (sigma * np.sqrt(np.pi))
    return pref * np.exp(-diff * diff).sum(axis=1)


def main() -> None:
    params = build_mis_segments()

    segments = {
        "metal": (params["onsite_m"], params["hop_m"]),
        "insulator": (params["onsite_i"], params["hop_i"]),
        "semiconductor": (params["onsite_s"], params["hop_s"]),
    }

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.5), sharex=True, constrained_layout=True)

    for ax, (label, (onsite, hops)) in zip(axes, segments.items()):
        H = build_tridiagonal_hamiltonian(onsite, hops)
        eigenvalues = np.linalg.eigvalsh(H.real)
        dos = gaussian_broadened_dos(eigenvalues, ENERGY_GRID, BROADENING)

        ax.plot(ENERGY_GRID, dos, lw=1.6)
        ax.axvline(FERMI_LEVEL, color="k", linestyle="--", linewidth=1.0)
        ax.set_ylabel("DOS (arb units)")
        ax.set_title(f"{label.capitalize()} segment")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Energy (eV)")

    fig.text(0.98, 0.5, f"mu = {FERMI_LEVEL:.2f} eV", rotation=90, va="center", ha="center")

    output_path = THIS_DIR / "segment_dos.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved segment DOS figure to {output_path}")


if __name__ == "__main__":
    main()
