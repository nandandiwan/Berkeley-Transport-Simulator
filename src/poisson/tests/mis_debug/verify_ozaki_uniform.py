from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # keep consistency if plotting is added later

THIS_DIR = Path(__file__).resolve().parent


def _find_src_root(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Unable to locate 'src' directory from {path}")


SRC_ROOT = _find_src_root(THIS_DIR)
sys.path.insert(0, str(SRC_ROOT))

from poisson.negf_coupling import Lead1D, OzakiNEGF, NEGFChargeProvider  # noqa: E402

KB_OVER_Q = 8.617333262145e-5  # Boltzmann constant in eV/K


def fermi(E: np.ndarray, mu: float, T: float) -> np.ndarray:
    beta = 1.0 / (KB_OVER_Q * T)
    return 1.0 / (1.0 + np.exp((E - mu) * beta))


def analytic_uniform_dos(E: np.ndarray, onsite: float, hopping: float) -> np.ndarray:
    t = abs(hopping)
    shifted = E - onsite
    inside = 4.0 * (t ** 2) - shifted ** 2
    rho = np.zeros_like(E)
    mask = inside > 0
    rho[mask] = 1.0 / (np.pi * np.sqrt(inside[mask]))
    return rho


def main() -> None:
    n_sites = 60
    onsite = 0.0
    hopping = -1.5
    temperature = 300.0
    mu = -0.2

    energy_grid = np.linspace(-4.5, 4.5, 2401)

    left_lead = Lead1D(onsite=onsite, hopping=hopping, attach_index=0)
    right_lead = Lead1D(onsite=onsite, hopping=hopping, attach_index=-1)

    negf = OzakiNEGF(
        onsite=np.full(n_sites, onsite),
        hoppings=np.full(n_sites - 1, hopping),
        energy_grid=energy_grid,
        left_lead=left_lead,
        right_lead=right_lead,
        temperature=temperature,
    )

    provider = NEGFChargeProvider(negf)

    potential = np.zeros(n_sites)
    n_lesser = provider.density_from_lesser(potential, mu)
    n_ozaki = provider.density_from_integral(potential, mu, conduction_only=False)

    rel_err = np.linalg.norm(n_lesser - n_ozaki) / np.linalg.norm(n_lesser)
    max_err = np.max(np.abs(n_lesser - n_ozaki))

    f_vals = fermi(energy_grid, mu, temperature)
    dos_vals = analytic_uniform_dos(energy_grid, onsite, hopping)
    analytic_density = np.trapz(dos_vals * f_vals, energy_grid)

    print("Uniform chain NEGF density check")
    print(f"  Sites: {n_sites}, hopping: {hopping:.3f} eV, mu: {mu:.3f} eV, T: {temperature:.1f} K")
    print(f"  Lesser vs Ozaki relative error: {rel_err:.3e}")
    print(f"  Lesser vs Ozaki max abs diff: {max_err:.3e}")
    print(f"  Mean density from lesser: {n_lesser.mean():.6e}")
    print(f"  Mean density from Ozaki:  {n_ozaki.mean():.6e}")
    print(f"  Analytic density (per site): {analytic_density:.6e}")
    print(f"  Analytic vs Ozaki diff: {abs(analytic_density - n_ozaki.mean()):.3e}")


if __name__ == "__main__":
    main()
