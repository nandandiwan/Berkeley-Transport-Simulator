"""Standalone script that builds and inspects a MIS tight-binding chain.

Run with the project root on PYTHONPATH or simply execute this file directly;
figures showing the DOS and carrier densities will be written next to the
script for quick visual inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipy.constants as spc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dolfinx import mesh
from mpi4py import MPI


THIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = THIS_DIR / "cache"


def _find_src_root(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Unable to locate 'src' directory from {path}")


SRC_ROOT = _find_src_root(THIS_DIR)
sys.path.insert(0, str(SRC_ROOT))
REPO_ROOT = SRC_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from poisson.negf_coupling import (  # noqa: E402
    Lead1D,
    OzakiNEGF,
    NEGFChargeProvider,
    build_tridiagonal_hamiltonian,
    effective_mass_1d_tb,
)
from poisson.fermi import EffectiveMassFermiEstimator, ChandrupatlaFermiSolver  # noqa: E402
from poisson.solver import DirichletBCSpec, NonlinearPoissonSolver  # noqa: E402
from poisson.scf import PoissonNEGFSCFSolver, SCFResult  # noqa: E402
from poisson.tests.mis_params import (  # noqa: E402
    MIS_LATTICE_A,
    MIS_SEGMENT_LENGTHS,
    build_mis_segments,
    LEAD_HOP,
    LEAD_ONSITE,
)


def build_mis_chain() -> tuple[OzakiNEGF, dict[str, object]]:
    lattice_a = MIS_LATTICE_A
    n_metal, n_insulator, n_semiconductor = MIS_SEGMENT_LENGTHS

    params = build_mis_segments()
    onsite = params["onsite"]
    hoppings = params["hoppings"]
    hop_s = params["hop_s"]
    hop_semiconductor_lead = params["hop_semiconductor_lead"]

    onsite = np.concatenate(([LEAD_ONSITE], onsite, [LEAD_ONSITE]))
    hoppings = np.concatenate(([LEAD_HOP], hoppings, [hop_semiconductor_lead]))

    energy_grid = np.linspace(-3.0, 3.0, 301)
    left_lead = Lead1D(
        onsite=LEAD_ONSITE,
        hopping=LEAD_HOP,
        attach_index=0,
        method="nanonet",
    )
    right_lead = Lead1D(
        onsite=LEAD_ONSITE,
        hopping=LEAD_HOP,
        attach_index=-1,
        method="nanonet",
    )

    negf = OzakiNEGF(
        onsite=onsite,
        hoppings=hoppings,
        energy_grid=energy_grid,
        left_lead=left_lead,
        right_lead=right_lead,
        temperature=300.0,
        ozaki_cutoff=80,
    )

    metadata = {
        "lattice_a": lattice_a,
        "lengths": (n_metal + 1, n_insulator, n_semiconductor + 1),
        "onsite": onsite,
        "hopping": hoppings,
        "hop_segments": (
            params["hop_m"],
            params["hop_i"],
            params["hop_s"],
        ),
    }
    return negf, metadata


def build_poisson_solver(
    n_sites: int,
    lattice_a: float,
    boundary_values: tuple[float, float] = (0.0, 0.0),
) -> NonlinearPoissonSolver:
    total_length = lattice_a * (n_sites - 1)
    domain = mesh.create_interval(MPI.COMM_SELF, n_sites - 1, [0.0, total_length])

    def at_left(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0], 0.0)

    def at_right(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0], total_length)

    left_bc, right_bc = (float(boundary_values[0]), float(boundary_values[1]))

    bcs = [
        DirichletBCSpec(left_bc, marker=at_left),
        DirichletBCSpec(right_bc, marker=at_right),
    ]

    solver = NonlinearPoissonSolver(
        domain=domain,
        permittivity=11.7,
        dirichlet_bcs=bcs,
        degree=1,
    )
    solver.update_initial_potential(0.0)
    return solver


def run_poisson_scf(
    provider: NEGFChargeProvider,
    onsite: np.ndarray,
    lattice_a: float,
    mu_ref: np.ndarray,
    net_doping: np.ndarray,
    fermi_solver: ChandrupatlaFermiSolver,
    boundary_values: tuple[float, float] = (0.0, 0.0),
    contact_fermi: Optional[tuple[float, float]] = None,
) -> SCFResult:
    poisson_solver = build_poisson_solver(onsite.size, lattice_a, boundary_values=boundary_values)
    scf_driver = PoissonNEGFSCFSolver(
        poisson_solver,
        provider,
        fermi_solver,
        net_doping=net_doping,
        conduction_band_edge=onsite,
        potential_mixing=0.5,
        tolerance=1e-6,
        max_iterations=10,
        enable_fermi_solver=False,
        conduction_only=True,
        density_scheme="integral",
    )

    initial_potential = np.linspace(boundary_values[0], boundary_values[1], onsite.size)
    result = scf_driver.run(
        initial_potential=initial_potential,
        initial_fermi=mu_ref,
        contact_fermi=contact_fermi,
    )

    print(f"SCF converged={result.converged} after {result.iterations} iterations")
    if result.history:
        last = result.history[-1]
        print(
            f"Final Δφ={last.delta_potential:.3e}, ΔEfn={last.delta_fermi:.3e}, "
            f"Poisson residual={result.poisson_result.residual_norm:.3e}"
        )
    potential = result.potential
    print(
        f"Potential range: min={potential.min():.3e} V, max={potential.max():.3e} V"
    )

    return result


def plot_debug(
    output_dir: Path,
    energy: np.ndarray,
    dos: np.ndarray,
    charge: np.ndarray,
    section_lengths: Sequence[int],
) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 6.5), constrained_layout=True)

    axs[0].plot(energy, dos, lw=1.8)
    axs[0].set_xlabel("Energy (eV)")
    axs[0].set_ylabel("DOS (arb units)")
    axs[0].set_title("Total DOS from LDOS integration")
    axs[0].grid(True, alpha=0.3)

    x = np.arange(charge.size)
    axs[1].plot(x, charge, "-o", lw=1.4, ms=4)
    axs[1].set_xlabel("Site index")
    axs[1].set_ylabel("Charge density (arb)")
    axs[1].set_title("Electron density from G^<")
    axs[1].grid(True, alpha=0.3)
    boundaries = np.cumsum([0, *section_lengths])
    ymax = max(charge.max(), 1.0) if charge.size else 1.0
    labels = ["M/I", "I/S", "end"]
    for xc, label in zip(boundaries[1:], labels):
        axs[1].axvline(x=xc, color="k", linestyle="--", alpha=0.25)
        axs[1].text(xc, ymax, label, va="bottom", ha="right", fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "mis_chain_debug.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved diagnostic figure to {fig_path}")


def main() -> None:
    negf, meta = build_mis_chain()
    provider = NEGFChargeProvider(negf)

    lattice_a = meta["lattice_a"]
    n_metal, n_insulator, n_semiconductor = meta["lengths"]
    onsite = meta["onsite"]
    hoppings = meta["hopping"]
    hop_m, hop_i, hop_s = meta["hop_segments"]

    H = build_tridiagonal_hamiltonian(onsite, hoppings)
    print(f"Hamiltonian shape: {H.shape}")

    potential = np.zeros_like(onsite)
    mu = 0.1

    ldos = negf.ldos_matrix(potential)
    total_dos = ldos.sum(axis=1)

    n_lesser = provider.density_from_lesser(potential, mu)
    n_ozaki = provider.density_from_integral(potential, mu, conduction_only=False)

    diff = np.linalg.norm(n_lesser - n_ozaki) / np.sqrt(n_lesser.size)
    print(f"Electron density check (||n_lesser - n_ozaki|| / sqrt(N)): {diff:.3e}")

    if hop_s.size > 1:
        t_eff = np.sqrt(abs(hop_s[0] * hop_s[1]))
    else:
        t_eff = hop_s[0]
    m_semic = effective_mass_1d_tb(t_eff, lattice_a)
    m_rel = m_semic / spc.m_e
    estimator = EffectiveMassFermiEstimator(m_effective_e=m_rel)
    solver = ChandrupatlaFermiSolver(estimator)

    def density_operator(Efn: np.ndarray, pot: np.ndarray) -> np.ndarray:
        return provider.density_from_integral(
            pot,
            Efn,
            conduction_only=False,
        )

    fermi_guess = np.full_like(onsite, mu)
    fermi_level = solver.solve(
        density_operator,
        target_density=n_lesser,
        Ec=np.zeros_like(onsite),
        net_doping=n_lesser,
        initial_guess=fermi_guess,
        extra_args=(potential,),
    )
    print(
        f"Quasi-Fermi uniformity check: min={fermi_level.min():.3f} eV, max={fermi_level.max():.3f} eV"
    )

    mu_source = float(mu)
    mu_ref = np.full_like(onsite, mu_source)
    doping_profile = np.zeros_like(onsite)
    print("Launching Poisson–NEGF SCF with zero net doping profile")
    scf_results: list[tuple[float, SCFResult]] = []
    result_eq = run_poisson_scf(
        provider,
        onsite,
        lattice_a,
        mu_ref,
        doping_profile,
        solver,
        boundary_values=(0.0, 0.0),
        contact_fermi=(mu_source, mu_source),
    )
    scf_results.append((0.0, result_eq))

    bias_values = np.linspace(0.0, 1, 8)
    for vd in bias_values[1:]:
        mu_drain = mu_source - vd
        mu_profile = np.linspace(mu_source, mu_drain, onsite.size)
        print(f"Running Poisson SCF for Vd={vd:.3f} V")
        result_bias = run_poisson_scf(
            provider,
            onsite,
            lattice_a,
            mu_profile,
            doping_profile,
            solver,
            boundary_values=(0.0, -vd),
            contact_fermi=(mu_source, mu_drain),
        )
        scf_results.append((vd, result_bias))

    output_dir = THIS_DIR
    res_fig, ax_res = plt.subplots(figsize=(6.5, 4.0))
    for vd, result in scf_results:
        if not result.history:
            continue
        iterations = [rec.iteration for rec in result.history]
        residuals = [rec.poisson_residual for rec in result.history]
        ax_res.plot(iterations, residuals, marker="o", label=f"Vd={vd:.2f} V")
    ax_res.set_xlabel("Iteration")
    ax_res.set_ylabel("Poisson residual")
    ax_res.set_yscale("log")
    ax_res.set_title("Poisson Solver Residuals")
    ax_res.grid(True, which="both", alpha=0.3)
    ax_res.legend()
    res_fig.tight_layout()
    residual_fig_path = output_dir / "mis_chain_poisson_residuals.png"
    res_fig.savefig(residual_fig_path, dpi=200)
    plt.close(res_fig)
    print(f"Saved Poisson residual plot to {residual_fig_path}")

    x_nm = np.arange(onsite.size) * lattice_a * 1e9
    pot_fig, ax_pot = plt.subplots(figsize=(6.5, 4.5))
    for vd, result in scf_results:
        ax_pot.plot(x_nm, result.potential, lw=1.5, label=f"Vd={vd:.2f} V")
    boundaries = np.cumsum([0, n_metal + 1, n_insulator, n_semiconductor + 1])
    for b in boundaries[1:-1]:
        ax_pot.axvline(x_nm[b], color="k", linestyle="--", alpha=0.2)
    ax_pot.set_xlabel("Position (nm)")
    ax_pot.set_ylabel("Potential (V)")
    ax_pot.set_title("Poisson Potential Profiles vs Drain Bias")
    ax_pot.grid(True, alpha=0.3)
    ax_pot.legend()
    pot_fig.tight_layout()
    pot_fig_path = output_dir / "mis_chain_poisson_profiles.png"
    pot_fig.savefig(pot_fig_path, dpi=200)
    plt.close(pot_fig)
    print(f"Saved potential profiles to {pot_fig_path}")

    charge_fig, ax_charge = plt.subplots(figsize=(6.5, 4.5))
    for vd, result in scf_results:
        charge_density = -spc.elementary_charge * result.electron_density
        ax_charge.plot(x_nm, charge_density, lw=1.5, label=f"Vd={vd:.2f} V")
    for b in boundaries[1:-1]:
        ax_charge.axvline(x_nm[b], color="k", linestyle="--", alpha=0.2)
    ax_charge.set_xlabel("Position (nm)")
    ax_charge.set_ylabel("Charge density (C per site)")
    ax_charge.set_title("NEGF Charge Density Profiles vs Drain Bias")
    ax_charge.grid(True, alpha=0.3)
    ax_charge.legend()
    charge_fig.tight_layout()
    charge_fig_path = output_dir / "mis_chain_charge_profiles.png"
    charge_fig.savefig(charge_fig_path, dpi=200)
    plt.close(charge_fig)
    print(f"Saved charge density profiles to {charge_fig_path}")

    plot_debug(
        output_dir,
        negf.energy_grid,
        total_dos,
        n_lesser,
        [n_metal + 1, n_insulator, n_semiconductor + 1],
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    transmission_eq = negf.transmission(potential)
    cache_path = CACHE_DIR / "transmission_cache.npz"
    np.savez(cache_path, energy=negf.energy_grid, transmission=transmission_eq)
    print(f"Cached transmission spectrum to {cache_path}")

    fig_t, ax_t = plt.subplots(figsize=(6.5, 4.0))
    ax_t.plot(negf.energy_grid, transmission_eq, lw=1.6)
    ax_t.set_xlabel("Energy (eV)")
    ax_t.set_ylabel("Transmission")
    ax_t.set_title("Transmission Spectrum")
    ax_t.grid(True, alpha=0.3)
    fig_t.tight_layout()
    trans_fig_path = output_dir / "mis_chain_transmission.png"
    fig_t.savefig(trans_fig_path, dpi=200)
    plt.close(fig_t)
    print(f"Saved transmission plot to {trans_fig_path}")

    vd_values = np.array([vd for vd, _ in scf_results])
    currents = []
    total_charge = []
    for vd, result in scf_results:
        mu_drain = mu_source - vd
        pot_bias = result.potential
        transmission_bias = negf.transmission(pot_bias)
        currents.append(
            negf.current(
                pot_bias,
                mu_source,
                mu_drain,
                transmission=transmission_bias,
            )
        )
        n_bias = negf.electron_density_non_equilibrium(pot_bias, mu_source, mu_drain)
        total_charge.append(-spc.elementary_charge * np.sum(n_bias))

    currents = np.asarray(currents)
    total_charge = np.asarray(total_charge)
    delta_charge = total_charge - total_charge[0]
    capacitance = np.gradient(delta_charge, vd_values, edge_order=2)

    cache_iv_path = CACHE_DIR / "transport_curves.npz"
    np.savez(
        cache_iv_path,
        vd=vd_values,
        current=currents,
        total_charge=total_charge,
        delta_charge=delta_charge,
        capacitance=capacitance,
    )
    print(f"Cached transport curves to {cache_iv_path}")

    fig_iv, axs = plt.subplots(2, 1, figsize=(6.5, 7.0), constrained_layout=True)
    axs[0].plot(vd_values, currents * 1e6, lw=1.6)
    axs[0].set_xlabel("Vd (V)")
    axs[0].set_ylabel("Current (µA)")
    axs[0].set_title("Id–Vd (with SCF potential)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(vd_values, capacitance * 1e18, lw=1.6)
    axs[1].set_xlabel("Vd (V)")
    axs[1].set_ylabel("Capacitance (aF)")
    axs[1].set_title("Differential Capacitance vs Vd")
    axs[1].grid(True, alpha=0.3)

    iv_fig_path = output_dir / "mis_chain_transport.png"
    fig_iv.savefig(iv_fig_path, dpi=200)
    plt.close(fig_iv)
    print(f"Saved transport plots to {iv_fig_path}")


if __name__ == "__main__":
    main()
