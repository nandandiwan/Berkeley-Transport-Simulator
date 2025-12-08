#!/usr/bin/env python3
"""Bulk silicon band structure along the standard FCC high-symmetry path."""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.abspath(os.path.dirname(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from hamiltonian import Hamiltonian  # noqa: E402
from hamiltonian.tb.constants import a_si, SPECIAL_K_POINTS_SI  # noqa: E402


def _silicon_conventional_cell(a_lattice: float) -> dict:
    coords = [
        ("Si1", (0.0, 0.0, 0.0)),
        ("Si2", (0.25 * a_lattice, 0.25 * a_lattice, 0.25 * a_lattice)),
        ("Si3", (0.0, 0.5 * a_lattice, 0.5 * a_lattice)),
        ("Si4", (0.25 * a_lattice, 0.75 * a_lattice, 0.75 * a_lattice)),
        ("Si5", (0.5 * a_lattice, 0.0, 0.5 * a_lattice)),
        ("Si6", (0.75 * a_lattice, 0.25 * a_lattice, 0.75 * a_lattice)),
        ("Si7", (0.5 * a_lattice, 0.5 * a_lattice, 0.0)),
        ("Si8", (0.75 * a_lattice, 0.75 * a_lattice, 0.25 * a_lattice)),
    ]
    return {
        "num_atoms": len(coords),
        "title": f"Silicon diamond conventional cell a={a_lattice:.3f} A",
        "atoms": [{label: pos} for label, pos in coords],
    }


def _silicon_primitive_cell(a_lattice: float) -> dict:
    coords = [
        ("Si1", (0.0, 0.0, 0.0)),
        ("Si2", (0.25 * a_lattice, 0.25 * a_lattice, 0.25 * a_lattice)),
    ]
    return {
        "num_atoms": len(coords),
        "title": f"Silicon diamond primitive cell a={a_lattice:.3f} A",
        "atoms": [{label: pos} for label, pos in coords],
    }


def _primitive_vectors(cell_type: str, a_lattice: float) -> list:
    if cell_type == "primitive":
        half = 0.5 * a_lattice
        return [
            [0.0, half, half],
            [half, 0.0, half],
            [half, half, 0.0],
        ]
    return [
        [a_lattice, 0.0, 0.0],
        [0.0, a_lattice, 0.0],
        [0.0, 0.0, a_lattice],
    ]


# Canonical fcc route touches every special point in SPECIAL_K_POINTS_SI.
_PATH_ORDER = ["L", "GAMMA", "X", "W", "K", "L", "W", "X", "U", "K", "GAMMA"]
_TICK_LABELS = {
    "GAMMA": "Γ",
    "X": "X",
    "W": "W",
    "K": "K",
    "L": "L",
    "U": "U",
}


def _build_k_path(points_per_segment: int):
    specials = {key: np.array(val, dtype=float) for key, val in SPECIAL_K_POINTS_SI.items()}
    k_path = []
    k_dist = []
    tick_positions = []
    tick_labels = []

    start_label = _PATH_ORDER[0]
    current = specials[start_label]
    cumulative = 0.0

    k_path.append(current)
    k_dist.append(cumulative)
    tick_positions.append(cumulative)
    tick_labels.append(_TICK_LABELS.get(start_label, start_label))

    for idx in range(len(_PATH_ORDER) - 1):
        left_label = _PATH_ORDER[idx]
        right_label = _PATH_ORDER[idx + 1]
        left = specials[left_label]
        right = specials[right_label]
        delta = right - left
        seg_len = np.linalg.norm(delta)
        if seg_len == 0.0:
            continue
        for step in range(1, points_per_segment + 1):
            frac = step / float(points_per_segment)
            point = left + frac * delta
            k_path.append(point)
            k_dist.append(cumulative + frac * seg_len)
        cumulative += seg_len
        tick_positions.append(cumulative)
        tick_labels.append(_TICK_LABELS.get(right_label, right_label))

    return np.array(k_path), np.array(k_dist), tick_positions, tick_labels


def _build_hamiltonian(cell_type: str) -> tuple[Hamiltonian, int]:
    if cell_type == "primitive":
        cell = _silicon_primitive_cell(a_si)
    else:
        cell = _silicon_conventional_cell(a_si)
    ham = Hamiltonian(xyz=cell, nn_distance=2.39, comp_overlap=False).initialize()
    ham.set_periodic_bc(_primitive_vectors(cell_type, a_si))
    return ham, cell["num_atoms"]


def _analyze_band_edges(energies: np.ndarray, k_dist: np.ndarray, atoms_per_cell: int,
                        electrons_per_atom: int = 4) -> dict:
    total_electrons = electrons_per_atom * atoms_per_cell
    occupied_bands = total_electrons // 2
    if occupied_bands < 1:
        raise ValueError("Not enough electrons to determine valence band.")
    if occupied_bands >= energies.shape[1]:
        raise ValueError(
            "Number of computed bands is insufficient to resolve the conduction band. "
            "Increase --num-bands."
        )

    valence_band = energies[:, occupied_bands - 1]
    conduction_band = energies[:, occupied_bands]

    v_idx = int(np.argmax(valence_band))
    c_idx = int(np.argmin(conduction_band))
    v_max = float(valence_band[v_idx])
    c_min = float(conduction_band[c_idx])
    gap = c_min - v_max

    return {
        "gap": gap,
        "occupied_bands": occupied_bands,
        "valence": {
            "energy": v_max,
            "k_index": v_idx,
            "k_dist": float(k_dist[v_idx]),
        },
        "conduction": {
            "energy": c_min,
            "k_index": c_idx,
            "k_dist": float(k_dist[c_idx]),
        },
    }


def compute_band_structure(points_per_segment: int, num_bands: int, cell_type: str):
    ham, atoms_per_cell = _build_hamiltonian(cell_type)
    diag_fn = getattr(ham, "diagonalize_periodic_bc", ham.diagonalize_k)
    k_vecs, k_dist, tick_positions, tick_labels = _build_k_path(points_per_segment)
    energies = []
    for k_vec in k_vecs:
        vals, _ = diag_fn(k_vec.tolist())
        energies.append(np.real(vals[:num_bands]))
    energies = np.array(energies)
    band_metrics = _analyze_band_edges(energies, k_dist, atoms_per_cell)
    return k_dist, energies, tick_positions, tick_labels, band_metrics


def plot_band_structure(k_dist, energies, ticks, labels, out_path, band_info):
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(k_dist, energies, color="#1f77b4", linewidth=0.7)
    for x_tick in ticks:
        ax.axvline(x=x_tick, color="0.8", linewidth=0.6, linestyle="--")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_title("Bulk silicon band structure")
    ax.axhline(0.0, color="0.2", linewidth=0.6, linestyle="-")
    ax.set_ylim(-5.0, 5.0)
    if band_info and band_info["gap"] > 0:
        v = band_info["valence"]
        c = band_info["conduction"]
        ax.scatter([v["k_dist"]], [v["energy"]], color="#d62728", marker="o", zorder=5)
        ax.scatter([c["k_dist"]], [c["energy"]], color="#9467bd", marker="o", zorder=5)
        ax.annotate(
            "VBM",
            xy=(v["k_dist"], v["energy"]),
            xytext=(-18, -20),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"),
        )
        ax.annotate(
            "CBM",
            xy=(c["k_dist"] if c["energy"] <= 5 else ticks[-2], min(c["energy"], 5)),
            xytext=(10, 18),
            textcoords="offset points",
            fontsize=9,
            color="#9467bd",
            arrowprops=dict(arrowstyle="->", color="#9467bd"),
        )
        gap_label = f"E_g = {band_info['gap']:.3f} eV"
        ax.text(
            0.02,
            0.94,
            gap_label,
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="top",
            color="#000",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot the bulk silicon band structure along FCC high-symmetry lines.")
    parser.add_argument("--points-per-segment", type=int, default=40,
                        help="Samples inserted between successive special k-points (default: 40).")
    parser.add_argument("--num-bands", type=int, default=24,
                        help="How many bands to retain in the output (default: 24).")
    parser.add_argument("--cell", choices=["conventional", "primitive"], default="conventional",
                        help="Choose between the 8-atom conventional cubic cell or the 2-atom primitive cell.")
    parser.add_argument("--output", default=os.path.join(_HERE, "outputs", "silicon_bulk", "si_bulk_band_structure.png"),
                        help="Destination PNG path (directories are created automatically).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    k_dist, energies, ticks, labels, band_info = compute_band_structure(
        args.points_per_segment, args.num_bands, args.cell
    )
    plot_band_structure(k_dist, energies, ticks, labels, args.output, band_info)

    npz_path = args.output[:-4] + ".npz"
    np.savez(
        npz_path,
        k_dist=k_dist,
        energies=energies,
        tick_positions=np.array(ticks),
        tick_labels=np.array(labels),
        band_gap=np.array(band_info["gap"]),
        valence_k=np.array(band_info["valence"]["k_dist"]),
        valence_energy=np.array(band_info["valence"]["energy"]),
        conduction_k=np.array(band_info["conduction"]["k_dist"]),
        conduction_energy=np.array(band_info["conduction"]["energy"]),
        occupied_bands=np.array(band_info["occupied_bands"]),
    )
    print("Band gap (eV):", f"{band_info['gap']:.3f}" if band_info["gap"] > 0 else "metallic/overlap")
    print("Valence band maximum at k = {:.4f}, Energy = {:.4f} eV".format(
        band_info["valence"]["k_dist"], band_info["valence"]["energy"]
    ))
    print("Conduction band minimum at k = {:.4f}, Energy = {:.4f} eV".format(
        band_info["conduction"]["k_dist"], band_info["conduction"]["energy"]
    ))
    print("Saved band structure plot:", args.output)
    print("Saved band structure data:", npz_path)


if __name__ == "__main__":
    main()
