from __future__ import annotations

import numpy as np

__all__ = [
    "MIS_LATTICE_A",
    "MIS_SEGMENT_LENGTHS",
    "build_mis_segments",
    "ssh_hopping_pattern",
    "LEAD_ONSITE",
    "LEAD_HOP",
    "HOP_SEMICONDUCTOR_LEAD",
]


MIS_LATTICE_A = 0.5e-9
MIS_SEGMENT_LENGTHS = (20, 6, 20)
LEAD_ONSITE = 0.0
LEAD_HOP = -1.0
HOP_SEMICONDUCTOR_LEAD = -0.9


def ssh_hopping_pattern(n_sites: int, t_strong: float, t_weak: float) -> np.ndarray:
    if n_sites < 2:
        raise ValueError("SSH chain requires at least two lattice sites")
    hops = np.empty(n_sites - 1, dtype=float)
    hops[0::2] = t_strong
    hops[1::2] = t_weak
    return hops


def build_mis_segments() -> dict[str, np.ndarray]:
    n_metal, n_insulator, n_semiconductor = MIS_SEGMENT_LENGTHS

    onsite_m = np.zeros(n_metal)
    hop_m = np.full(n_metal - 1, -1.0)

    onsite_i = np.full(n_insulator, 3.5)
    hop_i = np.full(n_insulator - 1, -1.0)

    onsite_s = np.full(n_semiconductor, 0.2)
    hop_s = ssh_hopping_pattern(n_semiconductor, t_strong=-1.6, t_weak=-0.8)

    hop_interfaces = np.array([-0.8, -0.9])

    onsite = np.concatenate([onsite_m, onsite_i, onsite_s])
    hoppings = np.concatenate([hop_m, hop_interfaces[:1], hop_i, hop_interfaces[1:], hop_s])

    return {
        "onsite_m": onsite_m,
        "onsite_i": onsite_i,
        "onsite_s": onsite_s,
        "hop_m": hop_m,
        "hop_i": hop_i,
        "hop_s": hop_s,
        "hop_interfaces": hop_interfaces,
        "onsite": onsite,
        "hoppings": hoppings,
        "lead_onsite": LEAD_ONSITE,
        "lead_hop": LEAD_HOP,
        "hop_semiconductor_lead": HOP_SEMICONDUCTOR_LEAD,
    }
