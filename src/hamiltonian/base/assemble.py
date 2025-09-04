"""Assembly routines for constructing block Hamiltonian pieces.

Current simplified approach:
 - Determine principal layer by slicing geometry along z (transport axis)
 - Build onsite block H0 as diagonal of onsite energies
 - Estimate inter-layer coupling Hl, Hr as simple constant (placeholder)

This will be replaced by full Slater-Koster evaluation later.
"""
from __future__ import annotations
import numpy as np
from ..geometry.xyz_loader import Geometry
from ..tb.orbitals import orbitals_for
from ..tb.parameters_silicon import onsite_energy
from .blocks import HamiltonianBlocks

def build_blocks(geom: Geometry, layer_indices: list[int]) -> HamiltonianBlocks:
    # For now assume entire geometry is single principal layer.
    # layer_indices ignored (placeholder for future multi-layer segmentation).
    basis_labels = []
    onsite = []
    for sym in geom.symbols:
        for orb in orbitals_for(sym):
            basis_labels.append(f"{sym}_{orb}")
            onsite.append(onsite_energy(sym, orb))
    onsite = np.array(onsite, dtype=float)
    n = onsite.size
    H0 = np.diag(onsite)
    # Placeholder couplings: small symmetric coupling between identical orbitals.
    coupling_strength = -0.1  # eV (placeholder)
    Hl = np.zeros((n, n))
    Hr = np.zeros((n, n))
    # Minimal example: couple s-s and p-p orbitals with same index.
    for i, label in enumerate(basis_labels):
        if label.endswith('_s') or '_p' in label:
            Hl[i, i] = coupling_strength
            Hr[i, i] = coupling_strength
    blocks = HamiltonianBlocks(Hl=Hl, H0=H0, Hr=Hr, orbitals_per_layer=n, basis_labels=basis_labels)
    blocks.validate()
    return blocks
