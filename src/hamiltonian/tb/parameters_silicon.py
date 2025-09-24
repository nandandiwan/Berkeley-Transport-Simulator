# """Silicon tight-binding onsite and hopping parameters (placeholder subset).

# For now provide only onsite energies; distance/angular dependent SK terms will be
# sourced later (or copied verbatim when needed). This lets us scaffold the block
# extraction pipeline. Replace with full parameterization for quantitative work.
# """
# from __future__ import annotations
# import numpy as np

# # Onsite (eV) example placeholders (NOT final calibrated values!)
# # Real implementation should load from calibrated dataset.
# ONSITE_SI = {
#     "s": -5.0,
#     "px": 0.0,
#     "py": 0.0,
#     "pz": 0.0,
#     "dxy": 4.0,
#     "dyz": 4.0,
#     "dzx": 4.0,
#     "dx2-y2": 4.0,
#     "dz2": 4.0,
#     "s*": 6.5,
# }

# ONSITE_H = {"s": -3.0}

# def onsite_energy(symbol: str, orbital: str) -> float:
#     if symbol == "Si":
#         return ONSITE_SI[orbital]
#     if symbol == "H":
#         return ONSITE_H[orbital]
#     raise KeyError(f"No onsite data for {symbol}")
