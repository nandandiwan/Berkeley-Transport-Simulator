from __future__ import annotations
"""Parametric silicon nanowire generator (tmp version).

This version procedurally recreates the reference (2,2,1) a=5.50 structure
WITHOUT reading the reference XYZ. For other sizes it still produces a
hydrogen-passivated prism cut from diamond lattice (may differ from any
external reference but remains deterministic).
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class GeneratedSimpleStructure:
    a: float; nx:int; ny:int; nz:int
    atom_positions: List[Tuple[float,float,float]]
    symbol: str = 'base'

    def all_atoms(self):
        atoms=[(self.symbol, *p) for p in self.atom_positions]
        return atoms
    def to_xyz(self,title=None):
        atoms=self.all_atoms()
        lines=[str(len(atoms)), title or f'nx={self.nx} ny={self.ny} nz={self.nz} a={self.a}']
        for lab,x,y,z in atoms:
            lines.append(f"{lab} {x:.6f} {y:.6f} {z:.6f}")
        return '\n'.join(lines)+'\n'

class SimpleStructureGenerator:
    @staticmethod
    def _one_d_wire(nx):
        x_max = nx
        atom_list = [None] * (nx + 1)
        for i in range(0, x_max + 1):
            atom_list[i] = (i)
        return atom_list

def generate_1d_wire_xyz(nx: int, a: float = 1.0, axis: str = 'x', symbol: str = 'base', title: str | None = None) -> str:
    """
    Generate a simple 1D wire with equally spaced atoms along a given axis.

    Parameters
    - nx: number of atoms along the axis
    - a: lattice spacing between neighboring atoms
    - axis: one of 'x', 'y', 'z'
    - symbol: atomic label to use (defaults to 'base' to work with Base orbitals)
    - title: optional XYZ title line

    Returns
    - xyz formatted string
    """
    if nx is None or nx <= 0:
        raise ValueError("nx must be a positive integer for 1D wire")
    axis = axis.lower()
    if axis not in ('x', 'y', 'z'):
        raise ValueError("axis must be 'x', 'y', or 'z'")
    coords: List[Tuple[float, float, float]] = []
    for i in range(nx):
        x = i * a if axis == 'x' else 0.0
        y = i * a if axis == 'y' else 0.0
        z = i * a if axis == 'z' else 0.0
        coords.append((x, y, z))
    g = GeneratedSimpleStructure(a=a, nx=nx, ny=1, nz=1, atom_positions=coords, symbol=symbol)
    return g.to_xyz(title or f"1D wire: {symbol}, nx={nx}, a={a}, axis={axis}")