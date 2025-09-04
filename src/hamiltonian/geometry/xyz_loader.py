"""XYZ geometry loader (minimal version).

Parses an XYZ file into (symbols, positions ndarray).
Future: extend with lattice vectors, region tags, passivation info.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pathlib import Path

def read_xyz(path: str | Path):
    path = Path(path)
    with path.open() as f:
        header = f.readline()
        try:
            n_atoms = int(header.strip())
        except ValueError as e:
            raise ValueError(f"First line must be atom count, got {header!r}") from e
        _comment = f.readline()  # ignore
        symbols = []
        coords = []
        for i in range(n_atoms):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF at atom {i}/{n_atoms}")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Bad XYZ line {i}: {line!r}")
            symbols.append(parts[0])
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError as e:
                raise ValueError(f"Non-numeric coordinate in line {i}: {line!r}") from e
            coords.append((x, y, z))
    positions = np.array(coords, dtype=float)
    return symbols, positions

@dataclass(slots=True)
class Geometry:
    symbols: list[str]
    positions: np.ndarray  # (N,3)

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    def slice(self, indices: np.ndarray | list[int]) -> "Geometry":
        idx = np.asarray(indices, dtype=int)
        return Geometry(symbols=[self.symbols[i] for i in idx], positions=self.positions[idx])

    def to_species_counts(self) -> dict[str, int]:
        out: dict[str,int] = {}
        for s in self.symbols:
            out[s] = out.get(s, 0) + 1
        return out


def load_geometry(xyz_path: str | Path) -> Geometry:
    symbols, pos = read_xyz(xyz_path)
    return Geometry(symbols=symbols, positions=pos)
