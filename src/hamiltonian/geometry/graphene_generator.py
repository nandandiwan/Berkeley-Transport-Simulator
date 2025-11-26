from __future__ import annotations
"""Procedural generator for graphene ribbons suitable for tight-binding runs."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np
from scipy.spatial import cKDTree

A_CC_DEFAULT = 1.42
_TOL = 1e-9
_C_H_BOND = 1.1


@dataclass
class GeneratedGraphene:
    nx: int
    ny: int
    orientation: str
    a_cc: float
    c_positions: List[Tuple[float, float, float]]
    h_positions: List[Tuple[float, float, float]]
    symbol: str = 'Cg'
    hydrogen_symbol: str = 'H'

    def all_atoms(self) -> List[Tuple[str, float, float, float]]:
        atoms = [(self.symbol, *p) for p in self.c_positions]
        atoms += [(self.hydrogen_symbol, *p) for p in self.h_positions]
        return atoms

    def to_xyz(self, title: str | None = None) -> str:
        atoms = self.all_atoms()
        lines = [
            str(len(atoms)),
            title or f"Graphene {self.orientation} nx={self.nx} ny={self.ny} a_cc={self.a_cc}",
        ]
        for label, x, y, z in atoms:
            lines.append(f"{label} {x:.6f} {y:.6f} {z:.6f}")
        return "\n".join(lines) + "\n"


class GrapheneGenerator:
    _SQRT3 = math.sqrt(3.0)

    @classmethod
    def generate(
        cls,
        nx: int,
        ny: int,
        orientation: str = 'zigzag',
        a_cc: float = A_CC_DEFAULT,
        periodic_dirs: str | None = 'x',
        vacuum: float = 0.0,
        symbol: str = 'Cg',
        passivate_edges: bool = False,
        passivate_x: bool = True,
    ) -> GeneratedGraphene:
        if nx is None or nx <= 0:
            raise ValueError('nx must be a positive integer for graphene generator')
        if ny is None or ny <= 0:
            raise ValueError('ny must be a positive integer for graphene generator')

        orientation_key = orientation.lower()
        carbon_coords = cls.build_C(
            nx=nx,
            ny=ny,
            orientation=orientation_key,
            a_cc=a_cc,
            periodic_dirs=periodic_dirs,
            vacuum=vacuum,
        )
        hydrogen_coords = cls.build_H(
            carbon_coords=carbon_coords,
            orientation=orientation_key,
            a_cc=a_cc,
            periodic_dirs=periodic_dirs,
            passivate_edges=passivate_edges,
            passivate_x=passivate_x,
        )

        carbon_positions = [tuple(map(float, row)) for row in carbon_coords]
        hydrogen_positions = [tuple(map(float, row)) for row in hydrogen_coords]
        return GeneratedGraphene(
            nx=nx,
            ny=ny,
            orientation=orientation_key,
            a_cc=a_cc,
            c_positions=carbon_positions,
            h_positions=hydrogen_positions,
            symbol=symbol,
        )

    @classmethod
    def write_xyz(cls, generated: GeneratedGraphene, path: str) -> None:
        atoms = []
        for i, (x, y, z) in enumerate(generated.c_positions, start=1):
            atoms.append((f"{generated.symbol}{i}", x, y, z))
        for i, (x, y, z) in enumerate(generated.h_positions, start=1):
            atoms.append((f"{generated.hydrogen_symbol}{i}", x, y, z))
        with open(path, 'w', encoding='ascii') as handle:
            handle.write(f"{len(atoms)}\n")
            handle.write("Generated graphene ribbon\n")
            for label, x, y, z in atoms:
                handle.write(f"{label:<4}{x:11.6f}{y:11.6f}{z:11.6f}\n")

    @classmethod
    def build_C(
        cls,
        nx: int,
        ny: int,
        orientation: str,
        a_cc: float,
        periodic_dirs: str | None,
        vacuum: float,
    ) -> np.ndarray:
        orientation_key = orientation.lower()
        if orientation_key == 'zigzag':
            coords = cls._build_zigzag_carbons(nx, ny, a_cc, vacuum)
        elif orientation_key == 'armchair':
            coords = cls._build_armchair_carbons(nx, ny, a_cc, vacuum)
        else:
            raise ValueError("should only be armchair or zigzag")

        coords = cls._apply_periodic_trim(coords, periodic_dirs)
        coords = cls._shift_to_origin(coords)
        return np.round(coords, 6)

    @classmethod
    def build_H(
        cls,
        carbon_coords: np.ndarray,
        orientation: str,
        a_cc: float,
        periodic_dirs: str | None,
        passivate_edges: bool,
        passivate_x: bool,
    ) -> np.ndarray:
        if not passivate_edges:
            return np.zeros((0, 3), dtype=float)

        coords = np.asarray(carbon_coords, dtype=float)
        if coords.size == 0:
            return np.zeros((0, 3), dtype=float)

        periodic_axes = cls._parse_periodic_axes(periodic_dirs)
        tol = 1e-6
        min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
        min_y, max_y = coords[:, 1].min(), coords[:, 1].max()

        tree = cKDTree(coords)
        neighbour_radius = a_cc * 1.1
        hydrogens: List[np.ndarray] = []

        for idx, pos in enumerate(coords):
            neighbour_indices = tree.query_ball_point(pos, r=neighbour_radius)
            neighbour_indices = [j for j in neighbour_indices if j != idx]
            if len(neighbour_indices) >= 3 or not neighbour_indices:
                continue

            neighbour_vectors = coords[neighbour_indices] - pos
            missing_vector = -neighbour_vectors.sum(axis=0)
            norm = np.linalg.norm(missing_vector)
            if norm < 1e-8:
                continue

            axis_type = 'y' if abs(missing_vector[1]) > 1e-5 else 'x'
            if axis_type == 'x' and 'x' in periodic_axes:
                continue
            if axis_type == 'y' and 'y' in periodic_axes:
                continue

            on_x_edge = (abs(pos[0] - min_x) <= tol) or (abs(pos[0] - max_x) <= tol)
            on_y_edge = (abs(pos[1] - min_y) <= tol) or (abs(pos[1] - max_y) <= tol)

            if axis_type == 'x':
                if passivate_x:
                    continue
                if not on_x_edge:
                    continue
            else:
                if not on_y_edge:
                    continue

            direction = missing_vector / norm
            hydrogen_pos = pos + direction * _C_H_BOND
            hydrogens.append(hydrogen_pos)

        if hydrogens:
            return np.round(np.asarray(hydrogens, dtype=float), 6)
        return np.zeros((0, 3), dtype=float)

    @classmethod
    def _build_zigzag_carbons(
        cls,
        nx: int,
        ny: int,
        a_cc: float,
        vacuum: float,
    ) -> np.ndarray:
        sqrt3 = cls._SQRT3
        slice_atoms: List[np.ndarray] = []
        for i in range(ny):
            y_val = (i + 0.5) * sqrt3 * a_cc
            slice_atoms.append(np.array([0.0, y_val, vacuum], dtype=float))

        slice_atoms.append(np.array([0.5 * a_cc, 0.0, vacuum], dtype=float))
        for i in range(ny):
            y_val = (i + 1.0) * sqrt3 * a_cc
            slice_atoms.append(np.array([0.5 * a_cc, y_val, vacuum], dtype=float))

        slice_atoms.append(np.array([1.5 * a_cc, 0.0, vacuum], dtype=float))
        for i in range(ny):
            y_val = (i + 1.0) * sqrt3 * a_cc
            slice_atoms.append(np.array([1.5 * a_cc, y_val, vacuum], dtype=float))

        for i in range(ny):
            y_val = (i + 0.5) * sqrt3 * a_cc
            slice_atoms.append(np.array([2.0 * a_cc, y_val, vacuum], dtype=float))

        tiled: List[np.ndarray] = []
        for ix in range(nx):
            shift = 3.0 * a_cc * ix
            for atom in slice_atoms:
                tiled.append(np.array([atom[0] + shift, atom[1], atom[2]], dtype=float))

        if tiled:
            return np.vstack(tiled)
        return np.zeros((0, 3), dtype=float)

    @classmethod
    def _build_armchair_carbons(
        cls,
        nx: int,
        ny: int,
        a_cc: float,
        vacuum: float,
    ) -> np.ndarray:
        if nx <= 0 or ny <= 0:
            return np.zeros((0, 3), dtype=float)

        sin60 = math.sqrt(3.0) / 2.0
        cos60 = 0.5
        base_points = (
            np.array([sin60, 0.0, 0.0], dtype=float),
            np.array([0.0, cos60, 0.0], dtype=float),
            np.array([0.0, 1.5, 0.0], dtype=float),
            np.array([sin60, 2.0, 0.0], dtype=float),
        )

        layer: List[np.ndarray] = []
        for j in range(ny):
            shift = np.array([0.0, 3.0 * j, 0.0], dtype=float)
            for point in base_points:
                layer.append(point + shift)

        tol = 1e-9
        exact_first = [atom.copy() for atom in layer if abs(atom[0]) <= tol]
        x_shift = (sin60 * 2.0) * nx
        for atom in exact_first:
            atom[0] += x_shift

        structure: List[np.ndarray] = []
        for ix in range(nx):
            delta_x = sin60 * 2.0 * ix
            for atom in layer:
                structure.append(np.array([atom[0] + delta_x, atom[1], 0.0], dtype=float))

        structure.extend([atom.copy() for atom in exact_first])

        coords = np.asarray(structure, dtype=float)
        if coords.size == 0:
            return coords
        coords[:, :2] *= a_cc
        coords[:, 2] = vacuum
        return coords

    @classmethod
    def _build_from_primitives(
        cls,
        nx: int,
        ny: int,
        combos: Dict[str, Tuple[int, int]],
        a1: np.ndarray,
        a2: np.ndarray,
        a_cc: float,
        vacuum: float,
    ) -> np.ndarray:
        basis = cls._basis_shifts(a_cc, vacuum)
        t_x = combos['combo_x'][0] * a1 + combos['combo_x'][1] * a2
        t_y = combos['combo_y'][0] * a1 + combos['combo_y'][1] * a2

        first_layer: List[np.ndarray] = []
        for iy in range(ny):
            origin = iy * t_y
            for shift in basis:
                first_layer.append(origin + shift)

        atoms: List[np.ndarray] = []
        for ix in range(nx):
            offset = ix * t_x
            for atom in first_layer:
                atoms.append(atom + offset)

        if atoms:
            return np.asarray(atoms, dtype=float)
        return np.zeros((0, 3), dtype=float)

    @classmethod
    def _rectify(
        cls,
        coords: np.ndarray,
        combo_x: Tuple[int, int],
        combo_y: Tuple[int, int],
        a1: np.ndarray,
        a2: np.ndarray,
    ) -> np.ndarray:
        if coords.size == 0:
            return coords
        vec_x = combo_x[0] * a1 + combo_x[1] * a2
        vec_y = combo_y[0] * a1 + combo_y[1] * a2

        ex = vec_x / np.linalg.norm(vec_x)
        vy = vec_y - np.dot(vec_y, ex) * ex
        ey_norm = np.linalg.norm(vy)
        if ey_norm < _TOL:
            raise ValueError('Graphene generator produced degenerate lattice vectors.')
        ey = vy / ey_norm

        x_coords = coords @ ex
        y_coords = coords @ ey
        return np.column_stack([x_coords, y_coords, coords[:, 2]])

    @classmethod
    def _apply_periodic_trim(cls, coords: np.ndarray, periodic_dirs: str | None) -> np.ndarray:
        if coords.size == 0 or not periodic_dirs:
            return coords
        axes = cls._parse_periodic_axes(periodic_dirs)
        if not axes:
            return coords
        maxima = coords.max(axis=0)
        mask = np.ones(len(coords), dtype=bool)
        if 'x' in axes:
            mask &= np.abs(coords[:, 0] - maxima[0]) >= _TOL
        if 'y' in axes:
            mask &= np.abs(coords[:, 1] - maxima[1]) >= _TOL
        return coords[mask]

    @staticmethod
    def _parse_periodic_axes(periodic_dirs: str | None) -> set[str]:
        if periodic_dirs is None:
            return set()
        return {ch for ch in str(periodic_dirs).lower() if ch in {'x', 'y'}}

    @staticmethod
    def _shift_to_origin(coords: np.ndarray) -> np.ndarray:
        if coords.size == 0:
            return coords
        shift = coords.min(axis=0)
        shifted = coords.copy()
        shifted -= shift
        return shifted

    @classmethod
    def _primitive_vectors(cls, a_cc: float) -> Tuple[np.ndarray, np.ndarray]:
        a1 = np.array([cls._SQRT3 * a_cc, 0.0, 0.0], dtype=float)
        a2 = np.array([0.5 * cls._SQRT3 * a_cc, 1.5 * a_cc, 0.0], dtype=float)
        return a1, a2

    @staticmethod
    def _basis_shifts(a_cc: float, vacuum: float) -> Tuple[np.ndarray, ...]:
        z = float(vacuum)
        return (
            np.array([0.0, 0.0, z], dtype=float),
            np.array([0.0, a_cc, z], dtype=float),
        )


def generate_graphene_xyz(
    nx: int,
    ny: int,
    orientation: str = 'zigzag',
    a_cc: float = A_CC_DEFAULT,
    periodic_dirs: str | None = 'x',
    vacuum: float = 0.0,
    symbol: str = 'Cg',
    passivate_edges: bool = False,
    passivate_x: bool = True,
    title: str | None = None,
) -> str:
    gen = GrapheneGenerator.generate(
        nx=nx,
        ny=ny,
        orientation=orientation,
        a_cc=a_cc,
        periodic_dirs=periodic_dirs,
        vacuum=vacuum,
        symbol=symbol,
        passivate_edges=passivate_edges,
        passivate_x=passivate_x,
    )
    return gen.to_xyz(title)


__all__ = ['GrapheneGenerator', 'GeneratedGraphene', 'generate_graphene_xyz']
