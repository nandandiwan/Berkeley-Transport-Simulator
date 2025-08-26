"""
Parametric Silicon Nanowire (SiNW) Generator with Robust Surface Definition.

This script generates hydrogen-passivated silicon nanowires for arbitrary
supercell dimensions (nx, ny, nz) and periodic boundary conditions.

Methodology:
1.  A silicon nanowire is defined as all atoms from a bulk diamond lattice
    that fall within a specified rectangular prism of size (nx*a, ny*a, nz*a).
    This is achieved by generating atoms in a larger volume and then "carving out"
    the desired block, which correctly includes atoms on the boundary planes.
2.  Periodic directions have their upper boundaries treated as exclusive to
    avoid duplicating atoms.
3.  A periodic-aware neighbor search correctly identifies coordination for all
    atoms, preventing incorrect passivation on periodic faces.
4.  A robust, two-pass algorithm places hydrogen atoms to passivate dangling
    bonds. Passivation on the x-faces can be disabled for transport simulations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import math
import os
from scipy.spatial import cKDTree


A_DEFAULT = 5.50  
SI_H_BOND_LENGTH = 1.48 
MIN_H_H_DISTANCE = 1.0 


_DIAMOND_BASIS = [
    (0.0, 0.0, 0.0),      
    (0.25, 0.25, 0.25),   
    (0.0, 0.5, 0.5),      
    (0.25, 0.75, 0.75),   
    (0.5, 0.0, 0.5),      
    (0.75, 0.25, 0.75),   
    (0.5, 0.5, 0.0),      
    (0.75, 0.75, 0.25),   
]


_TETRA_VECTORS_A = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
_TETRA_VECTORS_B = -_TETRA_VECTORS_A


@dataclass
class GeneratedNW:
    a: float
    nx: int
    ny: int
    nz: int
    si_positions: List[Tuple[float, float, float]]
    h_positions: List[Tuple[float, float, float]]

    def all_atoms(self) -> List[Tuple[str, float, float, float]]:
        """Returns a combined list of all atoms and their positions."""
        atoms = [('Si', *p) for p in self.si_positions]
        atoms.extend(('H', *p) for p in self.h_positions)
        return atoms

class SiNWGenerator:
    @staticmethod
    def generate(nx=2, ny=2, nz=1, a=A_DEFAULT, periodic_dirs: str | None = 'z', passivate_x: bool = True) -> GeneratedNW:
        """
        Generates a silicon nanowire of size (nx, ny, nz) with hydrogen passivation.
        """
        # generate silicon first using known positions
        tol = 1e-6
        x_max, y_max, z_max = nx * a, ny * a, nz * a
        x_min, y_min, z_min = -tol, -tol, -tol

        si_atoms_data_map = {} 
        for iz in range(-1, nz + 1):
            for iy in range(-1, ny + 1):
                for ix in range(-1, nx + 1):
                    for fx, fy, fz in _DIAMOND_BASIS:
                        x, y, z = (ix + fx) * a, (iy + fy) * a, (iz + fz) * a

                        in_bounds = (x_min <= x <= x_max + tol and
                                     y_min <= y <= y_max + tol and
                                     z_min <= z <= z_max + tol)
                        
                        if not in_bounds:
                            continue

                        if periodic_dirs and 'y' in periodic_dirs and y > y_max - tol:
                            continue
                        if periodic_dirs and 'z' in periodic_dirs and z > z_max - tol:
                            continue
                            
                        is_sublattice_A = round(2 * (fx + fy + fz)) % 2 == 0
                        pos_tuple = (round(x, 6), round(y, 6), round(z, 6))
                        si_atoms_data_map[pos_tuple] = {'pos': np.array(pos_tuple), 'is_A': is_sublattice_A}
        
        si_atoms_data = list(si_atoms_data_map.values())
        si_positions_np = np.array([d['pos'] for d in si_atoms_data])
        
        if si_positions_np.shape[0] == 0:
            return GeneratedNW(a, nx, ny, nz, [], [])

        si_si_bond_dist = a * math.sqrt(3) / 4.0
        search_radius = si_si_bond_dist * 1.05

        ghosts = []
        if periodic_dirs:
            if 'y' in periodic_dirs:
                y_shift = np.array([0, ny * a, 0])
                ghosts.extend([si_positions_np - y_shift, si_positions_np + y_shift])
            if 'z' in periodic_dirs:
                z_shift = np.array([0, 0, nz * a])
                ghosts.extend([si_positions_np - z_shift, si_positions_np + z_shift])
            if 'y' in periodic_dirs and 'z' in periodic_dirs:
                y_shift = np.array([0, ny * a, 0])
                z_shift = np.array([0, 0, nz * a])
                ghosts.extend([si_positions_np - y_shift - z_shift, si_positions_np + y_shift - z_shift,
                               si_positions_np - y_shift + z_shift, si_positions_np + y_shift + z_shift])

        search_points = np.vstack([si_positions_np] + ghosts) if ghosts else si_positions_np
        tree = cKDTree(search_points)
        neighbor_map = tree.query_ball_point(si_positions_np, r=search_radius)

        # hydrogens
        potential_h_positions = []
        for i, atom_data in enumerate(si_atoms_data):
            num_neighbors = len(neighbor_map[i]) - 1
            
            if num_neighbors >= 4:
                continue

            ideal_vectors = _TETRA_VECTORS_A if atom_data['is_A'] else _TETRA_VECTORS_B
            pos_i = atom_data['pos']
            
            existing_vectors = [search_points[j] - pos_i for j in neighbor_map[i] if np.linalg.norm(search_points[j] - pos_i) > tol]

            unoccupied_vectors = list(ideal_vectors)
            for exist_v in existing_vectors:
                if not unoccupied_vectors: break
                dot_products = [np.dot(exist_v, ideal_v) for ideal_v in unoccupied_vectors]
                best_match_idx = np.argmax(np.abs(dot_products))
                unoccupied_vectors.pop(best_match_idx)
            
            for d_vec in unoccupied_vectors:

                is_x_bond = np.argmax(np.abs(d_vec)) == 0
                if not passivate_x and is_x_bond:
                    continue

                norm_vec = d_vec / np.linalg.norm(d_vec)
                potential_h_positions.append(pos_i + norm_vec * SI_H_BOND_LENGTH)
        
        if not potential_h_positions:
            return GeneratedNW(a, nx, ny, nz, sorted([tuple(p) for p in si_positions_np]), [])

        # Pass 2: Cull duplicates by clustering nearby candidates
        h_pos_np = np.array(potential_h_positions)
        h_tree = cKDTree(h_pos_np)
        pairs = h_tree.query_pairs(r=MIN_H_H_DISTANCE)
        
        neighbors_map = {i: set() for i in range(len(h_pos_np))}
        for i, j in pairs:
            neighbors_map[i].add(j)
            neighbors_map[j].add(i)

        visited = set()
        final_h_positions = []
        for i in range(len(h_pos_np)):
            if i not in visited:
                final_h_positions.append(tuple(h_pos_np[i]))
                stack = [i]
                cluster_visited = {i}
                while stack:
                    current = stack.pop()
                    visited.add(current)
                    for neighbor in neighbors_map[current]:
                        if neighbor not in cluster_visited:
                            stack.append(neighbor)
                            cluster_visited.add(neighbor)
        
        return GeneratedNW(a, nx, ny, nz, sorted([tuple(p) for p in si_positions_np]), final_h_positions)

    @staticmethod
    def write_xyz(gen: GeneratedNW, path: str):
        """Writes the generated structure to an XYZ file."""
        atoms = gen.all_atoms()
        with open(path, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"Generated SiNW (nx={gen.nx}, ny={gen.ny}, nz={gen.nz}, a={gen.a})\n")
            atoms.sort(key=lambda atom: (atom[0], atom[3], atom[2], atom[1])) # Sort by z,y,x
            for i, (lab, x, y, z) in enumerate(atoms, 1):
                f.write(f"{lab:<2} {x:11.6f} {y:11.6f} {z:11.6f}\n")


def test_parametric():

    
    sinw = SiNWGenerator.generate()
    SiNWGenerator.write_xyz(sinw, "sinw_params_test.xyz")
    
test_parametric()