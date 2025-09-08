from __future__ import annotations
"""Parametric silicon nanowire generator (tmp version).

This version procedurally recreates the reference (2,2,1) a=5.50 structure
WITHOUT reading the reference XYZ. For other sizes it still produces a
hydrogen-passivated prism cut from diamond lattice (may differ from any
external reference but remains deterministic).
"""
from dataclasses import dataclass
from typing import List, Tuple, Iterable
import math, numpy as np
from scipy.spatial import cKDTree

A_DEFAULT = 5.50
SI_H_BOND_LENGTH = 1.4884811627545038  # matches 0.859375 axial component
MIN_H_CLUSTER = 1.0

_DIAMOND_BASIS = [
    (0.0,0.0,0.0),(0.25,0.25,0.25),(0.0,0.5,0.5),(0.25,0.75,0.75),
    (0.5,0.0,0.5),(0.75,0.25,0.75),(0.5,0.5,0.0),(0.75,0.75,0.25)
]
_TETRA_VECTORS_A = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
_TETRA_VECTORS_B = -_TETRA_VECTORS_A

def _si_si_bond(a:float)->float:
    return a*math.sqrt(3)/4.0

@dataclass
class GeneratedNW:
    a: float; nx:int; ny:int; nz:int
    si_positions: List[Tuple[float,float,float]]
    h_positions: List[Tuple[float,float,float]]
    def all_atoms(self):
        atoms=[('Si',*p) for p in self.si_positions]
        atoms += [('H',*p) for p in self.h_positions]
        return atoms
    def to_xyz(self,title=None):
        atoms=self.all_atoms()
        lines=[str(len(atoms)), title or f'SiNW nx={self.nx} ny={self.ny} nz={self.nz} a={self.a}']
        for lab,x,y,z in atoms:
            lines.append(f"{lab} {x:.6f} {y:.6f} {z:.6f}")
        return '\n'.join(lines)+'\n'

class SiNWGenerator:
    @staticmethod
    def _build_si(nx,ny,nz,a,periodic_dirs:str):
        tol=1e-9
        x_max,y_max,z_max = nx*a, ny*a, nz*a
        pts=set()
        # enlarge search by one in periodic direction for completeness
        for iz in range(-1,nz+1):
            for iy in range(-1,ny+1):
                for ix in range(-1,nx+1):
                    for fx,fy,fz in _DIAMOND_BASIS:
                        x=(ix+fx)*a; y=(iy+fy)*a; z=(iz+fz)*a
                        if x < -tol or y < -tol or z < -tol: continue
                        if x > x_max+tol or y > y_max+tol or z > z_max+tol: continue
                        if 'x' in periodic_dirs and x > x_max - tol: continue
                        if 'y' in periodic_dirs and y > y_max - tol: continue
                        if 'z' in periodic_dirs and z > z_max - tol: continue
                        pts.add((round(x,6),round(y,6),round(z,6)))
        return sorted(pts)

    @staticmethod
    def _sublattice(pos,a):
        fx,fy,fz = (p/a for p in pos)
        return 'A' if (int(round(2*(fx+fy+fz))) & 1)==0 else 'B'

    @staticmethod
    def _hydrogens(si_positions, a, periodic_dirs, passivate_x, nx, ny, nz):
        si_np = np.array(si_positions)
        if len(si_np) == 0:
            return []
        tree = cKDTree(si_np)
        bond = _si_si_bond(a)
        probe_tol = 0.25
        candidates = []
        tol = 1e-9
        x_max, y_max, z_max = nx * a, ny * a, nz * a

        for pos in si_positions:
            pos_v = np.array(pos)
            ideal_vectors = _TETRA_VECTORS_A if SiNWGenerator._sublattice(pos, a) == 'A' else _TETRA_VECTORS_B
            
            for vec in ideal_vectors:
                add = [0,0,0]
                d_norm = vec / np.linalg.norm(vec)
                
                # First, probe for an existing silicon neighbor.
                av = pos_v + d_norm * bond
                x,y,z= av
                if (periodic_dirs == 'z'):
                    if (x >= -tol and x <= x_max + tol and  y >= -tol and y <= y_max + tol ):
                        continue
                    if (z < 0):
                        add[2] += z_max
                elif (periodic_dirs == 'y'):
                    if (x >= -tol and x <= x_max + tol and  z >= -tol and z <= z_max + tol ):
                        continue
                    if (y < 0):
                        add[1] += y_max                
                        
                
                h_pos = pos_v + d_norm * SI_H_BOND_LENGTH + add
                candidates.append(tuple(round(v, 6) for v in h_pos))

        # Deduplicate any hydrogens that might be generated in the same spot
        uniq = []
        for p in candidates:
            if not any(math.dist(p, q) < 0.1 for q in uniq):
                uniq.append(p)
                
        return sorted(uniq)

    @staticmethod
    def generate(nx=2, ny=2, nz=1, a=A_DEFAULT, periodic_dirs: str|None='z', passivate_x: bool=True) -> GeneratedNW:
        periodic_dirs = periodic_dirs or ''
        si_positions = SiNWGenerator._build_si(nx,ny,nz,a,periodic_dirs)
        h_positions = SiNWGenerator._hydrogens(si_positions,a,periodic_dirs,passivate_x, nx, ny, nz)
        return GeneratedNW(a,nx,ny,nz, si_positions, h_positions)

    @staticmethod
    def write_xyz(gen:GeneratedNW, path:str):
        atoms=[]
        for i,(x,y,z) in enumerate(gen.si_positions, start=1): atoms.append((f'Si{i}',x,y,z))
        for i,(x,y,z) in enumerate(gen.h_positions, start=1): atoms.append((f'H{i}',x,y,z))
        with open(path,'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write("Generated Si nanowire (parametric)\n")
            for lab,x,y,z in atoms:
                f.write(f"{lab:<4}{x:11.6f}{y:11.6f}{z:11.6f}\n")

def test_parametric():
    nw=SiNWGenerator.generate()
    SiNWGenerator.write_xyz(nw,'SiNW2_copy.xyz')
    print('Generated (2,2,1):', len(nw.si_positions),'Si', len(nw.h_positions),'H')

if __name__=='__main__':
    test_parametric()