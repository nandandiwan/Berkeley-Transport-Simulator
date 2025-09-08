from __future__ import annotations
"""
Open (finite-in-transport) system structure designer template.

Goal
- Handle devices that are finite along a chosen transport axis (x|y|z)
  and either periodic or finite along the two transverse axes.
- Provide light-weight partitioning (Left, Device, Right) and neighbour queries
  that respect optional transverse periodic boundary conditions (PBC).
- Serve as a template to plug into the existing Hamiltonian builder.

Notes
- This is a template: some methods are intentionally simple and documented with
  TODOs where project-specific logic (like multi-orbital indexing) belongs.
- Uses real-space neighbour detection with an optional set of 1st-order image
  offsets in periodic transverse axes.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Literal, Optional
import numpy as np
from scipy.spatial import cKDTree

Axis = Literal['x','y','z']

@dataclass
class Partitions:
    L: List[int]
    D: List[int]
    R: List[int]

class OpenSystemDesigner:
    def __init__(
        self,
        labels: List[str],
        coords: np.ndarray,
        nn_distance: float,
        transport_axis: Axis = 'x',
        periodic_axes: Iterable[Axis] = ('y',),
        left_boundary: Optional[float] = None,
        right_boundary: Optional[float] = None,
        boundary_tol: float = 1e-6,
        lead_thickness: Optional[float] = None,
    ) -> None:
        self.labels = labels
        self.coords = np.asarray(coords, float)
        self.nn_distance = float(nn_distance)
        self.t_axis = {'x':0,'y':1,'z':2}[transport_axis]
        self.per_axes = {ax for ax in periodic_axes if ax in ('x','y','z') and {'x':0,'y':1,'z':2}[ax] != self.t_axis}
        self.boundary_tol = float(boundary_tol)
        # Determine finite extent along transport axis
        tvals = self.coords[:, self.t_axis]
        self.L_edge = float(np.min(tvals)) if left_boundary is None else float(left_boundary)
        self.R_edge = float(np.max(tvals)) if right_boundary is None else float(right_boundary)
        # Lead slab thickness (default to ~half a nearest-neighbour spacing if not provided)
        self.lead_thickness = float(lead_thickness) if lead_thickness is not None else max(0.5*self.nn_distance, 1e-6)
        # Transverse cell lengths for PBC images (estimate from min/max extents)
        self.cell_size = np.zeros(3)
        for ax, idx in (('x',0),('y',1),('z',2)):
            v = self.coords[:, idx]
            self.cell_size[idx] = float(np.max(v) - np.min(v))
        # Build KD-tree on the original coordinates
        self._tree = cKDTree(self.coords)
        # Precompute simple set of image offsets for transverse PBCs
        self._image_offsets = self._build_image_offsets()

    # -------------------- Partitioning --------------------
    def atom_classifier(self, coord: np.ndarray) -> Literal['L','D','R']:
        """Classify an atom by its position along the transport axis.
        L: within [L_edge, L_edge + lead_thickness]
        R: within [R_edge - lead_thickness, R_edge]
        D: otherwise
        """
        t = coord[self.t_axis]
        if t <= self.L_edge + self.lead_thickness + self.boundary_tol:
            return 'L'
        if t >= self.R_edge - self.lead_thickness - self.boundary_tol:
            return 'R'
        return 'D'

    def build_partitions(self) -> Partitions:
        L=[]; D=[]; R=[]
        for i, p in enumerate(self.coords):
            cls = self.atom_classifier(p)
            if cls == 'L': L.append(i)
            elif cls == 'R': R.append(i)
            else: D.append(i)
        return Partitions(L=L, D=D, R=R)

    # -------------------- Neighbours with optional transverse PBC --------------------
    def _build_image_offsets(self) -> np.ndarray:
        """First-order transverse image offsets for periodic axes.
        For axes not in per_axes or for the transport axis, the offset is 0.
        Returns an array of shape (M,3).
        """
        # Only -1,0,+1 along periodic transverse axes; 0 along transport axis.
        choices = {0:[0]}
        for ax, idx in (('x',0),('y',1),('z',2)):
            if idx == self.t_axis or ax not in self.per_axes:
                choices[idx] = [0]
            else:
                choices[idx] = [-1, 0, 1]
        offs=[]
        for ix in choices[0]:
            for iy in choices[1]:
                for iz in choices[2]:
                    if ix==iy==iz==0:
                        offs.append(np.zeros(3))
                    else:
                        offs.append(np.array([ix*self.cell_size[0], iy*self.cell_size[1], iz*self.cell_size[2]]))
        return np.array(offs, float)

    def get_neighbours(self, i: int) -> List[int]:
        """Return indices of neighbours of atom i within nn_distance.
        Includes interactions to 1st-order images for periodic transverse axes.
        NOTE: This returns indices of original atoms; image information is not retained.
        For Hamiltonian phase/shift handling, you can extend this to return (j, offset_vec).
        """
        origin = self.coords[i]
        neigh=set()
        # Always include original cell neighbours
        dists, inds = self._tree.query(origin, k=len(self.coords), distance_upper_bound=self.nn_distance+1e-12)
        for d,j in zip(dists, inds):
            if np.isfinite(d) and j!=i and j < len(self.coords) and d > self.nn_distance*0.1:
                neigh.add(int(j))
        # Periodic images in transverse axes
        for off in self._image_offsets:
            if not np.any(off):
                continue
            # Query around the shifted point; distances are computed in original cell, so we recompute explicitly
            dists, inds = self._tree.query(origin - off, k=len(self.coords), distance_upper_bound=self.nn_distance+1e-12)
            for d,j in zip(dists, inds):
                if not np.isfinite(d) or j>=len(self.coords) or d <= self.nn_distance*0.1:
                    continue
                # Recalculate exact distance between (origin) and (coords[j] + off)
                dr = origin - (self.coords[j] + off)
                if np.linalg.norm(dr) <= self.nn_distance + 1e-12:
                    neigh.add(int(j))
        return sorted(neigh)

    # -------------------- Block helpers for Hamiltonian partitioning --------------------
    def block_index_sets(self) -> Dict[str, List[int]]:
        """Return dict with index lists for L, D, R suitable for slicing Hamiltonian blocks.
        Example usage (pseudocode):
            S = block_index_sets(); H_LL = H[np.ix_(S['L'], S['L'])]
        """
        P = self.build_partitions()
        return {'L': P.L, 'D': P.D, 'R': P.R}

    def classify_edges(self) -> Dict[str, List[Tuple[int,int]]]:
        """Classify neighbour pairs into block-crossing categories for RGF-like assembly.
        Returns dict with keys: 'LL','DD','RR','LD','DR','LR' (LR should be empty if D connects the leads).
        """
        P = self.build_partitions()
        side = np.empty(len(self.coords), dtype='U1')
        side[P.L] = 'L'; side[P.D] = 'D'; side[P.R] = 'R'
        edges = {'LL':[], 'DD':[], 'RR':[], 'LD':[], 'DR':[], 'LR':[]}
        for i in range(len(self.coords)):
            for j in self.get_neighbours(i):
                a,b = side[i], side[j]
                key = None
                if a==b:
                    key = a+a
                elif (a,b) in (('L','D'),('D','L')):
                    key = 'LD'
                elif (a,b) in (('D','R'),('R','D')):
                    key = 'DR'
                else:
                    key = 'LR'
                if i < j:
                    edges[key].append((i,j))
        return edges

    # -------------------- Convenience --------------------
    @staticmethod
    def from_xyz(xyz_reader: str, nn_distance: float, **opts) -> 'OpenSystemDesigner':
        """Construct from XYZ text. Caller should use existing xyz2np to parse if preferred."""
        from ..io.xyz import xyz2np
        labels, coords = xyz2np(xyz_reader)
        return OpenSystemDesigner(labels=labels, coords=np.array(coords), nn_distance=nn_distance, **opts)
