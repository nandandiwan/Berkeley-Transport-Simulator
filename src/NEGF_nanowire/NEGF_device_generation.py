import numpy as np
import os
import sys
import heapq
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class Atom:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False
    def add(self, delta):
        return Atom(self.x+delta[0],self.y+delta[1],self.z+delta[2] )
    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)
    def getPos(self):
        return (self.x, self.y, self.z)
    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __repr__(self):
        return f"Atom({self.x}, {self.y}, {self.z})"
    
class UnitCell:
    @staticmethod
    def normalize(a):
        a = np.asarray(a)
        return a / np.linalg.norm(a)
    @staticmethod
    def _sublattice(atom):    
        return (round(atom.x*4) + round(atom.y*4) + round(atom.z*4)) & 1
    @staticmethod
    def directionalCosine(delta):
        dx,dy,dz = delta
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if norm != 0:
            l = dx / norm
            m = dy / norm
            n = dz / norm
        return l, m, n
    @staticmethod
    def createRotationMatrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    

    
    
    orientation_to_start ={
        (0,1,2,3) : Atom(0,0,0),
        (1,2,3,0) : Atom(0,0.25,0.25),
        (2,3,0,1) : Atom(0,0.5,0),
        (3,0,1,2) : Atom(0,0.75, 0.25),
        (0,3,2,1) : Atom(0,0,0),
        (1,0,3,1) : Atom(0,0.25,0.25),
        (2,1,0,3) : Atom(0,0.5,0),
        (3,2,1,0) : Atom(0,0.75, 0.25)
    } 


    @staticmethod
    def determine_hybridization(delta):
        sign_pattern = np.sign(delta)
        if np.array_equal(sign_pattern, [1, 1, 1]) or np.array_equal(sign_pattern, [-1, -1, -1]):       # Type a
            return 0
        elif np.array_equal(sign_pattern, [1, -1, -1]) or np.array_equal(sign_pattern, [-1, 1, 1]):   # Type b
            return 1
        elif np.array_equal(sign_pattern, [-1, 1, -1]) or np.array_equal(sign_pattern, [1, -1, 1]):   # Type c
            return 2
        elif np.array_equal(sign_pattern, [-1, -1, 1]) or np.array_equal(sign_pattern, [1, 1, -1]):   # Type d
            return 3
        else:
            raise Exception("error in hybridization")
    
    a = 0.5431e-9
    def classify_permutation(p: tuple) -> int | str:
        """
        Classifies a tuple if it's a cyclic permutation of (0,1,2,3) or (3,2,1,0).
        """
        if not isinstance(p, tuple) or sorted(list(p)) != [0, 1, 2, 3]:
            raise ValueError

        if all((p[i] + 1) % 4 == p[(i + 1) % 4] for i in range(4)):
            return 0
        if all((p[i] - 1) % 4 == p[(i + 1) % 4] for i in range(4)):
            
            return (3 - p[0] + 4) % 4 + 1
        
        # If the permutation is valid but doesn't fit the defined patterns.
        return "Error: Sequence is not cyclically ascending or descending."
    def __init__(self, channel_length : float, channel_width : float, channel_thickness : float, orientation = (0,1,2,3), equilibrium_GF = False):
        """
        Builds the nanowire unit cell. 
        
        """
       
        self.channel_length = channel_length
        self.channel_width = channel_width
        self.channel_thickness = channel_thickness 
        self.equilibrium_GF = equilibrium_GF
        
        self.Nz = round(channel_thickness / self.a * 100) /100
        self.Nx = round(self.channel_length / self.a * 100) / 100
        self.Ny = round(self.channel_width / self.a * 100) / 100
        self.raw_deltas = {
            ((0 + UnitCell.classify_permutation(orientation)) % 2): [(+0.25, +0.25, +0.25), (+0.25, -0.25, -0.25),
                (-0.25, +0.25, -0.25), (-0.25, -0.25, +0.25)],
            ((1 + UnitCell.classify_permutation(orientation)) % 2): [(-0.25, -0.25, -0.25), (-0.25, +0.25, +0.25),
                (+0.25, -0.25, +0.25), (+0.25, +0.25, -0.25)]
        }
        self.ATOM_POSITIONS = [] # the order of atoms in this cell is very important as this governs the layer structure 
        self.neighbors = {}
        self.danglingBonds = {}
        self.addAtoms(UnitCell.orientation_to_start[orientation])
        self.map_neighbors_and_dangling_bonds()
        

    def check_in_z_direction(self, atom : Atom):
        return not (atom.z >= self.Nz or atom.z < 0)
    def check_in_x_direction(self, atom : Atom):
        return not (atom.x >= self.Nx or atom.x < 0) 
    def check_in_y_direction(self, atom : Atom):
        return not (atom.y >= self.Ny or atom.y < 0) 
    def addAtoms(self, start_atom: Atom):
        """Priority queue traversal to fill ATOM_POSITIONS in (x, y, z) order."""
        import heapq
        visited = set()
        heap = []
        in_heap = set()
        heapq.heappush(heap, (start_atom.x, start_atom.y, start_atom.z, start_atom))
        in_heap.add(start_atom)

        while heap:
            x, y, z, atom = heapq.heappop(heap)
            in_heap.remove(atom)
            if atom in visited:
                continue
            visited.add(atom)
            self.ATOM_POSITIONS.append(atom)
            for neighbor in self.get_neighbors(atom):
                if neighbor not in visited and neighbor not in in_heap and self.check_in_x_direction(neighbor) and self.check_in_y_direction(neighbor) and self.check_in_z_direction(neighbor):
                    heapq.heappush(heap, (neighbor.x, neighbor.y, neighbor.z, neighbor))
                    in_heap.add(neighbor)
        # Ensure strict (x, y, z) order
        self.ATOM_POSITIONS.sort(key=lambda atom: (atom.x, atom.y, atom.z))
    def map_neighbors_and_dangling_bonds(self):
        """
        Populates self.neighbors and self.danglingBonds for all atoms in ATOM_POSITIONS.
        - self.neighbors[atom]: list of (neighbor, delta, l, m, n) for valid neighbors inside the cell
        - self.danglingBonds[atom]: list of (neighbor, hybridization) for neighbors outside in Y or Z
        """
        self.neighbors = {}
        self.danglingBonds = {}
        atom_set = set(self.ATOM_POSITIONS)
        for atom in self.ATOM_POSITIONS:
            sublattice = UnitCell._sublattice(atom)
            self.neighbors[atom] = []
            self.danglingBonds[atom] = []
            for delta in self.raw_deltas[sublattice]:
                neighbor = atom.add(delta)
                # Ignore dangling bonds in X direction

                if self.equilibrium_GF:
                    if not self.check_in_y_direction(neighbor) or not self.check_in_z_direction(neighbor) or not self.check_in_x_direction(neighbor):
                        self.danglingBonds[atom].append((neighbor, UnitCell.determine_hybridization(delta)))         
                    else:
                        if neighbor in atom_set:
                            l, m, n = UnitCell.directionalCosine(delta)
                            self.neighbors[atom].append((neighbor, delta, l, m, n))           
                else:
                    if not self.check_in_y_direction(neighbor) or not self.check_in_z_direction(neighbor):
                        self.danglingBonds[atom].append((neighbor, UnitCell.determine_hybridization(delta)))   
                    elif self.check_in_x_direction(neighbor):
                        # Only add as neighbor if inside cell in all directions
                        if neighbor in atom_set:
                            l, m, n = UnitCell.directionalCosine(delta)
                            self.neighbors[atom].append((neighbor, delta, l, m, n))
                    
    def get_neighbors(self, atom: Atom):
        """Return a list of neighboring Atom objects using raw_deltas and sublattice."""
        sublattice = UnitCell._sublattice(atom)
        deltas = self.raw_deltas[sublattice]
        neighbors = [atom.add(delta) for delta in deltas]
        return neighbors
        
class PeriodicUnitCell(UnitCell):
    """
    A UnitCell that implements periodic boundary conditions in the X-direction
    (transport direction) and treats Y and Z as finite with dangling bonds.

    This class is intended for generating the lead Hamiltonians (H00 and H01).
    """
    def __init__(self, channel_length: float, channel_width: float, channel_thickness: float, orientation=(0, 1, 2, 3)):
        # This dictionary will store the connections that cross the periodic boundary.
        # Key: atom in the cell, Value: list of (neighbor_in_next_cell, delta, l, m, n)
        self.periodic_neighbors = {}
        
        # Call the parent class's __init__ method.
        # We pass equilibrium_GF=False because we are manually handling all boundary logic here.
        super().__init__(channel_length, channel_width, channel_thickness, orientation, equilibrium_GF=False)

    def map_neighbors_and_dangling_bonds(self):
        """
        Overrides the parent method to implement periodic boundary conditions in X.

        - Populates self.neighbors for connections within the cell.
        - Populates self.periodic_neighbors for connections crossing the X-boundary.
        - Populates self.danglingBonds for connections crossing the Y/Z boundaries.
        """
        # Reset all neighbor dictionaries
        self.neighbors = {}
        self.danglingBonds = {}
        self.periodic_neighbors = {}
        
        atom_set = set(self.ATOM_POSITIONS)

        for atom in self.ATOM_POSITIONS:
            # Initialize lists for the current atom
            self.neighbors[atom] = []
            self.danglingBonds[atom] = []
            self.periodic_neighbors[atom] = []
            
            sublattice = self._sublattice(atom)
            
            for delta in self.raw_deltas[sublattice]:
                neighbor = atom.add(delta)

                # Case 1: The neighbor is a dangling bond on the Y or Z surface.
                if not self.check_in_y_direction(neighbor) or not self.check_in_z_direction(neighbor):
                    self.danglingBonds[atom].append((neighbor, self.determine_hybridization(delta)))
                
                # Case 2: The neighbor is inside the cell's X, Y, and Z bounds.
                elif self.check_in_x_direction(neighbor):
                    if neighbor in atom_set:
                        l, m, n = self.directionalCosine(delta)
                        self.neighbors[atom].append((neighbor, delta, l, m, n))
                
                # Case 3: The neighbor is outside the X-boundary (a periodic connection).
                else:
                    l, m, n = self.directionalCosine(delta)
                    # This neighbor connects to the next/previous unit cell.
                    self.periodic_neighbors[atom].append((neighbor, delta, l, m, n))