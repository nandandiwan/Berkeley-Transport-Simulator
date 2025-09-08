import numpy as np
import os
import sys
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
    
    raw_deltas = {
            0: [(+0.25, +0.25, +0.25), (+0.25, -0.25, -0.25),
                (-0.25, +0.25, -0.25), (-0.25, -0.25, +0.25)],
            1: [(-0.25, -0.25, -0.25), (-0.25, +0.25, +0.25),
                (+0.25, -0.25, +0.25), (+0.25, +0.25, -0.25)]
        }
    
    
    
    """We divide the atoms into 4 layers - 100 cleaving"""


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
    
    def __init__(self, vertical_blocks : int, channel_blocks : int, orientiation = (0,1,2,3), not_NEGF = False):
        self.Nz = vertical_blocks
        self.Nx = channel_blocks # this transport direction
        self.not_NEGF = not_NEGF
        position = {}
        for i, val in enumerate(orientiation):
            position[val] = i
        
        self.raw_atoms = {
            position[0] : [Atom(0.25 * position[0],0,0), Atom(0.25 * position[0], 0.5,0.5)],
            position[1] : [Atom(0.25 * position[1],0.25,0.25), Atom(0.25 * position[1],0.75,0.75)],
            position[2] : [Atom(0.25 * position[2],0.5,0), Atom(0.25 * position[2],0,0.5)],
            position[3] : [Atom(0.25 * position[3],0.75,0.25), Atom(0.25 * position[3], 0.25,0.75)]
        }

        self.ATOM_POSITIONS = []
        self.XZ_map = {}
        for layerX in range(1, self.Nx + 1):
            for layerZ in range(1, self.Nz + 1):
                self.ATOM_POSITIONS.extend(self.baseCoordinates(layerX, layerZ))

        for atom in self.ATOM_POSITIONS:
            nx,ny,nz = atom.x,atom.y,atom.z
            self.XZ_map[(nx,nz)] = atom
    
        self.danglingBondsZ = {} # this is for hydrogen passivation
        self.danglinbBondsX = set() # testing purposes (get handled through self energy terms)
        self.neighbors = self.mapNeighbors()
        
        
    
    

        
    def check_in_z_direction(self, atom : Atom):
        return not (atom.z >= self.Nz or atom.z < 0)
    def check_in_x_direction(self, atom : Atom):
        return not (atom.x >= self.Nx / 4 or atom.x < 0) 
    def check_in_y_direction(self, atom : Atom):
        return not (atom.y >= 1 or atom.y < 0) 
    
    def baseCoordinates(self, nx=1, nz = 1):
        "returns nth layer base coordinates" 
        atoms =  list(map(lambda x: x.add((int((nx - 1) // 4), 0, nz - 1)), self.raw_atoms[(nx - 1) % 4]))
        return atoms
    def mapNeighbors(self):
        """This gives the list of neighbors of an atom
        For example say i am working with atom at 0.25,.25,.25. I find the sublattice and then the neighbors.
        Each neighbors corresponds to an atom in the TB Hamiltonian - via periodicity or direct neighbor
        """
        atoms = self.ATOM_POSITIONS
        neighborsMap = {}

        for atom in atoms:
            sublattice = UnitCell._sublattice(atom)
            neighborsMap[atom] = []
            self.danglingBondsZ[atom] = []
            
            for delta in UnitCell.raw_deltas[sublattice]:
                neighbor = atom.add(tuple(delta))
                if not self.check_in_z_direction(neighbor):
                    self.danglingBondsZ[atom].append((neighbor, UnitCell.determine_hybridization(delta)))
                elif not self.check_in_x_direction(neighbor):
                    if self.not_NEGF:
                        # hydrogen passivation on the top too
                        self.danglingBondsZ[atom].append((neighbor, UnitCell.determine_hybridization(delta)))    
                    
                    self.danglinbBondsX.add(neighbor)
                else:
                    l,m,n = UnitCell.directionalCosine(delta)

                    mappedNeighbor = self.XZ_map[(neighbor.x, neighbor.z)]
                
                    neighborsMap[atom].append((mappedNeighbor, delta, l,m,n))
        return neighborsMap
        

