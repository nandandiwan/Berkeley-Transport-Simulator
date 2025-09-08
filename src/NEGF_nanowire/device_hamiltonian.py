import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from NEGF_device_generation import Atom, UnitCell ,PeriodicUnitCell 
from src.tight_binding import tight_binding_params as TBP
import numpy as np
from device import Device
import scipy.sparse as sp

class Hamiltonian:
    def __init__(self, device : Device):
        self.device = device
        self.unitCell = device.unitCell
        self.Nx = (int) (self.device.unitX)
        self.Ny = (int) (self.device.unitY)
        self.Nz = (int) (self.device.unitZ)
        
        self.U_orb_to_sp3 = 0.5*np.array([[1, 1, 1, 1],
                             [1, 1,-1,-1],
                             [1,-1, 1,-1],
                             [1,-1,-1, 1]])
        Es = TBP.E['s']
        Ep = TBP.E['px'] 
        
        H_orb = np.diag([Es, Ep, Ep, Ep])

        # change-of-basis matrices
        U_orb_to_sp3 = 0.5*np.array([[1, 1, 1, 1],
                                    [1, 1,-1,-1],
                                    [1,-1, 1,-1],
                                    [1,-1,-1, 1]])
        self.U_sp3_to_orb = U_orb_to_sp3.T  

        a = (Es + 3*Ep)/4.0
        b = (Es -   Ep)/4.0
        H_sp3_explicit = np.full((4,4), b)
        np.fill_diagonal(H_sp3_explicit, a)
        self.H_sp3_explicit = H_sp3_explicit
        
        # self energy
        """Which layer is at the end?"""
        self.layer_left_lead = 0 # 0,1,2,3 
        self.layer_right_lead = 0
        

    
    def get_H00_H01_H10(self, side="left", sparse=False):
        """
        Get H00, H01, and H10 matrices for both leads with proper symmetry.
        For symmetric DOS at zero bias, both leads should have identical structure.
        """
        if side == 'left':
            orientation = (0, 1, 2, 3)  # Always use same orientation
        elif side == "right":
            orientation = (3, 2, 1, 0)
        
        # Create 8-layer system: two adjacent 4-layer supercells
        # This allows us to extract H01 as coupling between supercells
        lead_unit_cell = PeriodicUnitCell(
            channel_length=0.5431e-9,  # Length of ONE repeating unit
            channel_width=self.device.channel_width,
            channel_thickness=self.device.channel_thickness,
            orientation=orientation
        )

        # 2. You will need a new Hamiltonian function that builds H00 and H01
        #    from the 'neighbors' and 'periodic_neighbors' dictionaries.
        H00, H01 = self.create_lead_hamiltonian(unitCell=lead_unit_cell, sparse=sparse)
        
        # H10 is the Hermitian conjugate of H01
        H10 = H01.conj().T
        

        newUnitCell = UnitCell(channel_length=2 * 0.5431e-9,channel_width=self.device.channel_width,channel_thickness=self.device.channel_thickness, orientation=orientation)
        HT = self.create_sparse_channel_hamlitonian(unitCell=newUnitCell, blocks=False)
        
        # Calculate proper block size: 8 atoms per block * total blocks perpendicular ot transport
        block_size = self.Nz * self.Ny * 4 * 10 * 2
        
        
        # H01: coupling from current unit cell to next unit cell  
        H01 = HT[:block_size, block_size:2*block_size]
        
        # H10: coupling from next unit cell to current unit cell (should be H01†)
        H10 = HT[block_size:2*block_size, :block_size]
        
        # Verify Hermitian relationship: H10 should equal H01†
        if not sparse:
            H00_dense = H00.toarray() if hasattr(H00, 'toarray') else H00
            H01_dense = H01.toarray() if hasattr(H01, 'toarray') else H01
            H10_dense = H10.toarray() if hasattr(H10, 'toarray') else H10
            
            # Check if H10 ≈ H01†
            diff = np.max(np.abs(H10_dense - H01_dense.conj().T))
            if diff > 1e-12:
                print(f"Warning: H10 != H01† for {side} lead, difference: {diff:.2e}")
            
            return H00_dense, H01_dense, H10_dense
            
        return H00, H01, H10
    
    
    def getMatrixSize(self):
        return len(self.unitCell.ATOM_POSITIONS) * 10
    
    def create_sparse_channel_hamlitonian(self, unitCell : UnitCell = None, blocks = True):
        if unitCell is None:
            unitCell = self.unitCell

        unitNeighbors = unitCell.neighbors
        danglingBonds = unitCell.danglingBonds
        numSilicon = len(unitNeighbors.keys())

        orbitals = ['s', 'px', 'py', 'pz', 'dxy','dyz','dzx','dx2y2','dz2', 's*']
        numOrbitals = len(orbitals)
        size = numSilicon * numOrbitals 
        A = np.zeros((size, size), dtype=complex)    
        
        atomToIndex = {}
        indexToAtom = {}
        for atom_index,atom in enumerate(unitNeighbors):
            atomToIndex[atom] = atom_index
            indexToAtom[atom_index] = atom
        
        potentialPerAtom = self.potential_correction()

        numSilicon = len(unitNeighbors.keys())

        orbitals = ['s', 'px', 'py', 'pz', 'dxy','dyz','dzx','dx2y2','dz2', 's*']
        numOrbitals = len(orbitals)
        size = numSilicon * numOrbitals 
        
        atomToIndex = {}
        indexToAtom = {}
        for atom_index,atom in enumerate(unitNeighbors):
            atomToIndex[atom] = atom_index
            indexToAtom[atom_index] = atom

        rows, cols, data = [], [], []                   # <-- sparse triplets

        # helper -----------------------------------------------------------
        def add(i, j, val):
            if val != 0.0:
                rows.append(i); cols.append(j); data.append(val)
                if i != j:                              # Hermitian
                    rows.append(j); cols.append(i); data.append(np.conj(val))
        # ------------------------------------------------------------------

        # ---------- on‑site (Si) ----------
        
        for atom_idx, atom in indexToAtom.items():

            hybridizationMatrix = self.H_sp3_explicit.copy() 
            danglingBondsList = danglingBonds[atom]
            for danglingBondAtom, position in danglingBondsList:
                hybridizationMatrix[position,position] += TBP.E['sp3']            
            
            # if there are no dangling bonds this returns the standard diag matrix with onsite energies 
        
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T 
            base = atom_idx * numOrbitals
    
            for i in range(4):
                for j in range(i ,4):
                    add(base + i, base + j, onsiteMatrix[i,j] + potentialPerAtom[atom_idx] * (i == j))
        
            for p in range(4, 9):                       # five d’s
                add(base + p, base + p, TBP.E['dxy'] + potentialPerAtom[atom_idx])
            add(base + 9, base + 9, TBP.E['s*']+ potentialPerAtom[atom_idx])
        
    
        for atom_index, atom in indexToAtom.items():
            base_i = atom_index * numOrbitals
            for atom2, delta, l,m,n in unitNeighbors[atom]:
                j = atomToIndex[atom2]
                if j < atom_index:             
                    continue                   

                phase = np.exp(2j*np.pi*(0*delta[1]))

                for o1, orb1 in enumerate(orbitals):
                    for o2, orb2 in enumerate(orbitals):
                        hop = TBP.SK[(orb1, orb2)](l, m, n, TBP.V) * phase
                        add(base_i + o1, j*numOrbitals + o2, hop)

                
    
        H = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()
        if blocks == False:
            return H
        total_size = H.shape[0]
        block_size = self.Nz * 20  *self.Ny # 2 atoms per single unit z layer 
        num_blocks = (int) (total_size // block_size)
        diagonal_blocks = [None] * num_blocks
        off_diagonal_blocks = [None] * (num_blocks - 1)
        
        for block in range(0, num_blocks - 1):
            s = block * block_size
            m = (block + 1) * block_size
            e = (block + 2) * block_size
            diagonal_blocks[block] = H[s : m, s : m] 
            off_diagonal_blocks[block] = H[s : m, m : e]
        diagonal_blocks[-1] = H[-block_size:, -block_size:]     

        return diagonal_blocks, off_diagonal_blocks
    
    def create_lead_hamiltonian(self, unitCell: PeriodicUnitCell, sparse=True):
        """
        Constructs the on-site (H00) and coupling (H01) Hamiltonian matrices
        for a periodic lead unit cell.

        Args:
            unitCell (PeriodicUnitCell): The lead's unit cell with periodic boundary conditions.
            sparse (bool): If True, returns sparse matrices.

        Returns:
            (scipy.sparse.csc_matrix, scipy.sparse.csc_matrix): A tuple containing H00 and H01.
        """
        unit_neighbors = unitCell.neighbors
        periodic_neighbors = unitCell.periodic_neighbors
        dangling_bonds = unitCell.danglingBonds
        
        num_silicon = len(unit_neighbors.keys())
        orbitals = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2y2', 'dz2', 's*']
        num_orbitals = len(orbitals)
        size = num_silicon * num_orbitals
        
        atom_to_index = {atom: i for i, atom in enumerate(unit_neighbors.keys())}

        rows_00, cols_00, data_00 = [], [], [] 
        rows_01, cols_01, data_01 = [], [], [] 

        def add_00(r, c, val):
            """Helper to add entries to H00, enforcing Hermiticity."""
            if val != 0.0:
                rows_00.append(r)
                cols_00.append(c)
                data_00.append(val)
                if r != c:  
                    rows_00.append(c)
                    cols_00.append(r)
                    data_00.append(np.conj(val))

        def add_01(r, c, val):
            """Helper to add entries to H01. H01 is not Hermitian."""
            if val != 0.0:
                rows_01.append(r)
                cols_01.append(c)
                data_01.append(val)


        potential_per_atom = self.potential_correction() 

        for atom, atom_idx in atom_to_index.items():
            hybridization_matrix = self.H_sp3_explicit.copy()
            for _, position in dangling_bonds[atom]:
                hybridization_matrix[position, position] += TBP.E['sp3']
            onsite_matrix = self.U_orb_to_sp3 @ hybridization_matrix @ self.U_orb_to_sp3.T
            
            base = atom_idx * num_orbitals
            potential = potential_per_atom[atom_idx]
            
            # Add sp3 block to H00
            for i in range(4):
                for j in range(i, 4):
                    add_00(base + i, base + j, onsite_matrix[i, j] + potential * (i == j))

            for p in range(4, 9):
                add_00(base + p, base + p, TBP.E['dxy'] + potential)
            add_00(base + 9, base + 9, TBP.E['s*'] + potential)

        for atom_i, i in atom_to_index.items():
            base_i = i * num_orbitals

            for atom_j_obj, delta, l, m, n in unit_neighbors[atom_i]:
                j = atom_to_index[atom_j_obj]
                if j < i: continue 

                for o1, orb1 in enumerate(orbitals):
                    for o2, orb2 in enumerate(orbitals):
                        hop = TBP.SK[(orb1, orb2)](l, m, n, TBP.V)
                        add_00(base_i + o1, j * num_orbitals + o2, hop)


            for atom_j_obj, delta, l, m, n in periodic_neighbors[atom_i]:

                atom_in_base_cell = Atom(atom_j_obj.x % unitCell.Nx, atom_j_obj.y, atom_j_obj.z)
                j = atom_to_index[atom_in_base_cell]

                for o1, orb1 in enumerate(orbitals):
                    for o2, orb2 in enumerate(orbitals):
                        hop = TBP.SK[(orb1, orb2)](l, m, n, TBP.V)
                        add_01(base_i + o1, j * num_orbitals + o2, hop)

        H00 = sp.coo_matrix((data_00, (rows_00, cols_00)), shape=(size, size))
        H01 = sp.coo_matrix((data_01, (rows_01, cols_01)), shape=(size, size))

        if sparse:
            return H00.tocsc(), H01.tocsc()
        else:
            return H00.toarray(), H01.toarray()
    def potential_correction(self):
        # For testing bandgap opening, return zeros to isolate dangling bond effects
        return [0.0] * len(self.unitCell.ATOM_POSITIONS) * 10
        
        # Or fix the original method:
        # voltage = self.device.Ec
        # L, H = voltage.shape
        # potential_diag = [0.0] * len(self.unitCell.ATOM_POSITIONS)  # Initialize with zeros
        
        # for atom_index, atom in enumerate(self.unitCell.neighbors):
        #     px, pz = atom.x, atom.z
        #     # Ensure proper grid mapping
        #     gx = min(max(int(px / self.Nx * L), 0), L-1)
        #     gz = min(max(int(pz / self.Nz * H), 0), H-1)
        #     potential_diag[atom_index] = voltage[gx, gz]
        
        # return potential_diag    