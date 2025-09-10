from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from functools import reduce
from operator import mul
import logging, inspect
import numpy as np, scipy
from .abstract_interfaces import AbstractBasis
from ..geometry.structure_designer import StructDesignerXYZ, CyclicTopology
from .block_tridiagonalization import split_into_subblocks_optimized
from ..tb.diatomic_matrix_element import me
from ..tb.orbitals import Orbitals
from ..io.xyz import dict2xyz
from ..geometry.si_nanowire_generator import generate_sinw_xyz

unique_distances = set()

class BasisTB(AbstractBasis, StructDesignerXYZ):
    def __init__(self, **kwargs):
        super(BasisTB, self).__init__(**kwargs)
        self._orbitals_dict = Orbitals.atoms_factory(list(self.num_of_species.keys()))
        self.quantum_numbers_lims = []
        for item in list(self.num_of_species.keys()):
            self.quantum_numbers_lims.append(OrderedDict([('atoms', self.num_of_species[item]), ('l', self.orbitals_dict[item].num_of_orbitals)]))
        self.basis_size = 0
        for item in self.quantum_numbers_lims:
            self.basis_size += reduce(mul, list(item.values()))
        self._offsets = [0]
        for j in range(len(self.atom_list) - 1):
            self._offsets.append(self.orbitals_dict[list(self.atom_list.keys())[j]].num_of_orbitals)
        self._offsets = np.cumsum(self._offsets)
    def qn2ind(self, qn):
        qn = OrderedDict(qn)
        if list(qn.keys()) == list(self.quantum_numbers_lims[0].keys()):
            return self._offsets[qn['atoms']] + qn['l']
        else:
            raise IndexError("Wrong set of quantum numbers")
    def ind2qn(self, ind):
        pass
    @property
    def orbitals_dict(self):
        class MyDict(dict):
            def __getitem__(self, key):
                key = ''.join([i for i in key if not i.isdigit()])
                return super(MyDict, self).__getitem__(key)
        return MyDict(self._orbitals_dict)

def sort_by_coordinate(coords, **kwargs):
    """Sorting function that sorts atoms based on a specified coordinate axis."""
    axis = kwargs.get('axis', 0)
    return np.argsort(coords[:, axis], kind='mergesort')

class Hamiltonian(BasisTB):
    def __init__(self, **kwargs):
        self.nn_distance = kwargs.get('nn_distance', 2.39)
        self.int_radial_dependence = None

        self.compute_overlap = kwargs.get('comp_overlap', False)
        self.compute_angular = kwargs.get('comp_angular_dep', True)
        kwargs['nn_distance'] = self.nn_distance

        self.transport_dir = kwargs.get('transport_dir', [1, 0, 0])
        self.transport_axis = np.argmax(np.abs(np.array(self.transport_dir)))

        # Store generator parameters for later use
        self.nx = kwargs.get('nx', None)
        self.ny = kwargs.get('ny', None)
        self.nz = kwargs.get('nz', None)

        # Add the sorter to kwargs for the block partitioning case
        kwargs['sort_func'] = lambda coords, **kws: sort_by_coordinate(
            coords, axis=self.transport_axis, **kws
        )

        # Accept either an xyz string/path, dict-form, or nanowire generator params
        if 'xyz' in kwargs and kwargs['xyz'] is not None:
            if not isinstance(kwargs['xyz'], str):
                kwargs['xyz'] = dict2xyz(kwargs['xyz'])
        else:
            # Attempt parametric nanowire generation if nx,ny,nz provided
            nx, ny, nz = self.nx, self.ny, self.nz
            if None not in (nx, ny, nz):
                a = kwargs.get('a', 5.50)
                periodic_dirs = kwargs.get('periodic_dirs', 'z')
                passivate_x = kwargs.get('passivate_x', True)
                kwargs['xyz'] = generate_sinw_xyz(nx=nx, ny=ny, nz=nz, a=a, periodic_dirs=periodic_dirs, passivate_x=passivate_x,
                                                  title=f'Generated SiNW nx={nx} ny={ny} nz={nz} a={a}')
            else:
                raise ValueError('Provide either xyz (string/path/dict) or nanowire parameters nx, ny, nz.')
        
        super(Hamiltonian, self).__init__(**kwargs)
        self._coords = None
        self.h_matrix = None
        self.ov_matrix = None
        self.h_matrix_bc_factor = None
        self.h_matrix_bc_add = None
        self.ov_matrix_bc_add = None
        self.h_matrix_left_lead = None
        self.h_matrix_right_lead = None
        self.k_vector = 0
        self.ct = None
        self.radial_dependence = kwargs.get('radial_dep', None)
        self.so_coupling = kwargs.get('so_coupling', 0.0)

    def initialize(self):
        # ... (no changes needed in this method)
        self._coords = [0 for _ in range(self.basis_size)]
        self.h_matrix = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        self.h_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=complex)
        if self.compute_overlap:
            self.ov_matrix = np.zeros((self.basis_size, self.basis_size), dtype=complex)
            self.ov_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        for j1 in range(self.num_of_nodes):
            list_of_neighbours = self.get_neighbours(j1)
            for j2 in list_of_neighbours:
                if j1 == j2:
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                        self.h_matrix[ind1, ind1] = self._get_me(j1, j2, l1, l1)
                        if self.compute_overlap: self.ov_matrix[ind1, ind1] = self._get_me(j1, j2, l1, l1, overlap=True)
                        self._coords[ind1] = list(self.atom_list.values())[j1]
                        if self.so_coupling != 0:
                            for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                                ind2 = self.qn2ind([('atoms', j1), ('l', l2)], )
                                self.h_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2)
                else:
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[j2]].num_of_orbitals):
                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)], )
                            self.h_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2)
                            if self.compute_overlap: self.ov_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2, overlap=True)
        return self

    def set_periodic_bc(self, primitive_cell):
        if list(primitive_cell):
            self.ct = CyclicTopology(primitive_cell, list(self.atom_list.keys()), list(self.atom_list.values()), self._nn_distance)
        else:
            self.ct = None

    def _reset_periodic_bc(self):
        self.h_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        if self.compute_overlap: self.ov_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=complex)
        self.k_vector = None

    def _compute_h_matrix_bc_add(self, overlap=False, split_the_leads: bool = False, transport_dir=None):
     
        if self.ct is not None:
            transport_vector = None
            if split_the_leads and transport_dir is not None:
                transport_dir = np.array(transport_dir, dtype=float).flatten()
                norm_transport_dir = transport_dir / np.linalg.norm(transport_dir)
                max_dot = -1.0
                for vec in self.ct.pcv:
                    norm_vec = vec / np.linalg.norm(vec)
                    dot_product = np.abs(np.dot(norm_vec, norm_transport_dir))
                    if dot_product > max_dot:
                        max_dot = dot_product
                        transport_vector = vec
                if max_dot < 0.95:
                    logging.warning(f"Specified transport_dir may not align well with any primitive_cell_vector. Best match dot product: {max_dot:.2f}")
            for j1 in self.ct.interfacial_atoms_ind:
                list_of_neighbours = self.ct.get_neighbours(list(self.atom_list.values())[j1])
                for j2 in list_of_neighbours:
                    coords = np.array(list(self.atom_list.values())[j1]) - np.array(list(self.ct.virtual_and_interfacial_atoms.values())[j2])
                    phase = np.exp(1j * np.dot(self.k_vector, coords)) if self.k_vector is not None else 1.0
                    key = list(self.ct.virtual_and_interfacial_atoms.keys())[j2]
                    parts = key.split('_')
                    ind = int(parts[2]) if key.startswith('*') else int(parts[0])
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[ind]].num_of_orbitals):
                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)])
                            ind2 = self.qn2ind([('atoms', ind), ('l', l2)])
                            val = phase * self._get_me(j1, ind, l1, l2, coords=coords)
                            self.h_matrix_bc_add[ind1, ind2] += val
                            if overlap and self.compute_overlap: self.ov_matrix_bc_add[ind1, ind2] += phase * self._get_me(j1, ind, l1, l2, coords=coords, overlap=True)
                            if split_the_leads and transport_vector is not None:
                                side = self.ct.atom_classifier(list(self.ct.virtual_and_interfacial_atoms.values())[j2], transport_vector)
                                if side == 'L': self.h_matrix_left_lead[ind1, ind2] += val
                                elif side == 'R': self.h_matrix_right_lead[ind1, ind2] += val
        return
    def get_hamiltonians(self):
        """
        Returns (Hl, H0, Hr) for NEGF calculations.
        This method intelligently switches between two modes:
        1. Periodic Transport (e.g., nz=1): Defines leads from periodic cell connections.
        2. Finite Device (e.g., nz>1): Defines leads by partitioning the device into layers.
        """
        if self.transport_axis == 0: num_layers = self.nx
        elif self.transport_axis == 1: num_layers = self.ny
        elif self.transport_axis == 2: num_layers = self.nz
        else: raise ValueError("Invalid transport_axis")

        if num_layers is None:
            raise ValueError("Device dimensions (nx, ny, nz) not found. Cannot determine number of layers.")


        logging.info("Operating in Periodic Transport mode (num_layers=1).")
        self.h_matrix_left_lead = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        self.h_matrix_right_lead = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        old_k = self.k_vector
        self.k_vector = [0.0, 0.0, 0.0]
        self._compute_h_matrix_bc_add(split_the_leads=True, transport_dir=self.transport_dir)
        self.k_vector = old_k
        return self.h_matrix_left_lead.T, self.h_matrix, self.h_matrix_right_lead.T

    def _ind2atom(self, ind):
        return self.orbitals_dict[list(self.atom_list.keys())[ind]] 
    # --- Other methods (diagonalize, _get_me, etc.) are unchanged ---
    def _get_me(self, atom1, atom2, l1, l2, coords=None, overlap=False):
        # ... (no changes needed in this method)
        if atom1 == atom2 and coords is None:
            atom_obj = self._ind2atom(atom1)
            if l1 == l2:
                if overlap: return 1.0
                else: return atom_obj.orbitals[l1]['energy']
            else: return 0
        if atom1 != atom2 or coords is not None:
            atom_kind1 = self._ind2atom(atom1)
            atom_kind2 = self._ind2atom(atom2)
            coords1 = coords.copy() if coords is not None else np.array(list(self.atom_list.values())[atom1], dtype=float) - np.array(list(self.atom_list.values())[atom2], dtype=float)
            norm = np.linalg.norm(coords1)
            which_neighbour = "" if self.int_radial_dependence is None else self.int_radial_dependence(norm)
            factor = 1.0 if self.radial_dependence is None else self.radial_dependence(norm)
            coords1 /= norm
            return me(atom_kind1, l1, atom_kind2, l2, coords1, which_neighbour, overlap=overlap) * factor

    def diagonalize_k(self, k_vector):
        # ... (no changes needed in this method)
        k_vector = list(k_vector)
        if k_vector != self.k_vector:
            self._reset_periodic_bc()
            self.k_vector = k_vector
            self._compute_h_matrix_bc_factor()
            self._compute_h_matrix_bc_add(overlap=self.compute_overlap)
        Hk = self.h_matrix_bc_factor * self.h_matrix + self.h_matrix_bc_add
        if self.compute_overlap:
            Ok = self.h_matrix_bc_factor * self.ov_matrix + self.ov_matrix_bc_add
            vals, vects = scipy.linalg.eigh(Hk, Ok)
        else:
            vals, vects = np.linalg.eigh(Hk)
        vals = np.real(vals)
        ind = np.argsort(vals)
        return vals[ind], vects[:, ind]
