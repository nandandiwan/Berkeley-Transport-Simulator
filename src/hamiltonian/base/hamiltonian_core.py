from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from functools import reduce
from operator import mul
import logging, inspect
import numpy as np, scipy
from .abstract_interfaces import AbstractBasis
from ..geometry.structure_designer import StructDesignerXYZ, CyclicTopology
from ..tb.diatomic_matrix_element import me
from ..tb.orbitals import Orbitals
from ..io.xyz import dict2xyz

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

class Hamiltonian(BasisTB):
    def __init__(self, **kwargs):
        nn_distance = kwargs.get('nn_distance', 2.39)
        self.int_radial_dependence = None
        nn_distance = self._set_nn_distances(nn_distance)
        self.compute_overlap = kwargs.get('comp_overlap', False)
        self.compute_angular = kwargs.get('comp_angular_dep', True)
        kwargs['nn_distance'] = nn_distance
        if not isinstance(kwargs['xyz'], str):
            kwargs['xyz'] = dict2xyz(kwargs['xyz'])
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
        self.radial_dependence = None
        self.so_coupling = kwargs.get('so_coupling', 0.0)
        radial_dep = kwargs.get('radial_dep', None)
        self.radial_dependence = radial_dep
    def initialize(self):
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
                        if self.compute_overlap:
                            self.ov_matrix[ind1, ind1] = self._get_me(j1, j2, l1, l1, overlap=True)
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
                            if self.compute_overlap:
                                self.ov_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2, overlap=True)
        return self
    def set_periodic_bc(self, primitive_cell):
        if list(primitive_cell):
            self.ct = CyclicTopology(primitive_cell, list(self.atom_list.keys()), list(self.atom_list.values()), self._nn_distance)
        else:
            self.ct = None
    def diagonalize(self):
        if self.compute_overlap:
            vals, vects = scipy.linalg.eigh(self.h_matrix, self.ov_matrix)
        else:
            vals, vects = np.linalg.eigh(self.h_matrix)
        vals = np.real(vals)
        ind = np.argsort(vals)
        return vals[ind], vects[:, ind]
    def set_periodic_bc(self, primitive_cell):
        if list(primitive_cell):
            self.ct = CyclicTopology(primitive_cell, list(self.atom_list.keys()), list(self.atom_list.values()), self._nn_distance)
        else:
            self.ct = None
    def _reset_periodic_bc(self):
        self.h_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        if self.compute_overlap:
            self.ov_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=complex)
        self.k_vector = None
    def _compute_h_matrix_bc_factor(self):
        for j1 in range(self.num_of_nodes):
            list_of_neighbours = self.get_neighbours(j1)
            for j2 in list_of_neighbours:
                if j1 != j2:
                    coords = np.array(list(self.atom_list.values())[j1], dtype=float) - np.array(list(self.atom_list.values())[j2], dtype=float)
                    phase = np.exp(1j * np.dot(self.k_vector, coords))
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[j2]].num_of_orbitals):
                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)])
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)])
                            self.h_matrix_bc_factor[ind1, ind2] = phase
    def _compute_h_matrix_bc_add(self, overlap=False):
        if self.ct is None: return
        for j1 in self.ct.interfacial_atoms_ind:
            list_of_neighbours = self.ct.get_neighbours(list(self.atom_list.values())[j1])
            for j2 in list_of_neighbours:
                coords = np.array(list(self.atom_list.values())[j1]) - np.array(list(self.ct.virtual_and_interfacial_atoms.values())[j2])
                phase = np.exp(1j * np.dot(self.k_vector, coords))
                key = list(self.ct.virtual_and_interfacial_atoms.keys())[j2]
                parts = key.split('_')
                # Key patterns:
                #   Real/interfacial: "<j>_<Label>"  -> index at parts[0]
                #   Virtual 1st order: "*_<count>_<j>_<Label>" -> index at parts[2]
                #   Virtual 2nd order: "**_<count>_<j>_<Label>" -> index at parts[2]
                if key.startswith('*'):
                    if len(parts) < 3:
                        raise ValueError(f"Unexpected virtual atom key pattern: {key}")
                    ind = int(parts[2])
                else:
                    ind = int(parts[0])
                for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                    for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[ind]].num_of_orbitals):
                        ind1 = self.qn2ind([('atoms', j1), ('l', l1)])
                        ind2 = self.qn2ind([('atoms', ind), ('l', l2)])
                        self.h_matrix_bc_add[ind1, ind2] += phase * self._get_me(j1, ind, l1, l2, coords=coords)
                        if overlap and self.compute_overlap:
                            self.ov_matrix_bc_add[ind1, ind2] += phase * self._get_me(j1, ind, l1, l2, coords=coords, overlap=True)
    def diagonalize_k(self, k_vector):
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
    def diagonalize_periodic_bc(self, k_vector):
        """Alias matching nanonet API for periodic band structure diagonalization."""
        return self.diagonalize_k(k_vector)
    def _set_nn_distances(self, nn_dist):
        if nn_dist is not None:
            if isinstance(nn_dist, list):
                nn_dist.sort()
                self._nn_distance = nn_dist[-1]
                def int_radial_dep(coords):
                    norm_of_coords = np.linalg.norm(coords)
                    ans = sum([norm_of_coords > item for item in nn_dist]) + 1
                    if norm_of_coords > nn_dist[-1]:
                        return 100
                    else:
                        return ans
                self.int_radial_dependence = int_radial_dep
            else:
                self._nn_distance = nn_dist
        return self._nn_distance
    def _ind2atom(self, ind):
        return self.orbitals_dict[list(self.atom_list.keys())[ind]]
    def _get_me(self, atom1, atom2, l1, l2, coords=None, overlap=False):
        if atom1 == atom2 and coords is None:
            atom_obj = self._ind2atom(atom1)
            if l1 == l2:
                if overlap:
                    return 1.0
                else:
                    return atom_obj.orbitals[l1]['energy']
            else:
                return 0
        if atom1 != atom2 or coords is not None:
            atom_kind1 = self._ind2atom(atom1)
            atom_kind2 = self._ind2atom(atom2)
            if coords is None:
                coords1 = np.array(list(self.atom_list.values())[atom1], dtype=float) - np.array(list(self.atom_list.values())[atom2], dtype=float)
            else:
                coords1 = coords.copy()
            norm = np.linalg.norm(coords1)
            if self.int_radial_dependence is None:
                which_neighbour = ""
            else:
                which_neighbour = self.int_radial_dependence(norm)
            if self.radial_dependence is None:
                factor = 1.0
            else:
                factor = self.radial_dependence(norm)
            coords1 /= norm
            return me(atom_kind1, l1, atom_kind2, l2, coords1, which_neighbour, overlap=overlap) * factor
