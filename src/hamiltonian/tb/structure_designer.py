from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import logging
import numpy as np
import scipy.spatial
from .abstract_interfaces import AbstractStructureDesigner
from .aux_functions import xyz2np, count_species, is_in_coords, print_dict

class StructDesignerXYZ(AbstractStructureDesigner):
    def __init__(self, **kwargs):
        xyz = kwargs.get('xyz', None)
        try:
            with open(xyz, 'r') as read_file:
                reader = read_file.read()
        except IOError:
            reader = xyz
        labels, coords = xyz2np(reader)
        num_lines = reader.count('\n')
        if num_lines > 11:
            logging.info("The xyz-file:\n {}".format('\n'.join(reader.split('\n')[:11])))
            logging.info("                  .                    ")
            logging.info("                  .                    ")
            logging.info("                  .                    ")
            logging.info("There are {} more coordinates".format(str(num_lines-10)))
        else:
            logging.info("The xyz-file:\n {}".format(reader))
        logging.info("---------------------------------\n")
        self._nn_distance = kwargs.get('nn_distance', 0)
        self._num_of_species = count_species(labels)
        self._num_of_nodes = sum(self.num_of_species.values())
        self._atom_list = OrderedDict(list(zip(labels, coords)))
        self._kd_tree = scipy.spatial.cKDTree(np.array(list(self._atom_list.values())), leafsize=1, balanced_tree=True)
        self.left_lead = kwargs.get('left_lead', [])
        self.right_lead = kwargs.get('right_lead', [])
        self.sort_func = kwargs.get('sort_func', None)
        self.reorder = None
        if self.sort_func is not None:
            self._sort(labels, coords)
    def _sort(self, labels, coords):
        coords = np.array(coords)
        h_matrix = np.zeros((coords.shape[0], coords.shape[0]))
        self._nn_distance = 2 * self._nn_distance
        for j in range(len(coords)):
            ans = self.get_neighbours(j)
            h_matrix[j, ans] = 1
        self._nn_distance = self._nn_distance / 2
        indices = self.sort_func(coords=coords, left_lead=self.left_lead, right_lead=self.right_lead, mat=h_matrix)
        self.reorder = indices
        if (isinstance(self.left_lead, list) or isinstance(self.left_lead, np.ndarray)) and len(self.left_lead) > 0:
            self.left_lead = np.squeeze(np.concatenate([np.where(indices == item) for item in self.left_lead]))
        if (isinstance(self.right_lead, list) or isinstance(self.right_lead, np.ndarray)) and len(self.right_lead) > 0:
            self.right_lead = np.squeeze(np.concatenate([np.where(indices == item) for item in self.right_lead]))
        coords = coords[indices]
        labels = [labels[i] for i in indices]
        self._atom_list = OrderedDict(list(zip(labels, coords)))
        self._kd_tree = scipy.spatial.cKDTree(np.array(list(self._atom_list.values())), leafsize=1, balanced_tree=True)
    def add_leads(self, left_lead, right_lead):
        self.left_lead = left_lead; self.right_lead = right_lead
    @property
    def atom_list(self):
        return self._atom_list
    @property
    def num_of_nodes(self):
        return self._num_of_nodes
    @property
    def num_of_species(self):
        return self._num_of_species
    def get_neighbours(self, query):
        ans = self._get_neighbours(query)
        if ans[0][0] == np.inf:
            ans1 = []
        else:
            ans1 = [ans[1][0]]
        for item in zip(ans[0], ans[1]):
            if self._nn_distance * 0.1 < item[0] < self._nn_distance:
                ans1.append(item[1])
        return ans1

class CyclicTopology(AbstractStructureDesigner):
    def __init__(self, primitive_cell_vectors, labels, coords, nn_distance):
        self._nn_distance = nn_distance
        self.pcv = primitive_cell_vectors
        self.sizes = [np.linalg.norm(item) for item in self.pcv]
        self.interfacial_atoms_ind = []
        self.virtual_and_interfacial_atoms = OrderedDict()
        self.shift = np.zeros(3)
        self._generate_atom_list(labels, coords)
        self._kd_tree = scipy.spatial.cKDTree(list(self.virtual_and_interfacial_atoms.values()), leafsize=100, balanced_tree=True)
        logging.info("Primitive_cell_vectors: \n {} \n".format(primitive_cell_vectors))
        logging.debug("Virtual and interfacial atoms: \n {} ".format(print_dict(self.virtual_and_interfacial_atoms)))
        logging.info("---------------------------------\n")
    @property
    def atom_list(self):
        return self.virtual_and_interfacial_atoms
    def _generate_atom_list(self, labels, coords):
        distances1 = np.empty((len(coords), len(self.pcv)), dtype=float)
        distances2 = np.empty((len(coords), len(self.pcv)), dtype=float)
        for j1, coord in enumerate(coords):
            for j2, basis_vec in enumerate(self.pcv):
                distances1[j1, j2] = np.inner(coord - self.shift, basis_vec) / self.sizes[j2]
                distances2[j1, j2] = np.inner(coord - self.shift - basis_vec, basis_vec) / self.sizes[j2]
        for item in distances1.T:
            self.shift += coords[np.argmin(item)]
        for j1, coord in enumerate(coords):
            for j2, basis_vec in enumerate(self.pcv):
                distances1[j1, j2] = np.inner(coord - self.shift, basis_vec) / self.sizes[j2]
                distances2[j1, j2] = np.inner(coord - self.shift - basis_vec, basis_vec) / self.sizes[j2]
        distances1 = np.abs(distances1 - np.min(distances1)) < self._nn_distance * 0.25
        distances2 = np.abs(np.abs(distances2) - np.min(np.abs(distances2))) < self._nn_distance * 0.25
        count = 0
        for j, item in enumerate(coords):
            if any(distances1[j]):
                self.virtual_and_interfacial_atoms.update({str(j) + "_" + labels[j]: item})
                self.interfacial_atoms_ind.append(j)
                for surf in np.where(distances1[j])[0]:
                    count = self._translate_atom_1st_order(item, np.array(self.pcv[surf]), "_" + str(j) + "_" + labels[j], coords, count)
                    count = self._translate_atom_2d_order(item, np.array(self.pcv[surf]), "_" + str(j) + "_" + labels[j], coords, count)
            if any(distances2[j]):
                self.virtual_and_interfacial_atoms.update({str(j) + "_" + labels[j]: item})
                self.interfacial_atoms_ind.append(j)
                for surf in np.where(distances2[j])[0]:
                    count = self._translate_atom_1st_order(item, -1 * np.array(self.pcv[surf]), "_" + str(j) + "_" + labels[j], coords, count)
                    count = self._translate_atom_2d_order(item, -1 * np.array(self.pcv[surf]), "_" + str(j) + "_" + labels[j], coords, count)
        self.interfacial_atoms_ind = list(set(self.interfacial_atoms_ind))
    def _translate_atom_1st_order(self, atom_coords, cell_vector, label, penalty_coords, count):
        try_coords = atom_coords + cell_vector
        if not is_in_coords(try_coords, penalty_coords) and not is_in_coords(try_coords, np.array(list(self.virtual_and_interfacial_atoms.values()))):
            self.virtual_and_interfacial_atoms.update({"*_" + str(count) + label: try_coords}); count += 1
        return count
    def _translate_atom_2d_order(self, atom_coords, cell_vector, label, penalty_coords, count):
        for vec in self.pcv:
            try_coords = atom_coords + cell_vector + vec
            if not is_in_coords(try_coords, penalty_coords) and not is_in_coords(try_coords, np.array(list(self.virtual_and_interfacial_atoms.values()))):
                self.virtual_and_interfacial_atoms.update({"**_" + str(count) + label: try_coords}); count += 1
            try_coords = atom_coords + cell_vector - vec
            if not is_in_coords(try_coords, penalty_coords) and not is_in_coords(try_coords, np.array(list(self.virtual_and_interfacial_atoms.values()))):
                self.virtual_and_interfacial_atoms.update({"**_" + str(count) + label: try_coords}); count += 1
        return count
    def get_neighbours(self, query):
        ans = self._get_neighbours(query); ans1 = []
        for item in zip(ans[0], ans[1]):
            if self._nn_distance * 0.1 < item[0] < self._nn_distance and list(self.virtual_and_interfacial_atoms.keys())[item[1]].startswith("*"):
                ans1.append(item[1])
        return ans1
    def atom_classifier(self, coords, leads):
        distance_to_surface1 = np.inner(coords - self.shift, leads) / np.linalg.norm(leads)
        distance_to_surface2 = np.inner(coords - self.shift - leads, leads) / np.linalg.norm(leads)
        flag = None
        if distance_to_surface1 < 0: flag = 'L'
        if distance_to_surface2 >= 0: flag = 'R'
        return flag
