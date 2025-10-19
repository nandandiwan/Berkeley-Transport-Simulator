from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from functools import reduce
from operator import mul
import logging, inspect
import numpy as np, scipy
import scipy.sparse as sp
from .abstract_interfaces import AbstractBasis
from ..geometry.structure_designer import StructDesignerXYZ, CyclicTopology
from .block_tridiagonalization import split_into_subblocks_optimized
from ..tb.diatomic_matrix_element import me
from ..tb.orbitals import Orbitals
from ..io.xyz import dict2xyz
from ..geometry.si_nanowire_generator import generate_sinw_xyz
from ..geometry.simple_structure_generator import generate_1d_wire_xyz

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

def sort_by_coordinate(coords, left_lead=None, right_lead=None, mat=None, axis=0, **kwargs):
    """Stable sort by a coordinate axis compatible with StructDesignerXYZ.

    Parameters
    ----------
    coords : np.ndarray (N x 3)
        Cartesian coordinates of atoms.
    left_lead, right_lead : sequence[int] | None
        Provided for API compatibility; not used by the default sorter.
    mat : np.ndarray (N x N) | None
        Connectivity/adjacency matrix; not used by the default sorter.
    axis : int, optional (default: 0)
        Axis index to sort along: `0 -> x`, `1 -> y`, `2 -> z`.

    Returns
    -------
    np.ndarray (N,)
        Indices that sort `coords` stably along the requested axis.
    """
    return np.argsort(np.asarray(coords)[:, axis], kind='mergesort')

class Hamiltonian(BasisTB):
    def __init__(self, **kwargs):
        self.nn_distance = kwargs.get('nn_distance', 2.39)
        self.periodic_dirs = kwargs.get('periodic_dirs')
        self.int_radial_dependence = None

        self.compute_overlap = kwargs.get('comp_overlap', False)
        self.compute_angular = kwargs.get('comp_angular_dep', True)
        kwargs['nn_distance'] = self.nn_distance

        self.transport_dir = tuple(kwargs.get('transport_dir', [1, 0, 0]))
        # Robust transport axis detection: accept unit basis tuples or any vector.
        if self.transport_dir in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            self.transport_axis = (0, 1, 2)[((1,0,0), (0,1,0), (0,0,1)).index(self.transport_dir)]
        else:
            self.transport_axis = int(np.argmax(np.abs(np.array(self.transport_dir, dtype=float))))
        # Store generator parameters for later use
        self.nx = kwargs.get('nx', None)
        self.ny = kwargs.get('ny', None)
        self.nz = kwargs.get('nz', None)

        # Sorting: allow a user-provided sort function; otherwise default to axis-based.
        # Default: sort along TRANSPORT axis. You can override with:
        # - sort_axis=int (0/1/2)
        # - sort_axis in {'x','y','z','transport','periodic'}
        # If 'periodic' is chosen and a single periodic axis is specified, that axis is used; otherwise falls back to transport.
        periodic_dirs_cfg = kwargs.get('periodic_dirs', None)
        self.periodic_dirs = periodic_dirs_cfg
        sort_axis_override = kwargs.get('sort_axis', None)
        axes_map = {'x': 0, 'y': 1, 'z': 2}
        default_axis = self.transport_axis
        if sort_axis_override is not None:
            if isinstance(sort_axis_override, str):
                key = sort_axis_override.lower()
                if key in axes_map:
                    default_axis = axes_map[key]
                elif key == 'transport':
                    default_axis = self.transport_axis
                elif key == 'periodic' and isinstance(periodic_dirs_cfg, str):
                    chars = [axes_map[c] for c in periodic_dirs_cfg.lower() if c in axes_map]
                    if len(chars) == 1:
                        default_axis = chars[0]
            else:
                default_axis = int(sort_axis_override)
        sort_func = kwargs.get('sort_func', None)
        if sort_func is None:
            sort_func = lambda coords, **kws: sort_by_coordinate(
                coords, axis=default_axis, **kws
            )
        elif not callable(sort_func):
            raise TypeError("sort_func must be callable with signature (coords: ndarray, **kwargs) -> index array")
        kwargs['sort_func'] = sort_func

        # Accept either an xyz string/path, dict-form, or structure generator params
        if 'xyz' in kwargs and kwargs['xyz'] is not None:
            if not isinstance(kwargs['xyz'], str):
                kwargs['xyz'] = dict2xyz(kwargs['xyz'])
        else:
            # Attempt generator based on 'structure' flag
            structure = str(kwargs.get('structure', 'sinw')).lower()
            self.structure = structure
            if structure in ('sinw', 'si-nw', 'silicon-nanowire', 'silicon'):
                
                # Parametric silicon nanowire generation requires nx,ny,nz
                nx, ny, nz = self.nx, self.ny, self.nz
                if None in (nx, ny, nz):
                    raise ValueError("For structure='sinw', provide nx, ny, nz.")
                a = kwargs.get('a', 5.50)
                periodic_dirs = kwargs.get('periodic_dirs', 'z')
                passivate_x = kwargs.get('passivate_x', True)
                kwargs['xyz'] = generate_sinw_xyz(
                    nx=nx, ny=ny, nz=nz, a=a, periodic_dirs=periodic_dirs, passivate_x=passivate_x,
                    title=f'Generated SiNW nx={nx} ny={ny} nz={nz} a={a}'
                )
            elif structure in ('1d', '1d-wire', 'wire-1d', 'line'):
                # Simple 1D wire only needs nx; spacing from 'a'
                nx = self.nx or kwargs.get('nx', None)
                a = kwargs.get('a', 1.0)
                axis = kwargs.get('wire_axis', {0:'x',1:'y',2:'z'}.get(self.transport_axis, 'x'))
                symbol = kwargs.get('wire_symbol', 'base')
                kwargs['xyz'] = generate_1d_wire_xyz(nx=nx, a=a, axis=axis, symbol=symbol,
                                                     title=f'Generated 1D wire nx={nx} a={a} axis={axis}')
                # For 1D wire, also set a sensible default nearest-neighbour distance
                if 'nn_distance' not in kwargs or kwargs['nn_distance'] is None:
                    kwargs['nn_distance'] = a * 1.01
            else:
                raise ValueError("Unknown structure type: {}. Supported: 'sinw', '1d-wire'".format(structure))
        
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
        # Sparse build options
        self.sparse_build = kwargs.get('sparse_build', True)
        self.return_sparse = kwargs.get('return_sparse', False)
        # Optionally build matrices immediately
        if kwargs.get('auto_initialize', False):
            self.initialize()

    def initialize(self):
        # Build matrices using a sparse assembler for speed, with optional dense output for compatibility.
        self._coords = [0 for _ in range(self.basis_size)]

        use_sparse_build: bool = True if 'sparse_build' not in self.__dict__ else True
        # Allow override via kwargs passed at construction time
        # Fallback to kwargs in instance dict if set by __init__ call
        use_sparse_build = bool(getattr(self, 'sparse_build', True)) if hasattr(self, 'sparse_build') else True
        return_sparse = bool(getattr(self, 'return_sparse', False)) if hasattr(self, 'return_sparse') else False

        if use_sparse_build:
            hi, hj, hv = [], [], []
            if self.compute_overlap:
                oi, oj, ov = [], [], []
            # Diagonal and off-diagonal assembly
            for j1 in range(self.num_of_nodes):
                list_of_neighbours = self.get_neighbours(j1)
                for j2 in list_of_neighbours:
                    if j1 == j2:
                        norb = self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals
                        for l1 in range(norb):
                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)])
                            val = self._get_me(j1, j2, l1, l1)
                            hi.append(ind1); hj.append(ind1); hv.append(val)
                            if self.compute_overlap:
                                valS = self._get_me(j1, j2, l1, l1, overlap=True)
                                oi.append(ind1); oj.append(ind1); ov.append(valS)
                            self._coords[ind1] = list(self.atom_list.values())[j1]
                            if self.so_coupling != 0:
                                for l2 in range(norb):
                                    ind2 = self.qn2ind([('atoms', j1), ('l', l2)])
                                    v = self._get_me(j1, j2, l1, l2)
                                    if v != 0:
                                        hi.append(ind1); hj.append(ind2); hv.append(v)
                    else:
                        norb1 = self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals
                        norb2 = self.orbitals_dict[list(self.atom_list.keys())[j2]].num_of_orbitals
                        for l1 in range(norb1):
                            for l2 in range(norb2):
                                ind1 = self.qn2ind([('atoms', j1), ('l', l1)])
                                ind2 = self.qn2ind([('atoms', j2), ('l', l2)])
                                v = self._get_me(j1, j2, l1, l2)
                                if v != 0:
                                    hi.append(ind1); hj.append(ind2); hv.append(v)
                                if self.compute_overlap:
                                    vS = self._get_me(j1, j2, l1, l2, overlap=True)
                                    if vS != 0:
                                        oi.append(ind1); oj.append(ind2); ov.append(vS)
            H = sp.coo_matrix((hv, (hi, hj)), shape=(self.basis_size, self.basis_size), dtype=complex).tocsr()
            self.h_matrix = H if return_sparse else H.toarray()
            # Boundary condition arrays remain dense for now
            self.h_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=complex)
            self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=complex)
            if self.compute_overlap:
                S = sp.coo_matrix((ov, (oi, oj)), shape=(self.basis_size, self.basis_size), dtype=complex).tocsr()
                self.ov_matrix = S if return_sparse else S.toarray()
                self.ov_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        else:
            # Fallback old dense builder
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

        if self.ct is None or not hasattr(self.ct, 'pcv'):
            mode = "open"
        else:
            mode = "periodic"
        
        if mode == "periodic":
            tdir = np.array(self.transport_dir, dtype=float).flatten()
            if np.linalg.norm(tdir) == 0:
                raise ValueError("transport_dir must be non-zero.")
            tdir /= np.linalg.norm(tdir)
            dots = [abs(np.dot(tdir, vec/np.linalg.norm(vec))) for vec in self.ct.pcv]
            max_dot = max(dots) if len(dots) else 0.0
            if max_dot < 0.95:
                raise ValueError(
                    "In periodic-transport mode, transport_dir must align with a primitive cell vector. "
                    f"Max alignment found: {max_dot:.3f}. Either (1) set periodic_dirs to include the transport axis, "
                    "or (2) construct a finite device along the transport axis (num_layers>1) and use layer partitioning."
                )
            self.h_matrix_left_lead = np.zeros((self.basis_size, self.basis_size), dtype=complex)
            self.h_matrix_right_lead = np.zeros((self.basis_size, self.basis_size), dtype=complex)
            old_k = self.k_vector
            self.k_vector = [0.0, 0.0, 0.0]
            self._compute_h_matrix_bc_add(split_the_leads=True, transport_dir=self.transport_dir)
            self.k_vector = old_k
            return self.h_matrix_left_lead.T, self.h_matrix, self.h_matrix_right_lead.T
        else:
            return self.determine_leads()
    def _ind2atom(self, ind):
        return self.orbitals_dict[list(self.atom_list.keys())[ind]]
    def determine_leads(self, tol: float = 1e-3, choose: str = 'center', redo=False):
        if self.h_matrix is None:
            raise ValueError("initialize")
        if redo == False and getattr(self, "hRC", None) != None:
            return self.h_matrix, self.hL0, self.hLC, self.hR0, self.hRC, self.h_periodic
        
        
        ham_d = self.h_matrix.toarray() if hasattr(self.h_matrix, 'toarray') else self.h_matrix
        old_size = ham_d.shape[0]
        # Build a 2-cell finite chain along transport to extract a single principal-layer and its coupling
        # Remove transport axis from periodic_dirs so we get an open chain along x
        pdirs = '' if self.periodic_dirs is None else str(self.periodic_dirs)
        axes_map = {'x':0,'y':1,'z':2}
        inv_axes = {0:'x',1:'y',2:'z'}
        t_axis_char = inv_axes.get(self.transport_axis, 'x')
        pdirs_nx = ''.join(ch for ch in pdirs if ch != t_axis_char)
        if pdirs_nx == '':
            pdirs_nx = None
        new_device = Hamiltonian(
            structure=self.structure,
            nx=2,
            ny=self.ny,
            nz=self.nz,
            periodic_dirs=self.periodic_dirs,
            passivate_x=False,
            nn_distance=2.4,
            transport_dir=[1, 0, 0],
            sort_axis='transport',
            auto_initialize=True,
        )
        hL = new_device.h_matrix.toarray() if hasattr(new_device.h_matrix, 'toarray') else new_device.h_matrix
        size = hL.shape[0]
        hL0 = hL[:size//2, :size//2]
        hLC = hL[:size//2, size//2:size]
        hR0 = hL0.copy()
        hRC = np.conjugate(hLC.T)  
        
        self.hL0 = hL0
        self.hLC = hLC 
        self.hR0 = hR0
        self.hRC = hRC

        if (self.periodic_dirs == "xy" or self.periodic_dirs == "y"):
            h_periodic  = self.get_periodic_coupling()
            
            self.h_periodic = h_periodic
        else:
            self.h_periodic = sp.csc_matrix(self.hL0.shape)

        return self.h_matrix, hL0, hLC, hR0, hRC, self.h_periodic
        
    # --- Other methods (diagonalize, _get_me, etc.) are unchanged ---
    def _get_me(self, atom1, atom2, l1, l2, coords=None, overlap=False):

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
    
    def get_device_dimensions(self):
        return self.nx + 2, self.ny, self.nz

    def get_periodic_coupling(self):
        # do something 
        return 
        
    
    def device_mapping_atomistic_to_custom_mesh():
        """"""
        return 
        