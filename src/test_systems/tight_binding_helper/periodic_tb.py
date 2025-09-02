"""Periodic tight-binding Hamiltonian (minimal, local) copied from nanonet
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as la
from .orbitals import SiliconSP3D5S, HydrogenS
from .diatomic_matrix_element import me
try:
    from .parametric_sinw import GeneratedNW  # optional import for type hints / factory method
except Exception:  # pragma: no cover
    GeneratedNW = None

_si = SiliconSP3D5S()
_h = HydrogenS()

def read_xyz(path):
    with open(path,'r') as f:
        lines = f.read().strip().splitlines()
    n = int(lines[0].strip())
    entries = []
    for line in lines[2:2+n]:
        p = line.split()
        entries.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return entries

class PeriodicTB:
    def __init__(self, xyz_path=None, *, entries=None, a_vec=(0,0,5.50), nn_distance=2.4, compute_angular=True,
                 periodic_axis='z', transport_axis=None, a_scalar=None):
        """Periodic tight-binding system.

        Provide either xyz_path (path to XYZ file) or pre-built entries list.
        entries format: list of (label, x, y, z).
        """
        if entries is None:
            if xyz_path is None:
                raise ValueError("Provide either xyz_path or entries")
            self.entries = read_xyz(xyz_path)
        else:
            self.entries = entries
        # Periodic (Bloch) translation vector used for k-phase
        self.a_vec = np.array(a_vec, dtype=float)
        self.periodic_axis = periodic_axis
        # Transport axis (principal layer direction) may differ; default to periodic_axis if not provided
        self.transport_axis = transport_axis or periodic_axis
        # Conventional lattice scalar (Angstrom) if provided; else infer from |a_vec| if non-zero
        self.a_scalar = a_scalar if a_scalar is not None else (np.linalg.norm(self.a_vec) if np.linalg.norm(self.a_vec) > 0 else 5.50)
        self.nn_distance = nn_distance
        self.compute_angular = compute_angular
        self.species_map = {'Si': _si, 'H': _h}
        self.positions = np.array([[x,y,z] for _,x,y,z in self.entries])
        self.site_species = [''.join([c for c in lab if not c.isdigit()]) for lab,_,_,_ in self.entries]
        self.orbital_counts = [self.species_map[s].num_of_orbitals for s in self.site_species]
        self.offsets = np.cumsum([0]+self.orbital_counts[:-1])
        self.basis_size = sum(self.orbital_counts)

        self._onsite = []
        self._Hc = []
        self._Hr = []
        self._Hl = []
        self._build_static()
        self._prepare_dense_blocks()
        # Transport Hamiltonian caches (lazy build)
        self._transport_cache = None

    @staticmethod
    def sort_smart(entries):
        """Placeholder ordering hook (currently identity)."""
        return entries
        
    
    @classmethod
    def from_generated_nw(cls, gen, *, periodic_axis='z', transport_axis=None, nn_distance=2.4, compute_angular=True):
        """Create PeriodicTB directly from a GeneratedNW object (no XYZ file).

        periodic_axis: 'x','y' or 'z' (default 'z'). The lattice translation
        length is set to gen.n{axis} * gen.a (e.g. gen.nz * gen.a for 'z').
        """
        if periodic_axis not in ('x', 'y', 'z'):
            raise ValueError("periodic_axis must be one of 'x','y','z'")
        # Map axis to multiplicity field on GeneratedNW
        mult = {'x': getattr(gen, 'nx', None), 'y': getattr(gen, 'ny', None), 'z': getattr(gen, 'nz', None)}[periodic_axis]
        if mult is None:
            raise ValueError('Provided object does not have nx/ny/nz attributes')
        length = float(mult) * float(gen.a)
        a_vec = {'x': (length, 0.0, 0.0), 'y': (0.0, length, 0.0), 'z': (0.0, 0.0, length)}[periodic_axis]

        # Build entries list from gen.all_atoms() -> returns list of (label,x,y,z)
        entries = [(lab, float(x), float(y), float(z)) for (lab, x, y, z) in gen.all_atoms()]
        entries = PeriodicTB.sort_smart(entries)
        return cls(entries=entries, a_vec=a_vec, nn_distance=nn_distance, compute_angular=compute_angular,
                   periodic_axis=periodic_axis, transport_axis=transport_axis, a_scalar=gen.a)

    # ---------------- Transport Hamiltonians (hl, h0, hr) -----------------
    def _classify_pairs_for_translation(self, t_vec):
        """Classify atom pairs relative to translation vector t_vec (like a_vec) returning lists.

        Returns (intra_list, plus_list) where plus_list are couplings to +t_vec. (Minus couplings implicit via Hermitian.)
        Each list element is {'tlist':[(gi,gj,val)]} similar to existing structures.
        """
        intra = []
        plus = []
        n_sites = len(self.entries)
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                base = self.positions[i] - self.positions[j]
                candidates = [(base,0),(base + t_vec,+1),(base - t_vec,-1)]
                cand_in = [(d,n) for d,n in candidates if 0 < np.linalg.norm(d) <= self.nn_distance]
                if not cand_in:
                    continue
                cand_in.sort(key=lambda x: (np.linalg.norm(x[0]), {0:0,+1:1,-1:2}[x[1]]))
                d_sel, n_sel = cand_in[0]
                tlist = self._compute_tlist(i,j,d_sel)
                if not tlist:
                    continue
                if n_sel == 0:
                    intra.append({'tlist': tlist})
                elif n_sel == +1:
                    plus.append({'tlist': tlist})
                else:  # n_sel == -1
                    rev = [(gj, gi, np.conjugate(val)) for gi, gj, val in tlist]
                    plus.append({'tlist': rev})
        return intra, plus

    def get_transport_hamiltonians(self, *, transport_axis=None):
        """Return (hl, h0, hr) for recursive Green's function along transport axis.

        hl = hr^†.
        h0 is Hermitian onsite+intra-cell term relative to chosen translation length a_scalar.

        transport_axis: override stored transport axis (x,y,z). Translation length = a_scalar.
        """
        axis = transport_axis or self.transport_axis
        if axis not in ('x','y','z'):
            raise ValueError('transport_axis must be x,y, or z')
        if self._transport_cache and self._transport_cache.get('axis') == axis:
            return self._transport_cache['hl'], self._transport_cache['h0'], self._transport_cache['hr']

        # Build translation vector of length a_scalar along axis
        length = self.a_scalar
        t_vec = {'x': np.array([length,0,0]), 'y': np.array([0,length,0]), 'z': np.array([0,0,length])}[axis]
        intra, plus = self._classify_pairs_for_translation(t_vec)
        n = self.basis_size
        h0 = np.zeros((n,n), dtype=complex)
        hr = np.zeros((n,n), dtype=complex)
        # Onsite energies in h0
        for idx,e in self._onsite:
            h0[idx, idx] = e
        # Intra symmetric
        for rec in intra:
            for gi, gj, val in rec['tlist']:
                h0[gi, gj] += val
                h0[gj, gi] += np.conjugate(val)
        # Plus couplings (do not add symmetric here)
        for rec in plus:
            for gi, gj, val in rec['tlist']:
                hr[gi, gj] += val
        hl = hr.conj().T  # Hermitian conjugate
        self._transport_cache = {'axis': axis, 'hl': hl, 'h0': h0, 'hr': hr}
        return hl, h0, hr


    def _compute_tlist(self, i, j, d_vec):
        dist = np.linalg.norm(d_vec)
        if dist == 0 or dist > self.nn_distance:
            return None
        direction = d_vec / dist if self.compute_angular else np.array([1.0,0.0,0.0])
        spec_i = self.species_map[self.site_species[i]]
        spec_j = self.species_map[self.site_species[j]]
        start_i = self.offsets[i]
        start_j = self.offsets[j]
        tlist = []
        for li in range(spec_i.num_of_orbitals):
            for lj in range(spec_j.num_of_orbitals):
                val = me(spec_i, li, spec_j, lj, direction, which_neighbour=0, overlap=False)
                if val != 0:
                    tlist.append((start_i+li, start_j+lj, val))
        return tlist if tlist else None

    def _build_static(self):
        # Onsite energies
        for i in range(len(self.entries)):
            spec = self.species_map[self.site_species[i]]
            start = self.offsets[i]
            for l in range(spec.num_of_orbitals):
                self._onsite.append((start+l, spec.orbitals[l]['energy']))
        # Classify each pair by shortest image (0,+1,-1)
        n_sites = len(self.entries)
        a = self.a_vec
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                base = self.positions[i] - self.positions[j]
                candidates = [(base,0),(base+a,+1),(base-a,-1)]
                cand_in = [(d,n) for d,n in candidates if 0 < np.linalg.norm(d) <= self.nn_distance]
                if not cand_in:
                    continue
                cand_in.sort(key=lambda x: (np.linalg.norm(x[0]), {0:0,+1:1,-1:2}[x[1]]))
                d_sel, n_sel = cand_in[0]
                tlist = self._compute_tlist(i,j,d_sel)
                if not tlist:
                    continue
                if n_sel == 0:
                    self._Hc.append({'tlist': tlist})
                elif n_sel == +1:
                    self._Hr.append({'tlist': tlist})
                else:
                    self._Hl.append({'tlist': tlist})

    def _prepare_dense_blocks(self):
        """Build dense Hermitian blocks Hc, Hr, Hl used for fast k combination.

        H(k) = Hc + exp(i k a) Hr + exp(-i k a) Hl^†
        We'll store full complex matrices for simplicity.
        """
        n = self.basis_size
        Hc = np.zeros((n,n), dtype=complex)
        Hr = np.zeros((n,n), dtype=complex)
        Hl = np.zeros((n,n), dtype=complex)
        # Onsite into Hc
        for idx,e in self._onsite:
            Hc[idx, idx] = e
        # Central symmetric
        for rec in self._Hc:
            for gi, gj, val in rec['tlist']:
                Hc[gi, gj] += val
                Hc[gj, gi] += np.conjugate(val)
        # Right couplings: store both directions (Hermitian) so later we can form combinations quickly
        for rec in self._Hr:
            for gi, gj, val in rec['tlist']:
                Hr[gi, gj] += val
                Hr[gj, gi] += np.conjugate(val)
        # Left couplings: store both directions of Hl^† (so use given val as from left cell; add Hermitian)
        for rec in self._Hl:
            for gi, gj, val in rec['tlist']:
                # In original assembly we added phase_l * val^* at (gi,gj); here keep raw val placeholder
                Hl[gi, gj] += np.conjugate(val)  # store the Hl^† pattern
                Hl[gj, gi] += val
        # Precompute combination helpers A,B for trig form if Hr and Hl^† needed
        self._Hc_dense = Hc
        self._Hr_dense = Hr
        self._Hl_dense = Hl
        # Precompute additive combos
        self._A_dense = Hr + Hl  # corresponds to (Hr + Hl^†)
        self._B_dense = Hr - Hl  # corresponds to (Hr - Hl^†)

    def h_of_k_fast(self, k_vec):
        """Fast H(k) using linear combination of dense precomputed blocks.

        H(k)= Hc + cos(k·a)*A + i sin(k·a)*B where A=Hr+Hl^†, B=Hr-Hl^†.
        (Derived from e^{ika}Hr + e^{-ika}Hl^†.)
        """
        k_vec = np.array(k_vec, dtype=float)
        ka = np.dot(k_vec, self.a_vec)
        c = np.cos(ka)
        s = np.sin(ka)
        # H = Hc + c*A + i s*B
        return self._Hc_dense + c * self._A_dense + 1j * s * self._B_dense

    def h_of_k(self, k_vec):
        k_vec = np.array(k_vec, dtype=float)
        a_vec = self.a_vec
        phase_r = np.exp(1j * np.dot(k_vec, a_vec))   # e^{+ik·a}
        phase_l = np.exp(-1j * np.dot(k_vec, a_vec))  # e^{-ik·a}
        H = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        # Onsite
        for idx,e in self._onsite:
            H[idx, idx] = e
        # Central (Hc) symmetric, no k phase (matching Hc term)
        for rec in self._Hc:
            for gi, gj, val in rec['tlist']:
                H[gi, gj] += val
                H[gj, gi] += np.conjugate(val)
        # Right couplings: Hr e^{+ik·a}
        for rec in self._Hr:
            for gi, gj, val in rec['tlist']:
                H[gi, gj] += phase_r * val
                H[gj, gi] += np.conjugate(phase_r * val)
        # Left couplings: Hl^† e^{-ik·a}  (add as hermitian conjugate with phase_l)
        for rec in self._Hl:
            for gi, gj, val in rec['tlist']:
                # Hl contributes Hl^† e^{-ik·a} => add e^{-ik·a} val* at (gi,gj) and its conjugate at (gj,gi)
                H[gi, gj] += phase_l * np.conjugate(val)
                H[gj, gi] += np.conjugate(phase_l * np.conjugate(val))
        return H

    def diagonalize(self, k_vec):
        # Use fast path dense combination
        Hk = self.h_of_k_fast(k_vec)
        w, _ = la.eigh(Hk)
        return np.real(w)
