"""Utility to compare generated Si nanowire (SiNW2_copy.xyz) vs reference NanoNet (SiNW2.xyz).

Enhanced comparison reports:
- Atom counts by species
- One-to-one nearest-neighbor matching (Si & H) with distance stats
- Unmatched (extra/missing) atoms lists (coordinates) per species
- Per-axis extrema and lattice extents
- Unique z-level histograms (Si / H) and differences
- Nearest-neighbor distance statistics (Si-Si, Si-H, H-H)
- Dangling bond stats
- Per-Si hydrogen coordination comparison (count & tetra direction pattern)
- Summary of problematic Si sites (missing or extra hydrogens)

CLI:
  python compare_sinw_structures.py --gen SiNW2_copy.xyz --ref SiNW2.xyz \
      --a 5.5 --tol 0.25 --verbose

Outputs JSON report to SiNW_compare_report.json (or --out path).
"""
from __future__ import annotations
import numpy as np, math, json, sys, pathlib, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from scipy.spatial import cKDTree

THIS_DIR = pathlib.Path(__file__).resolve().parent

SI_SI_BOND = 2.39  # ~ a*sqrt(3)/4 with a=5.5 -> 2.381; padded
SI_H_BOND_MAX = 1.65
H_H_NEAR = 2.2
DEFAULT_MATCH_TOL = 0.25

_DIAMOND_TETRA_A = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
_DIAMOND_TETRA_B = -_DIAMOND_TETRA_A

@dataclass
class Structure:
    labels: List[str]
    coords: np.ndarray  # (N,3)
    def species_indices(self, sp: str):
        if sp == 'Si':
            return [i for i,l in enumerate(self.labels) if l.startswith('Si')]
        return [i for i,l in enumerate(self.labels) if l.startswith(sp)]
    def subset(self, indices: List[int]):
        return self.coords[indices]


def read_xyz(path: pathlib.Path) -> Structure:
    with open(path) as f:
        lines = f.read().strip().splitlines()
    n = int(lines[0].strip())
    body = lines[2:2+n]
    labels=[]; coords=[]
    for ln in body:
        parts = ln.split()
        lab = parts[0]
        coords.append(tuple(map(float, parts[1:4])))
        labels.append(lab)
    return Structure(labels=labels, coords=np.array(coords))


def summarize(struct: Structure, name: str) -> Dict:
    coords = struct.coords
    si_idx = struct.species_indices('Si')
    h_idx = struct.species_indices('H')
    mins = coords.min(axis=0); maxs = coords.max(axis=0)
    return {
        'name': name,
        'counts': {'total': len(coords), 'Si': len(si_idx), 'H': len(h_idx)},
        'extent': {'xmin': float(mins[0]), 'xmax': float(maxs[0]), 'ymin': float(mins[1]), 'ymax': float(maxs[1]), 'zmin': float(mins[2]), 'zmax': float(maxs[2])}
    }


def neighbor_stats(struct: Structure):
    tree = cKDTree(struct.coords)
    si_idx = struct.species_indices('Si')
    h_idx = struct.species_indices('H')

    def collect(idx_a, idx_b_set, cutoff):
        res=[]
        for i in idx_a:
            dists, inds = tree.query(struct.coords[i], k=len(struct.coords), distance_upper_bound=cutoff)
            for d,j in zip(dists, inds):
                if not np.isfinite(d) or j==i or j>=len(struct.coords):
                    continue
                if j in idx_b_set:
                    res.append(d)
        return res
    si_si = collect(si_idx, set(si_idx), SI_SI_BOND+0.01)
    si_h = collect(si_idx, set(h_idx), SI_H_BOND_MAX+0.01)
    h_h = collect(h_idx, set(h_idx), H_H_NEAR+0.01)

    def stats(arr):
        if not arr: return {'n':0}
        a=np.array(arr)
        return {'n':int(len(a)), 'mean':float(a.mean()), 'min':float(a.min()), 'max':float(a.max())}
    return {'Si-Si': stats(si_si), 'Si-H': stats(si_h), 'H-H': stats(h_h)}


def dangling_bonds(struct: Structure):
    tree = cKDTree(struct.coords)
    si_idx = struct.species_indices('Si')
    danglers=0; degrees=[]
    for i in si_idx:
        # Si neighbors
        dists, inds = tree.query(struct.coords[i], k=len(struct.coords), distance_upper_bound=SI_SI_BOND+0.01)
        neigh=set()
        for d,j in zip(dists, inds):
            if not np.isfinite(d) or j==i or j>=len(struct.coords): continue
            if d < 0.5: continue
            if d <= SI_SI_BOND+1e-6:
                neigh.add(j)
        # H neighbors
        dists, inds = tree.query(struct.coords[i], k=len(struct.coords), distance_upper_bound=SI_H_BOND_MAX+0.01)
        for d,j in zip(dists, inds):
            if not np.isfinite(d) or j==i or j>=len(struct.coords): continue
            if struct.labels[j].startswith('H') and d <= SI_H_BOND_MAX+1e-6:
                neigh.add(j)
        deg=len(neigh); degrees.append(deg)
        if deg < 4: danglers+=1
    uniq, counts = np.unique(degrees, return_counts=True)
    hist = {int(k): int(c) for k, c in zip(uniq, counts)}
    return {'dangling_si': danglers, 'mean_coordination': float(np.mean(degrees)), 'degree_hist': hist}


def one_to_one_match(coords_a: np.ndarray, coords_b: np.ndarray, tol: float):
    """Greedy one-to-one nearest matching within tol. Returns matches list and unmatched indices.
    matches: list[(i_a,i_b,distance)] sorted by distance.
    """
    if len(coords_a)==0 or len(coords_b)==0:
        return [], list(range(len(coords_a))), list(range(len(coords_b)))
    tree = cKDTree(coords_b)
    candidate_pairs=[]
    # gather candidates (within tol)
    for i,p in enumerate(coords_a):
        idxs = tree.query_ball_point(p, tol)
        for j in idxs:
            d = np.linalg.norm(p - coords_b[j])
            candidate_pairs.append((d,i,j))
    candidate_pairs.sort()
    used_a=set(); used_b=set(); matches=[]
    for d,i,j in candidate_pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        matches.append((i,j,d))
    unmatched_a=[i for i in range(len(coords_a)) if i not in used_a]
    unmatched_b=[j for j in range(len(coords_b)) if j not in used_b]
    matches.sort(key=lambda x: x[2])
    return matches, unmatched_a, unmatched_b


def species_match_report(gen: Structure, ref: Structure, species: str, tol: float):
    g_idx = gen.species_indices(species)
    r_idx = ref.species_indices(species)
    g_coords = gen.subset(g_idx)
    r_coords = ref.subset(r_idx)
    matches, g_un, r_un = one_to_one_match(g_coords, r_coords, tol)
    distances = [d for *_ , d in matches]
    dist_stats = None
    if distances:
        arr=np.array(distances)
        dist_stats={'n':len(arr),'mean':float(arr.mean()),'max':float(arr.max()),'p95':float(np.percentile(arr,95))}
    # Lists of unmatched coordinates
    g_only = [tuple(map(float,g_coords[i])) for i in g_un]
    r_only = [tuple(map(float,r_coords[j])) for j in r_un]
    return {
        'species': species,
        'ref_count': len(r_coords),
        'gen_count': len(g_coords),
        'matched': len(matches),
        'gen_only': len(g_only),
        'ref_only': len(r_only),
        'distance_stats': dist_stats,
        'gen_only_coords': g_only[:20],  # cap to 20 for brevity
        'ref_only_coords': r_only[:20]
    }


def infer_sublattice(coord, a: float):
    x,y,z = coord
    fx,fy,fz = (x/a, y/a, z/a)
    return 'A' if (int(round(2*(fx+fy+fz))) & 1)==0 else 'B'


def classify_hydrogen_directions(struct: Structure, a: float):
    """Return mapping si_index -> {'pos':(x,y,z), 'nH':k, 'dirs':[dir_indices]} where dir_indices are 0..3 tetra slots."""
    coords = struct.coords
    si_idx = struct.species_indices('Si')
    h_idx = struct.species_indices('H')
    if not h_idx:
        return {}
    tree = cKDTree(coords)
    result={}
    for i in si_idx:
        pos = coords[i]
        sub = infer_sublattice(pos,a)
        ideals = _DIAMOND_TETRA_A if sub=='A' else _DIAMOND_TETRA_B
        ideals_n = np.array([v/np.linalg.norm(v) for v in ideals])
        # Query possible H neighbors
        dists, inds = tree.query(pos, k=len(coords), distance_upper_bound=SI_H_BOND_MAX+0.01)
        dirs=[]
        for d,j in zip(dists, inds):
            if not np.isfinite(d) or j>=len(coords) or j==i: continue
            if j in h_idx and d <= SI_H_BOND_MAX+1e-6:
                vec = coords[j] - pos
                vn = vec/np.linalg.norm(vec)
                dots = ideals_n @ vn
                k = int(np.argmax(dots))
                if dots[k] > 0.8:  # assign
                    dirs.append(k)
        result[i]={'pos': tuple(map(float,pos)), 'nH': len(dirs), 'dirs': sorted(dirs)}
    return result


def compare_hydrogen_coordination(gen: Structure, ref: Structure, a: float, tol: float):
    # Match Si first
    g_si_idx = gen.species_indices('Si')
    r_si_idx = ref.species_indices('Si')
    g_si = gen.subset(g_si_idx)
    r_si = ref.subset(r_si_idx)
    matches, g_un, r_un = one_to_one_match(g_si, r_si, tol)
    g_dir = classify_hydrogen_directions(gen,a)
    r_dir = classify_hydrogen_directions(ref,a)
    mismatches=[]
    for (ig, ir, d) in matches:
        g_global = g_si_idx[ig]; r_global = r_si_idx[ir]
        g_info = g_dir.get(g_global, {'nH':0,'dirs':[],'pos':tuple(map(float,g_si[ig]))})
        r_info = r_dir.get(r_global, {'nH':0,'dirs':[],'pos':tuple(map(float,r_si[ir]))})
        if g_info['nH'] != r_info['nH'] or g_info['dirs'] != r_info['dirs']:
            mismatches.append({
                'si_ref_pos': r_info['pos'],
                'si_gen_pos': g_info['pos'],
                'match_dist': float(d),
                'ref_nH': r_info['nH'], 'gen_nH': g_info['nH'],
                'ref_dirs': r_info['dirs'], 'gen_dirs': g_info['dirs']
            })
    return {
        'matched_si': len(matches),
        'unmatched_si_gen': len(g_un),
        'unmatched_si_ref': len(r_un),
        'si_with_h_mismatches': len(mismatches),
        'examples': mismatches[:25]
    }


def z_level_hist(coords: np.ndarray):
    levels, counts = np.unique(np.round(coords[:,2],6), return_counts=True)
    return {str(float(l)): int(c) for l,c in zip(levels, counts)}


def compare_z_layers(gen: Structure, ref: Structure):
    gen_h = gen.subset(gen.species_indices('H')) if gen.species_indices('H') else np.zeros((0,3))
    ref_h = ref.subset(ref.species_indices('H')) if ref.species_indices('H') else np.zeros((0,3))
    gen_si = gen.subset(gen.species_indices('Si'))
    ref_si = ref.subset(ref.species_indices('Si'))
    return {
        'si': {'generated': z_level_hist(gen_si), 'reference': z_level_hist(ref_si)},
        'h':  {'generated': z_level_hist(gen_h),  'reference': z_level_hist(ref_h)}
    }


def build_report(gen: Structure, ref: Structure, a: float, tol: float) -> Dict[str, Any]:
    report = {
        'summary': [summarize(gen,'generated'), summarize(ref,'reference')],
        'match': {
            'Si': species_match_report(gen, ref, 'Si', tol),
            'H': species_match_report(gen, ref, 'H', tol)
        },
        'neighbor_stats': {'generated': neighbor_stats(gen), 'reference': neighbor_stats(ref)},
        'dangling': {'generated': dangling_bonds(gen), 'reference': dangling_bonds(ref)},
        'z_layers': compare_z_layers(gen, ref),
        'hydrogen_coordination_diff': compare_hydrogen_coordination(gen, ref, a, tol),
    }
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--gen', default='SiNW2_copy.xyz')
    p.add_argument('--ref', default='SiNW2.xyz')
    p.add_argument('--out', default='SiNW_compare_report.json')
    p.add_argument('--a', type=float, default=5.50, help='Lattice constant (for sublattice & tetra classification)')
    p.add_argument('--tol', type=float, default=DEFAULT_MATCH_TOL, help='Position matching tolerance (Å)')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gen_path = (THIS_DIR / args.gen).resolve()
    ref_path = (THIS_DIR / args.ref).resolve()
    gen = read_xyz(gen_path)
    ref = read_xyz(ref_path)
    report = build_report(gen, ref, args.a, args.tol)
    out_path = THIS_DIR / args.out
    with open(out_path,'w') as f: json.dump(report,f,indent=2)
    print(json.dumps(report, indent=2))
    print(f'Wrote {out_path}')
    if args.verbose:
        # Print extra unmatched coordinate detail if requested
        si_match = report['match']['Si']
        h_match = report['match']['H']
        if si_match['gen_only'] or si_match['ref_only']:
            print('\n[Verbose] Some Si unmatched beyond tolerance: adjust --tol if expected to match.')
        if h_match['gen_only'] or h_match['ref_only']:
            print('[Verbose] Some H unmatched beyond tolerance.')

if __name__ == '__main__':
    main()
