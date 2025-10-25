from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

import numpy as np
import scipy.sparse as sp
import pymetis


def _diff_from_offsets(offsets: Iterable[int]) -> np.ndarray | None:
    arr = np.asarray(list(offsets), dtype=int)
    if arr.ndim != 1 or arr.size < 2:
        return None
    diffs = np.diff(arr)
    if np.any(diffs <= 0):
        return None
    return diffs.astype(int)


def compute_optimal_block_sizes(
    ham_device: np.ndarray | sp.spmatrix,
    H01: np.ndarray | sp.spmatrix | None = None,
    *,
    atom_offsets: Iterable[int] | None = None,
    tol: float = 1e-10,
) -> np.ndarray:
    """Heuristically determine block sizes for block-tridiagonal inversion.

    Preference order
    ----------------
    1. Explicit ``atom_offsets`` if provided (typical in tight-binding setups).
    2. Square coupling block ``H01`` (uniform block size).
    3. Fallback to unit-sized blocks (safe but possibly suboptimal).

    The routine is O(N + nnz) and returns a 1-D integer array whose sum
    equals the Hamiltonian dimension.
    """

    if atom_offsets is not None:
        candidate = _diff_from_offsets(atom_offsets)
        if candidate is not None and candidate.size > 0:
            return candidate

    n = int(ham_device.shape[0])
    if H01 is not None:
        h01 = H01.toarray() if sp.issparse(H01) else np.asarray(H01)
        if h01.ndim == 2 and h01.shape[0] == h01.shape[1] and h01.shape[0] > 0:
            block = int(h01.shape[0])
            if n % block == 0:
                return np.full(n // block, block, dtype=int)

    return np.ones(n, dtype=int)

    
def compute_block_sizes_block_tridiagonal(ham_device : sp.sparray):
    """Return diagonal block sizes for a block-tridiagonal matrix.

    The routine scans successive rows and groups together contiguous rows that
    share the same left-most and right-most non-zero column indices. For an
    ideal block-tridiagonal pattern (dense diagonal blocks with nearest-neighbour
    couplings) these pairs remain constant within a block and change exactly at
    block boundaries, allowing us to recover the block sizes in ``O(nnz)`` time.

    Parameters
    ----------
    ham_device : sp.sparray or array_like
        Square block-tridiagonal matrix (typically the device Hamiltonian).

    Returns
    -------
    np.ndarray
        1-D array of positive integers whose sum equals the matrix dimension.

    Raises
    ------
    ValueError
        If the input is not square, contains all-zero rows, or deviates from the
        expected block-tridiagonal sparsity pattern (detected via inconsistent
        row envelopes).
    """

    if ham_device is None:
        raise ValueError("ham_device must be provided.")

    sparray_type = getattr(sp, "sparray", None)

    # Convert to CSR array for efficient row access while accepting dense input.
    if sp.issparse(ham_device) or (sparray_type is not None and isinstance(ham_device, sparray_type)):
        csr = sp.csr_array(ham_device)
    else:
        arr = np.asarray(ham_device)
        if arr.ndim != 2:
            raise ValueError("ham_device must be a 2D square matrix.")
        csr = sp.csr_array(arr)

    n_rows, n_cols = csr.shape
    if n_rows != n_cols:
        raise ValueError("ham_device must be square to admit block-tridiagonal partitioning.")
    if n_rows == 0:
        return np.empty(0, dtype=int)

    indptr = csr.indptr
    indices = csr.indices
    row_min = np.empty(n_rows, dtype=int)
    row_max = np.empty(n_rows, dtype=int)

    for row in range(n_rows):
        start = indptr[row]
        end = indptr[row + 1]
        if start == end:
            raise ValueError("Encountered an all-zero row; matrix is not block-tridiagonal.")
        cols = indices[start:end]
        col_min = int(cols.min())
        col_max = int(cols.max())
        # Allow diagonal blocks whose explicit diagonal entries may be zero by
        # ensuring the current row index participates in the envelope.
        row_min[row] = min(col_min, row)
        row_max[row] = max(col_max, row)

    blocks: list[int] = []
    row = 0
    while row < n_rows:
        left = row_min[row]
        right = row_max[row]
        if left > row or right < row:
            raise ValueError("Matrix sparsity is inconsistent with block-tridiagonal structure.")

        end = row + 1
        while end < n_rows and row_min[end] == left and row_max[end] == right:
            end += 1

        block_size = end - row
        if block_size <= 0:
            raise ValueError("Failed to identify a positive block size during partitioning.")
        blocks.append(block_size)
        row = end

    if np.sum(blocks) != n_rows:
        raise ValueError("Recovered block sizes do not cover the full matrix dimension.")

    return np.asarray(blocks, dtype=int)
    
def compute_block_sizes_metis(
    ham_device: np.ndarray | sp.spmatrix,
    H00_left: np.ndarray | sp.spmatrix,
    H00_right: np.ndarray | sp.spmatrix,
    *,
    atom_offsets: Iterable[int],
    desired_partitions: int | None = None,
    min_block_orbitals: int | None = None,
) -> np.ndarray:
    """Partition a Hamiltonian into block-tridiagonal segments using PyMETIS.

    Parameters
    ----------
    ham_device : array_like or sparse matrix
        Device Hamiltonian in the current ordering.
    H00_left, H00_right : array_like or sparse matrix
        Lead onsite blocks. The resulting partition's first and last block sizes
        are constrained to match ``H00_left.shape[0]`` and ``H00_right.shape[0]``.
    atom_offsets : iterable of int
        Cumulative orbital counts per atom (length ``n_atoms + 1``). The offsets
        must align with the lead block sizes so that the left and right blocks
        comprise an integer number of atoms.
    desired_partitions : int, optional
        Explicit number of interior partitions for METIS. If ``None`` a heuristic
        based on the total number of orbitals is used.
    min_block_orbitals : int, optional
        Minimum number of orbitals permitted in an interior block. Defaults to the
        smaller of the two lead block sizes, but never less than two for sizeable
        devices.

    Returns
    -------
    np.ndarray
        Block sizes whose sum equals the Hamiltonian dimension.

    Notes
    -----
    If PyMETIS is unavailable or the partitioning fails, callers should fall back
    to a simpler heuristic (e.g. :func:`compute_optimal_block_sizes`).
    """

    if pymetis is None:  # pragma: no cover - depends on external package
        raise ImportError(
            "pymetis is required for compute_block_sizes_metis. Install via 'pip install pymetis'."
        )

    if ham_device is None:
        raise ValueError("ham_device must be provided.")

    atom_offsets = np.asarray(list(atom_offsets), dtype=int)
    if atom_offsets.ndim != 1 or atom_offsets.size < 2:
        raise ValueError("atom_offsets must be a 1-D iterable of cumulative counts.")

    n_orbitals = int(atom_offsets[-1])
    if sp.issparse(ham_device):
        ham_coo = ham_device.tocoo()
    else:
        ham_coo = sp.coo_array(np.asarray(ham_device))

    if ham_coo.shape[0] != ham_coo.shape[1]:
        raise ValueError("ham_device must be square to construct block partitions.")
    if ham_coo.shape[0] != n_orbitals:
        raise ValueError("atom_offsets do not match the Hamiltonian dimension.")

    left_size = int(H00_left.shape[0])
    right_size = int(H00_right.shape[0])
    if left_size <= 0 or right_size <= 0:
        raise ValueError("Lead onsite blocks must have positive dimension.")

    # Determine how many atoms are locked into the lead blocks.
    n_atoms = atom_offsets.size - 1
    left_atom_count = int(np.searchsorted(atom_offsets, left_size, side="left"))
    if left_atom_count <= 0 or atom_offsets[left_atom_count] != left_size:
        raise ValueError("Left lead block size does not align with atom_offsets.")

    right_atom_count = 0
    accum = 0
    while accum < right_size and (n_atoms - right_atom_count) > 0:
        atom_idx = n_atoms - 1 - right_atom_count
        accum += int(atom_offsets[atom_idx + 1] - atom_offsets[atom_idx])
        right_atom_count += 1
    if accum != right_size:
        raise ValueError("Right lead block size does not align with atom_offsets.")

    interior_atom_start = left_atom_count
    interior_atom_end = n_atoms - right_atom_count
    if interior_atom_start >= interior_atom_end:
        return np.asarray([left_size, right_size], dtype=int)

    interior_atoms = list(range(interior_atom_start, interior_atom_end))
    num_interior_atoms = len(interior_atoms)

    # Map orbitals -> atoms for fast lookup
    orbital_to_atom = np.empty(n_orbitals, dtype=int)
    for atom in range(n_atoms):
        start = int(atom_offsets[atom])
        end = int(atom_offsets[atom + 1])
        orbital_to_atom[start:end] = atom

    interior_index = {atom: idx for idx, atom in enumerate(interior_atoms)}
    edge_accumulator: List[dict[int, float]] = [defaultdict(float) for _ in range(num_interior_atoms)]

    rows = np.asarray(ham_coo.row, dtype=int)
    cols = np.asarray(ham_coo.col, dtype=int)
    data = np.asarray(ham_coo.data)
    for i, j, val in zip(rows, cols, data):
        if i >= j:
            continue  # accumulate each pair once
        atom_i = orbital_to_atom[i]
        atom_j = orbital_to_atom[j]
        if atom_i == atom_j:
            continue
        local_i = interior_index.get(atom_i)
        local_j = interior_index.get(atom_j)
        if local_i is None or local_j is None:
            continue  # skip couplings involving lead atoms
        weight = float(abs(val))
        if weight == 0.0:
            continue
        edge_accumulator[local_i][local_j] += weight
        edge_accumulator[local_j][local_i] += weight

    vertex_weights = [int(atom_offsets[atom + 1] - atom_offsets[atom]) for atom in interior_atoms]
    total_interior_orbitals = int(np.sum(vertex_weights))

    if min_block_orbitals is None:
        candidate = min(left_size, right_size)
        min_block_orbitals = max(1, candidate)
        if min_block_orbitals < 2 and total_interior_orbitals > 4:
            min_block_orbitals = 2
    else:
        min_block_orbitals = max(1, int(min_block_orbitals))

    if desired_partitions is not None:
        num_partitions = max(1, min(int(desired_partitions), num_interior_atoms))
    else:
        if total_interior_orbitals == 0:
            num_partitions = 1
        else:
            approx = max(1, total_interior_orbitals // max(min_block_orbitals, 1))
            num_partitions = min(max(1, approx), num_interior_atoms)

    all_weights = [w for mapping in edge_accumulator for w in mapping.values()]
    if all_weights:
        max_weight = max(all_weights)
        scale = max(max_weight / 100.0, 1e-12)
    else:
        scale = 1.0

    xadj: List[int] = [0]
    adjncy: List[int] = []
    eweights_flat: List[int] = []

    for mapping in edge_accumulator:
        if not mapping:
            xadj.append(len(adjncy))
            continue
        neighbors = sorted(mapping.keys())
        weights = [max(1, int(round(mapping[nbr] / scale))) for nbr in neighbors]
        adjncy.extend(int(n) for n in neighbors)
        eweights_flat.extend(int(w) for w in weights)
        xadj.append(len(adjncy))

    kwargs = {"vweights": [int(w) for w in vertex_weights]}
    if eweights_flat:
        kwargs["eweights"] = eweights_flat

    _, assignment = pymetis.part_graph(
        num_partitions,
        xadj=[int(v) for v in xadj],
        adjncy=adjncy,
        **kwargs,
    )

    block_sizes: List[int] = [left_size]
    if num_interior_atoms > 0:
        current_part = assignment[0]
        current_sum = 0
        for idx, atom in enumerate(interior_atoms):
            part = assignment[idx]
            if idx > 0 and part != current_part:
                if current_sum > 0:
                    block_sizes.append(current_sum)
                current_sum = 0
                current_part = part
            current_sum += vertex_weights[idx]
        if current_sum > 0:
            block_sizes.append(current_sum)

    block_sizes.append(right_size)

    # Merge undersized interior blocks to satisfy minimum size constraints.
    i = 1
    while i < len(block_sizes) - 1:
        if block_sizes[i] < min_block_orbitals and len(block_sizes) > 3:
            if i == 1:
                block_sizes[i + 1] += block_sizes[i]
                block_sizes.pop(i)
            elif i == len(block_sizes) - 2:
                block_sizes[i - 1] += block_sizes[i]
                block_sizes.pop(i)
                i -= 1
            else:
                if block_sizes[i - 1] <= block_sizes[i + 1]:
                    block_sizes[i - 1] += block_sizes[i]
                    block_sizes.pop(i)
                    i -= 1
                else:
                    block_sizes[i + 1] += block_sizes[i]
                    block_sizes.pop(i)
        else:
            i += 1

    if sum(block_sizes) != n_orbitals:
        raise RuntimeError("Partition sizes do not sum to Hamiltonian dimension.")

    return np.asarray(block_sizes, dtype=int)
