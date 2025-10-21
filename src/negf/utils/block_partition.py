from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp


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
