from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp


def _ensure_src_on_path() -> None:
    """Ensure the repository 'src' directory is on sys.path for absolute imports."""
    src_root = Path(__file__).resolve().parents[3]
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_path()
from negf.gf.recursive_greens_functions import gf_inverse
from negf.self_energy import greens_functions
from hamiltonian.tb.orbitals import Orbitals
from hamiltonian import Hamiltonian

from negf.utils.block_partition import (
    compute_block_sizes_block_tridiagonal,
    compute_block_sizes_metis,
)


def _has_pymetis() -> bool:
    try:
        import pymetis  # noqa: F401
    except Exception:
        return False
    return True


def _build_block_tridiagonal(block_sizes: list[int], *, seed: int = 0, sparse: bool = True):
    rng = np.random.default_rng(seed)
    dim = int(np.sum(block_sizes))
    mat = np.zeros((dim, dim), dtype=float)
    offset = 0
    for idx, block in enumerate(block_sizes):
        block_slice = slice(offset, offset + block)
        diag_block = rng.normal(size=(block, block))
        mat[block_slice, block_slice] = (diag_block + diag_block.T) / 2.0
        if idx < len(block_sizes) - 1:
            next_block = block_sizes[idx + 1]
            next_slice = slice(offset + block, offset + block + next_block)
            coupling = rng.normal(size=(block, next_block))
            mat[block_slice, next_slice] = coupling
            mat[next_slice, block_slice] = coupling.T
        offset += block
    if sparse:
        return sp.csr_array(mat)
    return mat


def _build_atom_hamiltonian(orbitals_per_atom: list[int]) -> tuple[sp.csr_array, np.ndarray]:
    offsets = np.concatenate(([0], np.cumsum(np.asarray(orbitals_per_atom, dtype=int))))
    dim = int(offsets[-1])
    mat = np.zeros((dim, dim), dtype=float)
    for atom, count in enumerate(orbitals_per_atom):
        start = int(offsets[atom])
        end = int(offsets[atom + 1])
        mat[start:end, start:end] = np.eye(count)
        if atom < len(orbitals_per_atom) - 1:
            next_start = int(offsets[atom + 1])
            next_end = int(offsets[atom + 2])
            coupling = np.ones((count, int(offsets[atom + 2] - offsets[atom + 1])), dtype=float)
            mat[start:end, next_start:next_end] = coupling
            mat[next_start:next_end, start:end] = coupling.T
    return sp.csr_array(mat), offsets


def _print_result(name: str, condition: bool, details: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"[{status}] {name}"
    if details:
        line += f" :: {details}"
    print(line)


def _case_sparse_variable_blocks() -> None:
    name = "sparse variable blocks"
    blocks = [2, 3, 1, 4]
    matrix = _build_block_tridiagonal(blocks, seed=42, sparse=True)
    result = compute_block_sizes_block_tridiagonal(matrix)
    ok = np.array_equal(result, np.asarray(blocks, dtype=int))
    _print_result(name, ok, f"result={result}")


def _case_dense_input() -> None:
    name = "dense input"
    blocks = [1, 5, 2]
    matrix = _build_block_tridiagonal(blocks, seed=7, sparse=False)
    result = compute_block_sizes_block_tridiagonal(matrix)
    ok = np.array_equal(result, np.asarray(blocks, dtype=int))
    _print_result(name, ok, f"result={result}")


def _case_zero_row_error() -> None:
    name = "zero row triggers ValueError"
    bad_matrix = sp.csr_array(np.zeros((3, 3), dtype=float))
    try:
        compute_block_sizes_block_tridiagonal(bad_matrix)
    except ValueError as exc:
        _print_result(name, True, details=str(exc))
    else:
        _print_result(name, False, details="expected ValueError")


def _case_one_d_chain() -> None:
    name = "1D chain with zero diagonal"
    n = 10
    data = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        data[i, i + 1] = 1.0
        data[i + 1, i] = 1.0
    matrix = sp.csr_array(data)
    result = compute_block_sizes_block_tridiagonal(matrix)
    ok = np.array_equal(result, np.ones(n, dtype=int))
    _print_result(name, ok, f"result={result}")


def _case_metis_one_d_chain() -> None:
    name = "METIS 1D chain"
    if not _has_pymetis():
        _print_result(name, True, details="skipped (pymetis not available)")
        return
    orbitals_per_atom = [1] * 6
    dim = len(orbitals_per_atom)
    mat = np.zeros((dim, dim), dtype=float)
    for i in range(dim - 1):
        mat[i, i + 1] = 1.0
        mat[i + 1, i] = 1.0
    matrix = sp.csr_array(mat)
    offsets = np.arange(dim + 1, dtype=int)
    H00 = np.ones((1, 1), dtype=float)
    result = compute_block_sizes_metis(
        matrix,
        H00,
        H00,
        atom_offsets=offsets,
        desired_partitions=7,
        min_block_orbitals=1,
    )
    ok = (
        np.issubdtype(result.dtype, np.integer)
        and result.sum() == dim
        and result[0] == 1
        and result[-1] == 1
        and np.all(result[1:-1] > 0)
    )
    _print_result(name, ok, f"result={result}")


def _case_metis_multi_orbital() -> None:
    name = "METIS multi-orbital chain"
    if not _has_pymetis():
        _print_result(name, True, details="skipped (pymetis not available)")
        return
    orbitals = [2, 2, 1, 3, 2, 1, 2, 2]
    matrix, offsets = _build_atom_hamiltonian(orbitals)
    H00 = np.eye(4, dtype=float)
    result = compute_block_sizes_metis(
        matrix,
        H00,
        H00,
        atom_offsets=offsets,
        desired_partitions=2,
        min_block_orbitals=1,
    )
    total_dim = int(offsets[-1])
    ok = (
        np.issubdtype(result.dtype, np.integer)
        and result.sum() == total_dim
        and result[0] == H00.shape[0]
        and result[-1] == H00.shape[0]
        and np.all(result[1:-1] >= 1)
    )
    _print_result(name, ok, f"result={result}")


def _case_metis_diatomic_chain() -> None:
    name = "METIS diatomic chain (A-B alternation)"
    if not _has_pymetis():
        _print_result(name, True, details="skipped (pymetis not available)")
        return

    atom_labels = ["A" if idx % 2 == 0 else "B" for idx in range(11)]
    dim = len(atom_labels)
    matrix = np.zeros((dim, dim), dtype=float)
    onsite_values = {"A": 0.5, "B": -0.5}
    for i, label in enumerate(atom_labels):
        matrix[i, i] = onsite_values[label]
        if i < dim - 1:
            matrix[i, i + 1] = 1.0
            matrix[i + 1, i] = 1.0
    matrix = sp.csr_array(matrix)

    offsets = np.arange(dim + 1, dtype=int)
    H00_left = np.array([[onsite_values["A"], 1.0], [1.0, onsite_values["B"]]], dtype=float)
    H00_right = np.array([[onsite_values["B"], 1.0], [1.0, onsite_values["A"]]], dtype=float)

    result = compute_block_sizes_metis(
        matrix,
        H00_left,
        H00_right,
        atom_offsets=offsets,
        desired_partitions=4,
        min_block_orbitals=2,
    )

    ok = (
        np.issubdtype(result.dtype, np.integer)
        and result.sum() == dim
        and result[0] == H00_left.shape[0]
        and result[-1] == H00_right.shape[0]
        and result.size >= 3
        and np.all(result[1:-1] >= 2)
    )

    boundaries = np.cumsum(result)[:-1]
    boundary_labels = [atom_labels[idx - 1] for idx in boundaries if 0 < idx < dim]
    details = f"result={result}" if ok else f"result={result}, boundaries={boundary_labels}"
    _print_result(name, ok, details)


if __name__ == "__main__":
    _case_sparse_variable_blocks()
    _case_dense_input()
    _case_zero_row_error()
    _case_one_d_chain()
    _case_metis_one_d_chain()
    _case_metis_multi_orbital()
    _case_metis_diatomic_chain()
