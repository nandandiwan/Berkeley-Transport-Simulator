from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import multiprocessing as mp

import numpy as np
import scipy.sparse as sp


ArrayLike = np.ndarray


@dataclass(slots=True)
class PairwiseInverseResult:
    diagonal: List[np.ndarray]
    upper: List[np.ndarray]
    lower: List[np.ndarray]
    block_sizes: Tuple[int, ...]
    full_inverse: np.ndarray | None = None


@dataclass(slots=True)
class _Segment:
    start: int
    end: int
    block_sizes: Tuple[int, ...]
    offsets: np.ndarray
    inverse: np.ndarray

    @property
    def dof(self) -> int:
        return int(self.inverse.shape[0])


def _to_ndarray(block: ArrayLike | sp.spmatrix) -> np.ndarray:
    if sp.issparse(block):
        return block.toarray()
    return np.asarray(block)


def _as_square(block: ArrayLike | sp.spmatrix, name: str) -> np.ndarray:
    arr = _to_ndarray(block).astype(np.complex128, copy=False)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D array; got shape {arr.shape}.")
    return arr


def _assemble_local_matrix(
    diag_blocks: Sequence[ArrayLike | sp.spmatrix],
    upper_blocks: Sequence[ArrayLike | sp.spmatrix],
    lower_blocks: Sequence[ArrayLike | sp.spmatrix],
) -> tuple[np.ndarray, Tuple[int, ...], np.ndarray]:
    n_blocks = len(diag_blocks)
    if len(upper_blocks) != max(0, n_blocks - 1) or len(lower_blocks) != max(0, n_blocks - 1):
        raise ValueError("Upper and lower block lists must have length len(diag_blocks) - 1.")

    diag = [_as_square(block, "diagonal block") for block in diag_blocks]
    sizes = tuple(block.shape[0] for block in diag)
    offsets = np.cumsum((0, *sizes))
    total = int(offsets[-1])
    matrix = np.zeros((total, total), dtype=np.complex128)

    for i, block in enumerate(diag):
        s = offsets[i]
        e = offsets[i + 1]
        matrix[s:e, s:e] = block
        if i < n_blocks - 1:
            upper = _to_ndarray(upper_blocks[i]).astype(np.complex128, copy=False)
            lower = _to_ndarray(lower_blocks[i]).astype(np.complex128, copy=False)
            next_size = sizes[i + 1]
            if upper.shape != (sizes[i], next_size):
                raise ValueError(
                    f"Upper coupling at index {i} has shape {upper.shape}, expected {(sizes[i], next_size)}."
                )
            if lower.shape != (next_size, sizes[i]):
                raise ValueError(
                    f"Lower coupling at index {i} has shape {lower.shape}, expected {(next_size, sizes[i])}."
                )
            s_next = offsets[i + 1]
            e_next = offsets[i + 2]
            matrix[s:e, s_next:e_next] = upper
            matrix[s_next:e_next, s:e] = lower
    return matrix, sizes, offsets


def _make_segment(
    start: int,
    diag_blocks: Sequence[ArrayLike | sp.spmatrix],
    upper_blocks: Sequence[ArrayLike | sp.spmatrix],
    lower_blocks: Sequence[ArrayLike | sp.spmatrix],
) -> _Segment:
    local_matrix, block_sizes, offsets = _assemble_local_matrix(diag_blocks, upper_blocks, lower_blocks)
    inverse = np.linalg.inv(local_matrix)
    end = start + len(block_sizes) - 1
    return _Segment(start=start, end=end, block_sizes=block_sizes, offsets=offsets, inverse=inverse)


def _segment_worker(payload: tuple[int, Sequence[ArrayLike | sp.spmatrix], Sequence[ArrayLike | sp.spmatrix], Sequence[ArrayLike | sp.spmatrix]]) -> _Segment:
    start, diag_blocks, upper_blocks, lower_blocks = payload
    return _make_segment(start, diag_blocks, upper_blocks, lower_blocks)


def _make_partition(num_blocks: int, processes: int | None) -> List[Tuple[int, int]]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if processes is None or processes <= 0:
        processes = num_blocks
    processes = min(processes, num_blocks)
    base = num_blocks // processes
    remainder = num_blocks % processes
    partition: List[Tuple[int, int]] = []
    start = 0
    for pid in range(processes):
        size = base + (1 if pid < remainder else 0)
        if size == 0:
            continue
        end = start + size - 1
        partition.append((start, end))
        start = end + 1
    if start != num_blocks:
        raise RuntimeError("Partitioning error: did not cover all blocks.")
    return partition


def _embed_coupling(
    left: _Segment,
    right: _Segment,
    upper: ArrayLike | sp.spmatrix,
    lower: ArrayLike | sp.spmatrix,
) -> tuple[np.ndarray, np.ndarray]:
    upper = _to_ndarray(upper).astype(np.complex128, copy=False)
    lower = _to_ndarray(lower).astype(np.complex128, copy=False)
    n_left = left.dof
    n_right = right.dof
    full_upper = np.zeros((n_left, n_right), dtype=np.complex128)
    full_lower = np.zeros((n_right, n_left), dtype=np.complex128)
    left_rows = slice(int(left.offsets[-2]), int(left.offsets[-1]))
    right_cols = slice(0, int(right.block_sizes[0]))
    full_upper[left_rows, right_cols] = upper
    full_lower[right_cols, left_rows] = lower
    return full_upper, full_lower


def _combine_segments(left: _Segment, right: _Segment, upper: ArrayLike | sp.spmatrix, lower: ArrayLike | sp.spmatrix) -> _Segment:
    full_upper, full_lower = _embed_coupling(left, right, upper, lower)
    temp = left.inverse @ full_upper
    middle = np.eye(right.dof, dtype=np.complex128) - right.inverse @ full_lower @ temp
    updated_right = np.linalg.solve(middle, right.inverse)
    left_update = temp @ updated_right @ full_lower @ left.inverse
    block_ll = left.inverse + left_update
    block_lr = -temp @ updated_right
    block_rl = -updated_right @ full_lower @ left.inverse
    block_rr = updated_right
    combined_size = left.dof + right.dof
    combined = np.zeros((combined_size, combined_size), dtype=np.complex128)
    combined[: left.dof, : left.dof] = block_ll
    combined[: left.dof, left.dof :] = block_lr
    combined[left.dof :, : left.dof] = block_rl
    combined[left.dof :, left.dof :] = block_rr
    block_sizes = (*left.block_sizes, *right.block_sizes)
    offsets = np.cumsum((0, *block_sizes))
    return _Segment(start=left.start, end=right.end, block_sizes=block_sizes, offsets=offsets, inverse=combined)


def _extract_tridiagonal(inverse: np.ndarray, block_sizes: Tuple[int, ...]) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    offsets = np.cumsum((0, *block_sizes))
    diagonal: List[np.ndarray] = []
    upper: List[np.ndarray] = []
    lower: List[np.ndarray] = []
    for idx in range(len(block_sizes)):
        s = offsets[idx]
        e = offsets[idx + 1]
        diagonal.append(inverse[s:e, s:e].copy())
        if idx < len(block_sizes) - 1:
            s_next = offsets[idx + 1]
            e_next = offsets[idx + 2]
            upper.append(inverse[s:e, s_next:e_next].copy())
            lower.append(inverse[s_next:e_next, s:e].copy())
    return diagonal, upper, lower


def pairwise_partial_inverse(
    diagonal_blocks: Sequence[np.ndarray],
    upper_blocks: Sequence[np.ndarray],
    lower_blocks: Sequence[np.ndarray],
    *,
    processes: int | None = None,
    return_full: bool = False,
) -> PairwiseInverseResult:
    """Compute block-tridiagonal elements of the inverse via pairwise merging.

    Parameters
    ----------
    diagonal_blocks:
        Sequence of square diagonal blocks defining the block tridiagonal matrix.
    upper_blocks:
        Sequence of couplings between block i and i+1 (same length as ``diagonal_blocks`` minus one).
    lower_blocks:
        Sequence of couplings between block i+1 and i.
    processes:
        Number of virtual processes used for the pairwise merging schedule.
    return_full:
        If True, include the dense inverse of the full matrix in the result.
    """
    num_blocks = len(diagonal_blocks)
    if num_blocks == 0:
        raise ValueError("At least one diagonal block is required.")
    if len(upper_blocks) != max(0, num_blocks - 1) or len(lower_blocks) != max(0, num_blocks - 1):
        raise ValueError("upper_blocks and lower_blocks must have length len(diagonal_blocks) - 1.")

    partition = _make_partition(num_blocks, processes)
    segments: List[_Segment] = []
    if processes is not None and processes > 1:
        ctx = mp.get_context("fork")
        tasks = []
        for start, end in partition:
            diag_slice = diagonal_blocks[start : end + 1]
            upper_slice = upper_blocks[start:end]
            lower_slice = lower_blocks[start:end]
            tasks.append((start, diag_slice, upper_slice, lower_slice))
        with ctx.Pool(processes=processes) as pool:
            segments = pool.map(_segment_worker, tasks)
    else:
        for start, end in partition:
            diag_slice = diagonal_blocks[start : end + 1]
            upper_slice = upper_blocks[start:end]
            lower_slice = lower_blocks[start:end]
            segments.append(_make_segment(start, diag_slice, upper_slice, lower_slice))

    while len(segments) > 1:
        next_level: List[_Segment] = []
        it = iter(segments)
        for left in it:
            try:
                right = next(it)
            except StopIteration:
                next_level.append(left)
                break
            coupling_idx = left.end
            combined = _combine_segments(left, right, upper_blocks[coupling_idx], lower_blocks[coupling_idx])
            next_level.append(combined)
        segments = next_level

    final_segment = segments[0]
    diagonal, upper, lower = _extract_tridiagonal(final_segment.inverse, final_segment.block_sizes)
    full_inverse = final_segment.inverse.copy() if return_full else None
    return PairwiseInverseResult(
        diagonal=diagonal,
        upper=upper,
        lower=lower,
        block_sizes=final_segment.block_sizes,
        full_inverse=full_inverse,
    )


def build_block_tridiagonal_matrix(
    diagonal_blocks: Sequence[np.ndarray],
    upper_blocks: Sequence[np.ndarray],
    lower_blocks: Sequence[np.ndarray],
) -> np.ndarray:
    matrix, _, _ = _assemble_local_matrix(diagonal_blocks, upper_blocks, lower_blocks)
    return matrix
