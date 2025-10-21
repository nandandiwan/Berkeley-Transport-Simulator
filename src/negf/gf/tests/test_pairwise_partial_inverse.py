from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from negf.gf.general_rgf.pairwise_partial_inverse import (
    build_block_tridiagonal_matrix,
    pairwise_partial_inverse,
)


def _random_blocks(block_sizes: list[int], seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    diagonal: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    lower: list[np.ndarray] = []
    for size in block_sizes:
        base = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        diagonal.append(base + (size + 1.0) * np.eye(size))
    for idx in range(len(block_sizes) - 1):
        rows = block_sizes[idx]
        cols = block_sizes[idx + 1]
        scale = 0.25
        upper.append(scale * (rng.normal(size=(rows, cols)) + 1j * rng.normal(size=(rows, cols))))
        lower.append(scale * (rng.normal(size=(cols, rows)) + 1j * rng.normal(size=(cols, rows))))
    return diagonal, upper, lower


def _extract_offsets(block_sizes: list[int]) -> np.ndarray:
    return np.cumsum([0, *block_sizes])


def _run_case(name: str, func) -> None:
    print(f"\n{name}")
    try:
        details = func()
    except AssertionError as exc:
        print(f"  FAILED: {exc}")
        raise
    else:
        if details:
            for line in details:
                print(f"  {line}")
        print("  PASSED")


def _case_pairwise_matches_direct_inverse() -> list[str]:
    block_sizes = [2, 3, 1, 2]
    diag, upper, lower = _random_blocks(block_sizes, seed=4)
    full = build_block_tridiagonal_matrix(diag, upper, lower)
    full_inv = np.linalg.inv(full)
    result = pairwise_partial_inverse(diag, upper, lower, processes=2, return_full=True)
    assert result.full_inverse is not None
    assert_allclose(result.full_inverse, full_inv, atol=1e-10, rtol=1e-10)
    diff = result.full_inverse - full_inv
    max_abs = float(np.max(np.abs(diff)))
    denom = np.maximum(np.abs(full_inv), 1e-14)
    max_rel = float(np.max(np.abs(diff) / denom))
    return [f"max |Δ| = {max_abs:.2e}", f"max rel |Δ| = {max_rel:.2e}"]


def _case_pairwise_block_entries_match_full_inverse() -> list[str]:
    summaries: list[str] = []
    for processes in [1, 2, 3, 5, 8]:
        block_sizes = [1, 2, 3, 2]
        diag, upper, lower = _random_blocks(block_sizes, seed=processes)
        full = build_block_tridiagonal_matrix(diag, upper, lower)
        full_inv = np.linalg.inv(full)
        result = pairwise_partial_inverse(diag, upper, lower, processes=processes)
        offsets = _extract_offsets(block_sizes)
        diag_err = 0.0
        upper_err = 0.0
        lower_err = 0.0
        for idx, block in enumerate(result.diagonal):
            s, e = offsets[idx], offsets[idx + 1]
            target = full_inv[s:e, s:e]
            diff = block - target
            diag_err = max(diag_err, float(np.max(np.abs(diff))))
            assert_allclose(block, target, atol=1e-10, rtol=1e-10)
        for idx, block in enumerate(result.upper):
            s, e = offsets[idx], offsets[idx + 1]
            s_next, e_next = offsets[idx + 1], offsets[idx + 2]
            target = full_inv[s:e, s_next:e_next]
            diff = block - target
            upper_err = max(upper_err, float(np.max(np.abs(diff))))
            assert_allclose(block, target, atol=1e-10, rtol=1e-10)
        for idx, block in enumerate(result.lower):
            s, e = offsets[idx], offsets[idx + 1]
            s_next, e_next = offsets[idx + 1], offsets[idx + 2]
            target = full_inv[s_next:e_next, s:e]
            diff = block - target
            lower_err = max(lower_err, float(np.max(np.abs(diff))))
            assert_allclose(block, target, atol=1e-10, rtol=1e-10)
        summaries.append(
            f"processes={processes}: max diag |Δ|={diag_err:.2e}, upper |Δ|={upper_err:.2e}, lower |Δ|={lower_err:.2e}"
        )
    return summaries


def _case_pairwise_handles_variable_block_sizes() -> list[str]:
    block_sizes = [3, 2, 4, 1, 3]
    diag, upper, lower = _random_blocks(block_sizes, seed=21)
    full = build_block_tridiagonal_matrix(diag, upper, lower)
    full_inv = np.linalg.inv(full)
    result = pairwise_partial_inverse(diag, upper, lower, processes=3, return_full=False)
    offsets = _extract_offsets(block_sizes)
    diag_err = 0.0
    for idx, block in enumerate(result.diagonal):
        s, e = offsets[idx], offsets[idx + 1]
        target = full_inv[s:e, s:e]
        diff = block - target
        diag_err = max(diag_err, float(np.max(np.abs(diff))))
        assert_allclose(block, target, atol=1e-10, rtol=1e-10)
    return [f"max diag |Δ| = {diag_err:.2e}"]


def _case_sparse_inputs_match_dense() -> list[str]:
    block_sizes = [2, 3, 2]
    diag, upper, lower = _random_blocks(block_sizes, seed=15)
    diag_sparse = [sp.csc_matrix(block) for block in diag]
    upper_sparse = [sp.csc_matrix(block) for block in upper]
    lower_sparse = [sp.csc_matrix(block) for block in lower]
    dense_result = pairwise_partial_inverse(diag, upper, lower, processes=2)
    sparse_result = pairwise_partial_inverse(diag_sparse, upper_sparse, lower_sparse, processes=2)
    offsets = _extract_offsets(block_sizes)
    summaries: list[str] = []
    for idx, block in enumerate(dense_result.diagonal):
        sparse_block = sparse_result.diagonal[idx]
        diff = sparse_block - block
        max_err = float(np.max(np.abs(diff)))
        summaries.append(f"block {idx}: max |Δ|={max_err:.2e}")
        assert_allclose(sparse_block, block, atol=1e-10, rtol=1e-10)
    return summaries


def main() -> None:
    print("Pairwise partial inverse validation")
    _run_case("Full inverse matches dense reference", _case_pairwise_matches_direct_inverse)
    _run_case("Block entries match dense inverse across process counts", _case_pairwise_block_entries_match_full_inverse)
    _run_case("Variable block sizes are handled", _case_pairwise_handles_variable_block_sizes)
    _run_case("Sparse inputs produce identical results", _case_sparse_inputs_match_dense)


if __name__ == "__main__":
    main()

