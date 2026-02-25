import copy
from typing import List

import numpy as np
import scipy.sparse as sp

try:  # Optional C++ acceleration for recursive inverse
    import cppimport  # type: ignore

    _cpp_recursive = cppimport.imp("negf.gf.recursive_inverse_ext")
except Exception:  # pragma: no cover - fallback to Python implementation
    _cpp_recursive = None
import scipy.constants as spc
#from negf.self_energy.surface import surface_greens_function
from negf.self_energy.surface import surface_greens_function
from negf.self_energy.greens_functions import surface_greens_function_nn
from negf.utils.common import fermi_dirac, smart_inverse, FD_minus_half
from negf.utils.block_partition import compute_block_sizes_block_tridiagonal
import scipy.linalg as linalg
from negf.gf.general_rgf.pairwise_partial_inverse import pairwise_partial_inverse

def mat_left_div(mat_a, mat_b):
    ans, resid, rank, s = linalg.lstsq(mat_a, mat_b, lapack_driver='gelsy')
    return ans


def _merge_blocks_for_edge(block_sizes: np.ndarray, target_dim: int, *, left: bool) -> np.ndarray:
    """Merge contiguous blocks so the edge block matches the lead size."""

    if target_dim <= 0:
        raise ValueError("Lead block dimension must be positive.")

    blocks = np.asarray(block_sizes, dtype=int)
    if blocks.ndim != 1 or blocks.size == 0:
        raise ValueError("block_sizes must be a 1-D array with at least one entry.")

    if left:
        total = 0
        idx = 0
        while idx < blocks.size and total < target_dim:
            total += int(blocks[idx])
            idx += 1
        if total != target_dim:
            raise ValueError("Unable to align block partition with left lead size.")
        if idx >= blocks.size:
            return np.array([target_dim], dtype=int)
        return np.concatenate((np.array([target_dim], dtype=int), blocks[idx:]))

    total = 0
    idx = blocks.size
    while idx > 0 and total < target_dim:
        idx -= 1
        total += int(blocks[idx])
    if total != target_dim:
        raise ValueError("Unable to align block partition with right lead size.")
    if idx <= 0:
        return np.array([target_dim], dtype=int)
    return np.concatenate((blocks[:idx], np.array([target_dim], dtype=int)))


def _recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, s_in=0, s_out=0, damp=0.000001j):
    for jj,item in enumerate(mat_d_list):
        mat_d_list[jj]=item - np.diag(energy*np.ones(mat_d_list[jj].shape[0]) + 1j*damp)
    num_of_matrices=len(mat_d_list)
    mat_shapes=[item.shape for item in mat_d_list]
    gr_left=[None for _ in range(num_of_matrices)]
    gr_left[0]=mat_left_div(-mat_d_list[0], np.eye(mat_shapes[0][0]))
    for q in range(num_of_matrices-1):
        gr_left[q+1]=mat_left_div((-mat_d_list[q+1]-mat_l_list[q].dot(gr_left[q]).dot(mat_u_list[q])), np.eye(mat_shapes[q+1][0]))
    grl=[None for _ in range(num_of_matrices-1)]
    gru=[None for _ in range(num_of_matrices-1)]
    grd=copy.copy(gr_left)
    g_trans=copy.copy(gr_left[len(gr_left)-1])
    for q in range(num_of_matrices-2,-1,-1):
        grl[q]=grd[q+1].dot(mat_l_list[q]).dot(gr_left[q])
        gru[q]=gr_left[q].dot(mat_u_list[q]).dot(grd[q+1])
        grd[q]=gr_left[q]+gr_left[q].dot(mat_u_list[q]).dot(grl[q])
        g_trans=gr_left[q].dot(mat_u_list[q]).dot(g_trans)
    for jj,item in enumerate(mat_d_list):
        mat_d_list[jj]=mat_d_list[jj]+np.diag(energy*np.ones(mat_d_list[jj].shape[0]) + 1j*damp)
    return g_trans, grd, grl, gru, gr_left

def recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, left_se=None, right_se=None, s_in=0, s_out=0, damp=0.000001j):
    if isinstance(left_se, np.ndarray):
        s01,s02=mat_d_list[0].shape; left_se=left_se[:s01,:s02]; mat_d_list[0]=mat_d_list[0]+left_se
    if isinstance(right_se, np.ndarray):
        s11,s12=mat_d_list[-1].shape; right_se=right_se[-s11:,-s12:]; mat_d_list[-1]=mat_d_list[-1]+right_se
    ans=_recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, s_in=s_in, s_out=s_out, damp=damp)
    if isinstance(left_se, np.ndarray):
        mat_d_list[0]=mat_d_list[0]-left_se
    if isinstance(right_se, np.ndarray):
        mat_d_list[-1]=mat_d_list[-1]-right_se
    return ans

def add_eta(E):
    return E + 1e-9j

def gf_inverse( # just directly all individual GF 
    E,
    H,
    H00,
    H01,
    mu1=0,
    mu2=0,
    block_size=None,
    block_size_list=None,
    method: str = "recursive",
    processes=1,
    H00_right=None,
    H01_right=None,
    occupation_mode: str = "fermi_dirac",
    occupation_prefactor: float | None = None,
    occupation_kbT: float | None = None,
    self_energy_damp: float = 1e-9,
    s_l: np.ndarray | None = None,
    s_r: np.ndarray | None = None,
    s_z: np.ndarray | None = None,
    s_l_right: np.ndarray | None = None,
    s_r_right: np.ndarray | None = None,
    s_z_right: np.ndarray | None = None,
):
    
    def _occupation_factor(energy, mu, mode, prefactor, kbT):
        if mode == "fermi_dirac":
            return fermi_dirac(energy, mu)
        if mode == "fd_half":
            if kbT is None or prefactor is None:
                raise ValueError("fd_half occupation requires kbT and prefactor")
            arg = -spc.elementary_charge * (np.real(energy) - mu) / kbT
            return prefactor * FD_minus_half(arg)
        raise ValueError(f"Unknown occupation mode '{mode}'")

    if H00_right is None:
        H00_right = H00
    if H01_right is None:
        H01_right = H01

    damp_complex = 1j * abs(float(self_energy_damp)) if self_energy_damp is not None else 1e-7j

    if H01 is None:
        raise ValueError("H01 must be provided to compute lead self-energies.")

    block_dim = H01.shape[0]
    s_l_arr = None if s_l is None else np.asarray(s_l, dtype=np.complex128)
    s_r_arr = None if s_r is None else np.asarray(s_r, dtype=np.complex128)

    if s_l_arr is None and s_r_arr is not None:
        s_l_arr = s_r_arr.conj().T
    if s_r_arr is None and s_l_arr is not None:
        s_r_arr = s_l_arr.conj().T

    s_l_right_arr = (
        np.asarray(s_l_right, dtype=np.complex128)
        if s_l_right is not None
        else (s_r_arr.conj().T if s_r_arr is not None else None)
    )
    s_r_right_arr = (
        np.asarray(s_r_right, dtype=np.complex128)
        if s_r_right is not None
        else (s_l_arr.conj().T if s_l_arr is not None else None)
    )

    s_z_full = None if s_z is None else np.asarray(s_z, dtype=np.complex128)
    s_z_right_full = (
        np.asarray(s_z_right, dtype=np.complex128)
        if s_z_right is not None
        else (s_z_full.copy() if s_z_full is not None else None)
    )

    s0_left = None
    s0_right = None
    if s_z_full is not None:
        if s_z_full.shape[0] == block_dim:
            s0_left = s_z_full
        elif s_z_full.shape[0] == H.shape[0]:
            s0_left = s_z_full[:block_dim, :block_dim]
        else:
            raise ValueError("s_z must be either block_dim x block_dim or match H dimensions")
    if s_z_right_full is not None:
        if s_z_right_full.shape[0] == block_dim:
            s0_right = s_z_right_full
        elif s_z_right_full.shape[0] == H.shape[0]:
            s0_right = s_z_right_full[-block_dim:, -block_dim:]
        else:
            raise ValueError("s_z_right must be either block_dim x block_dim or match H dimensions")

    Sigma_L_raw = surface_greens_function_nn(
        E - mu1,
        H01,
        H00,
        H01.conj().T,
        damp=damp_complex,
        s_l=s_r_arr,
        s_0=s0_left,
        s_r=s_l_arr,
    )
    
    if isinstance(Sigma_L_raw, tuple):
        # surface_greens_function_nn returns (sgf_r, sgf_l); select left component
        Sigma_L = Sigma_L_raw[1]
    else:
        Sigma_L = Sigma_L_raw
    
    Sigma_R_raw = surface_greens_function_nn(
        E - mu2,
        H01_right,
        H00_right,
        H01_right.conj().T,
        damp=damp_complex,
        s_l=s_r_right_arr,
        s_0=s0_right,
        s_r=s_l_right_arr,
    )
   
    if isinstance(Sigma_R_raw, tuple):
        # First tuple entry corresponds to the right-lead self-energy
        Sigma_R = Sigma_R_raw[0]
    else:
        Sigma_R = Sigma_R_raw
    #print(Sigma_L, Sigma_R)
    occ_left = _occupation_factor(E, mu1, occupation_mode, occupation_prefactor, occupation_kbT)
    occ_right = _occupation_factor(E, mu2, occupation_mode, occupation_prefactor, occupation_kbT)

    
    overlap_full = s_z_full if (s_z_full is not None and s_z_full.shape == H.shape) else None

    if method == "direct":
        return _direct_inverse(
            E,
            H,
            Sigma_L,
            Sigma_R,
            H01,
            mu1,
            mu2,
            block_size=block_size,
            block_size_list=block_size_list,
            occ_left=occ_left,
            occ_right=occ_right,
            overlap_matrix=overlap_full,
        )  
    if method == "recursive":
        return _recursive_inverse(
            E,
            H,
            H00,
            H01,
            Sigma_L,
            Sigma_R,
            mu1,
            mu2,
            block_size,
            occ_left=occ_left,
            occ_right=occ_right,
            overlap_matrix=overlap_full,
        )
    if method == "var_recursive":
        if block_size_list is None:
            try:
                block_sizes_var = compute_block_sizes_block_tridiagonal(H)
            except Exception as exc:
                raise ValueError(
                    "Unable to infer block_size_list automatically for method='var_recursive'."
                ) from exc
        else:
            block_sizes_var = np.asarray(block_size_list, dtype=int)

        block_sizes_var = np.asarray(block_sizes_var, dtype=int)
        if block_sizes_var.size == 0:
            raise ValueError("block_size_list must contain at least one block.")

        try:
            block_sizes_var = _merge_blocks_for_edge(block_sizes_var, Sigma_L.shape[0], left=True)
            block_sizes_var = _merge_blocks_for_edge(block_sizes_var, Sigma_R.shape[0], left=False)
        except ValueError as exc:
            raise ValueError(
                "Automatic block partition could not accommodate lead principal-layer dimensions; "
                "provide block_size_list explicitly."
            ) from exc

        return _variable_recursive_inverse(
            E,
            H,
            Sigma_L,
            Sigma_R,
            block_sizes_var,
            mu1,
            mu2,
            processes,
            overlap_matrix=overlap_full,
            return_full_inverse=False,
        )
    raise ValueError(f"Unknown method '{method}'. Use 'direct', 'recursive', or 'var_recursive'.")

def _direct_inverse(
    E,
    H,
    Sigma_L,
    Sigma_R,
    occ_left=0.0,
    occ_right=0.0,
    overlap_matrix: np.ndarray | None = None,
    *,
    eta: float = 1e-6,
    return_trace: bool = False,
):
    """Direct inversion identical in spirit to the working notebook snippet.

    Returns:
        G_R_diag : 1D ndarray (diagonal of full retarded Green's function)
        G_lesser_diag : 1D ndarray (diagonal of lesser GF under equilibrium two-terminal approximation)
        Gamma_L, Gamma_R : full broadening matrices (dense np.ndarray)
    """
    # Convert inputs to dense arrays matching notebook behavior
    H_dense = H.toarray() if sp.issparse(H) else np.asarray(H, dtype=complex)
    Sigma_L_dense = Sigma_L.toarray() if sp.issparse(Sigma_L) else np.asarray(Sigma_L, dtype=complex)
    Sigma_R_dense = Sigma_R.toarray() if sp.issparse(Sigma_R) else np.asarray(Sigma_R, dtype=complex)

    n = H_dense.shape[0]
    m = Sigma_L_dense.shape[0]

    # Embed into device corners
    Sigma_L_full = np.zeros((n, n), dtype=complex)
    Sigma_R_full = np.zeros((n, n), dtype=complex)
    Sigma_L_full[:m, :m] = Sigma_L_dense
    Sigma_R_full[-m:, -m:] = Sigma_R_dense

    if overlap_matrix is None:
        S_dense = np.eye(n, dtype=complex)
    else:
        S_dense = overlap_matrix if isinstance(overlap_matrix, np.ndarray) else np.asarray(overlap_matrix, dtype=complex)
        if S_dense.shape != (n, n):
            raise ValueError("overlap_matrix must match Hamiltonian dimensions in direct inversion.")

    # Build and solve for full G^R
    z = E + 1j * abs(float(eta))
    A = z * S_dense - (H_dense + Sigma_L_full + Sigma_R_full)
    G_R = np.linalg.solve(A, np.eye(n, dtype=complex))

    # Broadenings
    Gamma_L = 1j * (Sigma_L_full - Sigma_L_full.conj().T)
    Gamma_R = 1j * (Sigma_R_full - Sigma_R_full.conj().T)

    # Lesser
    Sigma_lesser = 1j * (Gamma_L * occ_left + Gamma_R * occ_right)
    G_A = G_R.conj().T
    G_lesser = G_R @ Sigma_lesser @ G_A

    block_size = Sigma_L.shape[0]
    block_sizes = np.full(n // block_size, int(block_size), dtype=int)

    offsets = np.cumsum(np.concatenate(([0], block_sizes)))
    # if offsets[-1] != n:
    #     raise ValueError("Sum of block sizes must match Hamiltonian dimension in direct inversion.")

    G_lesser_offdiag_right = []
    for idx in range(len(block_sizes) - 1):
        s = offsets[idx]
        e = offsets[idx + 1]
        s_next = offsets[idx + 1]
        e_next = offsets[idx + 2]
        G_block = G_lesser[s_next:e_next, s:e]
        G_lesser_offdiag_right.append(G_block)

    G_lesser_diag = np.diag(G_lesser)

    trace_gs = None
    if return_trace:
        trace_gs = np.trace(G_R @ S_dense)

    if return_trace:
        return G_R, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R, trace_gs
    return G_R, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R


def _recursive_inverse(
    E,
    H,
    Sigma_L,
    Sigma_R,
    compute_lesser=True,
    occ_left=0.0,
    occ_right=0.0,
    overlap_matrix: np.ndarray | None = None,
    eta: float = 1e-6,
    return_trace: bool = False,
    return_diag: bool = True,
    return_gamma: bool = True,
    return_g_trans: bool = False,
):
    # Attempt C++ accelerated path when available (cannot return g_trans there)
    if _cpp_recursive is not None and not return_g_trans:
        H_dense = H.toarray() if sp.issparse(H) else np.asarray(H, dtype=complex)
        Sigma_L_arr = np.asarray(Sigma_L, dtype=complex)
        Sigma_R_arr = np.asarray(Sigma_R, dtype=complex)
        overlap_obj = None if overlap_matrix is None else np.asarray(overlap_matrix, dtype=complex)
        res = _cpp_recursive.recursive_inverse_cpp(
            float(E),
            H_dense,
            Sigma_L_arr,
            Sigma_R_arr,
            overlap_obj if overlap_obj is not None else None,
            float(eta),
            compute_lesser,
            return_trace,
        )
        G_R_diag = np.asarray(res[0]).reshape(-1)
        G_lesser_diag = np.asarray(res[1]).reshape(-1)
        G_lesser_offdiag_right = [np.asarray(x) for x in res[2]]
        Gamma_L = np.asarray(res[3])
        Gamma_R = np.asarray(res[4])
        trace_gs = res[5] if return_trace else None
        if return_trace:
            return G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R, trace_gs
        return G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R

    if sp.issparse(H):
        H = H.toarray()
    n = H.shape[0]

    if overlap_matrix is None:
        S_full = np.eye(n, dtype=complex)
    else:
        S_full = np.asarray(overlap_matrix, dtype=complex)
        if S_full.shape != (n, n):
            raise ValueError("overlap_matrix must match Hamiltonian dimensions for recursive inverse.")

    block_size = Sigma_L.shape[0]
    if n % block_size != 0:
        raise ValueError("H dimension not divisible by block_size.")
    n_blocks = n // block_size
    E_eta = E + 1j * abs(float(eta))
    dagger = lambda A: np.conjugate(A.T)

    # Slice blocks
    A_ii = []
    A_ij = []
    for i in range(n_blocks):
        sl = slice(i * block_size, (i + 1) * block_size)
        sr = slice((i + 1) * block_size, (i + 2) * block_size)
        H_block = H[sl, sl].copy()
        S_block = S_full[sl, sl].copy()
        diag_block = E_eta * S_block - H_block
        A_ii.append(diag_block)
        if i < n_blocks - 1:
            H_cpl = H[sl, sr].copy()
            S_cpl = S_full[sl, sr].copy()
            A_ij.append(E_eta * S_cpl - H_cpl)

    Gamma_L = 1j * (Sigma_L - Sigma_L.conj().T) if return_gamma else None
    Gamma_R = 1j * (Sigma_R - Sigma_R.conj().T) if return_gamma else None
    if compute_lesser:
        if Gamma_L is None or Gamma_R is None:
            raise ValueError("compute_lesser=True requires return_gamma=True")
        Sigma_L_lesser = Gamma_L * occ_left
        Sigma_R_lesser = Gamma_R * occ_right

    g_R = []
    g_lesser = [] if compute_lesser else None

    # Forward sweep
    A00_eff = A_ii[0] - Sigma_L
    g0_R = smart_inverse(A00_eff)
    g_R.append(g0_R)

    if compute_lesser:
        g0_lesser = g0_R @ Sigma_L_lesser @ dagger(g0_R)
        assert g_lesser is not None
        g_lesser.append(g0_lesser)

    for i in range(1, n_blocks):
        A_i_im1 = dagger(A_ij[i - 1])
        sigma_recursive_R = A_i_im1 @ g_R[i - 1] @ A_ij[i - 1]
        g_i_R = smart_inverse(A_ii[i] - sigma_recursive_R)
        g_R.append(g_i_R)
        if compute_lesser:
            assert g_lesser is not None
            sigma_recursive_lesser = A_i_im1 @ g_lesser[i - 1] @ A_ij[i - 1]
            g_i_lesser = g_R[i] @ sigma_recursive_lesser @ dagger(g_R[i])
            g_lesser.append(g_i_lesser)

    G_R = [None] * n_blocks
    G_lesser = [None] * n_blocks if compute_lesser else None
    G_lesser_offdiag_right = [None] * (n_blocks - 1) if compute_lesser else None

    # Optional end-to-end retarded Green's function block (leftmost -> rightmost)
    # This is useful for transmission: T = Tr[Γ_L G_{1N} Γ_R G_{1N}^†]
    G_to_last = None

    # Backward sweep
    A_N_Nm1 = dagger(A_ij[-1])
    sigma_eff_R = A_N_Nm1 @ g_R[-2] @ A_ij[-1]
    GN_R = smart_inverse(A_ii[-1] - Sigma_R - sigma_eff_R)
    G_R[-1] = GN_R

    if return_g_trans:
        G_to_last = GN_R

    if compute_lesser:
        sigma_eff_lesser = A_N_Nm1 @ g_lesser[-2] @ A_ij[-1]
        total_sigma_lesser = Sigma_R_lesser + sigma_eff_lesser
        GN_lesser = G_R[-1] @ total_sigma_lesser @ dagger(G_R[-1])
        assert G_lesser is not None
        G_lesser[-1] = GN_lesser

    for i in range(n_blocks - 2, -1, -1):
        A_i_ip1 = A_ij[i]
        A_ip1_i = dagger(A_i_ip1)
        propagator = g_R[i] @ A_i_ip1 @ G_R[i + 1] @ A_ip1_i
        G_R[i] = g_R[i] + propagator @ g_R[i]

        if return_g_trans:
            assert G_to_last is not None
            # Off-diagonal block recurrence for a block-tridiagonal inverse.
            # For a 2x2 block system, G_{12} = -G_{11} A_{12} G_{22}.
            # Extending along the chain gives the end-to-end block by repeated application.
            G_to_last = -G_R[i] @ A_i_ip1 @ G_to_last
        if compute_lesser:
            assert G_lesser is not None
            assert G_lesser_offdiag_right is not None
            g_R_dag = dagger(g_R[i])
            assert g_lesser is not None
            term1 = g_lesser[i]
            term2 = propagator @ g_lesser[i]
            term3 = g_lesser[i] @ dagger(propagator)
            term4 = g_R[i] @ A_i_ip1 @ G_lesser[i + 1] @ A_ip1_i @ g_R_dag
            G_lesser[i] = term1 + term2 + term3 + term4

            G_ip1_A = dagger(G_R[i + 1])
            off_term_R = G_R[i] @ A_i_ip1 @ G_lesser[i + 1]
            off_term_L = G_lesser[i] @ A_i_ip1 @ G_ip1_A
            G_lesser_offdiag_right[i] = off_term_R + off_term_L

    G_R_diag = None
    if return_diag:
        G_R_diag = np.concatenate([np.diag(block) for block in G_R])

    G_lesser_diag = None
    if compute_lesser:
        assert G_lesser is not None
        G_lesser_diag = np.concatenate([np.diag(block) for block in G_lesser])

    trace_gs = None
    if return_trace:
        # Assemble trace(G·S) blockwise
        trace_val = 0.0 + 0.0j
        for idx, block in enumerate(G_R):
            s = idx * block_size
            e = (idx + 1) * block_size
            trace_val += np.trace(block @ S_full[s:e, s:e])
            if idx < n_blocks - 1:
                s_next = (idx + 1) * block_size
                e_next = (idx + 2) * block_size
                trace_val += np.trace(G_R[idx] @ (S_full[s:e, s_next:e_next]))
                trace_val += np.trace(G_R[idx + 1] @ (S_full[s_next:e_next, s:e]))
        trace_gs = trace_val

    g_trans = G_to_last if return_g_trans else None

    if return_trace:
        if return_g_trans:
            return G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R, trace_gs, g_trans
        return G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R, trace_gs
    if return_g_trans:
        return G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R, g_trans
    return G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R 

def _variable_recursive_inverse(
    E,
    H,
    Sigma_L,
    Sigma_R,
    block_size_list,
    processes,
    *,
    overlap_matrix: np.ndarray | None = None,
    return_full_inverse: bool = False,
    return_blocks: bool = True,
    return_diag: bool = True,
    return_gamma: bool = True,
    eta: float = 1e-6,
):
    if block_size_list is None:
        raise ValueError("block_size_list must be provided for method='var_recursive'.")

    block_sizes = np.asarray(block_size_list, dtype=int).astype(int)
    if block_sizes.ndim != 1:
        raise ValueError("block_size_list must be a 1D array of block sizes.")
    if np.any(block_sizes <= 0):
        raise ValueError("block_size_list entries must be positive integers.")

    if sp.issparse(H):
        H = H.toarray()


    n = H.shape[0]
    if overlap_matrix is None:
        S_full = None
    else:
        S_full = np.asarray(overlap_matrix, dtype=complex)
        if S_full.shape != (n, n):
            raise ValueError("overlap_matrix must match Hamiltonian dimensions for var_recursive.")

    offsets = np.cumsum(np.concatenate(([0], block_sizes)))
    if offsets[-1] != n:
        raise ValueError("Sum of block_size_list must match Hamiltonian dimension.")

    E_eta = E + 1j * abs(float(eta))

    # Ensure self-energies match the first/last block dimensions
    left_dim = int(block_sizes[0])
    right_dim = int(block_sizes[-1])
    Sigma_L = np.asarray(Sigma_L, dtype=complex)[:left_dim, :left_dim]
    Sigma_R = np.asarray(Sigma_R, dtype=complex)[-right_dim:, -right_dim:]
    Gamma_L = 1j * (Sigma_L - Sigma_L.conj().T) if return_gamma else None
    Gamma_R = 1j * (Sigma_R - Sigma_R.conj().T) if return_gamma else None

    # Build block tridiagonal representation of A = (E I - H - Sigma)
    diagonal_blocks: List[np.ndarray] = []
    upper_blocks: List[np.ndarray] = []
    lower_blocks: List[np.ndarray] = []

    for idx, size in enumerate(block_sizes):
        s = offsets[idx]
        e = offsets[idx + 1]
        H_block = H[s:e, s:e]
        if S_full is None:
            S_block = np.eye(size, dtype=complex)
        else:
            S_block = S_full[s:e, s:e]
        diag_block = E_eta * S_block - H_block
        if idx == 0:
            nL = int(Sigma_L.shape[0])
            diag_block[:nL, :nL] -= Sigma_L

        if idx == len(block_sizes) - 1:
            nR = int(Sigma_R.shape[0])
            diag_block[-nR:, -nR:] -= Sigma_R
            
        diagonal_blocks.append(diag_block)
        if idx < len(block_sizes) - 1:
            s_next = offsets[idx + 1]
            e_next = offsets[idx + 2]
            H_upper = H[s:e, s_next:e_next]
            H_lower = H[s_next:e_next, s:e]
            if S_full is None:
                S_upper = np.zeros_like(H_upper)
                S_lower = np.zeros_like(H_lower)
            else:
                S_upper = S_full[s:e, s_next:e_next]
                S_lower = S_full[s_next:e_next, s:e]
            upper_blocks.append(E_eta * S_upper - H_upper)
            lower_blocks.append(E_eta * S_lower - H_lower)

    result = pairwise_partial_inverse(
        diagonal_blocks,
        upper_blocks,
        lower_blocks,
        processes=processes,
        return_full=return_full_inverse,
    )

    G_R_diag = None
    if return_diag:
        G_R_diag = np.concatenate([np.diag(block) for block in result.diagonal])

    full_inverse = result.full_inverse if return_full_inverse else None

    trace_gs = None
    if overlap_matrix is not None:
        if full_inverse is not None:
            trace_gs = np.trace(full_inverse @ overlap_matrix)
        else:
            # overlap assumed block-tridiagonal aligned with block_sizes
            trace_val = 0.0 + 0.0j
            # Diagonal contribution
            for idx, block in enumerate(result.diagonal):
                s = offsets[idx]
                e = offsets[idx + 1]
                trace_val += np.trace(block @ overlap_matrix[s:e, s:e])
            # Upper / lower contributions (nearest neighbors)
            for idx, upper in enumerate(result.upper):
                s = offsets[idx]
                e = offsets[idx + 1]
                s_next = offsets[idx + 1]
                e_next = offsets[idx + 2]
                S_lower = overlap_matrix[s_next:e_next, s:e]
                trace_val += np.trace(upper @ S_lower)
            for idx, lower in enumerate(result.lower):
                s = offsets[idx]
                e = offsets[idx + 1]
                s_next = offsets[idx + 1]
                e_next = offsets[idx + 2]
                S_upper = overlap_matrix[s:e, s_next:e_next]
                trace_val += np.trace(lower @ S_upper)
            trace_gs = trace_val

    diag_blocks_out = result.diagonal if return_blocks else None
    full_inverse_out = full_inverse if return_full_inverse else None
    return G_R_diag, diag_blocks_out, full_inverse_out, Gamma_L, Gamma_R, trace_gs