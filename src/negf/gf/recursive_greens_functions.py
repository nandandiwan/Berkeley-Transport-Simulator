import copy
from typing import List

import numpy as np
import scipy.sparse as sp
#from negf.self_energy.surface import surface_greens_function
from negf.self_energy.surface import surface_greens_function
from negf.utils.common import fermi_dirac, smart_inverse
import scipy.linalg as linalg
from negf.gf.general_rgf.pairwise_partial_inverse import pairwise_partial_inverse

def mat_left_div(mat_a, mat_b):
    ans, resid, rank, s = linalg.lstsq(mat_a, mat_b, lapack_driver='gelsy')
    return ans

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

def gf_inverse(E, H, H00, H01, mu1 = 0, mu2 = 0, block_size = None, block_size_list = None, method : str= "recursive", processes = 1):
    if (method == "direct"):
        return _direct_inverse(E, H, H00, H01, mu1, mu2)  
    if (method == "recursive"):
        return _recursive_inverse(E, H, H00, H01, mu1, mu2, block_size)
    if (method == "var_recursive"):
        return _variable_recursive_inverse(E, H, H00, H01, block_size_list, mu1, mu2, processes)
    raise ValueError(f"Unknown method '{method}'. Use 'direct' or 'recursive'.")
    
def _direct_inverse(E, H, H00, H01, muL=0, muR=0):
    """Direct inversion identical in spirit to the working notebook snippet.

    Returns:
        G_R_diag : 1D ndarray (diagonal of full retarded Green's function)
        G_lesser_diag : 1D ndarray (diagonal of lesser GF under equilibrium two-terminal approximation)
        Gamma_L, Gamma_R : full broadening matrices (dense np.ndarray)
    """
    # Convert inputs to dense arrays matching notebook behavior
    if sp.issparse(H):
        H = H.toarray()
    if sp.issparse(H00):
        H00 = H00.toarray()
    if sp.issparse(H01):
        H01 = H01.toarray()
    n = H.shape[0]

    # Surface Green's functions (Sancho-Rubio already adds small imaginary part)
    Sigma_L, Sigma_R = surface_greens_function(E - muL,  H01.conj().T, H00, H01)
    # G00_R = surface_greens_function(E - muR, H00, H01)
    # if G00_L is None or G00_R is None:
    #     raise ValueError("surface_greens_function returned None (non-converged).")
    # G00_L = np.asarray(G00_L)
    # G00_R = np.asarray(G00_R)
    # if G00_L.ndim != 2:
    #     G00_L = np.atleast_2d(G00_L)
    # if G00_R.ndim != 2:
    #     G00_R = np.atleast_2d(G00_R)

    # # Self-energies blocks Σ = T^† G00 T with T = H01 (device↔lead principal layer coupling)
    # Sigma_L = H01.conj().T @ G00_L @ H01
    # Sigma_R = H01.conj().T @ G00_R @ H01
    m = Sigma_L.shape[0]

    # Embed into device corners
    Sigma_L_full = np.zeros((n, n), dtype=complex)
    Sigma_R_full = np.zeros((n, n), dtype=complex)
    Sigma_L_full[:m, :m] = Sigma_L
    Sigma_R_full[-m:, -m:] = Sigma_R

    # Build and solve for full G^R
    A = (E + 1e-9j) * np.eye(n, dtype=complex) - (H + Sigma_L_full + Sigma_R_full)
    G_R = np.linalg.solve(A, np.eye(n, dtype=complex))

    # Broadenings
    Gamma_L = 1j * (Sigma_L_full - Sigma_L_full.conj().T)
    Gamma_R = 1j * (Sigma_R_full - Sigma_R_full.conj().T)

    # Lesser (equilibrium two-terminal approximation)
    f_L = fermi_dirac(E, muL)
    f_R = fermi_dirac(E, muR)
    Sigma_lesser = 1j * (Gamma_L * f_L + Gamma_R * f_R)
    G_A = G_R.conj().T
    G_lesser = G_R @ Sigma_lesser @ G_A

    G_R_diag = np.diag(G_R)
    G_lesser_diag = np.diag(G_lesser)
    return G_R, G_lesser_diag, Gamma_L, Gamma_R


def _recursive_inverse(E, H, H00, H01, muL=0, muR=0, block_size=None, compute_lesser=True):
    if sp.issparse(H):
        H = H.toarray()
    n = H.shape[0]
    if block_size is None:
        block_size = H00.shape[0]
    if n % block_size != 0:
        raise ValueError("H dimension not divisible by block_size.")
    n_blocks = n // block_size
    E = add_eta(E)
    dagger = lambda A: np.conjugate(A.T)

    # Slice blocks
    H_ii = []
    H_ij = []
    for i in range(n_blocks):
        sl = slice(i*block_size, (i+1)*block_size)
        H_ii.append(H[sl, sl].copy())
        if i < n_blocks - 1:
            sr = slice((i+1)*block_size, (i+2)*block_size)
            H_ij.append(H[sl, sr].copy())

    dagger = lambda A: A.conj().T
    # Lead self energies
    G00_L = surface_greens_function(E - muL, H00, H01)
    G00_R = surface_greens_function(E - muR, H00, H01)
    Sigma_L = H01.conj().T @ G00_L @ H01
    Sigma_R = H01.conj().T @ G00_R @ H01
    Gamma_L = 1j * (Sigma_L - Sigma_L.conj().T)
    Gamma_R = 1j * (Sigma_R - Sigma_R.conj().T)

    f_L = fermi_dirac(E, muL)
    f_R = fermi_dirac(E, muR)
    Sigma_L_lesser = Gamma_L * f_L * 1j
    Sigma_R_lesser = Gamma_R * f_R * 1j
    g_R = []
    g_lesser = []

    # Forward sweep - using smart_inverse
    H00_eff = H_ii[0] + Sigma_L
    g0_R = smart_inverse(E * np.eye(block_size) - H00_eff)
    g_R.append(g0_R)

    if compute_lesser:
        g0_lesser = g0_R @ Sigma_L_lesser @ dagger(g0_R)
        g_lesser.append(g0_lesser)

    for i in range(1, n_blocks):
        H_i_im1 = dagger(H_ij[i-1])
        sigma_recursive_R = H_i_im1 @ g_R[i - 1] @ H_ij[i-1]
        g_i_R = smart_inverse(E * np.eye(block_size) - H_ii[i] - sigma_recursive_R)
        g_R.append(g_i_R)
        if compute_lesser:
            sigma_recursive_lesser = H_i_im1 @ g_lesser[i-1] @ H_ij[i-1]
            g_i_lesser = g_R[i] @ sigma_recursive_lesser @ dagger(g_R[i])
            g_lesser.append(g_i_lesser)

    G_R = [None] * n_blocks
    G_lesser = [None] * n_blocks
    # Optional storage for nearest-neighbor off-diagonal lesser blocks: G^<_{i,i+1}
    G_lesser_offdiag_right = [None] * (n_blocks - 1)

    # Backward sweep - using smart_inverse
    H_N_Nm1 = dagger(H_ij[-1])
    sigma_eff_R = H_N_Nm1 @ g_R[-2] @ H_ij[-1]
    GN_R = smart_inverse(E * np.eye(block_size) - H_ii[-1] - Sigma_R - sigma_eff_R)
    G_R[-1] = GN_R

    if compute_lesser:
        sigma_eff_lesser = H_N_Nm1 @ g_lesser[-2] @ H_ij[-1]
        total_sigma_lesser = Sigma_R_lesser + sigma_eff_lesser
        GN_lesser = G_R[-1] @ total_sigma_lesser @ dagger(G_R[-1])
        G_lesser[-1] = GN_lesser

    for i in range(n_blocks - 2, -1, -1):
        H_i_ip1 = H_ij[i]
        H_ip1_i = dagger(H_i_ip1)
        propagator = g_R[i] @ H_i_ip1 @ G_R[i + 1] @ H_ip1_i
        G_R[i] = g_R[i] + propagator @ g_R[i]
        if compute_lesser:
            g_R_dag = dagger(g_R[i])
            term1 = g_lesser[i]
            term2 = (propagator @ g_lesser[i])
            term3 = (g_lesser[i] @ dagger(propagator))
            term4 = (g_R[i] @ H_i_ip1 @ G_lesser[i + 1] @ H_ip1_i @ g_R_dag)
            G_lesser[i] = term1 + term2 + term3 + term4

            # Also compute nearest-neighbor off-diagonal lesser using full-G relation:
            # G_{i,i+1}^< = G_{i,i}^R H_{i,i+1} G_{i+1,i+1}^< + G_{i,i}^< H_{i,i+1} G_{i+1,i+1}^A
            G_ip1_A = dagger(G_R[i + 1])
            off_term_R = G_R[i] @ H_i_ip1 @ G_lesser[i + 1]
            off_term_L = G_lesser[i] @ H_i_ip1 @ G_ip1_A
            G_lesser_offdiag_right[i] = off_term_R + off_term_L

    G_R_diag = np.concatenate([np.diag(block) for block in G_R])
    G_lesser_diag = np.concatenate([np.diag(block) for block in G_lesser])

    return G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R 

def _variable_recursive_inverse(E, H, H00, H01, block_size_list, mu1, mu2, processes):
    if block_size_list is None:
        raise ValueError("block_size_list must be provided for method='var_recursive'.")

    block_sizes = np.asarray(block_size_list, dtype=int).astype(int)
    if block_sizes.ndim != 1:
        raise ValueError("block_size_list must be a 1D array of block sizes.")
    if np.any(block_sizes <= 0):
        raise ValueError("block_size_list entries must be positive integers.")

    if sp.issparse(H):
        H = H.toarray()
    if sp.issparse(H00):
        H00 = H00.toarray()
    if sp.issparse(H01):
        H01 = H01.toarray()

    n = H.shape[0]
    offsets = np.cumsum(np.concatenate(([0], block_sizes)))
    if offsets[-1] != n:
        raise ValueError("Sum of block_size_list must match Hamiltonian dimension.")

    E_eta = add_eta(E)

    Sigma_L, Sigma_R = surface_greens_function(E - mu1, H01.conj().T, H00, H01)
    # Ensure self-energies match the first/last block dimensions
    left_dim = int(block_sizes[0])
    right_dim = int(block_sizes[-1])
    Sigma_L = np.asarray(Sigma_L, dtype=complex)[:left_dim, :left_dim]
    Sigma_R = np.asarray(Sigma_R, dtype=complex)[-right_dim:, -right_dim:]
    Gamma_L = 1j * (Sigma_L - Sigma_L.conj().T)
    Gamma_R = 1j * (Sigma_R - Sigma_R.conj().T)

    f_L = fermi_dirac(E, mu1)
    f_R = fermi_dirac(E, mu2)
    Sigma_L_lesser = Gamma_L * f_L * 1j
    Sigma_R_lesser = Gamma_R * f_R * 1j

    # Build block tridiagonal representation of A = (E I - H - Sigma)
    diagonal_blocks: List[np.ndarray] = []
    upper_blocks: List[np.ndarray] = []
    lower_blocks: List[np.ndarray] = []

    for idx, size in enumerate(block_sizes):
        s = offsets[idx]
        e = offsets[idx + 1]
        H_block = H[s:e, s:e]
        diag_block = E_eta * np.eye(size, dtype=complex) - H_block
        if idx == 0:
            diag_block -= Sigma_L
        if idx == len(block_sizes) - 1:
            diag_block -= Sigma_R
        diagonal_blocks.append(diag_block)
        if idx < len(block_sizes) - 1:
            s_next = offsets[idx + 1]
            e_next = offsets[idx + 2]
            upper_blocks.append(-H[s:e, s_next:e_next])
            lower_blocks.append(-H[s_next:e_next, s:e])

    result = pairwise_partial_inverse(
        diagonal_blocks,
        upper_blocks,
        lower_blocks,
        processes=processes,
        return_full=True,
    )

    if result.full_inverse is None:
        raise RuntimeError("Pairwise inversion did not return the full inverse matrix.")

    full_inverse = result.full_inverse
    G_R_diag = np.concatenate([np.diag(block) for block in result.diagonal])

    first_slice = slice(offsets[0], offsets[1])
    last_slice = slice(offsets[-2], offsets[-1]) if len(block_sizes) > 1 else first_slice

    G_lesser_blocks: List[np.ndarray] = []
    G_lesser_offdiag_right: List[np.ndarray] = []

    for idx, size in enumerate(block_sizes):
        s = offsets[idx]
        e = offsets[idx + 1]
        Gi_first = full_inverse[s:e, first_slice]
        Gi_last = full_inverse[s:e, last_slice]
        block = np.zeros((size, size), dtype=complex)
        block += Gi_first @ Sigma_L_lesser @ Gi_first.conj().T
        block += Gi_last @ Sigma_R_lesser @ Gi_last.conj().T
        G_lesser_blocks.append(block)
        if idx < len(block_sizes) - 1:
            s_next = offsets[idx + 1]
            e_next = offsets[idx + 2]
            Gi_next_first = full_inverse[s_next:e_next, first_slice]
            Gi_next_last = full_inverse[s_next:e_next, last_slice]
            off_block = Gi_first @ Sigma_L_lesser @ Gi_next_first.conj().T
            off_block += Gi_last @ Sigma_R_lesser @ Gi_next_last.conj().T
            G_lesser_offdiag_right.append(off_block)

    G_lesser_diag = np.concatenate([np.diag(block) for block in G_lesser_blocks])

    return G_R_diag, G_lesser_diag, G_lesser_offdiag_right, Gamma_L, Gamma_R