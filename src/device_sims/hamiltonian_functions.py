import numpy as np 

# ====================================================================== #
#  HAMILTONIAN — swap this function to change the model                  #
# ====================================================================== #
def build_slice_square(W, t_hop=1.0):
    H = np.zeros((W, W))
    V = np.zeros((W, W))
    for r in range(W - 1):
        H[r, r+1] = -t_hop; H[r+1, r] = -t_hop
    for r in range(W):
        V[r, r] = -t_hop
    return H, V


def build_device_H(H_slice, V_slice, L_slices):
    n_per = H_slice.shape[0]
    N = L_slices * n_per
    H = np.zeros((N, N), dtype=complex)
    for s in range(L_slices):
        i0 = s * n_per; i1 = i0 + n_per
        H[i0:i1, i0:i1] = H_slice
        if s + 1 < L_slices:
            j0 = i1; j1 = j0 + n_per
            H[i0:i1, j0:j1] = V_slice
            H[j0:j1, i0:i1] = V_slice.conj().T
    return H

