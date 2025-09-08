import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import inv
import warnings

def smart_inverse(A, sparse_threshold=0.1):
    """
    Compute the inverse of a matrix A.
    If A is sparse and density < sparse_threshold, use spsolve.
    Otherwise, use dense inverse.
    Returns a dense or sparse matrix as appropriate.
    """
    if sp.issparse(A):
        density = A.nnz / (A.shape[0] * A.shape[1])
        if density < sparse_threshold:
            n = A.shape[0]
            I = sp.identity(n, dtype=A.dtype, format='csc')
            try:
                A_inv = spsolve(A, I)
                return sp.csc_matrix(A_inv)
            except Exception:
                warnings.warn("spsolve failed, falling back to dense inverse.")
                A = A.toarray()
        else:
            A = A.toarray()
    # Now A is dense
    try:
        return inv(A)
    except Exception:
        warnings.warn("Dense inverse failed, using pseudo-inverse.")
        return np.linalg.pinv(A)
