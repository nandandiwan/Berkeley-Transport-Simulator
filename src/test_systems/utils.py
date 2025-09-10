import numpy as np
from scipy.optimize import brentq
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
def FD_half(x):
    '''
    Approximation of the Fermi-Dirac integral of order 1/2.
    Reference: http://dx.doi.org/10.1063/1.4825209
    '''
    v = x**4 + 50 + 33.6 * x * (1 - 0.68 * np.exp(-0.17 * (x + 1)**2))
    return 1 / (np.exp(-x) + 3 * np.pi**0.5 / 4 * v**(-3/8))

# Define F_-1/2(x) as the derivative of F_1/2(x)
def FD_minus_half(x):
    dx = x * 1e-6  
    return (FD_half(x + dx) - FD_half(x - dx)) / (2 * dx)

def sparse_diag_product(A, B):
    """
    Compute diagonal elements of C = A * B efficiently for sparse matrices.

    Parameters:
        A (csr_matrix): sparse matrix in CSR format
        B (csc_matrix): sparse matrix in CSC format

    Returns:
        numpy.ndarray: diagonal elements of A*B
    """
    from scipy.sparse import csr_matrix, csc_matrix
    import numpy as np

    # Ensure A is CSR and B is CSC for efficient indexing
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    if not isinstance(B, csc_matrix):
        B = csc_matrix(B)

    n = A.shape[0]
    diag = np.zeros(n, dtype=complex)

    for i in range(n):
        # Get row i from A (CSR format)
        A_row_start, A_row_end = A.indptr[i], A.indptr[i+1]
        A_cols = A.indices[A_row_start:A_row_end]
        A_vals = A.data[A_row_start:A_row_end]

        # Get column i from B (CSC format)

        B_rows = B.indices[B.indptr[i]:B.indptr[i+1]]
        B_vals = B.data[B.indptr[i]:B.indptr[i+1]]

        # Compute intersection of indices efficiently
        ptr_a, ptr_b = 0, 0
        sum_diagonal = 0.0
        while ptr_a < len(A_cols) and ptr_b < len(B_rows):
            col_a, row_b = A_cols[ptr_a], B_rows[ptr_b]
            if col_a == row_b:
                sum_diagonal += A_vals[ptr_a] * B_vals[ptr_b]
                ptr_a += 1
                ptr_b += 1
            elif col_a < row_b:
                ptr_a += 1
            else:
                ptr_b += 1

        diag[i] = sum_diagonal

    return diag

def chandrupatla(f, x0, x1, verbose=False,
                 eps_m=None, eps_a=None,
                 rtol=1e-5, atol=0.0,
                 maxiter=50, return_iter=False, args=(),
                 allow_unbracketed=True,
                 f_tol=None,
                 stagnation_iters=5,
                 plateau_f_tol=None,
                 plateau_span_shrink=True):
    """Vectorized Chandrupatla root-finding method with optional plateau (flat residual) detection.

    Parameters
    ----------
    f : callable
        Function handle. Must accept (x,*args) where x can be a scalar or a
        NumPy array. The function must be vectorized w.r.t. x (element-wise).
    x0, x1 : scalar or array_like
        Bracketing interval bounds for each root. These can be scalars (then
        broadcast to the output shape) or arrays of the same shape as f(x0).
        For array use each pair (x0[i], x1[i]) must bracket a root: f(x0)*f(x1)<=0.
    verbose : bool or file-like, optional
        If True prints iteration diagnostics. If file-like writes there.
    eps_m, eps_a : float or array_like, optional (deprecated in favor of rtol/atol)
        Legacy names kept for backward compatibility. If provided they override
        rtol / atol. Relative (multiplicative) and absolute components of the
        stopping tolerance  tol = 2*eps_m*|x| + eps_a.
    rtol, atol : float or array_like, optional
        Preferred interface for relative/absolute tolerances. Default rtol=1e-5,
        atol=0.0. Can be broadcast to the shape of x. Effective tolerance used:
            tol = 2*rtol*|x| + atol
    maxiter : int, optional
        Maximum number of iterations.
    return_iter : bool, optional
        If True also return per-root iteration counts.
    args : tuple, optional
        Extra arguments passed to f.

    Parameters (additions)
    ---------------------
    plateau_f_tol : float, optional
        If provided, treat intervals where BOTH |f(a)| and |f(b)| <= plateau_f_tol as already
        converged even if they do not strictly bracket (same sign). The returned root for such
        intervals is the midpoint. This is useful for flat plateaus (e.g. band gaps) where any
        chemical potential inside is acceptable.
    plateau_span_shrink : bool, default True
        If True, when a plateau is detected the bracket is collapsed to its midpoint immediately
        to avoid unnecessary iterations / expansions.

    Returns
    -------
    roots : ndarray or scalar
        Approximated roots.
    iterations : ndarray (optional)
        Number of iterations performed for each root (only if return_iter).
    """
    # Evaluate function at initial end points (we copy to avoid mutating inputs)

    a0 = np.asarray(x0, dtype=float)
    b0 = np.asarray(x1, dtype=float)

    # Evaluate function at initial end points
    fa = f(a0, *args)
    fb = f(b0, *args)
    fa = np.asarray(fa)
    fb = np.asarray(fb)

    shape = fa.shape
    scalar_output = (shape == ())

    # Broadcasting diagnostics and actions
    if fa.shape != fb.shape:
        if verbose:
            print(f"Broadcasting fb and b0 from shape {fb.shape} to {fa.shape}")
        try:
            fb = np.broadcast_to(fb, fa.shape)
            b0 = np.broadcast_to(b0, fa.shape)
        except ValueError:
            raise ValueError("f(x0) and f(x1) shapes not broadcastable")

    if np.shape(a0) != shape:
        if verbose:
            print(f"Broadcasting a0 from shape {np.shape(a0)} to {shape}")
        a0 = np.broadcast_to(a0, shape).astype(float)
    if np.shape(b0) != shape:
        if verbose:
            print(f"Broadcasting b0 from shape {np.shape(b0)} to {shape}")
        b0 = np.broadcast_to(b0, shape).astype(float)


    a = a0.copy()
    b = b0.copy()

    # Early plateau detection BEFORE bracket validation
    plateau_mask = None
    if plateau_f_tol is not None:
        plateau_mask = (np.abs(fa) <= plateau_f_tol) & (np.abs(fb) <= plateau_f_tol)
        if plateau_mask.any():
            # Collapse those intervals to midpoint; this effectively "solves" them.
            mid = 0.5 * (a0[plateau_mask] + b0[plateau_mask]) if plateau_mask.ndim else 0.5 * (a0 + b0)
            if plateau_span_shrink:
                a0 = a0.copy(); b0 = b0.copy()
                if plateau_mask.ndim:
                    a0[plateau_mask] = mid
                    b0[plateau_mask] = mid
                else:
                    a0 = mid; b0 = mid
                fa = f(a0, *args)
                fb = f(b0, *args)

    # Bracket validation & controlled expansion (iterative, non-recursive)
    prod = np.sign(fa) * np.sign(fb)
    if not np.all(prod <= 0):
        # We will iteratively expand only those intervals that fail to bracket
        max_expansions = 12  # avoid runaway interval growth
        expansion_factor = 2.0  # multiplies half-width each expansion
        a_exp = a0.copy(); b_exp = b0.copy(); fa_exp = fa.copy(); fb_exp = fb.copy()
        for k in range(max_expansions):
            mask = (np.sign(fa_exp) * np.sign(fb_exp) > 0)
            # Exclude plateau intervals (already acceptable) from expansion attempts
            if plateau_f_tol is not None and plateau_mask is not None:
                if plateau_mask.ndim:
                    mask = mask & (~plateau_mask)
                elif plateau_mask:  # scalar True
                    mask = False
            if not np.any(mask):
                break
            if verbose:
                idx_list = np.where(mask)[0] if mask.ndim else 'scalar'
                msg = f"Bracket expansion iteration {k+1}: expanding {np.count_nonzero(mask)} interval(s) (indices {idx_list})"
                if verbose is True:
                    print(msg)
                else:
                    try:
                        print(msg, file=verbose)
                    except Exception:
                        print(msg)
            # Current centers & half-widths for masked indices only
            centers = 0.5 * (a_exp[mask] + b_exp[mask])
            half_widths = 0.5 * (b_exp[mask] - a_exp[mask]) * expansion_factor
            half_widths = np.where(half_widths == 0, 1.0, half_widths)
            a_exp[mask] = centers - half_widths
            b_exp[mask] = centers + half_widths
            # Recompute f over full arrays to preserve full-length outputs
            fa_exp = f(a_exp, *args)
            fb_exp = f(b_exp, *args)
        # Final check
        still_bad_mask = (np.sign(fa_exp) * np.sign(fb_exp) > 0)
        if plateau_f_tol is not None and plateau_mask is not None:
            # Keep plateau intervals out of the 'still bad' assessment
            if plateau_mask.ndim:
                still_bad_mask = still_bad_mask & (~plateau_mask)
            else:
                still_bad_mask = still_bad_mask and (not plateau_mask)
        if np.any(still_bad_mask):
            if allow_unbracketed:
                import warnings as _warnings
                _warnings.warn(f"Chandrupatla: proceeding without valid bracket for {np.count_nonzero(still_bad_mask)} index/indices; result may be approximate. Use tighter initial bounds or inspect function.")
                # Heuristic: replace one endpoint with midpoint to create artificial zero-width bracket
                mid = 0.5 * (a_exp[still_bad_mask] + b_exp[still_bad_mask])
                a_exp[still_bad_mask] = mid
                b_exp[still_bad_mask] = mid
                # Re-evaluate entire arrays to keep consistent shape
                fa_exp = f(a_exp, *args)
                fb_exp = f(b_exp, *args)
            else:
                bad = np.where(still_bad_mask)
                raise ValueError(f"Chandrupatla: failed to bracket roots after {max_expansions} expansions at indices {bad}")
        # Use expanded brackets
        a0, b0, fa, fb = a_exp, b_exp, fa_exp, fb_exp
        a = a0.copy(); b = b0.copy()

    c = a.copy()
    fc = fa.copy()

    # Determine effective tolerances (prefer new rtol/atol unless legacy given)
    if eps_m is not None:
        rtol = eps_m
    if eps_a is not None:
        atol = eps_a
    # Ensure arrays
    eps_m = np.asarray(rtol)  # rename for internal use
    eps_a = np.asarray(atol)
    if eps_m.shape not in ((), shape):
        if verbose:
            print(f"Broadcasting eps_m from shape {eps_m.shape} to {shape}")
        eps_m = np.broadcast_to(eps_m, shape)
    if eps_a.shape not in ((), shape):
        if verbose:
            print(f"Broadcasting eps_a from shape {eps_a.shape} to {shape}")
        eps_a = np.broadcast_to(eps_a, shape)

    t = np.full(shape, 0.5) if shape else 0.5
    terminate = np.zeros(shape, dtype=bool) if shape else False
    iterations = np.zeros(shape, dtype=int) if shape else 0

    last_fm = None
    stagnation_count = 0
    for iter_idx in range(maxiter):
        # Candidate new point
        xt = a + t * (b - a)
        ft = f(xt, *args)

        if verbose:
            output = f"t={t}\nxt={xt}\nft={ft}\na={a}\nfa={fa}\nb={b}\nfb={fb}\nc={c}\nfc={fc}"
            if verbose is True:
                print(output)
            else:
                try:
                    print(output, file=verbose)
                except Exception:
                    print(output)

        # Determine sign agreement
        samesign = np.sign(ft) == np.sign(fa)

        # Save old values for those not overwritten
        a_old, fa_old = a, fa
        b_old, fb_old = b.copy(), fb.copy()
        c_old, fc_old = c.copy(), fc.copy()

        # Update per scalar formulation using boolean masks
        # where samesign: c <- a_old; fc <- fa_old
        # else:          c <- b_old; fc <- fb_old; b <- a_old; fb <- fa_old
        if shape:
            c = np.where(samesign, a_old, b_old)
            fc = np.where(samesign, fa_old, fb_old)
            b = np.where(samesign, b_old, a_old)
            fb = np.where(samesign, fb_old, fa_old)
        else:
            if samesign:
                c = a_old
                fc = fa_old
            else:
                c = b_old
                fc = fb_old
                b = a_old
                fb = fa_old

        a = xt
        fa = ft

        # xm: point with smaller |f|
        fa_is_smaller = np.abs(fa) < np.abs(fb)
        if shape:
            xm = np.where(fa_is_smaller, a, b)
            fm = np.where(fa_is_smaller, fa, fb)
        else:
            xm = a if fa_is_smaller else b
            fm = fa if fa_is_smaller else fb

        tol = 2 * eps_m * np.abs(xm) + eps_a
        denom = np.abs(b - c)
        denom = np.where(denom == 0, 1.0, denom)  # avoid divide by zero
        tlim = tol / denom
        # Residual-based early termination
        if f_tol is not None:
            small_res = np.abs(fm) <= f_tol
        else:
            small_res = (fm == 0)

        # Plateau condition during iterations (updated endpoints
        if plateau_f_tol is not None:
            # Bracket endpoints small => plateau
            plateau_now = (np.abs(fa) <= plateau_f_tol) & (np.abs(fb) <= plateau_f_tol)
            new_terminate = small_res | (tlim > 0.5) | plateau_now | terminate
        else:
            new_terminate = small_res | (tlim > 0.5) | terminate
        if shape:
            iterations[~new_terminate] += 1
        else:
            if not new_terminate:
                iterations += 1
        terminate = new_terminate
        if np.all(terminate):
            break

        # Stagnation detection on fm magnitude (only where not yet terminated)
        if f_tol is not None:
            active = ~terminate
            if np.any(active):
                cur_fm_norm = np.max(np.abs(fm[active])) if np.ndim(fm) else abs(fm)
                if last_fm is not None and cur_fm_norm >= last_fm * 0.999:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                last_fm = cur_fm_norm
                if stagnation_count >= stagnation_iters:
                    # give up on active indices: mark them for termination
                    if shape:
                        terminate[active] = True
                    else:
                        terminate = True
                    break

        # Compute xi, phi
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = (a - b) / (c - b)
            phi = (fa - fb) / (fc - fb)

        # Inverse quadratic interpolation condition
        iqi = (phi ** 2 < xi) & ((1 - phi) ** 2 < 1 - xi)

        if shape:
            t = np.full(shape, 0.5)
            if np.any(iqi):
                # Only compute for those indices to avoid invalid divisions
                ai = a[iqi]; bi = b[iqi]; ci = c[iqi]
                fai = fa[iqi]; fbi = fb[iqi]; fci = fc[iqi]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ti = (fai / (fbi - fai)) * (fci / (fbi - fci)) + \
                         ((ci - ai) / (bi - ai)) * (fai / (fci - fai)) * (fbi / (fci - fbi))
                # Fallback to bisection if nan/inf
                bad = ~np.isfinite(ti)
                if np.any(bad):
                    ti[bad] = 0.5
                t[iqi] = ti
        else:
            if iqi and np.isfinite(fa) and np.isfinite(fb) and np.isfinite(fc):
                t = (fa / (fb - fa)) * (fc / (fb - fc)) + ((c - a) / (b - a)) * (fa / (fc - fa)) * (fb / (fc - fb))
            else:
                t = 0.5

        # Clamp t to (tlim, 1 - tlim)
        if shape:
            t = np.minimum(1 - tlim, np.maximum(tlim, t))
        else:
            t = min(1 - tlim, max(tlim, t))

    roots = xm if 'xm' in locals() else a  # fallback
    if scalar_output:
        roots = np.asarray(roots).item()
        iterations = int(iterations)
    if f_tol is not None:
        # Final residual screening
        fres = f(roots, *args)
        if shape:
            bad = np.where(np.abs(fres) > f_tol)
            if bad[0].size:
                roots = np.array(roots, copy=True)
                roots[bad] = np.nan
        else:
            if abs(fres) > f_tol:
                roots = np.nan
    if return_iter:
        return roots, iterations
    return roots
    
from unit_cell_generation import Atom
def gaussian_broadening_interpolation(atom_pos : list[Atom], values : np.ndarray, final_dim : tuple) -> np.ndarray:
    """maps from atomistic grid to evenly spaced grid of dimensions final_dim"""
    # Validate inputs
    if not isinstance(final_dim, tuple) or len(final_dim) not in (1, 2, 3):
        raise ValueError("final_dim must be a tuple of length 1, 2, or 3 (e.g., (Nx,), (Nx,Ny), (Nx,Ny,Nz))")

    if len(atom_pos) == 0:
        return np.zeros(final_dim, dtype=values.dtype if isinstance(values, np.ndarray) else float)

    values = np.asarray(values)
    if values.shape[0] != len(atom_pos):
        raise ValueError("values length must match number of atoms in atom_pos")

    # Prepare positions array based on requested dimensionality
    dims = len(final_dim)
    pos = np.empty((len(atom_pos), dims), dtype=float)
    for i, a in enumerate(atom_pos):
        if dims == 1:
            pos[i, 0] = a.x
        elif dims == 2:
            pos[i, 0] = a.x; pos[i, 1] = a.y
        else:
            pos[i, 0] = a.x; pos[i, 1] = a.y; pos[i, 2] = a.z

    # Build evenly spaced grid axes spanning the atom positions
    axes = []
    for d in range(dims):
        vmin = float(np.min(pos[:, d]))
        vmax = float(np.max(pos[:, d]))
        n = final_dim[d]
        if n <= 0:
            raise ValueError("final_dim entries must be positive integers")
        if n == 1:
            axes.append(np.array([(vmin + vmax) * 0.5], dtype=float))
        else:
            axes.append(np.linspace(vmin, vmax, n, dtype=float))

    # Heuristic Gaussian sigma (isotropic) from median nearest-neighbor spacing in the used subspace
    # Prefer SciPy cKDTree if available; else fall back to grid-based spacing
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(pos)
        # k=2 because the closest point to a point is itself (distance 0) at k=1
        dists, _ = tree.query(pos, k=2)
        nn = dists[:, 1]
        med_nn = float(np.median(nn[nn > 0])) if np.any(nn > 0) else 0.0
        sigma = med_nn / 2.0 if med_nn > 0 else 0.0
    except Exception:
        # Fallback: use average grid spacing magnitude across axes
        spacings = []
        for ax in axes:
            if ax.size > 1:
                spacings.append((ax[-1] - ax[0]) / (ax.size - 1))
        sigma = float(np.mean(spacings)) if spacings else 0.0

    # If sigma is degenerate (e.g., single atom), set sigma to one grid step to avoid singularities
    if not np.isfinite(sigma) or sigma <= 0:
        # robust default: average grid spacing or small epsilon
        spacings = []
        for ax in axes:
            if ax.size > 1:
                spacings.append((ax[-1] - ax[0]) / (ax.size - 1))
        sigma = float(np.mean(spacings)) if spacings else 1e-12

    cutoff = 2.5  # truncate Gaussian at ~3.5 sigma (~0.00087 tail mass)
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

    # Output grid
    out = np.zeros(final_dim, dtype=values.dtype)

    # Utility to find index window [i0, i1) within cutoff for coordinate p along axis ax
    def window(ax: np.ndarray, p: float, rad: float):
        # searchsorted gives insertion positions; we expand by +/- rad
        left = p - rad
        right = p + rad
        i0 = int(np.searchsorted(ax, left, side='left'))
        i1 = int(np.searchsorted(ax, right, side='right'))
        if i0 < 0:
            i0 = 0
        if i1 > ax.size:
            i1 = ax.size
        return i0, i1

    # Accumulate contributions per atom using separable Gaussian weights
    rad = cutoff * sigma

    if dims == 1:
        ax = axes[0]
        for (x,), v in zip(pos, values):
            i0, i1 = window(ax, x, rad)
            if i0 >= i1:
                continue
            dx = ax[i0:i1] - x
            w = np.exp(-dx * dx * inv_two_sigma2)
            sw = w.sum()
            if sw > 0:
                out[i0:i1] += v * (w / sw)
    elif dims == 2:
        ax, ay = axes
        for (x, y), v in zip(pos, values):
            ix0, ix1 = window(ax, x, rad)
            iy0, iy1 = window(ay, y, rad)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            dx = ax[ix0:ix1] - x
            dy = ay[iy0:iy1] - y
            wx = np.exp(-dx * dx * inv_two_sigma2)
            wy = np.exp(-dy * dy * inv_two_sigma2)
            # separable 2D Gaussian weights via outer product
            w = np.multiply.outer(wy, wx)
            sw = w.sum()
            if sw > 0:
                out[ix0:ix1, iy0:iy1] += (v * w / sw).T  # transpose to match (x,y) indexing
    else:  # dims == 3
        ax, ay, az = axes
        for (x, y, z), v in zip(pos, values):
            ix0, ix1 = window(ax, x, rad)
            iy0, iy1 = window(ay, y, rad)
            iz0, iz1 = window(az, z, rad)
            if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
                continue
            dx = ax[ix0:ix1] - x
            dy = ay[iy0:iy1] - y
            dz = az[iz0:iz1] - z
            wx = np.exp(-dx * dx * inv_two_sigma2)
            wy = np.exp(-dy * dy * inv_two_sigma2)
            wz = np.exp(-dz * dz * inv_two_sigma2)
            # separable 3D Gaussian weights via tensor outer product
            # w[z,y,x] = wz[:,None,None] * wy[None,:,None] * wx[None,None,:]
            w = np.multiply.outer(wz, np.multiply.outer(wy, wx))
            sw = w.sum()
            if sw > 0:
                # out indexed as [Nx, Ny, Nz]; map local block accordingly
                out[ix0:ix1, iy0:iy1, iz0:iz1] += (v * w / sw).transpose(2, 1, 0)

    return out