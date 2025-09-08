import numpy as np
from scipy.optimize import brentq
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import inv
import warnings


def smart_inverse(A, sparse_threshold=0.1):
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
    try:
        return inv(A)
    except Exception:
        warnings.warn("Dense inverse failed, using pseudo-inverse.")
        return np.linalg.pinv(A)


def FD_half(x):
    v = x ** 4 + 50 + 33.6 * x * (1 - 0.68 * np.exp(-0.17 * (x + 1) ** 2))
    return 1 / (np.exp(-x) + 3 * np.pi ** 0.5 / 4 * v ** (-3 / 8))


def FD_minus_half(x):
    dx = x * 1e-6
    return (FD_half(x + dx) - FD_half(x - dx)) / (2 * dx)


def sparse_diag_product(A, B):
    from scipy.sparse import csr_matrix, csc_matrix
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    if not isinstance(B, csc_matrix):
        B = csc_matrix(B)
    n = A.shape[0]
    diag = np.zeros(n, dtype=complex)
    for i in range(n):
        A_row_start, A_row_end = A.indptr[i], A.indptr[i + 1]
        A_cols = A.indices[A_row_start:A_row_end]
        A_vals = A.data[A_row_start:A_row_end]
        B_rows = B.indices[B.indptr[i]:B.indptr[i + 1]]
        B_vals = B.data[B.indptr[i]:B.indptr[i + 1]]
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
    a0 = np.asarray(x0, dtype=float)
    b0 = np.asarray(x1, dtype=float)
    fa = f(a0, *args)
    fb = f(b0, *args)
    fa = np.asarray(fa)
    fb = np.asarray(fb)
    shape = fa.shape
    scalar_output = (shape == ())
    if fa.shape != fb.shape:
        try:
            fb = np.broadcast_to(fb, fa.shape)
            b0 = np.broadcast_to(b0, fa.shape)
        except ValueError:
            raise ValueError("f(x0) and f(x1) shapes not broadcastable")
    if np.shape(a0) != shape:
        a0 = np.broadcast_to(a0, shape).astype(float)
    if np.shape(b0) != shape:
        b0 = np.broadcast_to(b0, shape).astype(float)
    a = a0.copy()
    b = b0.copy()
    plateau_mask = None
    if plateau_f_tol is not None:
        plateau_mask = (np.abs(fa) <= plateau_f_tol) & (np.abs(fb) <= plateau_f_tol)
        if plateau_mask.any():
            mid = 0.5 * (a0[plateau_mask] + b0[plateau_mask]) if plateau_mask.ndim else 0.5 * (a0 + b0)
            if plateau_span_shrink:
                a0 = a0.copy(); b0 = b0.copy()
                if plateau_mask.ndim:
                    a0[plateau_mask] = mid
                    b0[plateau_mask] = mid
                else:
                    a0 = mid; b0 = mid
                fa = f(a0, *args); fb = f(b0, *args)
    prod = np.sign(fa) * np.sign(fb)
    if not np.all(prod <= 0):
        max_expansions = 12
        expansion_factor = 2.0
        a_exp = a0.copy(); b_exp = b0.copy(); fa_exp = fa.copy(); fb_exp = fb.copy()
        for _ in range(max_expansions):
            mask = (np.sign(fa_exp) * np.sign(fb_exp) > 0)
            if plateau_f_tol is not None and plateau_mask is not None:
                if plateau_mask.ndim:
                    mask = mask & (~plateau_mask)
                elif plateau_mask:
                    mask = False
            if not np.any(mask):
                break
            centers = 0.5 * (a_exp[mask] + b_exp[mask])
            half_widths = 0.5 * (b_exp[mask] - a_exp[mask]) * expansion_factor
            half_widths = np.where(half_widths == 0, 1.0, half_widths)
            a_exp[mask] = centers - half_widths
            b_exp[mask] = centers + half_widths
            fa_exp = f(a_exp, *args)
            fb_exp = f(b_exp, *args)
        still_bad_mask = (np.sign(fa_exp) * np.sign(fb_exp) > 0)
        if plateau_f_tol is not None and plateau_mask is not None:
            if plateau_mask.ndim:
                still_bad_mask = still_bad_mask & (~plateau_mask)
            else:
                still_bad_mask = still_bad_mask and (not plateau_mask)
        if np.any(still_bad_mask):
            if allow_unbracketed:
                import warnings as _warnings
                _warnings.warn("Chandrupatla: proceeding without valid bracket for some indices")
                mid = 0.5 * (a_exp[still_bad_mask] + b_exp[still_bad_mask])
                a_exp[still_bad_mask] = mid
                b_exp[still_bad_mask] = mid
                fa_exp = f(a_exp, *args)
                fb_exp = f(b_exp, *args)
            else:
                bad = np.where(still_bad_mask)
                raise ValueError(f"Chandrupatla: failed to bracket roots at indices {bad}")
        a0, b0, fa, fb = a_exp, b_exp, fa_exp, fb_exp
        a = a0.copy(); b = b0.copy()
    c = a.copy(); fc = fa.copy()
    if eps_m is not None:
        rtol = eps_m
    if eps_a is not None:
        atol = eps_a
    eps_m = np.asarray(rtol)
    eps_a = np.asarray(atol)
    if eps_m.shape not in ((), shape):
        eps_m = np.broadcast_to(eps_m, shape)
    if eps_a.shape not in ((), shape):
        eps_a = np.broadcast_to(eps_a, shape)
    t = np.full(shape, 0.5) if shape else 0.5
    terminate = np.zeros(shape, dtype=bool) if shape else False
    iterations = np.zeros(shape, dtype=int) if shape else 0
    last_fm = None
    stagnation_count = 0
    for _ in range(maxiter):
        xt = a + t * (b - a)
        ft = f(xt, *args)
        samesign = np.sign(ft) == np.sign(fa)
        a_old, fa_old = a, fa
        b_old, fb_old = b.copy(), fb.copy()
        c_old, fc_old = c.copy(), fc.copy()
        if shape:
            c = np.where(samesign, a_old, b_old)
            fc = np.where(samesign, fa_old, fb_old)
            b = np.where(samesign, b_old, a_old)
            fb = np.where(samesign, fb_old, fa_old)
        else:
            if samesign:
                c = a_old; fc = fa_old
            else:
                c = b_old; fc = fb_old; b = a_old; fb = fa_old
        a = xt; fa = ft
        fa_is_smaller = np.abs(fa) < np.abs(fb)
        if shape:
            xm = np.where(fa_is_smaller, a, b)
            fm = np.where(fa_is_smaller, fa, fb)
        else:
            xm = a if fa_is_smaller else b
            fm = fa if fa_is_smaller else fb
        tol = 2 * eps_m * np.abs(xm) + eps_a
        denom = np.abs(b - c)
        denom = np.where(denom == 0, 1.0, denom)
        tlim = tol / denom
        if f_tol is not None:
            small_res = np.abs(fm) <= f_tol
        else:
            small_res = (fm == 0)
        if plateau_f_tol is not None:
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
                    if shape:
                        terminate[active] = True
                    else:
                        terminate = True
                    break
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = (a - b) / (c - b)
            phi = (fa - fb) / (fc - fb)
        iqi = (phi ** 2 < xi) & ((1 - phi) ** 2 < 1 - xi)
        if shape:
            t = np.full(shape, 0.5)
            if np.any(iqi):
                ai = a[iqi]; bi = b[iqi]; ci = c[iqi]
                fai = fa[iqi]; fbi = fb[iqi]; fci = fc[iqi]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ti = (fai / (fbi - fai)) * (fci / (fbi - fci)) + \
                         ((ci - ai) / (bi - ai)) * (fai / (fci - fai)) * (fbi / (fci - fbi))
                bad = ~np.isfinite(ti)
                if np.any(bad):
                    ti[bad] = 0.5
                t[iqi] = ti
        else:
            if iqi and np.isfinite(fa) and np.isfinite(fb) and np.isfinite(fc):
                t = (fa / (fb - fa)) * (fc / (fb - fc)) + ((c - a) / (b - a)) * (fa / (fc - fa)) * (fb / (fc - fb))
            else:
                t = 0.5
        if shape:
            t = np.minimum(1 - tlim, np.maximum(tlim, t))
        else:
            t = min(1 - tlim, max(tlim, t))
    roots = xm if 'xm' in locals() else a
    if scalar_output:
        roots = np.asarray(roots).item()
        iterations = int(iterations)
    if f_tol is not None:
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
