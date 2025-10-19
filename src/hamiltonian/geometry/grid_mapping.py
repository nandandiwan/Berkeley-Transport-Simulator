import numpy as np
from typing import Tuple, Optional, Dict, Sequence

try:
    from scipy.spatial import cKDTree
    _HAS_CKD = True
except Exception:
    _HAS_CKD = False


def _auto_sigma(coords: np.ndarray, axes_idx: Sequence[int], grid_shape: Tuple[int, ...]) -> float:
    """Heuristic Gaussian sigma based on median nearest-neighbor spacing in used axes.

    Falls back to grid step if KDTree is unavailable or degenerate geometry.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N,3)")
    # Project onto used axes
    sub = coords[:, axes_idx]
    sigma = None
    if _HAS_CKD and sub.shape[0] >= 2:
        try:
            tree = cKDTree(sub)
            # k=2 -> first neighbor is the point itself; second is nearest neighbor
            dists, _ = tree.query(sub, k=2, workers=-1)
            nn = dists[:, 1]
            # Robust median and avoid zeros
            sigma = float(np.median(nn))
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = None
        except Exception:
            sigma = None
    if sigma is None:
        # Fallback: use grid spacing average across axes
        spans = []
        for ax, n in zip(axes_idx, grid_shape):
            low = float(np.min(coords[:, ax]))
            high = float(np.max(coords[:, ax]))
            if n <= 1:
                continue
            spans.append((high - low) / (n - 1))
        if spans:
            sigma = float(np.mean(spans))
        else:
            sigma = 1.0
    # Smoothness knob: smaller than NN to keep sharp but stable
    return max(1e-9, 0.5 * sigma)


def _make_axes(coords: np.ndarray,
               grid_shape: Tuple[int, ...],
               dims: int,
               bounds: Optional[Dict[str, Tuple[float, float]]] = None,
               axes_order: Tuple[int, ...] = (0, 1, 2)) -> Tuple[Tuple[np.ndarray, ...], Tuple[int, ...]]:
    """Create evenly spaced grid axes spanning atom positions or provided bounds.

    Returns (axes, axes_idx) where axes is a tuple of 1D arrays (len=dims) and
    axes_idx are the corresponding coordinate indices (0:x,1:y,2:z) used.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N,3)")
    if dims not in (1, 2, 3):
        raise ValueError("dims must be 1, 2, or 3")

    # Map dims to axis indices: 1D -> (x), 2D -> (x,z), 3D -> (x,y,z) by default
    if dims == 1:
        axes_idx = (axes_order[0],)
    elif dims == 2:
        # As specified: 2D grid is X-Z slice (y squashed)
        axes_idx = (axes_order[0], axes_order[2])
    else:
        axes_idx = axes_order

    ax_names = ['x', 'y', 'z']
    sel_names = tuple(ax_names[i] for i in axes_idx)
    axes = []
    for name, ax, n in zip(sel_names, axes_idx, grid_shape):
        if bounds and name in bounds:
            lo, hi = bounds[name]
        else:
            lo = float(np.min(coords[:, ax]))
            hi = float(np.max(coords[:, ax]))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = -0.5, 0.5
            if lo == hi:
                lo -= 0.5
                hi += 0.5
        if n <= 1:
            axis = np.array([(lo + hi) * 0.5], dtype=float)
        else:
            axis = np.linspace(lo, hi, int(n), dtype=float)
        axes.append(axis)
    return tuple(axes), axes_idx


def atomistic_to_grid(coords: np.ndarray,
                      values: np.ndarray,
                      grid_shape: Tuple[int, ...],
                      sigma: Optional[float] = None,
                      cutoff: float = 2.5,
                      bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                      transport_axis: Optional[int] = None,
                      combine: str = "sum",
                      eps: float = 1e-15) -> np.ndarray:
    """Map atomistic values (per-atom) onto a regular grid using Gaussian broadening.

    - 1D grid: axis = x (or transport_axis if provided)
    - 2D grid: axes = (x, z), y is squashed (ignored)
    - 3D grid: axes = (x, y, z)

    Parameters
    ----------
    coords : (N,3) array
        Atom positions in Cartesian coordinates (units consistent with bounds).
    values : (N,) array
        Values defined on atomistic grid (e.g., LDOS or charge density per site).
    grid_shape : tuple[int, ...]
        Desired output grid shape (nx,), (nx,nz), or (nx,ny,nz).
    sigma : float, optional
        Gaussian sigma. If None, estimated from NN spacing/grid spacing.
    cutoff : float
        Truncate Gaussian beyond cutoff*sigma in each axis.
    bounds : dict, optional
        Optional axis bounds, e.g., {'x':(xmin,xmax),'z':(zmin,zmax)}.
    transport_axis : int, optional
        If provided and dims==1, uses this axis index (0/1/2) as the 1D axis.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N,3)")
    if values.ndim != 1 or values.shape[0] != coords.shape[0]:
        raise ValueError("values must be (N,) aligned with coords")

    dims = len(grid_shape)
    if dims == 1 and transport_axis is not None:
        axes_order = (int(transport_axis),) + tuple(ax for ax in (0,1,2) if ax != int(transport_axis))
    else:
        axes_order = (0, 1, 2)

    axes, axes_idx = _make_axes(coords, grid_shape, dims, bounds=bounds, axes_order=axes_order)
    if sigma is None:
        sigma = _auto_sigma(coords, axes_idx, grid_shape)
    sigma = float(max(1e-12, sigma))
    rad = cutoff * sigma
    inv2sig2 = 1.0 / (2.0 * sigma * sigma)

    # Output grid (+ optional normalization for mean)
    out = np.zeros(tuple(int(n) for n in grid_shape), dtype=float)
    denom = None
    combine = str(combine).lower()
    if combine not in ("sum", "mean", "max", "maxabs"):
        raise ValueError("combine must be one of {'sum','mean','max','maxabs'}")
    if combine == "mean":
        denom = np.zeros_like(out)

    # Helper: index window on a 1D axis within radius
    def window(ax: np.ndarray, p: float, r: float) -> Tuple[int, int]:
        lo = p - r
        hi = p + r
        i0 = int(np.searchsorted(ax, lo, side='left'))
        i1 = int(np.searchsorted(ax, hi, side='right'))
        i0 = max(0, i0)
        i1 = min(ax.size, i1)
        return i0, i1

    if dims == 1:
        ax = axes[0]
        for p, v in zip(coords[:, axes_idx[0]], values):
            i0, i1 = window(ax, float(p), rad)
            if i0 >= i1:
                continue
            dx = ax[i0:i1] - float(p)
            w = np.exp(-dx*dx * inv2sig2)
            if combine == "sum" or combine == "mean":
                out[i0:i1] += v * w
                if denom is not None:
                    denom[i0:i1] += w
            elif combine == "max":
                out[i0:i1] = np.maximum(out[i0:i1], v * w)
            else:  # maxabs
                out[i0:i1] = np.where(np.abs(v) * w > np.abs(out[i0:i1]), v * w, out[i0:i1])
        if denom is not None:
            out = out / (denom + eps)
        return out

    if dims == 2:
        ax, az = axes  # x, z
        xs, zs = axes_idx[0], axes_idx[1]
        for p, v in zip(coords, values):
            px = float(p[xs]); pz = float(p[zs])
            ix0, ix1 = window(ax, px, rad)
            iz0, iz1 = window(az, pz, rad)
            if ix0 >= ix1 or iz0 >= iz1:
                continue
            dx = ax[ix0:ix1] - px
            dz = az[iz0:iz1] - pz
            wx = np.exp(-dx*dx * inv2sig2)
            wz = np.exp(-dz*dz * inv2sig2)
            # Outer product accumulation
            if combine == "sum" or combine == "mean":
                out[ix0:ix1, iz0:iz1] += v * (wx[:, None] * wz[None, :])
                if denom is not None:
                    denom[ix0:ix1, iz0:iz1] += (wx[:, None] * wz[None, :])
            elif combine == "max":
                out[ix0:ix1, iz0:iz1] = np.maximum(out[ix0:ix1, iz0:iz1], v * (wx[:, None] * wz[None, :]))
            else:  # maxabs
                cand = v * (wx[:, None] * wz[None, :])
                mask = np.abs(cand) > np.abs(out[ix0:ix1, iz0:iz1])
                out[ix0:ix1, iz0:iz1][mask] = cand[mask]
        if denom is not None:
            out = out / (denom + eps)
        return out

    # dims == 3
    ax, ay, az = axes
    xs, ys, zs = axes_idx
    for p, v in zip(coords, values):
        px = float(p[xs]); py = float(p[ys]); pz = float(p[zs])
        ix0, ix1 = window(ax, px, rad)
        iy0, iy1 = window(ay, py, rad)
        iz0, iz1 = window(az, pz, rad)
        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue
        dx = ax[ix0:ix1] - px
        dy = ay[iy0:iy1] - py
        dz = az[iz0:iz1] - pz
        wx = np.exp(-dx*dx * inv2sig2)
        wy = np.exp(-dy*dy * inv2sig2)
        wz = np.exp(-dz*dz * inv2sig2)
        if combine == "sum" or combine == "mean":
            kernel = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
            out[ix0:ix1, iy0:iy1, iz0:iz1] += v * kernel
            if denom is not None:
                denom[ix0:ix1, iy0:iy1, iz0:iz1] += kernel
        elif combine == "max":
            out[ix0:ix1, iy0:iy1, iz0:iz1] = np.maximum(out[ix0:ix1, iy0:iy1, iz0:iz1],
                                                        v * (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]))
        else:  # maxabs
            cand = v * (wx[:, None, None] * wy[None, :, None] * wz[None, None, :])
            mask = np.abs(cand) > np.abs(out[ix0:ix1, iy0:iy1, iz0:iz1])
            out[ix0:ix1, iy0:iy1, iz0:iz1][mask] = cand[mask]
    if denom is not None:
        out = out / (denom + eps)
    return out


def grid_to_atomistic(coords: np.ndarray,
                      grid: np.ndarray,
                      bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                      transport_axis: Optional[int] = None) -> np.ndarray:
    """Map grid values to atomistic positions via nearest/bilinear/trilinear sampling.

    - If grid.ndim == 1: sample along x (or transport_axis if provided)
    - If grid.ndim == 2: sample using (x,z) slice, y is ignored (constant along y)
    - If grid.ndim == 3: trilinear sampling (x,y,z)

    Parameters
    ----------
    coords : (N,3) array
        Atom positions.
    grid : ndarray
        Grid values. Shape determines sampling dimensionality.
    bounds : dict, optional
        Axis bounds used to generate the grid. If omitted, inferred from coords min/max.
    transport_axis : int, optional
        For 1D grids, choose axis index (0/1/2). Default: x.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N,3)")
    grid = np.asarray(grid)
    dims = int(grid.ndim)
    if dims not in (1, 2, 3):
        raise ValueError("grid must be 1D, 2D, or 3D")

    # Recreate axes used for the grid
    grid_shape = grid.shape
    if dims == 1 and transport_axis is not None:
        axes_order = (int(transport_axis),) + tuple(ax for ax in (0,1,2) if ax != int(transport_axis))
    else:
        axes_order = (0, 1, 2)
    axes, axes_idx = _make_axes(coords, grid_shape, dims, bounds=bounds, axes_order=axes_order)

    # Helpers
    def _index_and_frac(axis: np.ndarray, p: float) -> Tuple[int, float]:
        # Map p to cell index i such that p in [axis[i], axis[i+1]] and fraction t
        if axis.size == 1:
            return 0, 0.0
        # Clip to bounds
        if p <= axis[0]:
            return 0, 0.0
        if p >= axis[-1]:
            return axis.size - 2, 1.0
        i1 = int(np.searchsorted(axis, p, side='right'))
        i0 = i1 - 1
        x0 = axis[i0]; x1 = axis[i1]
        t = 0.0 if x1 == x0 else float((p - x0) / (x1 - x0))
        return i0, t

    N = coords.shape[0]
    out = np.zeros(N, dtype=float)

    if dims == 1:
        ax = axes[0]
        idx = axes_idx[0]
        for i in range(N):
            p = float(coords[i, idx])
            i0, t = _index_and_frac(ax, p)
            v0 = float(grid[i0])
            v1 = float(grid[i0 + 1]) if ax.size > 1 else v0
            out[i] = (1.0 - t) * v0 + t * v1
        return out

    if dims == 2:
        ax, az = axes  # x, z
        ix, iz = axes_idx[0], axes_idx[1]
        for i in range(N):
            px = float(coords[i, ix])
            pz = float(coords[i, iz])
            ix0, tx = _index_and_frac(ax, px)
            iz0, tz = _index_and_frac(az, pz)
            # Bilinear
            g00 = float(grid[ix0, iz0])
            g10 = float(grid[ix0 + 1, iz0]) if ax.size > 1 else g00
            g01 = float(grid[ix0, iz0 + 1]) if az.size > 1 else g00
            g11 = float(grid[ix0 + 1, iz0 + 1]) if (ax.size > 1 and az.size > 1) else g00
            out[i] = (1 - tx) * (1 - tz) * g00 + tx * (1 - tz) * g10 + (1 - tx) * tz * g01 + tx * tz * g11
        return out

    # dims == 3: trilinear
    ax, ay, az = axes
    ix, iy, iz = axes_idx
    for i in range(N):
        px = float(coords[i, ix])
        py = float(coords[i, iy])
        pz = float(coords[i, iz])
        ix0, tx = _index_and_frac(ax, px)
        iy0, ty = _index_and_frac(ay, py)
        iz0, tz = _index_and_frac(az, pz)
        # Fetch corners with clamps
        def G(ixx, iyy, izz):
            ixx2 = min(ixx, grid.shape[0] - 1)
            iyy2 = min(iyy, grid.shape[1] - 1)
            izz2 = min(izz, grid.shape[2] - 1)
            return float(grid[ixx2, iyy2, izz2])
        c000 = G(ix0, iy0, iz0)
        c100 = G(ix0 + 1, iy0, iz0)
        c010 = G(ix0, iy0 + 1, iz0)
        c110 = G(ix0 + 1, iy0 + 1, iz0)
        c001 = G(ix0, iy0, iz0 + 1)
        c101 = G(ix0 + 1, iy0, iz0 + 1)
        c011 = G(ix0, iy0 + 1, iz0 + 1)
        c111 = G(ix0 + 1, iy0 + 1, iz0 + 1)
        # Trilinear interpolation
        c00 = (1 - tx) * c000 + tx * c100
        c01 = (1 - tx) * c001 + tx * c101
        c10 = (1 - tx) * c010 + tx * c110
        c11 = (1 - tx) * c011 + tx * c111
        c0 = (1 - ty) * c00 + ty * c10
        c1 = (1 - ty) * c01 + ty * c11
        out[i] = (1 - tz) * c0 + tz * c1
    return out


# --- Hamiltonian-aware convenience wrappers ---
def _ham_coords(ham) -> np.ndarray:
    """Extract (N,3) coordinates from a Hamiltonian-like object.

    Expects ham.atom_list to be an OrderedDict label->coord and returns coords in order.
    """
    if not hasattr(ham, 'atom_list'):
        raise AttributeError("Hamiltonian object must have 'atom_list' attribute")
    atom_list = ham.atom_list
    # OrderedDict values in current sorted order
    coords = np.array(list(atom_list.values()), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Extracted coordinates must be (N,3)")
    return coords


def atomistic_to_grid_ham(ham,
                          values: np.ndarray,
                          grid_shape: Tuple[int, ...],
                          sigma: Optional[float] = None,
                          cutoff: float = 2.5,
                          bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                          combine: str = "sum",
                          eps: float = 1e-15) -> np.ndarray:
    """Map per-atom values to a regular grid using positions from a Hamiltonian.

    Parameters
    ----------
    ham : Hamiltonian-like
        Object exposing 'atom_list' (ordered) and 'transport_axis'.
    values : (N,) array
        Per-atom data in the same order as the Hamiltonian's current sorting.
    grid_shape : tuple[int,...]
        Target grid shape (1D/2D/3D). For 2D, uses x–z slice (y squashed).
    sigma, cutoff, bounds : see atomistic_to_grid
    """
    coords = _ham_coords(ham)
    values = np.asarray(values)
    if values.ndim != 1 or values.shape[0] != coords.shape[0]:
        raise ValueError("values must be (N,) with N equal to number of atoms in Hamiltonian")
    t_axis = int(getattr(ham, 'transport_axis', 0))
    return atomistic_to_grid(coords, values, grid_shape,
                             sigma=sigma, cutoff=cutoff, bounds=bounds,
                             transport_axis=t_axis, combine=combine, eps=eps)


def grid_to_atomistic_ham(ham,
                          grid: np.ndarray,
                          bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
    """Sample a grid back onto atomistic positions from a Hamiltonian.

    For 1D grids, transport axis from the Hamiltonian is used. For 2D, treats the grid as an x–z slice
    (constant along y). For 3D, standard trilinear sampling is applied.
    """
    coords = _ham_coords(ham)
    t_axis = int(getattr(ham, 'transport_axis', 0))
    return grid_to_atomistic(coords, np.asarray(grid), bounds=bounds, transport_axis=t_axis)
