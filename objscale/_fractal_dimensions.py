from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import numba
from numba.typed import List
from scipy.ndimage import label
import warnings
from warnings import warn
from ._object_analysis import (
    label_structures,
    _merge_periodic_labels,
    remove_structures_touching_border_nan,
    remove_structure_holes,
    get_structure_areas,
    get_structure_perimeters,
    get_structure_height_width,
)
from ._utils import linear_regression, encase_in_value

__all__ = [
    'ensemble_correlation_dimension',
    'ensemble_box_dimension',
    'ensemble_information_dimension',
    'ensemble_box_renyi_dimension',
    'ensemble_sandbox_renyi_dimension',
    'individual_fractal_dimension',
    'get_coords_of_boundaries',
    'get_locations_from_pixel_sizes',
    'coarsen_array',
    'total_perimeter',
    'total_number',
    'individual_correlation_dimension',
    'isolate_nth_largest_structure',
    'label_size',
]


@numba.njit(parallel=True, fastmath=True)
def _box_sum_2d(arr: NDArray, factor: int) -> NDArray:
    """Sum a 2D array into ``(h//factor, w//factor)`` boxes of size ``factor``.

    Parallel over output rows; each thread reads a contiguous strip of input
    rows so the inner loop runs against private L2 (no strided cache aliasing
    on Zen 5, no thread contention). Caller is responsible for trimming the
    input so ``arr.shape[0]`` and ``arr.shape[1]`` are exact multiples of
    ``factor``.
    """
    h, w = arr.shape
    out_h = h // factor
    out_w = w // factor
    out = np.zeros((out_h, out_w), dtype=np.int64)
    for i in numba.prange(out_h):
        i0 = i * factor
        for j in range(out_w):
            j0 = j * factor
            s = 0
            for di in range(factor):
                for dj in range(factor):
                    s += arr[i0 + di, j0 + dj]
            out[i, j] = s
    return out


def ensemble_correlation_dimension(
    arrays: NDArray | list[NDArray],
    x_sizes: NDArray | None = None,
    y_sizes: NDArray | None = None,
    minlength: str | float = 'auto',
    maxlength: str | float = 'auto',
    interior_circles_only: bool = True,
    return_C_l: bool = False,
    bins: NDArray | int | None = None,
    point_reduction_factor: float = 1,
    nbins: int = 50
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    Calculate the correlation dimension D where C_l ∝ l^D for binary arrays.

    The resulting dimension is for the set of object edge points.

    This function is a thin wrapper around
    :func:`ensemble_sandbox_renyi_dimension` at ``q=2``: the
    Grassberger-Procaccia correlation integral is exactly the ``q=2`` case
    of the sandbox partition function (``M_i^{q-1} = M_i`` and
    ``sum_i M_i`` is the pair count). The function is preserved under its
    historical name because Grassberger-Procaccia is the standard
    designation for the ``q=2`` correlation dimension. For arbitrary
    ``q``, call :func:`ensemble_sandbox_renyi_dimension` directly.

    Parameters
    ----------
    arrays : list or np.ndarray
        List of binary arrays to calculate correlation dimension of.
    x_sizes : np.ndarray, optional
        Pixel sizes in the x direction. If None, assume all pixel dimensions are 1.
        This single grid is used for every array in ``arrays``.
    y_sizes : np.ndarray, optional
        Pixel sizes in the y direction. If None, assume all pixel dimensions are 1.
        This single grid is used for every array in ``arrays``.
    minlength : str or float, default='auto'
        Minimum length scale for correlation calculation. If 'auto', uses 8 times
        the minimum pixel size.
    maxlength : str or float, default='auto'
        Maximum length scale for correlation calculation. If 'auto', uses 0.33 times
        the minimum array dimension.
    interior_circles_only : bool, default=True
        If True, only use circle centers that are at least maxlength distance from
        all array edges to avoid boundary effects. Recommended!
    return_C_l : bool, default=False
        If True, return dimension, error, bins, C_l. Otherwise, return dimension, error.
    bins : None, int, or array-like, optional
        Values of l to use for the regression. Can be:
        - None: automatically calculate as logarithmically spaced intervals between
          3*minimum length and the array width or height using nbins points
        - int: number of logarithmically spaced bins to generate automatically
        - array-like: explicit bin edges to use
    point_reduction_factor : float, default=1
        Draw N/point_reduction_factor circles, where N is the total number of
        available circles. Choose the circle centers randomly. Must be >= 1.
    nbins : int, default=50
        Number of bins to use when bins=None or when bins is an int. Only used
        for automatic bin generation.

    Returns
    -------
    dimension : float
        The correlation dimension.
    error : float
        Error estimate for the dimension (95% CI half-width).
    bins : np.ndarray, optional
        The bins used for calculation. Only returned if return_C_l=True.
    C_l : np.ndarray, optional
        The correlation integral values (= sandbox ``Z_{q=2}`` =
        ``sum_i M_i``). Only returned if return_C_l=True.

    Raises
    ------
    ValueError
        If arrays contain NaN values, if pixel sizes are invalid, or if scale
        range is insufficient.

    See Also
    --------
    ensemble_sandbox_renyi_dimension : The general sandbox-method Rényi
        family for arbitrary ``q``. This function is the ``q=2`` case.
    ensemble_box_renyi_dimension : The box-counting Rényi family.
    """
    result = ensemble_sandbox_renyi_dimension(
        binary_arrays=arrays,
        q=2.0,
        set='edge',
        x_sizes=x_sizes,
        y_sizes=y_sizes,
        minlength=minlength,
        maxlength=maxlength,
        interior_circles_only=interior_circles_only,
        nbins=nbins,
        bins=bins,
        point_reduction_factor=point_reduction_factor,
        return_values=return_C_l,
    )
    # Preserve the pre-refactor failure-mode contract: when the fit
    # returns nan, the historical `ensemble_correlation_dimension` emitted
    # a warning and returned ``(nan, nan, [nan], [nan])`` from
    # ``return_C_l=True``. The sandbox wrapper instead returns
    # ``(nan, nan, full_bins, full_Z)``; rewrap here so that
    # downstream code keyed to the old contract keeps working.
    if return_C_l:
        dimension, error, bins_used, Z = result
        if not np.isfinite(dimension):
            warn('Not enough data to estimate correlation dimension, returning nan')
            return np.nan, np.nan, np.array([np.nan]), np.array([np.nan])
        return dimension, error, bins_used, Z
    dimension, error = result
    if not np.isfinite(dimension):
        warn('Not enough data to estimate correlation dimension, returning nan')
    return dimension, error


def individual_correlation_dimension(
    array: NDArray,
    n: int = 1,
    x_sizes: NDArray | None = None,
    y_sizes: NDArray | None = None,
    minlength: str | float = 'auto',
    maxlength: str | float = 'auto',
    return_C_l: bool = False,
    point_reduction_factor: float = 1,
    nbins: int = 50,
    filled: bool = True,
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    Calculate the correlation dimension of the Nth largest structure in an array.

    Removes border-touching structures, isolates the Nth largest remaining
    structure, crops to its bounding box, and computes the correlation dimension
    via :func:`ensemble_correlation_dimension`.

    Parameters
    ----------
    array : np.ndarray
        2-D binary array. May optionally have np.nan at borders; if not, a NaN
        border is added internally.
    n : int, default=1
        Which structure to analyze, ranked by pixel count (1 = largest).
    x_sizes : np.ndarray, optional
        Pixel sizes in the x direction. If None, assume all pixel dimensions are 1.
    y_sizes : np.ndarray, optional
        Pixel sizes in the y direction. If None, assume all pixel dimensions are 1.
    minlength : str or float, default='auto'
        Minimum length scale for correlation calculation. If 'auto', uses 8 times
        the minimum pixel size.
    maxlength : str or float, default='auto'
        Maximum length scale for correlation calculation. If 'auto', uses
        ``0.33 * max(bbox_height, bbox_width)`` of the isolated structure in
        physical units.
    return_C_l : bool, default=False
        If True, return dimension, error, bins, C_l. Otherwise, return dimension,
        error.
    filled : bool, default=True
        If True, fill interior holes in the isolated structure before computing
        the correlation dimension, so that only the outer boundary contributes.
        If False, holes are left as-is and interior boundaries are included.
    point_reduction_factor : float, default=1
        Draw N/point_reduction_factor circles, where N is the total number of
        available circles. Must be >= 1.
    nbins : int, default=50
        Number of logarithmically spaced bins for the correlation integral.

    Returns
    -------
    dimension : float
        The correlation dimension.
    error : float
        Error estimate for the dimension (95% confidence interval).
    bins : np.ndarray, optional
        The bins used for calculation. Only returned if return_C_l=True.
    C_l : np.ndarray, optional
        The correlation integral values. Only returned if return_C_l=True.

    Raises
    ------
    ValueError
        If array is not 2-D, n < 1, or n exceeds the number of available
        structures after border removal.
    """
    if array.ndim != 2:
        raise ValueError('array must be 2-dimensional')
    if n < 1:
        raise ValueError('n must be >= 1')

    # Pad with NaN border if not already present
    if not np.any(np.isnan(array)):
        array = np.pad(array.astype(np.float32), pad_width=1, mode='constant',
                       constant_values=np.nan)
        if x_sizes is not None:
            x_sizes = np.pad(x_sizes, pad_width=1, mode='edge')
        if y_sizes is not None:
            y_sizes = np.pad(y_sizes, pad_width=1, mode='edge')

    # Remove border-touching structures and clean NaN
    cleaned = remove_structures_touching_border_nan(array)
    binary = np.nan_to_num(cleaned, nan=0.0).astype(bool)

    # Isolate the Nth largest structure
    isolated = isolate_nth_largest_structure(binary, n=n)

    # Optionally fill interior holes so only the outer boundary contributes
    if filled:
        isolated = remove_structure_holes(isolated.astype(float)).astype(bool)

    # Crop to bounding box
    rows = np.any(isolated, axis=1)
    cols = np.any(isolated, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = isolated[rmin:rmax + 1, cmin:cmax + 1].astype(np.float32)

    # Pad with 1px of zeros to prevent toroidal wrap artifacts
    padded = np.pad(cropped, pad_width=1, mode='constant', constant_values=0)

    # Handle pixel sizes
    if x_sizes is not None:
        cropped_x = x_sizes[rmin:rmax + 1, cmin:cmax + 1]
        cropped_x = np.pad(cropped_x, pad_width=1, mode='edge')
    else:
        cropped_x = None

    if y_sizes is not None:
        cropped_y = y_sizes[rmin:rmax + 1, cmin:cmax + 1]
        cropped_y = np.pad(cropped_y, pad_width=1, mode='edge')
    else:
        cropped_y = None

    # Compute maxlength from crop dimensions if auto
    if maxlength == 'auto':
        if cropped_x is not None and cropped_y is not None:
            locs_x, locs_y = get_locations_from_pixel_sizes(cropped_x, cropped_y)
            h, w = cropped_x.shape
            phys_width = locs_x[h // 2, w - 1] - locs_x[h // 2, 0]
            phys_height = locs_y[h - 1, w // 2] - locs_y[0, w // 2]
            maxlength = 0.33 * max(phys_width, phys_height)
        else:
            crop_h, crop_w = cropped.shape
            maxlength = 0.33 * float(max(crop_h, crop_w))

    return ensemble_correlation_dimension(
        [padded],
        x_sizes=cropped_x,
        y_sizes=cropped_y,
        minlength=minlength,
        maxlength=maxlength,
        interior_circles_only=False,
        return_C_l=return_C_l,
        point_reduction_factor=point_reduction_factor,
        nbins=nbins,
    )


def _one_sided_edge_mask(binary: NDArray) -> NDArray:
    """One-sided 4-connected edge mask: 1-pixels with at least one 0-neighbor.

    Pixels on the array boundary count as having a "0 neighbor" outside the
    domain, matching the convention of `set='ones'` for boundary handling.
    Returns an int8 mask.
    """
    b = binary.astype(np.int8, copy=False)
    h, w = b.shape
    # neighbor[i,j] = OR of (b[i,j] == 1 AND any 4-neighbor == 0)
    has_zero_nbr = np.zeros((h, w), dtype=bool)
    # Up neighbor (treat domain edge as 0)
    has_zero_nbr[1:, :] |= (b[:-1, :] == 0)
    has_zero_nbr[0, :] = True
    # Down
    has_zero_nbr[:-1, :] |= (b[1:, :] == 0)
    has_zero_nbr[-1, :] = True
    # Left
    has_zero_nbr[:, 1:] |= (b[:, :-1] == 0)
    has_zero_nbr[:, 0] = True
    # Right
    has_zero_nbr[:, :-1] |= (b[:, 1:] == 0)
    has_zero_nbr[:, -1] = True
    return ((b == 1) & has_zero_nbr).astype(np.int8)


def _box_renyi_from_set(
    set_arrays: list[NDArray],
    q_arr: NDArray,
    box_sizes: NDArray,
    box_origin_shift: tuple[float, float],
) -> tuple[NDArray, NDArray, NDArray]:
    """Core box-Rényi routine: count 1-pixels per box, fit slopes.

    Parameters
    ----------
    set_arrays : list of int8 arrays
        Binary arrays where 1 = pixel in the set being measured. The caller
        is responsible for whatever preprocessing turns the user's input into
        a "set" (e.g. one-sided edge mask for ``set='edge'``).
    q_arr : np.ndarray
        1-D array of Rényi orders.
    box_sizes : np.ndarray
        Integer box sizes (in pixels), already filtered to be valid for the
        input shapes.
    box_origin_shift : tuple of float
        ``(sx, sy)`` fractional shifts of the box grid origin, in units of
        the current box size.

    Returns
    -------
    D_q : np.ndarray, shape (len(q_arr),)
        Estimated Rényi dimension for each ``q``.
    err : np.ndarray, shape (len(q_arr),)
        95% CI half-width for each estimate.
    partition : np.ndarray, shape (len(q_arr), len(box_sizes))
        ``Z_q^(p)(eps)`` for q != 1, ``-S_1(eps)`` for q == 1, in the same
        order as ``q_arr``. Useful for plotting and diagnostics.
    """
    nq = q_arr.shape[0]
    nb = box_sizes.shape[0]
    partition = np.full((nq, nb), np.nan, dtype=np.float64)

    sx, sy = box_origin_shift

    for k, factor in enumerate(box_sizes):
        factor = int(factor)
        # Pool box counts and total interior pixels across the ensemble.
        # The caller (ensemble_box_renyi_dimension) is responsible for ensuring
        # that every array fits at every factor; if h_full or w_full ever
        # came out 0 here, the ensemble would silently shrink at that factor
        # and bias the slope. Treat that as an internal error.
        pooled_n: list[NDArray] = []
        V_total = 0  # total interior pixel area summed across arrays
        for arr_idx, arr in enumerate(set_arrays):
            sy_pix = int(round(sy * factor))
            sx_pix = int(round(sx * factor))
            shifted = arr[sy_pix:, sx_pix:]
            h_full = (shifted.shape[0] // factor) * factor
            w_full = (shifted.shape[1] // factor) * factor
            if h_full == 0 or w_full == 0:
                raise AssertionError(
                    f'Internal error: array {arr_idx} of shape {arr.shape} cannot '
                    f'fit a single box of size {factor} after shift {(sx_pix, sy_pix)}. '
                    f'The wrapper should have filtered this box size out.'
                )
            trimmed = shifted[:h_full, :w_full]
            if not trimmed.flags['C_CONTIGUOUS']:
                trimmed = np.ascontiguousarray(trimmed)
            box_counts = _box_sum_2d(trimmed, factor)
            pooled_n.append(box_counts.ravel())
            V_total += h_full * w_full

        if not pooled_n or V_total == 0:
            continue

        n = np.concatenate(pooled_n)
        n_sum = float(n.sum())
        if n_sum <= 0:
            continue

        for qi in range(nq):
            q = float(q_arr[qi])
            if abs(q - 1.0) < 1e-10:
                # Entropy form: requires probability normalization (sum = 1).
                # Use log10 so the slope vs log10(eps) is directly -D_1.
                p = n[n > 0] / n_sum
                S1 = -float(np.sum(p * np.log10(p)))
                partition[qi, k] = S1
            else:
                # Geometric / interior-pixel-area normalization.
                # Z_q = sum_i (n_i / V) ** q
                if q <= 0:
                    n_pos = n[n > 0]
                    Zq = float(np.sum((n_pos / V_total) ** q))
                else:
                    Zq = float(np.sum((n / V_total) ** q))
                partition[qi, k] = Zq

    # Fit slopes
    log_eps = np.log10(box_sizes.astype(np.float64))
    D_q = np.full(nq, np.nan, dtype=np.float64)
    err = np.full(nq, np.nan, dtype=np.float64)
    for qi in range(nq):
        q = float(q_arr[qi])
        y = partition[qi].copy()
        if abs(q - 1.0) < 1e-10:
            # S_1 itself is linear in log_eps with slope -D_1.
            yfit = y
        else:
            # Z_q is non-negative; log it.
            yfit = np.where(y > 0, np.log10(y), np.nan)
        (slope, _), (slope_err, _) = linear_regression(log_eps, yfit)
        if not np.isfinite(slope):
            continue
        if abs(q - 1.0) < 1e-10:
            D_q[qi] = -slope
            err[qi] = slope_err
        else:
            D_q[qi] = slope / (q - 1.0)
            err[qi] = slope_err / abs(q - 1.0)

    return D_q, err, partition


def ensemble_box_renyi_dimension(
    binary_arrays: NDArray | list[NDArray],
    q: float | NDArray = 0.0,
    set: str = 'edge',
    box_sizes: str | NDArray = 'default',
    max_box_size: int | None = None,
    min_box_size: int = 8,
    box_origin_shift: tuple[float, float] = (0.0, 0.0),
    return_values: bool = False,
):
    """Calculate the generalized Rényi dimension D_q via box counting.

    The Rényi dimensions form a one-parameter family of fractal dimensions
    indexed by an order ``q``, generalizing the box-counting dimension
    (``q=0``), information dimension (``q=1``), and correlation dimension
    (``q=2``). For a scale-invariant set, the partition function

    .. math::
        Z_q(\\varepsilon) = \\sum_i p_i^q \\propto \\varepsilon^{(q-1) D_q}

    where ``p_i`` is the (probability or density) measure of the set in box
    ``i`` of size ``\\varepsilon``. The slope of ``log Z_q`` vs
    ``log \\varepsilon`` is ``(q-1) D_q``. At ``q=1`` the formula has a
    removable singularity that resolves to the Shannon entropy form
    ``S_1 = -sum p_i log p_i``, which scales as ``-D_1 log \\varepsilon``.

    For monofractal sets (e.g. level sets of fractional Brownian motion),
    ``D_q`` is constant in ``q``. For multifractal sets, ``D_q`` is a
    monotonically decreasing function of ``q`` whose values quantify the
    distribution's heterogeneity at different moment orders.

    Parameters
    ----------
    binary_arrays : np.ndarray or list of np.ndarray
        2D binary arrays. May contain 0/1 integers, booleans, or floats.
    q : float or np.ndarray, default=0.0
        Rényi order(s). May be a scalar or a 1-D array of values. ``q=0``
        gives the box-counting dimension; ``q=1`` gives the information
        dimension; ``q=2`` gives the correlation dimension.
    set : {'edge', 'ones'}, default='edge'
        Which set is being measured.

        * ``'edge'``: the boundary of the binary array, computed as a
          one-sided 4-connected edge mask (1-pixels that have at least one
          0-neighbor; pixels on the array border count as having a "0
          neighbor" outside the domain).
        * ``'ones'``: the set of 1-pixels themselves, no preprocessing.
    box_sizes : 'default' or array-like, default='default'
        Integer box sizes in pixels. If 'default', powers of 2 from
        ``min_box_size`` up to ``max_box_size``.
    max_box_size : int or None, default=None
        Largest box size in pixels. If None, uses the smaller array
        dimension (i.e. the largest box that fits at all).
    min_box_size : int, default=8
        Smallest box size in pixels. The default of 8 keeps the fit
        away from pixel-discretization noise at the smallest scales.
    box_origin_shift : tuple of (float, float), default=(0.0, 0.0)
        Fractional shift ``(sx, sy)`` of the box-grid origin, in units of
        the current box size. At each box size ``factor``, the actual
        integer pixel shift is ``int(round(sx * factor))`` along x and
        ``int(round(sy * factor))`` along y. Used to probe sensitivity to
        the box-grid alignment.
    return_values : bool, default=False
        If True, also return the partition function values used in the fit.

    Returns
    -------
    D_q : float or np.ndarray
        Estimated Rényi dimension. Scalar if ``q`` was scalar, array
        otherwise.
    err : float or np.ndarray
        95% CI half-width for the estimate(s).
    box_sizes : np.ndarray, optional
        The box sizes used. Returned only if ``return_values=True``.
    partition : np.ndarray, optional
        The partition function values: ``Z_q^(p)(eps)`` for ``q != 1`` (with
        geometric normalization ``p_i = n_i / V``, where ``V`` is the total
        interior pixel area at this ``eps``), and the Shannon entropy
        ``S_1(eps)`` for ``q == 1``. Shape ``(len(q), len(box_sizes))`` if
        ``q`` is array, ``(len(box_sizes),)`` if scalar. Returned only if
        ``return_values=True``.

    Notes
    -----
    **Interior boxes only.** Boxes that would extend past the domain edge
    are not counted. The function trims each input array to a multiple of
    the current box size (after applying any ``box_origin_shift``) and
    discards the remainder. This matches the philosophy of
    ``ensemble_correlation_dimension(interior_circles_only=True)``.

    **Normalization.** For ``q != 1``, ``p_i = n_i / V(\\varepsilon)`` where
    ``V`` is the total interior pixel area at this box size — a purely
    geometric normalization that removes coverage-fluctuation bias from the
    fitted slope. For ``q == 1``, the Shannon entropy requires probability
    normalization ``p_i = n_i / sum_j n_j``; the two coincide for
    uniform-density fields.

    **Boundary convention for** ``set='edge'``. The edge mask is one-sided:
    a pixel is in the edge set iff it is 1 and has at least one 0-neighbor.
    This differs slightly from the convention used by older versions of
    ``ensemble_box_dimension``, which counted a coarsened cell as "edge"
    iff it contained both 0s and 1s.

    See Also
    --------
    ensemble_box_dimension : Equivalent to ``q=0``.
    ensemble_information_dimension : Equivalent to ``q=1``.
    ensemble_correlation_dimension : Theoretically equivalent to ``q=2`` for
        the same set, computed via the Grassberger-Procaccia pairwise method.
    """
    # Normalize inputs
    if isinstance(binary_arrays, np.ndarray):
        binary_arrays = [binary_arrays]
    if len(binary_arrays) == 0:
        raise ValueError('binary_arrays must be non-empty')

    if np.any([np.any(np.isnan(arr)) for arr in binary_arrays]):
        raise ValueError('arrays must not contain NaN values')

    if set not in ('edge', 'ones'):
        raise ValueError(f'set={set!r} not supported (supported values are "edge" or "ones")')

    q_scalar = np.isscalar(q)
    q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))

    # Resolve box_sizes. The maximum box size is bounded by the *smallest*
    # array in the ensemble (after subtracting any box_origin_shift overhead),
    # so that every array contributes a fully-tiled interior region at every
    # box size. Otherwise, smaller arrays would silently drop out at large
    # box sizes and bias the slope.
    if isinstance(box_sizes, str):
        if box_sizes != 'default':
            raise ValueError(f'box_sizes={box_sizes} not supported')
        box_sizes_arr = 2 ** np.arange(1, 15)
    else:
        box_sizes_arr = np.asarray(box_sizes)

    smallest_dim = min(min(arr.shape) for arr in binary_arrays)
    # box_origin_shift can push the start past the first row/col by up to
    # factor-1 pixels, so the effective "available" extent is one factor
    # short in the worst case. Require shape >= factor + (factor - 1) =
    # 2*factor - 1, i.e. factor <= (shape + 1) // 2 in the worst case.
    # For shift=(0, 0) (the common case), the bound is just factor <= shape.
    sx, sy = box_origin_shift
    max_shift_frac = max(abs(sx), abs(sy))
    if max_shift_frac >= 1.0:
        raise ValueError(
            f'box_origin_shift fractions must lie in [0, 1), got {box_origin_shift}'
        )
    # Worst-case shift in pixels at factor F is round(max_shift_frac * F).
    # Require F + round(max_shift_frac * F) <= smallest_dim.
    # Rearranging: F <= smallest_dim / (1 + max_shift_frac).
    auto_max = int(smallest_dim / (1.0 + max_shift_frac))
    if max_box_size is None:
        max_box_size_eff = auto_max
    else:
        max_box_size_eff = min(int(max_box_size), auto_max)

    box_sizes_arr = box_sizes_arr[box_sizes_arr <= max_box_size_eff]
    box_sizes_arr = box_sizes_arr[box_sizes_arr >= min_box_size]
    box_sizes_arr = np.unique(box_sizes_arr.astype(np.int64))
    if box_sizes_arr.size < 3:
        raise ValueError(
            f'Need at least 3 valid box sizes for a slope fit, got {box_sizes_arr.size}. '
            f'Smallest array dimension is {smallest_dim}, max usable box size is '
            f'{max_box_size_eff}. Reduce min_box_size, raise max_box_size, increase '
            f'array size, or pass an explicit box_sizes array.'
        )

    # Build the set arrays
    set_arrays: list[NDArray] = []
    for arr in binary_arrays:
        if arr.ndim != 2:
            raise ValueError('binary_arrays must be 2-dimensional')
        if set == 'ones':
            set_arrays.append((np.asarray(arr) > 0).astype(np.int8))
        else:  # 'edge'
            set_arrays.append(_one_sided_edge_mask(np.asarray(arr) > 0))

    D_q, err, partition = _box_renyi_from_set(
        set_arrays, q_arr, box_sizes_arr, box_origin_shift
    )

    if q_scalar:
        D_q_out = float(D_q[0])
        err_out = float(err[0])
        partition_out = partition[0]
    else:
        D_q_out = D_q
        err_out = err
        partition_out = partition

    if return_values:
        return D_q_out, err_out, box_sizes_arr, partition_out
    return D_q_out, err_out


def ensemble_box_dimension(
    binary_arrays: NDArray | list[NDArray],
    set: str = 'edge',
    max_box_size: int | None = None,
    min_box_size: int = 8,
    box_sizes: str | NDArray = 'default',
    return_values: bool = False
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    Calculate the ensemble box-counting dimension of binary arrays.

    Equivalent to ``ensemble_box_renyi_dimension(..., q=0)``. Kept as a
    convenience wrapper because box counting is the most familiar special
    case.

    Parameters
    ----------
    binary_arrays : list of np.ndarray or np.ndarray
        A list of 2D binary arrays or a single 2D binary array.
    set : str, default='edge'
        Specifies which set to consider for box counting:
        - 'edge': Box dimension of the set of boundaries (1-pixels with at
          least one 0-neighbor).
        - 'ones': Box dimension of the set of 1-pixels.
    max_box_size : int or None, default=None
        Largest box size in pixels. If None, uses the smaller array dimension.
    min_box_size : int, default=8
        Smallest box size in pixels. The default of 8 keeps the fit
        away from pixel-discretization noise at the smallest scales.
    box_sizes : array-like or 'default', default='default'
        Box sizes used. If 'default', uses powers of 2 from ``min_box_size``
        up to ``max_box_size``.
    return_values : bool, default=False
        If True, return additional data used in the calculation.

    Returns
    -------
    dimension : float
        The estimated box-counting dimension.
    error : float
        The error of the estimate.
    box_sizes : np.ndarray, optional
        Box sizes used. Only returned if return_values=True.
    number_boxes : np.ndarray, optional
        Number of nonempty boxes (pooled across the input ensemble) at each
        box size. Only returned if return_values=True.

    Raises
    ------
    ValueError
        If an unsupported value is provided for 'set' or if arrays contain NaN
        values.
    """
    return ensemble_box_renyi_dimension(
        binary_arrays,
        q=0.0,
        set=set,
        box_sizes=box_sizes,
        max_box_size=max_box_size,
        min_box_size=min_box_size,
        return_values=return_values,
    )


def ensemble_information_dimension(
    binary_arrays: NDArray | list[NDArray],
    method: str = 'sandbox',
    set: str = 'edge',
    return_values: bool = False,
    **kwargs,
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    Calculate the ensemble information dimension D_1 of binary arrays.

    The information dimension is the ``q=1`` member of the Rényi-dimension
    family. It has a removable singularity at ``q=1`` that resolves to the
    Shannon entropy form

    .. math::
        S_1(\\varepsilon) = -\\sum_i p_i \\log p_i \\sim -D_1 \\log \\varepsilon
        \\qquad (\\text{box estimator})

    or, equivalently, the set-centered log average

    .. math::
        \\langle \\log M(r) \\rangle \\sim D_1 \\log r
        \\qquad (\\text{sandbox estimator}).

    Both estimators converge to the same ``D_1``.

    Two methods are supported:

    * ``method='sandbox'`` (default): forwards to
      :func:`ensemble_sandbox_renyi_dimension` at ``q=1``. Set-centered
      balls at continuous radii. Lower grid-quantization noise; recommended
      for most use cases.
    * ``method='box'``: forwards to :func:`ensemble_box_renyi_dimension`
      at ``q=1``. Fixed-grid box counting with the Shannon entropy form.

    Parameters
    ----------
    binary_arrays : list of np.ndarray or np.ndarray
        A list of 2D binary arrays or a single 2D binary array.
    method : {'sandbox', 'box'}, default='sandbox'
        Which estimator to use. ``'sandbox'`` is the default because it
        avoids the grid-alignment noise that box counting suffers from at
        ``q=1``.
    set : {'edge', 'ones'}, default='edge'
        Which set to measure. See :func:`ensemble_sandbox_renyi_dimension`
        or :func:`ensemble_box_renyi_dimension`.
    return_values : bool, default=False
        If True, also return the bins and partition values used in the fit.
    **kwargs
        Method-specific options forwarded to the underlying estimator.

        For ``method='sandbox'``: ``x_sizes``, ``y_sizes``, ``minlength``,
        ``maxlength``, ``interior_circles_only``, ``nbins``, ``bins``,
        ``point_reduction_factor``.

        For ``method='box'``: ``max_box_size``, ``min_box_size``,
        ``box_sizes``, ``box_origin_shift``.

    Returns
    -------
    dimension : float
        The estimated information dimension.
    error : float
        The 95% CI half-width.
    bins : np.ndarray, optional
        Distance bins (sandbox) or box sizes (box). Only returned if
        ``return_values=True``.
    partition : np.ndarray, optional
        Partition function values: ``\\langle log10 M(r)\\rangle``
        (sandbox) or Shannon entropy ``S_1(eps)`` (box). Only returned if
        ``return_values=True``.

    See Also
    --------
    ensemble_sandbox_renyi_dimension : Sandbox-method Rényi family.
    ensemble_box_renyi_dimension : Box-counting Rényi family.
    ensemble_box_dimension : The ``q=0`` box-counting wrapper.
    ensemble_correlation_dimension : The ``q=2`` sandbox-method wrapper.
    """
    if method == 'sandbox':
        return ensemble_sandbox_renyi_dimension(
            binary_arrays,
            q=1.0,
            set=set,
            return_values=return_values,
            **kwargs,
        )
    elif method == 'box':
        return ensemble_box_renyi_dimension(
            binary_arrays,
            q=1.0,
            set=set,
            return_values=return_values,
            **kwargs,
        )
    else:
        raise ValueError(
            f"method={method!r} not supported (use 'sandbox' or 'box')"
        )


def ensemble_sandbox_renyi_dimension(
    binary_arrays: NDArray | list[NDArray],
    q: float | NDArray = 2.0,
    set: str = 'edge',
    x_sizes: NDArray | None = None,
    y_sizes: NDArray | None = None,
    minlength: str | float = 'auto',
    maxlength: str | float = 'auto',
    interior_circles_only: bool | str = True,
    nbins: int = 50,
    bins: NDArray | int | None = None,
    point_reduction_factor: float = 1,
    return_values: bool = False,
):
    """Sandbox-method estimate of the Rényi dimension D_q of binary arrays.

    For each set point i (the "sandbox center"), count the number M_i(r) of
    other set points within distance r, then aggregate

    .. math::
        Z_q(r) = \\sum_i M_i(r)^{q-1}    \\quad (q \\ne 1)

        Z_1(r) = \\sum_i \\log_{10} M_i(r)  \\quad (q = 1, set-point average)

    For a multifractal set, ``\\langle M(r)^{q-1}\\rangle \\sim r^{(q-1) D_q}``,
    so the slope of ``log Z_q`` vs ``log r`` is ``(q-1) D_q`` for ``q != 1``,
    recovered as ``D_q = slope / (q - 1)``. At ``q = 1`` the formula has a
    removable singularity that resolves to ``\\langle \\log M(r)\\rangle \\sim
    D_1 \\log r``, fit directly.

    The sandbox method is a strict generalization of the
    Grassberger-Procaccia correlation integral to arbitrary ``q``: at
    ``q = 2``, ``M_i^{q-1} = M_i`` and ``sum_i M_i = `` the pair count, so
    ``D_2`` is the Grassberger-Procaccia correlation dimension. The function
    :func:`ensemble_correlation_dimension` is now a thin wrapper around this
    one at ``q=2``.

    Compared to the box-counting Rényi family
    (:func:`ensemble_box_renyi_dimension`), sandbox samples the partition
    function at continuous radii instead of fixed-grid tiles, and uses a
    set-centered measure that bounds ``M_i >= 1`` and so behaves better at
    ``q < 1`` where box counting suffers from grid-quantization noise.

    References
    ----------
    Tél, Fülöp, Vicsek 1989, *Physica A* 159, 155–166.
    Vicsek 1992, *Fractal Growth Phenomena*, Ch. 3.

    Parameters
    ----------
    binary_arrays : np.ndarray or list of np.ndarray
        2D binary arrays. May contain 0/1 integers, booleans, or floats.
    q : float or np.ndarray, default=2.0
        Rényi order(s). Scalar or 1-D array. ``q == 1`` (within ``1e-10``)
        is special-cased to the log form.
    set : {'edge', 'ones'}, default='edge'
        Which set is being measured. Same convention as
        :func:`ensemble_box_renyi_dimension`.

        * ``'edge'`` — one-sided 4-connected edge mask of the binary array
          (1-pixels with at least one 0-neighbor; pixels on the array border
          count as having a "0 neighbor" outside the domain).
        * ``'ones'`` — the set of 1-pixels itself.
    x_sizes, y_sizes : np.ndarray, optional
        Per-pixel physical sizes. If None, uniform unit pixels.
    minlength, maxlength : str or float, default='auto'
        Distance scale range. ``'auto'`` = ``8 * min pixel size`` and
        ``0.33 * min array dimension`` respectively. The default
        ``minlength`` of 8× pixel size keeps the fit away from
        pixel-discretization noise at the smallest scales.
    interior_circles_only : bool or {'truncate'}, default=True
        Boundary handling for sandbox circles. Three modes are supported:

        * ``True`` — only sandbox centers at least ``maxlength`` from every
          domain edge contribute, so every circle fits entirely inside the
          domain. Avoids truncation bias but discards small-radius
          information from set points near the boundary.
        * ``False`` — every set point is used as a center; circles that
          extend past a domain edge are silently truncated, biasing
          ``M_i(r)`` low at large ``r``.
        * ``'truncate'`` — every set point is used as a center, but each
          center's contribution is capped at radii ``r <= d_edge(center)``,
          where ``d_edge`` is the distance to the nearest domain edge. ``Z``
          remains a raw sum over admissible centers (no per-bin
          normalization), so the count-vs-radius signature includes the
          fact that the contributing-center population shrinks with ``r``;
          the fitted slope therefore picks up an extra
          ``d log N(r) / d log r`` term on top of ``(q-1) D_q``.
    nbins : int, default=50
        Number of log-spaced distance bins (used when ``bins`` is None).
    bins : np.ndarray or int, optional
        Explicit bin array (ascending physical distances) or an int
        (number of log-spaced bins).
    point_reduction_factor : float, default=1
        Subsample sandbox centers by this factor (must be ``>= 1``).
    return_values : bool, default=False
        If True, return ``(D_q, err, bins, Z)`` instead of ``(D_q, err)``.
        For scalar ``q``, ``Z`` has shape ``(B,)``. For vector ``q``, ``Z``
        has shape ``(len(q), B)``. At the ``q=1`` index, ``Z`` holds the
        per-center mean ``\\langle log10 M_i\\rangle`` (already normalized).

    Returns
    -------
    D_q : float or np.ndarray
        Rényi dimension estimate(s). Scalar if ``q`` is scalar.
    err : float or np.ndarray
        95% CI half-width.
    bins : np.ndarray, optional
        Distance bins used. Returned only if ``return_values=True``.
    Z : np.ndarray, optional
        Sandbox partition function values.

    See Also
    --------
    ensemble_box_renyi_dimension : Box-based Rényi dimension family.
    ensemble_correlation_dimension : Thin wrapper around this function at
        ``q=2``, kept under its historical Grassberger-Procaccia name.
    ensemble_information_dimension : ``q=1`` wrapper, supports both
        ``method='sandbox'`` (default) and ``method='box'``.
    """
    # ----- normalize inputs -----
    if isinstance(binary_arrays, np.ndarray):
        binary_arrays = [binary_arrays]
    if len(binary_arrays) == 0:
        raise ValueError('binary_arrays must be non-empty')

    if set not in ('edge', 'ones'):
        raise ValueError(f"set={set!r} not supported (use 'edge' or 'ones')")

    if point_reduction_factor < 1:
        raise ValueError('point_reduction_factor must be >= 1')

    q_scalar = np.isscalar(q)
    q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
    Q = q_arr.shape[0]

    # Mask of q entries within tolerance of 1.0 (entropy-form special case).
    # Stored as a per-element bool so duplicate q==1 entries are all handled.
    is_q1 = np.abs(q_arr - 1.0) < 1e-10

    # Use the first array's shape to set up the physical grid (caller is
    # responsible for passing arrays of compatible shape; preserved from GP).
    if x_sizes is None:
        x_sizes = np.ones(binary_arrays[0].shape, dtype=np.float32)
    if y_sizes is None:
        y_sizes = np.ones(binary_arrays[0].shape, dtype=np.float32)
    locations_x, locations_y = get_locations_from_pixel_sizes(x_sizes, y_sizes)

    h = x_sizes.shape[0]
    w = x_sizes.shape[1]

    # ----- resolve scale range -----
    if maxlength == 'auto':
        maxlength = 0.33 * min(
            (locations_x[int(h / 2), w - 1] - locations_x[int(h / 2), 0]),
            (locations_y[h - 1, int(w / 2)] - locations_y[0, int(w / 2)]),
        )
    if minlength == 'auto':
        minlength = 8 * min(np.nanmin(x_sizes), np.nanmin(y_sizes))

    # ----- validate -----
    if np.any([np.any(np.isnan(arr)) for arr in binary_arrays]):
        raise ValueError('arrays must not contain NaN values')
    if np.any(np.isnan(x_sizes)) or np.any(np.isnan(y_sizes)):
        raise ValueError('x_sizes and y_sizes cannot contain NaN values')
    if np.any(x_sizes <= 0) or np.any(y_sizes <= 0):
        raise ValueError('x_sizes and y_sizes must be positive')

    if bins is None:
        bins = np.geomspace(minlength, maxlength, nbins)
    elif isinstance(bins, int):
        bins = np.geomspace(minlength, maxlength, bins)
    bins = np.asarray(bins, dtype=np.float64)

    if bins[-1] <= bins[0]:
        raise ValueError(
            f'bin maximum length ({bins[-1]:.3f}) must be greater than bin '
            f'minimum length ({bins[0]:.3f}); or if bins are passed, they '
            f'must be increasing. Did you pass invalid values for '
            f'minlength/maxlength?'
        )
    if bins[-1] / bins[0] < 10:
        raise ValueError(
            f'Available scale ratio ({maxlength / minlength:.2f}) is less '
            f'than 10. Need at least one order of magnitude separation for '
            f'reliable dimension estimation.'
        )

    # Resolve the boundary-handling mode. Accept booleans for backward
    # compatibility plus the string 'truncate' for the per-center cap.
    if isinstance(interior_circles_only, str):
        if interior_circles_only != 'truncate':
            raise ValueError(
                f"interior_circles_only={interior_circles_only!r} not "
                f"supported (use True, False, or 'truncate')"
            )
        boundary_mode = 'truncate'
    elif interior_circles_only:
        boundary_mode = 'interior'
    else:
        boundary_mode = 'open'

    if boundary_mode == 'interior':
        domain_width = locations_x[int(h / 2), w - 1] - locations_x[int(h / 2), 0]
        domain_height = locations_y[h - 1, int(w / 2)] - locations_y[0, int(w / 2)]
        interior_width = domain_width - 2 * maxlength
        interior_height = domain_height - 2 * maxlength
        min_pixel_x = np.min(x_sizes)
        min_pixel_y = np.min(y_sizes)
        interior_pixels_x = interior_width / min_pixel_x if min_pixel_x > 0 else 0
        interior_pixels_y = interior_height / min_pixel_y if min_pixel_y > 0 else 0
        if interior_pixels_x < 5 or interior_pixels_y < 5:
            raise ValueError(
                f'interior_circles_only=True requires that maxlength leaves a usable '
                f'interior region, but maxlength={maxlength:.1f} with a domain of '
                f'{domain_width:.1f} x {domain_height:.1f} leaves only '
                f'{max(interior_pixels_x, 0):.0f} x {max(interior_pixels_y, 0):.0f} '
                f'pixels of interior (need at least 5x5). Reduce maxlength or set '
                f'interior_circles_only=False.'
            )

    bins_sq = bins ** 2
    max_bin = float(bins[-1])
    max_bin_sq = float(bins_sq[-1])

    # ----- per-array kernel call, accumulate Z_total -----
    Z_total = np.zeros((Q, bins.shape[0]), dtype=np.float64)
    N_per_bin_total = np.zeros(bins.shape[0], dtype=np.int64)
    n_centers_total = 0

    for array in binary_arrays:
        if np.any(array.shape != x_sizes.shape):
            raise ValueError(
                f'All arrays must be same shape as pixel sizes (currently '
                f'{array.shape} and {x_sizes.shape}, respectively)'
            )

        # Build the set mask, then extract pixel coordinates of set points.
        if set == 'edge':
            set_mask = _one_sided_edge_mask(np.asarray(array) > 0)
        else:  # set == 'ones'
            set_mask = (np.asarray(array) > 0).astype(np.int8)

        all_set_coords = np.argwhere(set_mask > 0)

        if all_set_coords.shape[0] == 0:
            continue

        # Per-set-point distance to the nearest domain edge (used by both
        # the interior filter and the truncate per-center cap).
        coord_locations_x = locations_x[all_set_coords[:, 0], all_set_coords[:, 1]]
        coord_locations_y = locations_y[all_set_coords[:, 0], all_set_coords[:, 1]]
        dist_to_left = coord_locations_x - locations_x[int(h / 2), 0]
        dist_to_right = locations_x[int(h / 2), w - 1] - coord_locations_x
        dist_to_top = coord_locations_y - locations_y[0, int(w / 2)]
        dist_to_bottom = locations_y[h - 1, int(w / 2)] - coord_locations_y
        min_dist_to_any_edge = np.minimum.reduce(
            [dist_to_left, dist_to_right, dist_to_top, dist_to_bottom]
        )
        del coord_locations_x, coord_locations_y
        del dist_to_left, dist_to_right, dist_to_top, dist_to_bottom

        if boundary_mode == 'interior':
            interior_mask = min_dist_to_any_edge >= maxlength
            sandbox_centers_idx = all_set_coords[interior_mask]
            sandbox_centers_dedge = min_dist_to_any_edge[interior_mask]
        else:
            sandbox_centers_idx = all_set_coords
            sandbox_centers_dedge = min_dist_to_any_edge

        if sandbox_centers_idx.shape[0] == 0:
            continue

        # Subsample sandbox centers
        if point_reduction_factor > 1:
            n_keep = int(len(sandbox_centers_idx) / point_reduction_factor)
            if n_keep == 0:
                continue
            chosen = np.random.choice(
                np.arange(len(sandbox_centers_idx)), n_keep, replace=False
            )
            sandbox_centers_idx = sandbox_centers_idx[chosen]
            sandbox_centers_dedge = sandbox_centers_dedge[chosen]

        # Convert to physical coordinates
        boundary_phys_x = locations_x[all_set_coords[:, 0], all_set_coords[:, 1]]
        boundary_phys_y = locations_y[all_set_coords[:, 0], all_set_coords[:, 1]]
        boundary_phys = np.column_stack([boundary_phys_x, boundary_phys_y])
        del boundary_phys_x, boundary_phys_y

        center_phys_x = locations_x[sandbox_centers_idx[:, 0], sandbox_centers_idx[:, 1]]
        center_phys_y = locations_y[sandbox_centers_idx[:, 0], sandbox_centers_idx[:, 1]]
        center_phys = np.column_stack([center_phys_x, center_phys_y])
        n_centers_here = center_phys.shape[0]
        del set_mask, all_set_coords, sandbox_centers_idx
        del center_phys_x, center_phys_y, min_dist_to_any_edge

        # Per-center maximum squared radius. For 'truncate' we cap at the
        # squared distance to the nearest edge; otherwise we set the cap
        # to the largest bin so every bin admits every center.
        if boundary_mode == 'truncate':
            r_max_sq_per_center = np.minimum(
                sandbox_centers_dedge.astype(np.float64) ** 2, max_bin_sq
            )
        else:
            r_max_sq_per_center = np.full(
                n_centers_here, max_bin_sq, dtype=np.float64
            )
        del sandbox_centers_dedge

        # Sort boundary points by physical y for the kernel's bbox trick
        sort_order = np.argsort(boundary_phys[:, 1])
        sorted_boundary_phys = boundary_phys[sort_order]
        del sort_order, boundary_phys

        Z_chunk, N_chunk = _sandbox_partition(
            center_phys, sorted_boundary_phys, bins_sq, max_bin, q_arr, is_q1,
            r_max_sq_per_center,
        )
        Z_total += Z_chunk
        N_per_bin_total += N_chunk
        n_centers_total += n_centers_here

    # All modes: Z is the raw partition-function sum (sum_i M_i^(q-1) for
    # q!=1, sum_i log10 M_i for q==1). For q==1 we convert to a per-center
    # mean so the slope vs log r is D_1 directly. In 'truncate' mode the
    # contributing-center count varies with r; we deliberately do NOT
    # divide by N_per_bin, since the user wants Z to remain a count and
    # the resulting d log N(r) / d log r contribution to the slope is
    # part of the methodological signature of the truncate approach.
    if n_centers_total > 0 and is_q1.any():
        Z_total[is_q1, :] /= float(n_centers_total)

    # ----- linear regression per q -----
    log_r = np.log10(bins)
    D_q = np.full(Q, np.nan, dtype=np.float64)
    err_arr = np.full(Q, np.nan, dtype=np.float64)
    for qi in range(Q):
        y = Z_total[qi].copy()
        q_val = float(q_arr[qi])
        if is_q1[qi]:
            # Z[qi] is already <log10 M_i> linear in log10 r.
            y_fit = y
        else:
            y_fit = np.full_like(y, np.nan)
            pos = y > 0
            y_fit[pos] = np.log10(y[pos])
        (slope, _), (slope_err, _) = linear_regression(log_r, y_fit)
        if not np.isfinite(slope):
            continue
        if is_q1[qi]:
            D_q[qi] = slope
            err_arr[qi] = slope_err
        else:
            D_q[qi] = slope / (q_val - 1.0)
            err_arr[qi] = slope_err / abs(q_val - 1.0)

    # Unbox return shape
    if q_scalar:
        D_out = float(D_q[0])
        err_out = float(err_arr[0])
        Z_out = Z_total[0]
    else:
        D_out = D_q
        err_out = err_arr
        Z_out = Z_total

    if return_values:
        return D_out, err_out, bins, Z_out
    return D_out, err_out


def ensemble_coarsening_dimension(
    arrays: NDArray | list[NDArray],
    x_sizes: NDArray | list[NDArray] | None = None,
    y_sizes: NDArray | list[NDArray] | None = None,
    cloudy_threshold: float = 0.5,
    min_pixels: int = 30,
    return_values: bool = False,
    coarsening_factors: str | NDArray = 'default',
    count_exterior: bool = False
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    NOT RECOMMENDED due to ambiguity in coarsening a binary array

    Calculate the ensemble fractal dimension by coarsening image resolution and 
    calculating total perimeter as a function of resolution.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray
        Array, or list of arrays, to coarsen, apply cloudy_threshold to make binary,
        then calculate total perimeter.
    x_sizes : np.ndarray or list, optional
        Pixel sizes in the x direction. If None, assume all pixel dimensions are 1.
        If np.ndarray, use these for each array in 'arrays'. If list, assume
        x_sizes[i] corresponds to arrays[i].
    y_sizes : np.ndarray or list, optional
        Pixel sizes in the y direction. If None, assume all pixel dimensions are 1.
        If np.ndarray, use these for each array in 'arrays'. If list, assume
        y_sizes[i] corresponds to arrays[i].
    cloudy_threshold : float, default=0.5
        Threshold for making arrays binary.
    min_pixels : int, default=30
        Limit the coarsening factors such that coarsened matrices always have
        shape >= (min_pixels, min_pixels).
    return_values : bool, default=False
        If True, return additional data used in the calculation.
    coarsening_factors : str or array-like, default='default'
        Coarsening factors to use. If 'default', automatically determined.
    count_exterior : bool, default=False
        Whether to count exterior perimeter.

    Returns
    -------
    D_e : float
        The ensemble fractal dimension.
    error : float
        Error estimate (95% confidence).
    coarsening_factors : np.ndarray, optional
        The coarsening factors used. Only returned if return_values=True.
    mean_total_perimeters : np.ndarray, optional
        Mean total perimeters. Only returned if return_values=True.
    """
    raise NotImplementedError(
        "ensemble_coarsening_dimension has been removed due to ambiguity in "
        "coarsening binary arrays. Use ensemble_correlation_dimension (q=2), "
        "ensemble_sandbox_renyi_dimension (general q), or "
        "ensemble_box_dimension (q=0) instead."
    )


_UNSET = object()

_INDIVIDUAL_METHODS = {
    'filled perimeter vs filled area',
    'summed perimeter vs unfilled area',
    'filled perimeter vs width',
    'filled perimeter vs height',
    'summed perimeter vs width',
    'summed perimeter vs height',
}


def individual_fractal_dimension(
    arrays: NDArray | list[NDArray],
    x_sizes: NDArray | list[NDArray] | None = None,
    y_sizes: NDArray | list[NDArray] | None = None,
    min_length_scale: float = _UNSET,
    max_length_scale: float = _UNSET,
    bins: int | None = 30,
    return_values: bool = False,
    method: str = _UNSET,
    filled: bool = _UNSET,
    min_a: float = _UNSET,
    max_a: float = _UNSET,
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    Calculate the individual fractal dimension Df of objects within arrays.

    The method uses linear regression on log(length scale) vs. log(perimeter),
    omitting structures touching the array edge.

    Parameters
    ----------
    arrays : list of np.ndarray
        List of boolean 2D arrays.
    x_sizes : np.ndarray or list, optional
        Pixel sizes in the x direction. If None, assume all pixel dimensions are 1.
        If np.ndarray, use these for each array in 'arrays'. If list, assume
        x_sizes[i] corresponds to arrays[i].
    y_sizes : np.ndarray or list, optional
        Pixel sizes in the y direction. If None, assume all pixel dimensions are 1.
        If np.ndarray, use these for each array in 'arrays'. If list, assume
        y_sizes[i] corresponds to arrays[i].
    min_length_scale : float, default=3
        Minimum length scale to include. Filters on the x-axis quantity:
        sqrt(area) for area methods, width or height for those methods.
    max_length_scale : float, default=np.inf
        Maximum length scale to include.
    bins : int or None, default=30
        Number of bins along the log10 length scale for averaging. The
        regression is performed on the bin-averaged values. If None, fit on
        all individual points without binning.
    return_values : bool, default=False
        If True, return additional data used in the calculation.
    method : str, default='filled perimeter vs filled area'
        Which perimeter and length-scale combination to use. Options:

        - ``'filled perimeter vs filled area'``: fill holes, regress
          log(perimeter) vs log(sqrt(area)). Default.
        - ``'summed perimeter vs unfilled area'``: no hole filling, perimeter
          includes inner boundaries, area excludes holes.
        - ``'filled perimeter vs width'``: fill holes, regress log(perimeter)
          vs log(bounding-box width).
        - ``'filled perimeter vs height'``: fill holes, regress log(perimeter)
          vs log(bounding-box height).
        - ``'summed perimeter vs width'``: no hole filling, regress
          log(perimeter) vs log(bounding-box width).
        - ``'summed perimeter vs height'``: no hole filling, regress
          log(perimeter) vs log(bounding-box height).
    filled : bool, optional
        .. deprecated::
            Use ``method`` instead. ``filled=True`` maps to
            ``'filled perimeter vs filled area'``; ``filled=False`` maps to
            ``'summed perimeter vs unfilled area'``.
    min_a : float, optional
        .. deprecated::
            Use ``min_length_scale`` instead. Converted via ``sqrt(min_a)``.
    max_a : float, optional
        .. deprecated::
            Use ``max_length_scale`` instead. Converted via ``sqrt(max_a)``.

    Returns
    -------
    Df : float
        The individual fractal dimension.
    uncertainty : float
        Uncertainty estimate (95% confidence).
    log10_length_scale : np.ndarray, optional
        Log10 of the length-scale values (bin centers if bins is not None).
        Only returned if return_values=True.
    log10_p : np.ndarray, optional
        Log10 of perimeter values (bin means if bins is not None).
        Only returned if return_values=True.

    Raises
    ------
    ValueError
        If array shapes don't match pixel size shapes, if an invalid method
        is given, or if deprecated and new parameters are mixed.
    """
    # --- resolve deprecated 'filled' parameter ---
    if filled is not _UNSET and method is not _UNSET:
        raise ValueError("Cannot pass both 'filled' and 'method'.")
    if filled is not _UNSET:
        warnings.warn(
            "The 'filled' parameter is deprecated. Use method='filled perimeter"
            " vs filled area' or method='summed perimeter vs unfilled area'.",
            DeprecationWarning, stacklevel=2,
        )
        method = ('filled perimeter vs filled area' if filled
                  else 'summed perimeter vs unfilled area')
    elif method is _UNSET:
        method = 'filled perimeter vs filled area'

    if method not in _INDIVIDUAL_METHODS:
        raise ValueError(
            f"method={method!r} not recognized. Supported methods: "
            + ', '.join(sorted(_INDIVIDUAL_METHODS))
        )

    # --- resolve deprecated min_a / max_a ---
    if min_a is not _UNSET and min_length_scale is not _UNSET:
        raise ValueError("Cannot pass both 'min_a' and 'min_length_scale'.")
    if max_a is not _UNSET and max_length_scale is not _UNSET:
        raise ValueError("Cannot pass both 'max_a' and 'max_length_scale'.")
    if min_a is not _UNSET:
        warnings.warn(
            "The 'min_a' parameter is deprecated. Use 'min_length_scale' instead.",
            DeprecationWarning, stacklevel=2,
        )
        min_length_scale = np.sqrt(min_a)
    if max_a is not _UNSET:
        warnings.warn(
            "The 'max_a' parameter is deprecated. Use 'max_length_scale' instead.",
            DeprecationWarning, stacklevel=2,
        )
        max_length_scale = np.sqrt(max_a)
    if min_length_scale is _UNSET:
        min_length_scale = 3
    if max_length_scale is _UNSET:
        max_length_scale = np.inf

    # --- parse method ---
    fill_holes = method.startswith('filled perimeter')
    if 'vs filled area' in method or 'vs unfilled area' in method:
        length_scale_type = 'sqrt_area'
    elif 'vs width' in method:
        length_scale_type = 'width'
    else:
        length_scale_type = 'height'

    # --- collect per-structure data ---
    length_scales, perimeters = [], []
    if not isinstance(arrays, list):
        arrays = [arrays]

    if x_sizes is None:
        x_sizes = np.ones_like(arrays[0])
    if y_sizes is None:
        y_sizes = np.ones_like(arrays[0])

    for i in range(len(arrays)):
        array = arrays[i]
        if isinstance(x_sizes, list):
            xs = x_sizes[i]
        else:
            xs = x_sizes
        if isinstance(y_sizes, list):
            ys = y_sizes[i]
        else:
            ys = y_sizes

        if np.any(array.shape != xs.shape):
            raise ValueError('Each array shape must match corresponding pixel sizes shape')

        array = remove_structures_touching_border_nan(array)
        if fill_holes:
            array = remove_structure_holes(array)
        lab, nm, nl = label_structures(array, wrap='both')
        if lab is None:
            continue

        new_p = get_structure_perimeters(lab, nm, nl, xs, ys)
        if length_scale_type == 'sqrt_area':
            new_a = get_structure_areas(lab, nm, nl, xs, ys)
            new_ls = np.sqrt(new_a)
            valid = (new_a > 0) & (new_p > 0)
        else:
            heights, widths = get_structure_height_width(lab, nm, nl, xs, ys)
            new_ls = widths if length_scale_type == 'width' else heights
            valid = (new_ls > 0) & (new_p > 0)

        length_scales.extend(new_ls[valid])
        perimeters.extend(new_p[valid])

    length_scales = np.array(length_scales)
    perimeters = np.array(perimeters)
    mask = (length_scales > min_length_scale) & (length_scales < max_length_scale)
    length_scales, perimeters = length_scales[mask], perimeters[mask]

    log_ls = np.log10(length_scales)
    log_p = np.log10(perimeters)

    if bins is not None:
        bin_edges = np.linspace(log_ls.min(), log_ls.max(), bins + 1)
        bin_indices = np.digitize(log_ls, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = np.array([log_p[bin_indices == i].mean()
                              if np.any(bin_indices == i) else np.nan
                              for i in range(bins)])
        valid = np.isfinite(bin_means)
        log_ls = bin_centers[valid]
        log_p = bin_means[valid]

    (slope, _), (err, _) = linear_regression(log_ls, log_p)

    if return_values:
        return slope, err, log_ls, log_p
    else:
        return slope, err


# Helper functions for fractal dimension calculations

def get_coords_of_boundaries(array: NDArray) -> NDArray:
    """
    Find coordinates of pixels with value 1 that are adjacent to pixels with value 0.

    Parameters
    ----------
    array : np.ndarray
        2-D binary array.

    Returns
    -------
    np.ndarray
        Array of shape (n_boundaries, 2) where each element is a pair of indices
        corresponding to the locations of pixels with value 1 that are adjacent
        to a pixel of value 0.

    Notes
    -----
    Topology is toroidal, so pixels on one edge are considered adjacent to pixels
    on the opposite edge.

    Examples
    --------
    >>> array1 = np.zeros((10, 10))
    >>> array1[6:8, 6:8] = 1
    >>> array1[3:4, 2:5] = 1
    >>> array2 = np.zeros((10, 10))
    >>> for i, j in get_coords_of_boundaries(array1):
    ...     array2[i, j] = 1
    >>> np.all(array2 == array1)
    True
    """
    array = array.astype(np.int16)
    shifted_right = np.roll(array, shift=1, axis=1)
    shifted_down = np.roll(array, 1, axis=0)
    diff_right = shifted_right - array
    diff_down = shifted_down - array
    del shifted_right, shifted_down

    right_side_of_pixel = np.argwhere(diff_right == 1)
    right_side_of_pixel[:, 1] -= 1
    left_side_of_pixel = np.argwhere(diff_right == -1)
    bottom_of_pixel = np.argwhere(diff_down == 1)
    bottom_of_pixel[:, 0] -= 1
    top_of_pixel = np.argwhere(diff_down == -1)

    all_coords = np.append(right_side_of_pixel, left_side_of_pixel, axis=0)
    all_coords = np.append(all_coords, top_of_pixel, axis=0)
    all_coords = np.append(all_coords, bottom_of_pixel, axis=0)

    # remove duplicates
    all_coords = np.unique(all_coords, axis=0)

    return all_coords


def get_locations_from_pixel_sizes(
    pixel_sizes_x: NDArray,
    pixel_sizes_y: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Convert pixel sizes to cumulative location coordinates.

    Parameters
    ----------
    pixel_sizes_x : np.ndarray
        2-D array of pixel sizes in x direction.
    pixel_sizes_y : np.ndarray
        2-D array of pixel sizes in y direction.

    Returns
    -------
    locations_x : np.ndarray
        Cumulative x locations.
    locations_y : np.ndarray
        Cumulative y locations.
    """
    return np.nancumsum(pixel_sizes_x, 1), np.nancumsum(pixel_sizes_y, 0)


@numba.njit(parallel=True, fastmath=True)
def _sandbox_partition(
    centers_phys,        # (N_c, 2) sandbox centers (physical x, y), drawn from set
    sorted_boundary,     # (N_s, 2) set points (physical x, y), sorted by y
    bins_sq,             # (B,) ascending squared distance bin edges
    max_bin,             # float: sqrt(bins_sq[-1])
    qs,                  # (Q,) Rényi orders
    is_q1,               # (Q,) bool: True where qs[qi] is within tolerance of 1.0
    r_max_sq_per_center, # (N_c,) per-center maximum squared radius. For the legacy
                         #   modes, fill with bins_sq[-1] so every bin is admissible.
                         #   For the 'truncate' mode, fill with squared distance to
                         #   the nearest domain edge — bins above this are skipped.
):
    """For each center, compute M_i(r_k) = #set points within distance r_k.
    Accumulate Z[qi, k] across centers and counts of contributing centers
    per bin N_per_bin[k]:
        Z[qi, k] = sum_{i: bins[k] <= r_max_i} M_i(r_k)^(qs[qi] - 1)        (q != 1)
        Z[qi, k] = sum_{i: bins[k] <= r_max_i} log10(M_i(r_k))              (q == 1)
        N_per_bin[k] = #{i: bins[k] <= r_max_i}

    For the legacy boundary modes, ``r_max_sq_per_center[i] = bins_sq[-1]``
    for every center, so every bin admits every center and ``N_per_bin``
    equals the total center count for all k — caller can ignore it.

    Centers are assumed to be drawn from the same set as sorted_boundary,
    so each center self-counts at distance 0 (M_i >= 1 always at any
    positive radius). This matches the convention of the legacy
    `correlation_integral` kernel and makes the q=2 output bit-identical
    to the previous Grassberger-Procaccia implementation:

        sum_i M_i(r_k) ^ (2 - 1) = sum_i M_i(r_k) = pair count = old C_l[k].

    Binning convention: `searchsorted(bins_sq, dist_sq, side='right')` —
    bin index is the smallest k with bins_sq[k] > dist_sq, i.e. the bin
    "first reached" by this pair. After per-center cumulative sum,
    M_i(bins[k]) = #neighbors with dist < bins[k]. Same as old kernel.
    """
    N_c = centers_phys.shape[0]
    B = bins_sq.shape[0]
    Q = qs.shape[0]
    n_threads = numba.config.NUMBA_NUM_THREADS

    # Per-thread accumulators (avoid races, no atomics needed)
    Z_per_thread = np.zeros((n_threads, Q, B), dtype=np.float64)
    N_per_bin_per_thread = np.zeros((n_threads, B), dtype=np.int64)
    # Per-thread scratch histograms, reused across centers
    hist_per_thread = np.zeros((n_threads, B), dtype=np.int64)

    sorted_y = sorted_boundary[:, 1].copy()

    for i in numba.prange(N_c):
        thread_id = numba.get_thread_id()
        cx = centers_phys[i, 0]
        cy = centers_phys[i, 1]
        r_max_sq_i = r_max_sq_per_center[i]
        # Per-center search radius: never larger than the global max_bin,
        # but tightened by the per-center cap to save work.
        if r_max_sq_i < bins_sq[B - 1]:
            search_r = np.sqrt(r_max_sq_i)
        else:
            search_r = max_bin

        # k_max_i = #{k : bins_sq[k] <= r_max_sq_i}
        # All bins k < k_max_i are admissible for this center.
        k_max_i = np.searchsorted(bins_sq, r_max_sq_i, side='right')
        if k_max_i > B:
            k_max_i = B
        if k_max_i == 0:
            # Nearest edge is closer than the smallest bin: skip center.
            continue

        # Zero this thread's scratch histogram (only up to k_max_i needed)
        for k in range(k_max_i):
            hist_per_thread[thread_id, k] = 0

        # y bounding box via binary search
        lo = np.searchsorted(sorted_y, cy - search_r, side='left')
        hi = np.searchsorted(sorted_y, cy + search_r, side='right')

        for j in range(lo, hi):
            bx = sorted_boundary[j, 0]
            if abs(bx - cx) > search_r:
                continue
            dx = cx - bx
            dy = cy - sorted_boundary[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq > r_max_sq_i:
                continue
            bin_idx = np.searchsorted(bins_sq, dist_sq, side='right')
            if bin_idx < k_max_i:
                hist_per_thread[thread_id, bin_idx] += 1

        # Per-center cumulative sum gives M_i(bins[k]) at each k.
        # Then accumulate M_i^(q-1) (or log10(M_i) at q==1) into Z[qi, k].
        # Only k < k_max_i is admissible: bins above the per-center cap
        # would extend beyond the domain edge.
        M = 0
        for k in range(k_max_i):
            M += hist_per_thread[thread_id, k]
            N_per_bin_per_thread[thread_id, k] += 1
            if M > 0:
                M_f = float(M)
                for qi in range(Q):
                    if is_q1[qi]:
                        Z_per_thread[thread_id, qi, k] += np.log10(M_f)
                    else:
                        exponent = qs[qi] - 1.0
                        Z_per_thread[thread_id, qi, k] += M_f ** exponent

    # Reduce across threads
    Z = np.zeros((Q, B), dtype=np.float64)
    N_per_bin = np.zeros(B, dtype=np.int64)
    for t in range(n_threads):
        for qi in range(Q):
            for k in range(B):
                Z[qi, k] += Z_per_thread[t, qi, k]
        for k in range(B):
            N_per_bin[k] += N_per_bin_per_thread[t, k]
    return Z, N_per_bin


def coarsen_array(array: NDArray, factor: int) -> NDArray:
    """
    Coarsen an array by averaging superpixel regions.

    Takes an input array and reduces it by a given factor along both the x and y
    dimensions. The coarsening is achieved by summing 'superpixel' regions of the
    original array and dividing by the number of pixels in each region.

    Parameters
    ----------
    array : np.ndarray
        The input array to be coarsened.
    factor : int
        The coarsening factor for reducing the array resolution. Must be a positive integer.

    Returns
    -------
    np.ndarray
        The coarsened array with reduced resolution.

    Examples
    --------
    >>> original_array = np.array([[1, 2], [3, 4]])
    >>> coarsened = coarsen_array(original_array, factor=2)
    >>> coarsened
    array([[2.5]])
    """
    coarsened_array = np.add.reduceat(array, np.arange(array.shape[0], step=factor), axis=0)
    coarsened_array = np.add.reduceat(coarsened_array, np.arange(array.shape[1], step=factor), axis=1)

    # The number of pixels that are coarsened is usually factor**2, but not for edge superpixels
    # if the factor does not evenly divide into array size. Solution:
    pixel_counts = np.add.reduceat(np.ones(array.shape), np.arange(array.shape[0], step=factor), axis=0)
    pixel_counts = np.add.reduceat(pixel_counts, np.arange(array.shape[1], step=factor), axis=1)

    coarsened_array = coarsened_array / pixel_counts

    return coarsened_array


@numba.njit()
def total_perimeter(array, x_sizes, y_sizes):
    """
    Calculate the total perimeter of a binary array.

    Given a binary array, calculate the total perimeter. Only counts perimeter
    along edges between 1 and 0. Assumes periodic boundary conditions; for other
    boundary conditions, pad inputs with 0s or nans.

    Parameters
    ----------
    array : np.ndarray
        Binary array (0s and 1s).
    x_sizes : np.ndarray
        Pixel sizes in x direction.
    y_sizes : np.ndarray
        Pixel sizes in y direction.

    Returns
    -------
    float
        Total perimeter length.

    Raises
    ------
    ValueError
        If x_sizes or y_sizes is nan where array is 1.
    """
    perimeter = 0
    for (i, j), value in np.ndenumerate(array):
        if value == 1:
            if np.isnan(x_sizes[i, j]) or np.isnan(y_sizes[i, j]):
                raise ValueError('x_sizes or y_sizes is nan where array is 1')
            if i != array.shape[0] - 1 and array[i + 1, j] == 0:
                perimeter += x_sizes[i, j]
            elif i == array.shape[0] - 1 and array[0, j] == 0:
                perimeter += x_sizes[i, j]

            if i != 0 and array[i - 1, j] == 0:
                perimeter += x_sizes[i, j]
            elif i == 0 and array[array.shape[0] - 1, j] == 0:
                perimeter += x_sizes[i, j]

            if j != array.shape[1] - 1 and array[i, j + 1] == 0:
                perimeter += y_sizes[i, j]
            elif j == array.shape[1] - 1 and array[i, 0] == 0:
                perimeter += y_sizes[i, j]

            if j != 0 and array[i, j - 1] == 0:
                perimeter += y_sizes[i, j]
            elif j == 0 and array[i, array.shape[1] - 1] == 0:
                perimeter += y_sizes[i, j]

    return perimeter


def total_number(
    array: NDArray,
    structure: NDArray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
) -> int:
    """
    Count the number of connected objects in an array.

    Given a 2-D array with 0's, nans, and 1's, calculate number of objects of
    connected 1's where connectivity is defined by structure.

    Parameters
    ----------
    array : np.ndarray
        2-D array with 0s, 1s, and optionally nans.
    structure : np.ndarray, default=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        Connectivity structure.

    Returns
    -------
    int
        Number of connected objects.
    """
    clean = np.where(np.isnan(array), 0, array)
    _, n_structures = label(clean.astype(bool), structure, output=np.float32)
    return n_structures


def isolate_nth_largest_structure(
    binary_array: NDArray,
    n: int = 1,
    structure: NDArray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
) -> NDArray[np.bool_]:
    """
    Isolate the Nth largest connected structure in a binary array.

    Parameters
    ----------
    binary_array : np.ndarray
        Binary input array.
    n : int, default=1
        Which structure to return, ranked by pixel count (1 = largest).
    structure : np.ndarray, default=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        Connectivity structure.

    Returns
    -------
    np.ndarray
        Boolean array with only the Nth largest structure set to True.

    Raises
    ------
    ValueError
        If ``binary_array`` contains no structures or ``n`` exceeds the number
        of structures.
    """
    if n < 1:
        raise ValueError('n must be >= 1')
    labelled_array = label(binary_array, structure)[0]
    cloud_values = labelled_array[labelled_array != 0]  # remove background
    if cloud_values.size == 0:
        raise ValueError('binary_array contains no structures')
    values, counts = np.unique(cloud_values, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    if n > len(values):
        raise ValueError(f'Requested n={n} but only {len(values)} structures exist')
    selected = values[sorted_indices[n - 1]]
    return labelled_array == selected


def isolate_largest_structure(*args, **kwargs):
    """Removed. Use :func:`isolate_nth_largest_structure` instead."""
    raise NotImplementedError(
        'isolate_largest_structure has been renamed to '
        'isolate_nth_largest_structure(binary_array, n=1)'
    )


def label_size(
    array: NDArray,
    variable: str = 'area',
    wrap: str | None = 'both',
    x_sizes: NDArray | None = None,
    y_sizes: NDArray | None = None
) -> NDArray:
    """
    Label structures with their size values.

    Creates a labelled array where each structure is labelled with its size
    (area, summed perimeter, width, or height) instead of a unique identifier.

    Parameters
    ----------
    array : np.ndarray
        Binary array of structures: 2-d array, padded with 0's or np.nan's.
    variable : str, default='area'
        Which variable to use for 'size'. Options: ``'area'``,
        ``'summed perimeter'``, ``'width'``, ``'height'``.
        ``'perimeter'`` is accepted but deprecated.
    wrap : str or None, default='both'
        Boundary wrapping options: None, 'sides', 'both'.
        If 'sides', connect structures that span the left/right edge.
        If 'both', connect structures that span all edges.
    x_sizes : np.ndarray, optional
        Pixel sizes in x direction. If None, assume all sizes are 1.
    y_sizes : np.ndarray, optional
        Pixel sizes in y direction. If None, assume all sizes are 1.

    Returns
    -------
    np.ndarray
        Array where structures are labelled with their size value and background is 0.

    Raises
    ------
    ValueError
        If variable or wrap is not a supported value.

    Notes
    -----
    If x_sizes or y_sizes are not uniform, the width will be the sum of the average
    pixel widths of the pixels in the column and in the object. Similarly, the height
    will be the sum of the average pixel heights of the pixels in the row and in the object.
    """
    binary = np.where(np.isnan(array), 0, array).astype(bool)
    labelled_array, n_structures = label(binary, output=np.float32)

    if variable == 'perimeter':
        warnings.warn(
            "variable='perimeter' is deprecated, use 'summed perimeter' instead.",
            DeprecationWarning, stacklevel=2,
        )
        variable = 'summed perimeter'
    if variable not in ['area', 'summed perimeter', 'width', 'height']:
        raise ValueError(f'variable={variable} not supported (supported values are "area", "summed perimeter", "width", "height")')

    if x_sizes is None:
        x_sizes = np.ones_like(labelled_array)
    if y_sizes is None:
        y_sizes = np.ones_like(labelled_array)

    if wrap is None:
        pass
    elif wrap == 'sides':
        # set those on right to the same i.d. as those on left
        for j, value in enumerate(labelled_array[:, 0]):
            if value != 0:
                if labelled_array[j, labelled_array.shape[1] - 1] != 0 and labelled_array[j, labelled_array.shape[1] - 1] != value:
                    labelled_array[labelled_array == labelled_array[j, labelled_array.shape[1] - 1]] = value

    if wrap is None:
        pass
    elif wrap == 'both' or wrap == 'sides':
        labelled_array = _merge_periodic_labels(labelled_array, wrap)
    else:
        raise ValueError(f'wrap={wrap} not supported')

    # Flatten arrays to find their indices.
    values = np.sort(labelled_array.ravel())
    original_locations = np.argsort(labelled_array.ravel())
    indices_2d = np.array(np.unravel_index(original_locations, labelled_array.shape)).T

    labelled_array[np.isnan(array)] = np.nan  # Turn this back to nan so perimeter along it is not included
    split_here = np.roll(values, shift=-1) - values  # Split where the values changed.
    split_here[-1] = 0  # Last value rolled over from first

    separated_structure_indices = np.split(indices_2d, np.where(split_here != 0)[0] + 1)
    separated_structure_indices = separated_structure_indices[1:]  # Remove the locations that were 0 (not structure)
    if len(separated_structure_indices) == 0:
        return np.zeros(array.shape, dtype=int)

    labelled_with_sizes = np.zeros(array.shape, dtype=np.float32)

    # must use numba.typed.List here for Numba compatibility
    # https://numba.readthedocs.io/en/stable/reference/pysupported.html#feature-typed-list
    labelled_with_sizes = _label_size_helper(labelled_array, List(separated_structure_indices), labelled_with_sizes, variable, x_sizes, y_sizes)
    return labelled_with_sizes


@numba.njit()
def _label_size_helper(labelled_array, separated_structure_indices, labelled_with_sizes, variable, x_sizes, y_sizes):
    for indices in separated_structure_indices:
        perimeter = 0
        area = 0

        y_coords_structure = np.array([c[0] for c in indices])
        x_coords_structure = np.array([c[1] for c in indices])
        unique_y_coords = []
        unique_x_coords = []
        height = 0
        width = 0

        for (i, j) in indices:
            # Height, Width
            if i not in unique_y_coords:
                unique_y_coords.append(i)
                mask = (y_coords_structure == i)
                y_sizes_here = []
                for loc, take in enumerate(mask):
                    if take:
                        y_sizes_here.append(y_sizes[y_coords_structure[loc], x_coords_structure[loc]])
                y_sizes_here = np.array(y_sizes_here)
                height += np.mean(y_sizes_here)
            if j not in unique_x_coords:
                unique_x_coords.append(j)
                mask = (x_coords_structure == j)
                x_sizes_here = []
                for loc, take in enumerate(mask):
                    if take:
                        x_sizes_here.append(x_sizes[y_coords_structure[loc], x_coords_structure[loc]])
                x_sizes_here = np.array(x_sizes_here)
                width += np.mean(x_sizes_here)

            # Perimeter:
            if i != labelled_array.shape[0] - 1 and labelled_array[i + 1, j] == 0:
                perimeter += x_sizes[i, j]
            elif i == labelled_array.shape[0] - 1 and labelled_array[0, j] == 0:
                perimeter += x_sizes[i, j]

            if i != 0 and labelled_array[i - 1, j] == 0:
                perimeter += x_sizes[i, j]
            elif i == 0 and labelled_array[labelled_array.shape[0] - 1, j] == 0:
                perimeter += x_sizes[i, j]

            if j != labelled_array.shape[1] - 1 and labelled_array[i, j + 1] == 0:
                perimeter += y_sizes[i, j]
            elif j == labelled_array.shape[1] - 1 and labelled_array[i, 0] == 0:
                perimeter += y_sizes[i, j]

            if j != 0 and labelled_array[i, j - 1] == 0:
                perimeter += y_sizes[i, j]
            elif j == 0 and labelled_array[i, labelled_array.shape[1] - 1] == 0:
                perimeter += y_sizes[i, j]

            # Area:
            area += y_sizes[i, j] * x_sizes[i, j]

        for (i, j) in indices:
            if variable == 'summed perimeter':
                labelled_with_sizes[i, j] = perimeter
            elif variable == 'area':
                labelled_with_sizes[i, j] = area
            elif variable == 'width':
                labelled_with_sizes[i, j] = width
            elif variable == 'height':
                labelled_with_sizes[i, j] = height

    return labelled_with_sizes
