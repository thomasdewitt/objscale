from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import numba
from numba.typed import List
from scipy.ndimage import label
from warnings import warn
from ._object_analysis import (
    label_structures,
    _merge_periodic_labels,
    remove_structures_touching_border_nan,
    remove_structure_holes,
    get_structure_areas,
    get_structure_perimeters,
)
from ._utils import linear_regression, encase_in_value

__all__ = [
    'ensemble_correlation_dimension',
    'ensemble_box_dimension',
    'ensemble_information_dimension',
    'ensemble_renyi_dimension',
    'individual_fractal_dimension',
    'get_coords_of_boundaries',
    'get_locations_from_pixel_sizes',
    'correlation_integral',
    'coarsen_array',
    'total_perimeter',
    'total_number',
    'individual_correlation_dimension',
    'isolate_nth_largest_structure',
    'label_size',
]


@numba.njit(parallel=True, fastmath=True, cache=True)
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

    Requires that each array has the same pixel sizes, although across the array
    they may be nonuniform (e.g. increasing from left to right).

    Note that the resulting dimension is for the set of *object edge* points.

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
        Minimum length scale for correlation calculation. If 'auto', uses 3 times
        the minimum pixel size.
    maxlength : str or float, default='auto'
        Maximum length scale for correlation calculation. If 'auto', uses 0.33 times
        the minimum array dimension.
    interior_circles_only : bool, default=True
        If True, only use circle centers that are at least maxlength distance from
        all array edges to avoid boundary effects. In other words, only use circles
        that are fully contained within the array. Recommended!
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
        Error estimate for the dimension.
    bins : np.ndarray, optional
        The bins used for calculation. Only returned if return_C_l=True.
    C_l : np.ndarray, optional
        The correlation integral values. Only returned if return_C_l=True.

    Raises
    ------
    ValueError
        If arrays contain NaN values, if pixel sizes are invalid, or if scale
        range is insufficient.

    Notes
    -----
    This function uses the Grassberger-Procaccia pairwise-distance method:
    it counts pairs of set points separated by less than ``l`` and fits a
    power law. This is theoretically equivalent to ``q=2`` of the Rényi
    family, ``ensemble_renyi_dimension(..., q=2, set='edge')``, but the two
    will not give bit-identical numbers on finite-resolution data because
    they are sensitive to discretization in different ways. The
    Grassberger-Procaccia form converges in scale faster on small images,
    so we keep both implementations.
    """
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if x_sizes is None:
        x_sizes = np.ones(arrays[0].shape, dtype=np.float32)
    if y_sizes is None:
        y_sizes = np.ones(arrays[0].shape, dtype=np.float32)
    locations_x, locations_y = get_locations_from_pixel_sizes(x_sizes, y_sizes)

    h = x_sizes.shape[0]
    w = x_sizes.shape[1]

    if maxlength == 'auto':
        # ~One third of min(width, height) of entire array, where width, height are calculated in the center
        maxlength = 0.33 * min((locations_x[int(h / 2), w - 1] - locations_x[int(h / 2), 0]), (locations_y[h - 1, int(w / 2)] - locations_y[0, int(w / 2)]))

    if minlength == 'auto':
        minlength = 3 * min(np.nanmin(x_sizes), np.nanmin(y_sizes))

    # Basic validation checks
    if np.any(np.isnan(arrays)):
        raise ValueError('arrays must not contain NaN values')

    if np.any(np.isnan(x_sizes)) or np.any(np.isnan(y_sizes)):
        raise ValueError('x_sizes and y_sizes cannot contain NaN values')

    if np.any(x_sizes <= 0) or np.any(y_sizes <= 0):
        raise ValueError('x_sizes and y_sizes must be positive')

    if bins is None:
        bins = np.geomspace(minlength, maxlength, nbins)
    elif isinstance(bins, int):
        bins = np.geomspace(minlength, maxlength, bins)

    # range of scale checks
    if bins[-1] <= bins[0]:
        raise ValueError(f'bin maximum length ({bins[-1]:.3f}) must be greater than bin minimum length ({bins[0]:.3f}); or if bins are passed, they must be increasing. Did you pass invalid values for minlength/maxlength?')

    if bins[-1] / bins[0] < 10:
        raise ValueError(f'Available scale ratio ({maxlength / minlength:.2f}) is less than 10. Need at least one order of magnitude separation for reliable dimension estimation.')

    if interior_circles_only:
        # Check that maxlength leaves a usable interior region (at least 5x5 pixels)
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

    C_l = np.zeros(bins.shape)

    for array in arrays:
        if np.any(array.shape != x_sizes.shape):
            raise ValueError(f'All arrays must be same shape as pixel sizes (currently {array.shape} and {x_sizes.shape}, respectively)')

        array = array.astype(np.float16)

        all_boundary_coordinates = get_coords_of_boundaries(array)

        if interior_circles_only:
            # Calculate distance from each boundary coordinate to all array edges
            coord_locations_x = locations_x[all_boundary_coordinates[:, 0], all_boundary_coordinates[:, 1]]
            coord_locations_y = locations_y[all_boundary_coordinates[:, 0], all_boundary_coordinates[:, 1]]

            # Distance to each edge for all coordinates
            dist_to_left = coord_locations_x - locations_x[int(h / 2), 0]
            dist_to_right = locations_x[int(h / 2), w - 1] - coord_locations_x
            dist_to_top = coord_locations_y - locations_y[0, int(w / 2)]
            dist_to_bottom = locations_y[h - 1, int(w / 2)] - coord_locations_y

            # Find coordinates that are at least maxlength from ALL edges
            min_dist_to_any_edge = np.minimum.reduce([dist_to_left, dist_to_right, dist_to_top, dist_to_bottom])
            interior_mask = min_dist_to_any_edge >= maxlength

            circle_centers = all_boundary_coordinates[interior_mask]

        else:
            circle_centers = all_boundary_coordinates

        if len(circle_centers) == 0:
            continue

        if point_reduction_factor > 1:
            circle_centers = circle_centers[np.random.choice(np.arange(len(circle_centers)), int(len(circle_centers) / point_reduction_factor), replace=False)]
        elif point_reduction_factor < 1:
            raise ValueError('point_reduction_factor must be >= 1')

        if len(circle_centers) == 0:
            continue

        # Convert index coordinates to physical coordinates for bounding box
        boundary_phys_x = locations_x[all_boundary_coordinates[:, 0], all_boundary_coordinates[:, 1]]
        boundary_phys_y = locations_y[all_boundary_coordinates[:, 0], all_boundary_coordinates[:, 1]]
        boundary_phys = np.column_stack([boundary_phys_x, boundary_phys_y])

        center_phys_x = locations_x[circle_centers[:, 0], circle_centers[:, 1]]
        center_phys_y = locations_y[circle_centers[:, 0], circle_centers[:, 1]]
        center_phys = np.column_stack([center_phys_x, center_phys_y])

        # Sort boundary points by physical y for binary-search bounding box
        sort_order = np.argsort(boundary_phys[:, 1])
        sorted_boundary_phys = boundary_phys[sort_order]

        max_bin = bins[-1]
        bins_sq = bins ** 2

        C_l += correlation_integral(center_phys, sorted_boundary_phys,
                                    bins_sq, max_bin)

    # Perform linear regression to estimate dimension
    x, y = np.log10(bins), np.log10(C_l)
    index = np.isfinite(x) & np.isfinite(y)
    if len(x[index]) < 3:
        warn('Not enough data to estimate correlation dimension, returning nan')
        if return_C_l:
            return np.nan, np.nan, np.array([np.nan]), np.array([np.nan])
        else:
            return np.nan, np.nan
    coefficients, cov = np.polyfit(x[index], y[index], 1, cov=True)
    fit_error = np.sqrt(np.diag(cov))
    dimension, error = coefficients[0], 2 * fit_error[0]  # fit error is for 95% conf. int.

    if return_C_l:
        return dimension, error, bins, C_l
    else:
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
        Minimum length scale for correlation calculation. If 'auto', uses 3 times
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
        array = np.pad(array.astype(float), pad_width=1, mode='constant',
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
    cropped = isolated[rmin:rmax + 1, cmin:cmax + 1].astype(np.float64)

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


def _renyi_dimension_from_set(
    set_arrays: list[NDArray],
    q_arr: NDArray,
    box_sizes: NDArray,
    box_origin_shift: tuple[float, float],
) -> tuple[NDArray, NDArray, NDArray]:
    """Core Rényi-dimension routine: count 1-pixels per box, fit slopes.

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
        # Pool box counts and total interior pixels across the ensemble
        pooled_n: list[NDArray] = []
        V_total = 0  # total interior pixel area summed across arrays
        for arr in set_arrays:
            sy_pix = int(round(sy * factor))
            sx_pix = int(round(sx * factor))
            shifted = arr[sy_pix:, sx_pix:]
            h_full = (shifted.shape[0] // factor) * factor
            w_full = (shifted.shape[1] // factor) * factor
            if h_full == 0 or w_full == 0:
                continue
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


def ensemble_renyi_dimension(
    binary_arrays: NDArray | list[NDArray],
    q: float | NDArray = 0.0,
    set: str = 'edge',
    box_sizes: str | NDArray = 'default',
    min_pixels: int = 1,
    min_box_size: int = 2,
    box_origin_shift: tuple[float, float] = (0.0, 0.0),
    return_values: bool = False,
):
    """Calculate the generalized Rényi dimension D_q of binary arrays.

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
        ``min_box_size`` up to the largest size that satisfies ``min_pixels``.
    min_pixels : int, default=1
        Largest box size, in units of the number of boxes required to cover
        the smaller array dimension.
    min_box_size : int, default=2
        Smallest box size in pixels.
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

    # Resolve box_sizes (mirrors ensemble_box_dimension)
    if isinstance(box_sizes, str):
        if box_sizes != 'default':
            raise ValueError(f'box_sizes={box_sizes} not supported')
        box_sizes_arr = 2 ** np.arange(1, 15)
    else:
        box_sizes_arr = np.asarray(box_sizes)
    max_factor = min(binary_arrays[0].shape) / max(min_pixels, 1)
    box_sizes_arr = box_sizes_arr[box_sizes_arr <= max_factor]
    box_sizes_arr = box_sizes_arr[box_sizes_arr >= min_box_size]
    box_sizes_arr = np.unique(box_sizes_arr.astype(np.int64))
    if box_sizes_arr.size < 3:
        raise ValueError(
            f'Need at least 3 valid box sizes for a slope fit, got {box_sizes_arr.size}. '
            f'Reduce min_box_size or min_pixels, or pass an explicit box_sizes array.'
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

    D_q, err, partition = _renyi_dimension_from_set(
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
    min_pixels: int = 1,
    min_box_size: int = 2,
    box_sizes: str | NDArray = 'default',
    return_values: bool = False
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    Calculate the ensemble box-counting dimension of binary arrays.

    Equivalent to ``ensemble_renyi_dimension(..., q=0)``. Kept as a
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
    min_pixels : int, default=1
        Largest box size, in units of the number of boxes required to cover the
        array in the smaller direction.
    min_box_size : int, default=2
        Smallest box size.
    box_sizes : array-like or 'default', default='default'
        Box sizes used. If 'default', uses powers of 2 up to 2^14 that satisfy
        the above criteria.
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
    return ensemble_renyi_dimension(
        binary_arrays,
        q=0.0,
        set=set,
        box_sizes=box_sizes,
        min_pixels=min_pixels,
        min_box_size=min_box_size,
        return_values=return_values,
    )


def ensemble_information_dimension(
    binary_arrays: NDArray | list[NDArray],
    set: str = 'edge',
    min_pixels: int = 1,
    min_box_size: int = 2,
    box_sizes: str | NDArray = 'default',
    return_values: bool = False
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    Calculate the ensemble information dimension of binary arrays.

    Equivalent to ``ensemble_renyi_dimension(..., q=1)``. The information
    dimension is the ``q=1`` member of the Rényi-dimension family, defined
    via the Shannon entropy of the box-mass distribution rather than a
    moment of the count: as ``eps -> 0``,

    .. math::
        S_1(\\varepsilon) = -\\sum_i p_i \\log p_i \\sim -D_1 \\log\\varepsilon,

    where ``p_i = n_i / sum_j n_j`` is the probability that a randomly
    chosen set point lies in box ``i``. The information dimension is the
    ``q -> 1`` limit of the Rényi family; it has to be computed via the
    entropy form because the moment-based formula has a removable
    singularity at ``q=1``.

    Parameters
    ----------
    binary_arrays : list of np.ndarray or np.ndarray
        A list of 2D binary arrays or a single 2D binary array.
    set : str, default='edge'
        Which set to measure. See :func:`ensemble_renyi_dimension`.
    min_pixels, min_box_size, box_sizes, return_values
        Same as :func:`ensemble_box_dimension`.

    Returns
    -------
    dimension : float
        The estimated information dimension.
    error : float
        The 95% CI half-width.
    box_sizes : np.ndarray, optional
        Box sizes used. Only returned if return_values=True.
    entropy : np.ndarray, optional
        Shannon entropy ``S_1(eps)`` (in log10) at each box size, pooled
        over the input ensemble. Only returned if return_values=True.

    See Also
    --------
    ensemble_renyi_dimension : The general Rényi family for arbitrary ``q``.
    ensemble_box_dimension : The ``q=0`` special case.
    ensemble_correlation_dimension : Theoretically equivalent to ``q=2``,
        computed via the Grassberger-Procaccia pairwise method.
    """
    return ensemble_renyi_dimension(
        binary_arrays,
        q=1.0,
        set=set,
        box_sizes=box_sizes,
        min_pixels=min_pixels,
        min_box_size=min_box_size,
        return_values=return_values,
    )


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
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    if x_sizes is None:
        x_sizes = np.ones(arrays[0].shape, dtype=np.float32)
    if y_sizes is None:
        y_sizes = np.ones(arrays[0].shape, dtype=np.float32)

    if isinstance(coarsening_factors, str):  # to not try elementwise comparison
        if coarsening_factors != 'default':
            raise ValueError(f'coarsening_factors={coarsening_factors} not supported')
        if np.count_nonzero((arrays[0] < 1) & (arrays[0] > 0)) == 0:
            # If a binary array, even coarsening factors can be ambiguous because sometimes the superpixel is half cloudy
            coarsening_factors = 3 ** np.arange(0, 10)
        else:
            # Otherwise, use more coarsening factors:
            coarsening_factors = 2 ** np.arange(0, 15)
    else:
        coarsening_factors = np.array(coarsening_factors)

    max_coarsening_factor = min(arrays[0].shape) / min_pixels
    coarsening_factors = coarsening_factors[coarsening_factors <= max_coarsening_factor]

    mean_total_perimeters = np.empty((0, coarsening_factors.size), dtype=np.float32)

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

        total_perimeters = []
        for factor in coarsening_factors:
            # Coarsen
            coarsened_array = encase_in_value(coarsen_array(array, factor), np.nan)
            coarsened_x_sizes = factor * encase_in_value(coarsen_array(xs, factor), 0)
            coarsened_y_sizes = factor * encase_in_value(coarsen_array(ys, factor), 0)

            nanmask = (np.isnan(coarsened_array) | np.isnan(coarsened_x_sizes) | np.isnan(coarsened_y_sizes))
            # Make binary
            coarsened_array_binary = (coarsened_array > cloudy_threshold).astype(np.float32)

            # To not count exterior perimeter, set to nan, to count it, set to 0
            if count_exterior:
                padding = 0
            else:
                padding = np.nan
            coarsened_array_binary[nanmask] = padding

            total_p = total_perimeter(coarsened_array_binary, coarsened_x_sizes, coarsened_y_sizes)
            total_perimeters.append(total_p)
        mean_total_perimeters = np.append(mean_total_perimeters, [total_perimeters], axis=0)

    mean_total_perimeters = np.mean(mean_total_perimeters, axis=0)
    mean_total_perimeters[mean_total_perimeters == 0] = np.nan  # eliminate warning when logging 0

    (slope, _), (error, _) = linear_regression(np.log10(coarsening_factors), np.log10(mean_total_perimeters))
    if return_values:
        return 1 - slope, error, coarsening_factors, mean_total_perimeters

    return 1 - slope, error


def individual_fractal_dimension(
    arrays: NDArray | list[NDArray],
    x_sizes: NDArray | list[NDArray] | None = None,
    y_sizes: NDArray | list[NDArray] | None = None,
    min_a: float = 10,
    max_a: float = np.inf,
    bins: int | None = 30,
    return_values: bool = False,
    filled: bool = True,
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    Calculate the individual fractal dimension Df of objects within arrays.

    The method uses linear regression on log a vs. log p, omitting structures
    touching the array edge. By default, interior holes are filled before
    computing areas and perimeters (see ``filled``).

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
    min_a : float, default=10
        Minimum structure area to include in calculation.
    max_a : float, default=np.inf
        Maximum structure area to include in calculation.
    bins : int or None, default=30
        Number of bins along log10(sqrt(area)) for averaging. The regression
        is performed on the bin-averaged values. If None, fit on all individual
        points without binning.
    return_values : bool, default=False
        If True, return additional data used in the calculation.
    filled : bool, default=True
        If True, fill interior holes in structures before computing areas and
        perimeters. If False, holes are left as-is, so perimeters include
        interior boundaries and areas exclude hole pixels.

    Returns
    -------
    Df : float
        The individual fractal dimension.
    uncertainty : float
        Uncertainty estimate (95% confidence).
    log10_sqrt_a : np.ndarray, optional
        Log10 of sqrt(area) values (bin centers if bins is not None).
        Only returned if return_values=True.
    log10_p : np.ndarray, optional
        Log10 of perimeter values (bin means if bins is not None).
        Only returned if return_values=True.

    Raises
    ------
    ValueError
        If array shapes don't match pixel size shapes.
    """
    areas, perimeters = [], []
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
        if filled:
            array = remove_structure_holes(array)
        lab, nm, nl = label_structures(array, wrap='both')
        if lab is None:
            continue
        new_a = get_structure_areas(lab, nm, nl, xs, ys)
        new_p = get_structure_perimeters(lab, nm, nl, xs, ys)
        # Filter out labels with zero area or zero perimeter (e.g. NaN-surrounded)
        valid = (new_a > 0) & (new_p > 0)
        areas.extend(new_a[valid])
        perimeters.extend(new_p[valid])

    areas, perimeters = np.array(areas), np.array(perimeters)
    mask = (areas > min_a) & (areas < max_a)
    areas, perimeters = areas[mask], perimeters[mask]

    log_sqrt_a = np.log10(np.sqrt(areas))
    log_p = np.log10(perimeters)

    if bins is not None:
        bin_edges = np.linspace(log_sqrt_a.min(), log_sqrt_a.max(), bins + 1)
        bin_indices = np.digitize(log_sqrt_a, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = np.array([log_p[bin_indices == i].mean()
                              if np.any(bin_indices == i) else np.nan
                              for i in range(bins)])
        valid = np.isfinite(bin_means)
        log_sqrt_a = bin_centers[valid]
        log_p = bin_means[valid]

    (slope, _), (err, _) = linear_regression(log_sqrt_a, log_p)

    if return_values:
        return slope, err, log_sqrt_a, log_p
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


@numba.njit(parallel=True)
def correlation_integral(centers_phys, sorted_boundary_phys, bins_sq, max_bin):
    """
    Calculate correlation integral for boundary coordinates.

    For each center, count how many boundary points are within each distance
    threshold. Uses physical coordinates directly for correct bounding on any
    grid geometry (including non-uniform and 2D-varying pixel sizes).

    Parameters
    ----------
    centers_phys : np.ndarray
        Physical (x, y) coordinates of circle centers, shape (N, 2).
    sorted_boundary_phys : np.ndarray
        Physical (x, y) coordinates of boundary points, shape (M, 2),
        sorted by y coordinate.
    bins_sq : np.ndarray
        1-D array of squared distance thresholds (bins**2), sorted ascending.
    max_bin : float
        Maximum bin distance (sqrt of bins_sq[-1]).

    Returns
    -------
    C_l : np.ndarray
        Correlation integral values, same shape as bins_sq.

    Notes
    -----
    Uses three optimizations over the naive approach:
    1. Bounding box in physical space via sort + binary search on y coordinate.
    2. Squared distances to avoid sqrt in the inner loop.
    3. Binary-search binning with forward cumulative sum instead of linear scan.
    """
    sorted_y = sorted_boundary_phys[:, 1].copy()
    num_bins = bins_sq.shape[0]

    # Each thread gets its own histogram to eliminate race conditions
    hist_per_thread = np.zeros((numba.config.NUMBA_NUM_THREADS, num_bins))

    for i in numba.prange(centers_phys.shape[0]):
        thread_id = numba.get_thread_id()
        cx = centers_phys[i, 0]
        cy = centers_phys[i, 1]

        # Binary search for physical y bounding box
        lo = np.searchsorted(sorted_y, cy - max_bin, side='left')
        hi = np.searchsorted(sorted_y, cy + max_bin, side='right')

        for j in range(lo, hi):
            bx = sorted_boundary_phys[j, 0]

            # Physical x bounding box check
            if abs(bx - cx) > max_bin:
                continue

            dx = cx - bx
            dy = cy - sorted_boundary_phys[j, 1]
            dist_sq = dx * dx + dy * dy

            # Binary search: find first bin where bins_sq[bin_idx] > dist_sq
            bin_idx = np.searchsorted(bins_sq, dist_sq, side='right')
            if bin_idx < num_bins:
                hist_per_thread[thread_id, bin_idx] += 1

    # Sum across threads
    hist = np.sum(hist_per_thread, axis=0)

    # Forward cumulative sum: convert histogram to cumulative counts
    # hist[k] = count of pairs whose first qualifying bin index is k
    # C_l[k] = count of pairs with dist_sq < bins_sq[k] = sum of hist[0..k]
    C_l = np.zeros(num_bins)
    cumsum = 0.0
    for k in range(num_bins):
        cumsum += hist[k]
        C_l[k] = cumsum
    return C_l


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
    array_copy = array.copy()
    array_copy[np.isnan(array_copy)] = 0
    _, n_structures = label(array_copy.astype(bool), structure, output=np.float32)
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
    (area, perimeter, width, or height) instead of a unique identifier.

    Parameters
    ----------
    array : np.ndarray
        Binary array of structures: 2-d array, padded with 0's or np.nan's.
    variable : str, default='area'
        Which variable to use for 'size'. Options: 'area', 'perimeter', 'width', 'height'.
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

    if variable not in ['area', 'perimeter', 'width', 'height']:
        raise ValueError(f'variable={variable} not supported (supported values are "area", "perimeter", "width", "height")')

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
    values = np.sort(labelled_array.flatten())
    original_locations = np.argsort(labelled_array.flatten())
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
            if variable == 'perimeter':
                labelled_with_sizes[i, j] = perimeter
            elif variable == 'area':
                labelled_with_sizes[i, j] = area
            elif variable == 'width':
                labelled_with_sizes[i, j] = width
            elif variable == 'height':
                labelled_with_sizes[i, j] = height

    return labelled_with_sizes
