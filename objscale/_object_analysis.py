from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
import numba
from numba import njit, prange
from skimage.segmentation import clear_border
from ._utils import encase_in_value

__all__ = [
    'label_structures',
    'get_structure_props',
    'get_structure_areas',
    'get_structure_perimeters',
    'get_structure_height_width',
    'get_every_boundary_perimeter',
    'remove_structures_touching_border_nan',
    'clear_border_adjacent',
    'remove_structure_holes',
]

DEFAULT_STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


# =============================================================================
# Labeling
# =============================================================================

def label_structures(
    array: NDArray,
    structure: NDArray = DEFAULT_STRUCTURE,
    wrap: str | None = 'both',
) -> tuple[NDArray | None, NDArray | None, int]:
    """
    Label connected components in a binary array.

    Wrapper on ``scipy.ndimage.label`` with NaN handling and optional
    periodic boundary merging.

    Parameters
    ----------
    array : np.ndarray
        2-D binary array (0s, 1s, and optionally NaN).
    structure : np.ndarray, default=4-connectivity cross
        Connectivity kernel passed to ``scipy.ndimage.label``.
    wrap : str or None, default='both'
        Periodic boundary handling:
        - ``'both'``: merge labels across left-right and top-bottom edges.
        - ``'sides'``: merge labels across left-right edges only.
        - ``None``: no periodic merging.

    Returns
    -------
    labelled_array : np.ndarray or None
        Float32 array where each unique positive value is a connected
        component label. Pixels that were NaN in the input are 0.
        ``None`` if no structures exist.
    nan_mask : np.ndarray or None
        Boolean array indicating NaN locations in the input.
        ``None`` if no structures exist.
    n_labels : int
        Number of connected components found (0 if none).
    """
    nan_mask = np.isnan(array)
    if nan_mask.any():
        no_nans = array.copy()
        no_nans[nan_mask] = 0
    else:
        no_nans = array
    if np.count_nonzero(no_nans) == 0:
        return None, None, 0
    labelled_array, n_labels = label(no_nans.astype(bool), structure, output=np.float32)

    if wrap == 'both' or wrap == 'sides':
        labelled_array = _merge_periodic_labels(labelled_array, wrap)
    elif wrap is not None:
        raise ValueError(f'wrap={wrap!r} not supported')

    return labelled_array, nan_mask, n_labels


def _merge_periodic_labels(labelled_array: NDArray, wrap: str) -> NDArray:
    """Merge labels that span periodic boundaries (internal helper)."""
    if wrap == 'sides' or wrap == 'both':
        for j, value in enumerate(labelled_array[:, 0]):
            if value != 0:
                if labelled_array[j, labelled_array.shape[1] - 1] != 0 and labelled_array[j, labelled_array.shape[1] - 1] != value:
                    labelled_array[labelled_array == labelled_array[j, labelled_array.shape[1] - 1]] = value

    if wrap == 'both':
        for i, value in enumerate(labelled_array[0, :]):
            if value != 0:
                if labelled_array[labelled_array.shape[0] - 1, i] != 0 and labelled_array[labelled_array.shape[0] - 1, i] != value:
                    labelled_array[labelled_array == labelled_array[labelled_array.shape[0] - 1, i]] = value

    return labelled_array


def _validate_inputs(array, x_sizes, y_sizes):
    """Validate that array, x_sizes, y_sizes have matching shapes and no bad nans."""
    if array.shape != x_sizes.shape or array.shape != y_sizes.shape:
        raise ValueError(
            f'array, x_sizes, and y_sizes must all be same shape. '
            f'Currently {array.shape},{x_sizes.shape},{y_sizes.shape}'
        )
    if np.count_nonzero((np.isnan(x_sizes) | np.isnan(y_sizes)) & np.isfinite(array)):
        raise ValueError('x or y sizes are nan in locations where array is not')


def _validate_labelled(labelled_array):
    """Raise TypeError if the array looks binary instead of labelled."""
    if labelled_array.dtype == bool:
        raise TypeError(
            'labelled_array is boolean. '
            'Pass a labelled array from label_structures() instead.'
        )


# =============================================================================
# Area: pure numpy bincount — O(n)
# =============================================================================

def get_structure_areas(
    labelled_array: NDArray,
    nan_mask: NDArray,
    n_labels: int,
    x_sizes: NDArray,
    y_sizes: NDArray,
) -> NDArray:
    """
    Calculate areas of labelled structures.

    Parameters
    ----------
    labelled_array : np.ndarray
        Labelled array from :func:`label_structures`.
    nan_mask : np.ndarray
        Boolean NaN mask from :func:`label_structures`.
    n_labels : int
        Number of labels from :func:`label_structures`.
    x_sizes : np.ndarray
        Pixel sizes in horizontal direction, same shape as labelled_array.
    y_sizes : np.ndarray
        Pixel sizes in vertical direction, same shape as labelled_array.

    Returns
    -------
    areas : np.ndarray
        1-D array of shape ``(n_labels,)`` where ``areas[i]`` is the area of
        label ``i + 1``. Guarantees index alignment with other
        ``get_structure_*`` functions called on the same labelled array.
    """
    _validate_labelled(labelled_array)
    return _compute_areas(labelled_array, x_sizes, y_sizes, n_labels)


def _compute_areas(labelled_array, x_sizes, y_sizes, n_labels):
    """Compute per-label areas via np.bincount.

    Returns array of shape (n_labels,) — index i is label i+1.
    Merged labels from periodic wrapping may have area 0.
    """
    pixel_areas = (x_sizes * y_sizes).ravel()
    labels_flat = labelled_array.ravel()
    mask = labels_flat > 0
    areas = np.bincount(
        labels_flat[mask].astype(np.intp),
        weights=pixel_areas[mask],
        minlength=n_labels + 1,
    )
    return areas[1:].astype(np.float32)


# =============================================================================
# Perimeter: Numba parallel row-chunked — O(n)
# =============================================================================

def get_structure_perimeters(
    labelled_array: NDArray,
    nan_mask: NDArray,
    n_labels: int,
    x_sizes: NDArray,
    y_sizes: NDArray,
) -> NDArray:
    """
    Calculate perimeters of labelled structures.

    Perimeter between a structure and NaN is not counted.

    Parameters
    ----------
    labelled_array : np.ndarray
        Labelled array from :func:`label_structures`.
    nan_mask : np.ndarray
        Boolean NaN mask from :func:`label_structures`.
    n_labels : int
        Number of labels from :func:`label_structures`.
    x_sizes : np.ndarray
        Pixel sizes in horizontal direction, same shape as labelled_array.
    y_sizes : np.ndarray
        Pixel sizes in vertical direction, same shape as labelled_array.

    Returns
    -------
    perimeters : np.ndarray
        1-D array of shape ``(n_labels,)`` where ``perimeters[i]`` is the
        perimeter of label ``i + 1``. Guarantees index alignment with other
        ``get_structure_*`` functions called on the same labelled array.
    """
    _validate_labelled(labelled_array)
    return _compute_perimeters(labelled_array, nan_mask, x_sizes, y_sizes, n_labels)


@njit(parallel=True)
def _compute_perimeters(labelled_array, nan_mask, x_sizes, y_sizes, n_labels):
    """Compute per-label perimeters via parallel row-chunked scan.

    Only counts edges where a structure pixel neighbors a 0 (background).
    Edges adjacent to NaN (outside domain) are not counted as perimeter.
    """
    nrows, ncols = labelled_array.shape
    n_pixels = nrows * ncols

    # Memory guard: per-thread buffers are n_threads × (n_labels+1) floats.
    # Cap n_threads so total buffer stays under 4× array size.
    n_threads = numba.config.NUMBA_NUM_THREADS
    max_threads = max(1, (4 * n_pixels) // (n_labels + 1))
    n_threads = min(n_threads, max_threads)

    chunk_size = (nrows + n_threads - 1) // n_threads
    local_p = np.zeros((n_threads, n_labels + 1), dtype=np.float32)

    for t in prange(n_threads):
        start = t * chunk_size
        end = min(start + chunk_size, nrows)
        for i in range(start, end):
            for j in range(ncols):
                lab = labelled_array[i, j]
                if lab <= 0:
                    continue
                lab_int = np.intp(lab)

                # Down neighbor: count if neighbor is 0 and not nan
                ni = i + 1 if i < nrows - 1 else 0
                if labelled_array[ni, j] == 0 and not nan_mask[ni, j]:
                    local_p[t, lab_int] += x_sizes[i, j]

                # Up neighbor
                ni = i - 1 if i > 0 else nrows - 1
                if labelled_array[ni, j] == 0 and not nan_mask[ni, j]:
                    local_p[t, lab_int] += x_sizes[i, j]

                # Right neighbor
                nj = j + 1 if j < ncols - 1 else 0
                if labelled_array[i, nj] == 0 and not nan_mask[i, nj]:
                    local_p[t, lab_int] += y_sizes[i, j]

                # Left neighbor
                nj = j - 1 if j > 0 else ncols - 1
                if labelled_array[i, nj] == 0 and not nan_mask[i, nj]:
                    local_p[t, lab_int] += y_sizes[i, j]

    # Sum across threads
    perimeters = np.zeros(n_labels + 1, dtype=np.float32)
    for t in range(n_threads):
        for lab in range(n_labels + 1):
            perimeters[lab] += local_p[t, lab]

    # Skip label 0
    return perimeters[1:]


# =============================================================================
# Height/width: parallel row-chunked bounding-box kernel
# =============================================================================

def get_structure_height_width(
    labelled_array: NDArray,
    nan_mask: NDArray,
    n_labels: int,
    x_sizes: NDArray,
    y_sizes: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    Calculate heights and widths of labelled structures as bounding-box extents.

    The height of a structure is the physical extent spanned by its
    bounding-box rows; the width is the physical extent spanned by its
    bounding-box columns. For structures that span a periodic boundary (when
    ``labelled_array`` was produced with ``wrap='both'`` or ``'sides'``), the
    smallest wrap-aware extent is used.

    Parameters
    ----------
    labelled_array : np.ndarray
        Labelled array from :func:`label_structures`.
    nan_mask : np.ndarray
        Boolean NaN mask from :func:`label_structures`. Currently unused; kept
        for signature compatibility with the other ``get_structure_*`` functions.
    n_labels : int
        Number of labels from :func:`label_structures`.
    x_sizes : np.ndarray
        Pixel sizes in horizontal direction, same shape as labelled_array.
    y_sizes : np.ndarray
        Pixel sizes in vertical direction, same shape as labelled_array.

    Returns
    -------
    heights : np.ndarray
        1-D array of shape ``(n_labels,)`` where ``heights[i]`` is the height
        of label ``i + 1``.
    widths : np.ndarray
        1-D array of shape ``(n_labels,)`` where ``widths[i]`` is the width
        of label ``i + 1``.

    Notes
    -----
    Widths are well-defined only when ``x_sizes`` is constant within each
    column (``x_sizes[:, j]`` does not vary for any ``j``); heights are
    well-defined only when ``y_sizes`` is constant within each row. If either
    invariant is violated, a :class:`UserWarning` is emitted and a per-row /
    per-column nanmean is used as the canonical pixel size.
    """
    _validate_labelled(labelled_array)

    # Ambiguity check for non-uniform pixel sizes within columns/rows.
    # All-NaN pad rows/cols are treated as "no information" and skipped.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        xmin_col = np.nanmin(x_sizes, axis=0)
        xmax_col = np.nanmax(x_sizes, axis=0)
        ymin_row = np.nanmin(y_sizes, axis=1)
        ymax_row = np.nanmax(y_sizes, axis=1)
    x_both_finite = np.isfinite(xmin_col) & np.isfinite(xmax_col)
    y_both_finite = np.isfinite(ymin_row) & np.isfinite(ymax_row)
    if np.any((xmin_col != xmax_col) & x_both_finite):
        warnings.warn(
            'x_sizes varies within at least one column; widths are ambiguous. '
            'Using per-column nanmean of x_sizes as the canonical column width.',
            stacklevel=2,
        )
    if np.any((ymin_row != ymax_row) & y_both_finite):
        warnings.warn(
            'y_sizes varies within at least one row; heights are ambiguous. '
            'Using per-row nanmean of y_sizes as the canonical row height.',
            stacklevel=2,
        )

    nrows, ncols = labelled_array.shape

    # Canonical per-row y_size and per-col x_size, NaN→0 for pad rows/cols.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_per_row = np.nanmean(y_sizes, axis=1).astype(np.float32)
        x_per_col = np.nanmean(x_sizes, axis=0).astype(np.float32)
    y_per_row = np.nan_to_num(y_per_row, nan=0.0)
    x_per_col = np.nan_to_num(x_per_col, nan=0.0)

    # Cumulative sums for O(1) range extent lookup.
    cumy = np.concatenate(([np.float32(0.0)], np.cumsum(y_per_row, dtype=np.float32)))
    cumx = np.concatenate(([np.float32(0.0)], np.cumsum(x_per_col, dtype=np.float32)))

    if n_labels == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    min_r, max_r, min_c, max_c, t_r0, t_rN, t_c0, t_cN = _compute_bbox(
        labelled_array, n_labels
    )

    heights = np.zeros(n_labels, dtype=np.float32)
    widths = np.zeros(n_labels, dtype=np.float32)

    for l in range(n_labels):
        if max_r[l] < 0:
            # no pixels for this label (e.g. merged away by periodic wrap)
            continue

        if t_r0[l] and t_rN[l]:
            heights[l] = _wrap_aware_extent(
                labelled_array, np.float32(l + 1), axis=0, cum=cumy, length=nrows
            )
        else:
            heights[l] = cumy[max_r[l] + 1] - cumy[min_r[l]]

        if t_c0[l] and t_cN[l]:
            widths[l] = _wrap_aware_extent(
                labelled_array, np.float32(l + 1), axis=1, cum=cumx, length=ncols
            )
        else:
            widths[l] = cumx[max_c[l] + 1] - cumx[min_c[l]]

    return heights, widths


@njit(parallel=True)
def _compute_bbox(labelled_array, n_labels):
    """Parallel row-chunked bbox kernel.

    Returns per-label min/max row/col and flags for whether the label touches
    the first/last row or column. Index ``l`` corresponds to label value ``l+1``.
    """
    nrows, ncols = labelled_array.shape
    n_pixels = nrows * ncols

    # Memory guard: per-thread buffers are 20 bytes per label (4×int32 + 4×bool).
    # Same "<4× array bytes" budget as the perimeter kernel.
    n_threads = numba.config.NUMBA_NUM_THREADS
    max_threads = max(1, (4 * n_pixels) // ((n_labels + 1) * 5))
    n_threads = min(n_threads, max_threads)
    if n_threads < 1:
        n_threads = 1

    chunk_size = (nrows + n_threads - 1) // n_threads

    local_min_r = np.full((n_threads, n_labels + 1), nrows, dtype=np.int32)
    local_max_r = np.full((n_threads, n_labels + 1), -1, dtype=np.int32)
    local_min_c = np.full((n_threads, n_labels + 1), ncols, dtype=np.int32)
    local_max_c = np.full((n_threads, n_labels + 1), -1, dtype=np.int32)
    local_t_r0 = np.zeros((n_threads, n_labels + 1), dtype=np.bool_)
    local_t_rN = np.zeros((n_threads, n_labels + 1), dtype=np.bool_)
    local_t_c0 = np.zeros((n_threads, n_labels + 1), dtype=np.bool_)
    local_t_cN = np.zeros((n_threads, n_labels + 1), dtype=np.bool_)

    for t in prange(n_threads):
        start = t * chunk_size
        end = min(start + chunk_size, nrows)
        for i in range(start, end):
            for j in range(ncols):
                lab = labelled_array[i, j]
                if lab <= 0:
                    continue
                li = np.intp(lab)
                if i < local_min_r[t, li]:
                    local_min_r[t, li] = i
                if i > local_max_r[t, li]:
                    local_max_r[t, li] = i
                if j < local_min_c[t, li]:
                    local_min_c[t, li] = j
                if j > local_max_c[t, li]:
                    local_max_c[t, li] = j
                if i == 0:
                    local_t_r0[t, li] = True
                if i == nrows - 1:
                    local_t_rN[t, li] = True
                if j == 0:
                    local_t_c0[t, li] = True
                if j == ncols - 1:
                    local_t_cN[t, li] = True

    # Reduction across threads (index l corresponds to label value l+1).
    min_r = np.full(n_labels, nrows, dtype=np.int32)
    max_r = np.full(n_labels, -1, dtype=np.int32)
    min_c = np.full(n_labels, ncols, dtype=np.int32)
    max_c = np.full(n_labels, -1, dtype=np.int32)
    t_r0 = np.zeros(n_labels, dtype=np.bool_)
    t_rN = np.zeros(n_labels, dtype=np.bool_)
    t_c0 = np.zeros(n_labels, dtype=np.bool_)
    t_cN = np.zeros(n_labels, dtype=np.bool_)

    for l in range(n_labels):
        li = l + 1
        for t in range(n_threads):
            if local_min_r[t, li] < min_r[l]:
                min_r[l] = local_min_r[t, li]
            if local_max_r[t, li] > max_r[l]:
                max_r[l] = local_max_r[t, li]
            if local_min_c[t, li] < min_c[l]:
                min_c[l] = local_min_c[t, li]
            if local_max_c[t, li] > max_c[l]:
                max_c[l] = local_max_c[t, li]
            if local_t_r0[t, li]:
                t_r0[l] = True
            if local_t_rN[t, li]:
                t_rN[l] = True
            if local_t_c0[t, li]:
                t_c0[l] = True
            if local_t_cN[t, li]:
                t_cN[l] = True

    return min_r, max_r, min_c, max_c, t_r0, t_rN, t_c0, t_cN


def _wrap_aware_extent(labelled_array, lab_value, axis, cum, length):
    """Compute wrap-aware bbox extent for a label that touches both edges of ``axis``.

    ``axis=0`` → row extent (height), ``axis=1`` → column extent (width).
    The structure is assumed to occupy the first and last index along the axis,
    so the largest *non-wrap* gap between consecutive occupied indices identifies
    the seam to wrap around.
    """
    if axis == 0:
        mask = (labelled_array == lab_value).any(axis=1)
    else:
        mask = (labelled_array == lab_value).any(axis=0)
    occ = np.nonzero(mask)[0]
    if len(occ) == 0:
        return np.float32(0.0)
    if len(occ) == 1:
        return cum[occ[0] + 1] - cum[occ[0]]

    gaps = occ[1:] - occ[:-1] - 1
    k = int(np.argmax(gaps))
    # bbox wraps around the seam: covers occ[k+1]..length-1 then 0..occ[k]
    return (cum[length] - cum[occ[k + 1]]) + (cum[occ[k] + 1] - cum[0])


# =============================================================================
# Backward-compatible wrapper
# =============================================================================

def get_structure_props(
    array: NDArray,
    x_sizes: NDArray,
    y_sizes: NDArray,
    structure: NDArray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    print_none: bool = False,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Calculate properties of structures in a binary array.

    Assumes toroidal (periodic) boundary conditions. For non-periodic domains,
    pad edges with 0 or np.nan before calling. Any perimeter between structure
    and nan is not counted.

    Parameters
    ----------
    array : np.ndarray
        Binary array of structures: 2-d array, padded with 0's or np.nan's.
    x_sizes : np.ndarray
        Sizes of pixels in horizontal direction, same shape as array.
    y_sizes : np.ndarray
        Sizes of pixels in vertical direction, same shape as array.
    structure : np.ndarray, default=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        Defines connectivity.
    print_none : bool, default=False
        Print message if no structures found.

    Returns
    -------
    perimeter : np.ndarray
        1-D array, each element the perimeter of an individual structure.
    area : np.ndarray
        1-D array, each element the area of an individual structure.
    height : np.ndarray
        1-D array, each element the height of an individual structure.
    width : np.ndarray
        1-D array, each element the width of an individual structure.

    Raises
    ------
    ValueError
        If array, x_sizes, and y_sizes are not the same shape.
        If x or y sizes are nan where array is not nan.

    Notes
    -----
    If x_sizes or y_sizes are not uniform, the width will be the sum of the average
    pixel widths of the pixels in the column and in the object. Similarly, the height
    will be the sum of the average pixel heights of the pixels in the row and in the object.

    For better performance when only a subset of properties is needed, use the
    individual functions: get_structure_areas, get_structure_perimeters,
    get_structure_height_width.
    """
    _validate_inputs(array, x_sizes, y_sizes)
    labelled_array, nan_mask, n_labels = label_structures(array, structure, wrap='both')
    if labelled_array is None:
        if print_none:
            print('No structures found')
        return np.array([]), np.array([]), np.array([]), np.array([])

    a = get_structure_areas(labelled_array, nan_mask, n_labels, x_sizes, y_sizes)
    p = get_structure_perimeters(labelled_array, nan_mask, n_labels, x_sizes, y_sizes)
    h, w = get_structure_height_width(labelled_array, nan_mask, n_labels, x_sizes, y_sizes)

    # Filter out labels with zero area (from periodic wrapping merging labels)
    valid = a > 0
    return p[valid], a[valid], h[valid], w[valid]




def get_every_boundary_perimeter(
    array: NDArray,
    x_sizes: NDArray,
    y_sizes: NDArray,
    return_nlevels: bool = False
) -> list | tuple[list, int]:
    """
    Return perimeters of each boundary between 0s and 1s.

    Each individual boundary between 0s and 1s in the array is treated as a unique
    perimeter. For example, a donut of 1s gives 2 values: one for the inner circle
    and one for the outer circle.

    Parameters
    ----------
    array : np.ndarray
        2-D binary array containing only 0s and 1s.
    x_sizes : np.ndarray
        Sizes of pixels in horizontal direction, same shape as array.
    y_sizes : np.ndarray
        Sizes of pixels in vertical direction, same shape as array.
    return_nlevels : bool, default=False
        If True, also return the number of nesting levels processed.

    Returns
    -------
    perimeters : list
        List of perimeter values for each boundary.
    nlevels : int, optional
        Number of nesting levels. Only returned if return_nlevels=True.

    Raises
    ------
    ValueError
        If more than 100 nesting levels are found (likely infinite loop).
    """
    # Encase once up front: pad + pixel-size arrays never change shape or content.
    # Interior NaNs (e.g. satellite "no data" inside a cloud) MUST be preserved
    # across iterations — `remove_structure_holes` treats NaN as not-a-hole, and
    # `NaN - NaN = NaN` keeps the pattern through the layer subtraction. If we
    # collapsed them to 0, a surrounded NaN would be filled on iteration 1 and
    # then emerge as a spurious structure on iteration 2.
    work = encase_in_value(array).astype(np.float32, copy=False)
    enc_xs = encase_in_value(x_sizes)
    enc_ys = encase_in_value(y_sizes)

    # NaN mask (pad + interior NaN) is static across iterations because NaN
    # positions never move under the layer-subtract.
    nan_mask_fixed = np.isnan(work)

    perimeters = []
    counter = 0
    # (work == 1).any() is NaN-safe (NaN == 1 is False) and avoids the O(n)
    # reduction np.nansum would do.
    while (work == 1).any():
        counter += 1
        if counter > 100:
            raise ValueError('Hole layer limit reached: 100 layers')
        all_holes_filled = remove_structure_holes(work)
        lab, _, nl = label_structures(all_holes_filled, wrap='both')
        if lab is not None:
            # Pass the pre-computed NaN mask explicitly so the perim kernel
            # correctly skips edges to NaN even though we reuse it each iteration.
            p = get_structure_perimeters(lab, nan_mask_fixed, nl, enc_xs, enc_ys)
            perimeters.extend(p[p > 0])

        # Next layer: filled minus original. Since filled ≥ original on the
        # finite pixels, the old `new_array[all_holes_filled == 0] = 0` masking
        # was redundant. NaN pixels stay NaN via NaN - NaN = NaN.
        work = all_holes_filled - work
        # Now what were previously holes are clouds. What were previously clouds in holes are now holes in the "new" clouds.
    if return_nlevels:
        return perimeters, counter
    return perimeters


def remove_structures_touching_border_nan(array: NDArray) -> NDArray:
    """
    Remove structures that touch the NaN border of an array.

    Parameters
    ----------
    array : np.ndarray
        2-D array consisting of 0s, 1s, and np.nan. All values at the array edge
        should be np.nan.

    Returns
    -------
    np.ndarray
        2-D array consisting of 0s, 1s, and np.nan with any structure in contact
        with the nan values around the outer edge of the good data removed.
        Contact is defined using adjacent connectivity (4-connectivity).

    Raises
    ------
    ValueError
        If array is not 2-dimensional.
    """
    if array.ndim != 2:
        raise ValueError('array not 2-dimensional')

    nanmask = np.isnan(array).astype(np.int8)
    edge_nan_mask = (nanmask - clear_border_adjacent(nanmask)).astype(bool)

    with_edge = array.copy()
    with_edge[edge_nan_mask] = 1

    cleared = clear_border_adjacent(with_edge).astype(np.float32)
    cleared[edge_nan_mask] = np.nan
    cleared[np.isnan(array)] = np.nan
    return cleared


def clear_border_adjacent(
    array: NDArray,
    structure: NDArray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
) -> NDArray[np.bool_]:
    """
    Remove connected regions that touch the array border.

    Similar to skimage.segmentation.clear_border but allows custom connectivity
    structure.

    Parameters
    ----------
    array : np.ndarray
        2-D array consisting of 0s and 1s.
    structure : np.ndarray, default=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        Defines connectivity for determining connected regions.

    Returns
    -------
    np.ndarray
        2-D boolean array with border-touching structures removed.

    Examples
    --------
    For a structure of ``np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])``:

    - ``[[0,0,0,0], [0,1,1,0], [0,0,0,1], [0,0,0,0]]`` -> keeps middle structure
    - ``[[0,0,0,0], [0,1,1,0], [0,0,1,1], [0,0,0,0]]`` -> removes all (connected to border)
    - ``[[0,0,0,0], [0,1,0,0], [1,0,0,0], [0,0,0,0]]`` -> keeps middle structure
    """
    border_cleared = clear_border(label(array.astype(bool), structure)[0])
    border_cleared[border_cleared > 0] = 1
    return border_cleared.astype(bool)


def remove_structure_holes(
    array: NDArray,
    periodic: bool | str = False
) -> NDArray:
    """
    Fill in all holes in all structures within the array.

    Sets any value of 0 that is not connected to the largest connected structure
    of 0s (the background) to 1. Assumes the largest contiguous area of 0s is
    the background.

    Parameters
    ----------
    array : np.ndarray
        2D array with values either 0, 1, or np.nan.
    periodic : bool or str, default=False
        Boundary condition handling. Options:
        - False: Holes connected to the edge are filled (as if padded with 1s).
        - 'sides': Periodic boundary on left/right edges.
        - 'both': Periodic boundary on all edges.

    Returns
    -------
    np.ndarray
        Array with all structure holes filled.

    Raises
    ------
    ValueError
        If array is not a numpy array or contains values other than 0, 1, or np.nan.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError('array must be a np.ndarray object')
    filled = array.copy()
    filled[np.isnan(filled)] = 0
    if np.any(filled > 1):
        raise ValueError('array can only have values 0, 1, or np.nan')

    # invert and label
    labelled, _ = label((1 - filled))
    if periodic is not False:
        labelled = _merge_periodic_labels(labelled, periodic)
    # Find the background (largest 0-component) via bincount — O(n), no sort.
    counts = np.bincount(labelled.ravel())
    counts[0] = 0  # ignore the "labelled == 0" bucket (cloud pixels)
    label_of_background = counts.argmax()

    filled[(labelled != 0) & (labelled != label_of_background)] = 1

    if np.count_nonzero(np.isnan(array)) > 0:
        filled[np.isnan(array)] = np.nan

    return filled
