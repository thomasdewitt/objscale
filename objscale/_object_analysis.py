from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
import numba
from numba import njit, prange
from numba.typed import List
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
    no_nans = array.copy()
    no_nans[nan_mask] = 0
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
# Height/width: existing per-structure parallel loop
# =============================================================================

def get_structure_height_width(
    labelled_array: NDArray,
    nan_mask: NDArray,
    n_labels: int,
    x_sizes: NDArray,
    y_sizes: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    Calculate heights and widths of labelled structures.

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
    heights : np.ndarray
        1-D array of shape ``(n_labels,)`` where ``heights[i]`` is the height
        of label ``i + 1``.
    widths : np.ndarray
        1-D array of shape ``(n_labels,)`` where ``widths[i]`` is the width
        of label ``i + 1``.

    Notes
    -----
    If x_sizes or y_sizes are not uniform, the width will be the sum of the average
    pixel widths of the pixels in the column and in the object. Similarly, the height
    will be the sum of the average pixel heights of the pixels in the row and in the object.
    """
    _validate_labelled(labelled_array)

    separated, label_ids = _get_separated_structure_indices(labelled_array)
    if len(separated) == 0:
        return np.zeros(n_labels, dtype=np.float32), np.zeros(n_labels, dtype=np.float32)

    h_sparse, w_sparse = _compute_height_width(labelled_array, List(separated), x_sizes, y_sizes)

    # Map sparse results back to label-indexed arrays
    heights = np.zeros(n_labels, dtype=np.float32)
    widths = np.zeros(n_labels, dtype=np.float32)
    for k, lab in enumerate(label_ids):
        heights[lab - 1] = h_sparse[k]
        widths[lab - 1] = w_sparse[k]
    return heights, widths


def _get_separated_structure_indices(labelled_array):
    """Get list of 2D index arrays, one per structure label.

    Returns (separated, label_ids) where label_ids[k] is the label value
    for separated[k].
    """
    values = np.sort(labelled_array.flatten())
    original_locations = np.argsort(labelled_array.flatten())
    indices_2d = np.array(np.unravel_index(original_locations, labelled_array.shape)).T

    split_here = np.roll(values, shift=-1) - values
    split_here[-1] = 0

    split_points = np.where(split_here != 0)[0] + 1
    separated = np.split(indices_2d, split_points)
    separated = separated[1:]  # Remove label 0

    # Extract the label id for each group
    unique_labels = np.unique(values)
    label_ids = unique_labels[unique_labels > 0].astype(int)
    return separated, label_ids


@njit(parallel=True)
def _compute_height_width(labelled_array, separated_structure_indices, x_sizes, y_sizes):
    """Compute height and width per structure via per-structure parallel loop."""
    n_structures = len(separated_structure_indices)
    h = np.empty(n_structures, dtype=np.float32)
    w = np.empty(n_structures, dtype=np.float32)

    for iteration in prange(n_structures):
        iteration = np.int64(iteration)
        structure_coords = separated_structure_indices[iteration]

        y_coords_structure = np.array([c[0] for c in structure_coords])
        x_coords_structure = np.array([c[1] for c in structure_coords])
        unique_y_coords = []
        unique_x_coords = []
        height = 0.0
        width = 0.0

        for i, j in structure_coords:
            if i not in unique_y_coords:
                unique_y_coords.append(i)
                indices = (y_coords_structure == i)
                y_sizes_here = []
                for loc, take in enumerate(indices):
                    if take:
                        y_sizes_here.append(y_sizes[y_coords_structure[loc], x_coords_structure[loc]])
                y_sizes_here = np.array(y_sizes_here)
                height += np.mean(y_sizes_here)
            if j not in unique_x_coords:
                unique_x_coords.append(j)
                indices = (x_coords_structure == j)
                x_sizes_here = []
                for loc, take in enumerate(indices):
                    if take:
                        x_sizes_here.append(x_sizes[y_coords_structure[loc], x_coords_structure[loc]])
                x_sizes_here = np.array(x_sizes_here)
                width += np.mean(x_sizes_here)

        h[iteration] = height
        w[iteration] = width

    return h, w


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
    perimeters = []
    counter = 0
    while np.nansum(array) != 0:
        counter += 1
        if counter > 100:
            raise ValueError('Hole layer limit reached: 100 layers')
        all_holes_filled = remove_structure_holes(array)
        encased = encase_in_value(all_holes_filled)
        enc_xs = encase_in_value(x_sizes)
        enc_ys = encase_in_value(y_sizes)
        lab, nm, nl = label_structures(encased, wrap='both')
        if lab is not None:
            p = get_structure_perimeters(lab, nm, nl, enc_xs, enc_ys)
            perimeters.extend(p[p > 0])

        # remove one layer
        new_array = all_holes_filled - array
        new_array[all_holes_filled == 0] = 0
        array = new_array
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

    nanmask = np.isnan(array).astype(int)
    edge_nan_mask = (nanmask - clear_border_adjacent(nanmask)).astype(bool)

    with_edge = array.copy()
    with_edge[edge_nan_mask] = 1

    cleared = clear_border_adjacent(with_edge).astype(float)
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
    # largest structure will be the background or the cloudy areas.
    unique_values, unique_counts = np.unique(labelled.flatten(), return_counts=True)
    # Make sure we don't identify the cloudy areas as the background.
    unique_counts, unique_values = unique_counts[unique_values != 0], unique_values[unique_values != 0]
    label_of_background = unique_values[unique_counts.argmax()]

    filled[(labelled != 0) & (labelled != label_of_background)] = 1

    if np.count_nonzero(np.isnan(array)) > 0:
        filled[np.isnan(array)] = np.nan

    return filled
