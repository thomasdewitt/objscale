from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
from numba import njit, prange
from numba.typed import List
from warnings import warn
from skimage.segmentation import clear_border
from ._utils import encase_in_value

__all__ = [
    'get_structure_props',
    'label_periodic_boundaries',
    'get_every_boundary_perimeter',
    'remove_structures_touching_border_nan',
    'clear_border_adjacent',
    'remove_structure_holes',
]


def get_structure_props(
    array: NDArray,
    x_sizes: NDArray,
    y_sizes: NDArray,
    structure: NDArray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    print_none: bool = False,
    wrap: str | None = None
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Calculate properties of structures in a binary array.

    Given an array and the sizes of each pixel in each direction, calculate
    properties of structures. Any perimeter between structure and nan is not counted.

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
    wrap : str or None, default=None
        Boundary wrapping options: None, 'sides', 'both'.
        If 'sides', connect structures that span the left/right edge.
        If 'both', connect structures that span the left/right edge and top/bottom edge.

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
        If wrap is not None, 'sides', or 'both'.

    Notes
    -----
    If x_sizes or y_sizes are not uniform, the width will be the sum of the average
    pixel widths of the pixels in the column and in the object. Similarly, the height
    will be the sum of the average pixel heights of the pixels in the row and in the object.
    """
    if array.shape != x_sizes.shape or array.shape != y_sizes.shape:
        raise ValueError(f'array, x_sizes, and y_sizes must all be same shape. Currently {array.shape},{x_sizes.shape},{y_sizes.shape}')

    if np.count_nonzero((np.isnan(x_sizes) | np.isnan(y_sizes)) & np.isfinite(array)):
        raise ValueError('x or y sizes are nan in locations where array is not')

    no_nans = array.copy()
    no_nans[np.isnan(array)] = 0   # so we don't consider nans structures and also so they don't connect multiple structures
    if np.count_nonzero(no_nans) == 0:
        if print_none:
            print('No structures found')
        return np.array([]), np.array([]), np.array([]), np.array([])
    labelled_array, n_structures = label(no_nans.astype(bool), structure, output=np.float32)

    if wrap is None:
        pass
    elif wrap == 'both' or wrap == 'sides':
        labelled_array = label_periodic_boundaries(labelled_array, wrap)
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
        return np.array([]), np.array([]), np.array([]), np.array([])

    # must use numba.typed.List here for Numba compatibility
    # https://numba.readthedocs.io/en/stable/reference/pysupported.html#feature-typed-list
    p, a, h, w = _get_structure_props_helper(labelled_array, List(separated_structure_indices), x_sizes, y_sizes)
    nanmask = np.logical_or(np.logical_or(np.isnan(p), np.isnan(a)), np.logical_or(np.isnan(h), np.isnan(w)))
    if np.count_nonzero(nanmask) > 0:
        raise ValueError(f'Nan values found: {np.count_nonzero(nanmask)} out of {len(p)}')
    p, a, h, w = np.array(p), np.array(a), np.array(h), np.array(w)
    p, a, h, w = p[~nanmask], a[~nanmask], h[~nanmask], w[~nanmask]
    return p, a, h, w


@njit(parallel=True)
def _get_structure_props_helper(labelled_array, separated_structure_indices, x_sizes, y_sizes):
    # Preallocate arrays
    n_structures = len(separated_structure_indices)
    p = np.empty(n_structures, dtype=np.float32)
    a = np.empty(n_structures, dtype=np.float32)
    h = np.empty(n_structures, dtype=np.float32)
    w = np.empty(n_structures, dtype=np.float32)

    for iteration in prange(len(separated_structure_indices)):
        iteration = np.int64(iteration)
        structure_coords = separated_structure_indices[iteration]
        perimeter = 0
        area = 0

        y_coords_structure = np.array([c[0] for c in structure_coords])
        x_coords_structure = np.array([c[1] for c in structure_coords])
        unique_y_coords = []
        unique_x_coords = []
        height = 0
        width = 0

        for i, j in structure_coords:
            # Height, Width
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

        if area != 0:
            p[iteration] = perimeter
            a[iteration] = area
            h[iteration] = height
            w[iteration] = width

    # Return only the valid entries
    valid_mask = (a > 0)
    return p[valid_mask], a[valid_mask], h[valid_mask], w[valid_mask]


def label_periodic_boundaries(labelled_array: NDArray, wrap: str) -> NDArray:
    """
    Make labelled structures that span a periodic boundary have the same label.

    Parameters
    ----------
    labelled_array : np.ndarray
        A 2D array where each unique non-zero element represents a distinct label.
        Should be the output of scipy.ndimage.label().
    wrap : str
        Determines how the boundaries of the array should be wrapped.
        'sides': Sets labels on the right boundary to match those on the left.
        'both': Also sets labels on the top boundary to match those on the bottom.

    Returns
    -------
    np.ndarray
        The input array with periodic boundaries labelled according to the wrap parameter.

    Raises
    ------
    ValueError
        If wrap is neither 'sides' nor 'both'.
    """
    if wrap == 'sides' or wrap == 'both':
        # set those on right to the same i.d. as those on left
        for j, value in enumerate(labelled_array[:, 0]):
            if value != 0:
                if labelled_array[j, labelled_array.shape[1] - 1] != 0 and labelled_array[j, labelled_array.shape[1] - 1] != value:
                    labelled_array[labelled_array == labelled_array[j, labelled_array.shape[1] - 1]] = value

    if wrap == 'both':
        # set those on top to the same i.d. as those on bottom
        for i, value in enumerate(labelled_array[0, :]):
            if value != 0:
                if labelled_array[labelled_array.shape[0] - 1, i] != 0 and labelled_array[labelled_array.shape[0] - 1, i] != value:
                    labelled_array[labelled_array == labelled_array[labelled_array.shape[0] - 1, i]] = value

    if wrap not in ['sides', 'both']:
        raise ValueError(f'wrap = {wrap} not supported')
    return labelled_array


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
        exterior_perimeters, _, _, _ = get_structure_props(encase_in_value(all_holes_filled), encase_in_value(x_sizes), encase_in_value(y_sizes))
        perimeters.extend(exterior_perimeters)

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
        labelled = label_periodic_boundaries(labelled, periodic)
    # largest structure will be the background or the cloudy areas.
    unique_values, unique_counts = np.unique(labelled.flatten(), return_counts=True)
    # Make sure we don't identify the cloudy areas as the background.
    unique_counts, unique_values = unique_counts[unique_values != 0], unique_values[unique_values != 0]
    label_of_background = unique_values[unique_counts.argmax()]

    filled[(labelled != 0) & (labelled != label_of_background)] = 1

    if np.count_nonzero(np.isnan(array)) > 0:
        filled[np.isnan(array)] = np.nan

    return filled
