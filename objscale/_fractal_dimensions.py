from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import numba
from numba.typed import List
from scipy.ndimage import label
from warnings import warn
from ._object_analysis import (
    remove_structures_touching_border_nan,
    remove_structure_holes,
    get_structure_props,
    label_periodic_boundaries,
)
from ._utils import linear_regression, encase_in_value

__all__ = [
    'ensemble_correlation_dimension',
    'ensemble_box_dimension',
    'individual_fractal_dimension',
    'get_coords_of_boundaries',
    'get_locations_from_pixel_sizes',
    'correlation_integral',
    'coarsen_array',
    'total_perimeter',
    'total_number',
    'isolate_largest_structure',
    'label_size',
]


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

        C_l += correlation_integral(circle_centers, all_boundary_coordinates, locations_x, locations_y, bins)

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

    Estimates the box-counting dimension (also known as Minkowski-Bouligand dimension)
    for a list of binary arrays. It averages the results across multiple arrays.

    Parameters
    ----------
    binary_arrays : list of np.ndarray or np.ndarray
        A list of 2D binary arrays or a single 2D binary array.
    set : str, default='edge'
        Specifies which set to consider for box counting:
        - 'edge': Box dimension of the set of boundaries between 0 and 1.
        - 'ones': Box dimension of the set of values equal to 1.
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
    mean_number_boxes : np.ndarray, optional
        Mean box counts. Only returned if return_values=True.

    Raises
    ------
    ValueError
        If an unsupported value is provided for 'set' or if arrays contain NaN values.

    Notes
    -----
    The function uses linear regression on log-log plot of box counts vs. scale
    to estimate the box-counting dimension. The slope of this regression gives
    the negative of the box-counting dimension.
    """
    if isinstance(binary_arrays, np.ndarray):
        binary_arrays = [binary_arrays]

    if np.any(np.isnan(binary_arrays)):
        raise ValueError('arrays must not contain NaN values')

    if isinstance(box_sizes, str):
        if box_sizes != 'default':
            raise ValueError(f'box_sizes={box_sizes} not supported')
        box_sizes = 2 ** np.arange(1, 15)  # assumed any array is smaller than 32768 pixels
    else:
        box_sizes = np.array(box_sizes)

    max_coarsening_factor = min(binary_arrays[0].shape) / min_pixels
    box_sizes = box_sizes[box_sizes <= max_coarsening_factor]
    box_sizes = box_sizes[box_sizes >= min_box_size]

    mean_number_boxes = np.empty((0, box_sizes.size), dtype=np.float32)

    for array in binary_arrays:
        number_boxes = []
        for factor in box_sizes:
            # Coarsen
            coarsened_array = encase_in_value(coarsen_array(array, factor), np.nan)

            # Count boxes
            if set == 'edge':
                nboxes = np.count_nonzero((coarsened_array > 0) & (coarsened_array < 1))
            elif set == 'ones':
                nboxes = np.count_nonzero(coarsened_array > 0)
            else:
                raise ValueError(f'set={set} not supported (supported values are "edge" or "ones")')

            number_boxes.append(nboxes)

        mean_number_boxes = np.append(mean_number_boxes, [number_boxes], axis=0)

    mean_number_boxes = np.mean(mean_number_boxes, axis=0)
    mean_number_boxes[mean_number_boxes == 0] = np.nan  # eliminate warning when logging 0

    (slope, _), (error, _) = linear_regression(np.log10(box_sizes), np.log10(mean_number_boxes))

    if return_values:
        return -slope, error, box_sizes, mean_number_boxes
    return -slope, error


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
    return_values: bool = False
) -> tuple[float, float] | tuple[float, float, NDArray, NDArray]:
    """
    Calculate the individual fractal dimension Df of objects within arrays.

    The method uses linear regression on log a vs. log p, where a and p are
    calculated not including structure holes, and omitting structures touching
    the array edge.

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
    return_values : bool, default=False
        If True, return additional data used in the calculation.

    Returns
    -------
    Df : float
        The individual fractal dimension.
    uncertainty : float
        Uncertainty estimate (95% confidence).
    log10_sqrt_a : np.ndarray, optional
        Log10 of sqrt(area) values. Only returned if return_values=True.
    log10_p : np.ndarray, optional
        Log10 of perimeter values. Only returned if return_values=True.

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
        array = remove_structure_holes(array)
        new_p, new_a, _, _ = get_structure_props(array, xs, ys)
        areas.extend(new_a)
        perimeters.extend(new_p)

    areas, perimeters = np.array(areas), np.array(perimeters)
    areas, perimeters = areas[(areas > min_a) & (areas < max_a)], perimeters[(areas > min_a) & (areas < max_a)]

    (slope, _), (err, _) = linear_regression(np.log10(np.sqrt(areas)), np.log10(perimeters))

    if return_values:
        return slope, err, np.log10(np.sqrt(areas)), np.log10(perimeters)
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
def correlation_integral(coordinates_to_check, coordinates_to_count, locations_x, locations_y, bins):
    """
    Calculate correlation integral for boundary coordinates.

    For each coordinates_to_check, calculate how many coordinates_to_count are
    less than a distance bin_i of the chosen coordinate, for each bin_i in bins.

    Parameters
    ----------
    coordinates_to_check : np.ndarray
        Array of coordinates of boundaries of shape [[x1,y1], [x2,y2], [x3,y3], ...].
    coordinates_to_count : np.ndarray
        Array of coordinates to count within distance bins.
    locations_x : np.ndarray
        X locations of each pixel.
    locations_y : np.ndarray
        Y locations of each pixel.
    bins : np.ndarray
        1-D array of distances to bin.

    Returns
    -------
    C_l : np.ndarray
        Correlation integral values, same shape as bins.

    Notes
    -----
    For example, np.sqrt((locations_x[i,j]-locations_x[p,q])**2 +
    (locations_y[i,j]-locations_y[p,q])**2) should represent the physical
    distance between pixel locations at i,j and p,q.
    """
    # Each thread gets its own copy to eliminate race condition during parallelization
    C_l_per_thread = np.zeros((numba.config.NUMBA_NUM_THREADS, bins.shape[0]))

    for i in numba.prange(coordinates_to_check.shape[0]):
        thread_id = numba.get_thread_id()
        for j in range(coordinates_to_count.shape[0]):
            p, q = coordinates_to_check[i]
            r, s = coordinates_to_count[j]

            dx = (locations_x[p, q] - locations_x[r, s])
            dy = (locations_y[p, q] - locations_y[r, s])

            distance = np.sqrt((dx ** 2) + (dy ** 2))
            for bin_index, bin in enumerate(bins):
                if distance < bin:
                    C_l_per_thread[thread_id, bin_index] += 1

    # Return total over all threads
    return np.sum(C_l_per_thread, axis=0)


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


def isolate_largest_structure(
    binary_array: NDArray,
    structure: NDArray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
) -> NDArray[np.bool_]:
    """
    Isolate the largest connected structure in a binary array.

    Parameters
    ----------
    binary_array : np.ndarray
        Binary input array.
    structure : np.ndarray, default=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        Connectivity structure.

    Returns
    -------
    np.ndarray
        Boolean array with only the largest structure set to True.

    Raises
    ------
    ValueError
        If ``binary_array`` contains no structures (no non-zero pixels).
    """
    labelled_array = label(binary_array, structure)[0]
    cloud_values = labelled_array[labelled_array != 0]  # remove background
    if cloud_values.size == 0:
        raise ValueError('binary_array contains no structures')
    values, counts = np.unique(cloud_values, return_counts=True)
    most_common = values[np.argmax(counts)]
    return labelled_array == most_common


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
    labelled_array, n_structures = label(array.astype(bool), output=np.float32)

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
        return np.zeros(array.shape, dtype=int)

    labelled_with_sizes = np.zeros(array.shape, dtype=int)

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
