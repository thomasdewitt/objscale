"""
Functions for calculating size distributions in 2-D domains while taking into account finite size effects.
By Thomas DeWitt (https://github.com/thomasdewitt/)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from warnings import warn
from ._object_analysis import remove_structures_touching_border_nan, get_structure_props, get_every_boundary_perimeter
from ._utils import linear_regression, encase_in_value

__all__ = [
    'finite_array_size_distribution',
    'finite_array_powerlaw_exponent',
    'array_size_distribution',
]


def finite_array_size_distribution(
    arrays: NDArray | list[NDArray],
    variable: str,
    x_sizes: NDArray | list[NDArray] | None = None,
    y_sizes: NDArray | list[NDArray] | None = None,
    bins: int | NDArray = 100,
    bin_logs: bool = True,
    min_threshold: float = 10,
    truncation_threshold: float = 0.5
) -> tuple[NDArray, NDArray, NDArray, int]:
    """
    Calculate size distributions for structures within binary arrays.

    Returns the size distributions for truncated objects and nontruncated objects
    and the index where truncated objects begin to dominate. Works for binary arrays
    and also for binary arrays where the data boundary is demarcated by nans, enabling
    the domain boundary to be an arbitrary shape.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray
        2-D arrays where objects of interest have value 1, background has value 0,
        and no data has np.nan. Interior nans are treated like 0's, except the
        perimeter along them is not counted.
    variable : str
        Object attribute to bin by. Options: 'area', 'perimeter', 'nested perimeter',
        'height', 'width'. See Notes for definitions.
    x_sizes : np.ndarray or list, optional
        Pixel sizes in the x direction. If None, assume all pixel dimensions are 1.
        If np.ndarray, use these for each array in 'arrays'. If list, assume
        x_sizes[i] corresponds to arrays[i].
    y_sizes : np.ndarray or list, optional
        Pixel sizes in the y direction. If None, assume all pixel dimensions are 1.
        If np.ndarray, use these for each array in 'arrays'. If list, assume
        y_sizes[i] corresponds to arrays[i].
    bins : int or array-like, default=100
        If int, auto calculate bin locations and make that number of bins.
        If 1-D array: use these as bin edges or log10(bin edges). They must be uniformly
        linearly or logarithmically spaced (depending on bin_logs).
        If using per-array ``x_sizes``/``y_sizes`` lists with very different overall
        scales, pass explicit ``bins`` to avoid auto-bin ranges being dominated by
        the first array.
    bin_logs : bool, default=True
        If True, bin log10(variable) into logarithmically-spaced bins. If False, bin
        variable into linearly spaced bins.
    min_threshold : float, default=10
        Smallest bin edge. If bin edges are passed, this arg is ignored.
    truncation_threshold : float, default=0.5
        Float between 0 and 1. Bins with a larger fraction of truncated objects than
        this are omitted from the regression.

    Returns
    -------
    bin_middles : np.ndarray
        Bin middle values. If bin_logs is True, this is actually log10(bin_middles).
    nontruncated_counts : np.ndarray
        Counts of non-truncated objects in each bin.
    truncated_counts : np.ndarray
        Counts of truncated objects in each bin.
    truncation_index : int
        Index where truncated objects begin to dominate.

    Notes
    -----
    Variable definitions:

    - 'perimeter': Sum of pixel edge lengths between all pixels within a structure and
      neighboring values of 0. Does not include perimeter adjacent to a nan.
      A donut shaped structure returns a single value.
    - 'nested perimeter': Sum of pixel edge lengths between all pixels that are between a structure
      and a neighboring region of 0s. Does not include perimeter adjacent to a nan.
      A donut shaped structure returns two values: one for the inner circle and one for the outer.
    - 'area': Sum of individual pixel areas constituting the structure.
    - 'length' or 'width': Overall distance between the farthest two points in a structure in
      the x- or y- direction.
    """
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    if x_sizes is None:
        x_sizes = np.ones(arrays[0].shape, dtype=np.float32)
    if y_sizes is None:
        y_sizes = np.ones(arrays[0].shape, dtype=np.float32)

    if isinstance(bins, int):
        if isinstance(x_sizes, list):
            max_value = np.nansum(x_sizes[0] * y_sizes[0])
        else:
            max_value = np.nansum(x_sizes * y_sizes)
        if bin_logs:
            bin_edges = np.linspace(np.log10(min_threshold), np.log10(max_value), bins + 1)
        else:
            bin_edges = np.linspace(min_threshold, max_value, bins + 1)
    else:
        bin_edges = bins

    truncated_counts = np.zeros(bin_edges.size - 1)
    nontruncated_counts = np.zeros(bin_edges.size - 1)

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

        # Encase the array in nans to ensure objects in contact with the edge are considered truncated
        array = encase_in_value(array)

        if variable in ['perimeter','area','height','width']:
            no_truncated = remove_structures_touching_border_nan(array)
            truncated_only = array - no_truncated

            truncated_counts += array_size_distribution(truncated_only, x_sizes=encase_in_value(xs), y_sizes=encase_in_value(ys), variable=variable, wrap=None, bins=bin_edges, bin_logs=bin_logs)[1]
            nontruncated_counts += array_size_distribution(no_truncated, x_sizes=encase_in_value(xs), y_sizes=encase_in_value(ys), variable=variable, wrap=None, bins=bin_edges, bin_logs=bin_logs)[1]
        elif variable == 'nested perimeter':  # nested perimeter
            # For this case, an edge-touching cloud may have a hole but the hole does not touch the edge. 
            # We want to count the perimeter of the hole in the non-edge-touching histogram.
            # print(array)
            no_truncated = remove_structures_touching_border_nan(array)
            truncated_but_with_holes_that_are_not_truncated = array - no_truncated

            nontruncated_counts += array_size_distribution(no_truncated, x_sizes=encase_in_value(xs), y_sizes=encase_in_value(ys), variable=variable, wrap=None, bins=bin_edges, bin_logs=bin_logs)[1]




            cloud_holes_truncated_and_nontruncated = 1-truncated_but_with_holes_that_are_not_truncated
            cloud_holes_nontruncated = remove_structures_touching_border_nan(cloud_holes_truncated_and_nontruncated)
            # print(cloud_holes_nontruncated)
            # exit()

            nontruncated_counts += array_size_distribution(cloud_holes_nontruncated, x_sizes=encase_in_value(xs), y_sizes=encase_in_value(ys), variable=variable, wrap=None, bins=bin_edges, bin_logs=bin_logs)[1]

            # cloud_holes_truncated = cloud_holes_truncated_and_nontruncated - cloud_holes_nontruncated

            # Now fill in those nontruncated holes to obtain holes+clouds that are only truncated
            truncated_only = truncated_but_with_holes_that_are_not_truncated + cloud_holes_nontruncated

            truncated_counts += array_size_distribution(truncated_only, x_sizes=encase_in_value(xs), y_sizes=encase_in_value(ys), variable=variable, wrap=None, bins=bin_edges, bin_logs=bin_logs)[1]
        else: raise ValueError(f'variable {variable} not supported')

    # Find index where number of edge clouds is greater than threshold times total number of clouds
    truncation_index = np.argwhere(truncated_counts > truncation_threshold * (truncated_counts + nontruncated_counts))
    if truncation_index.size == 0:  # then there is no need to truncate
        truncation_index = len(bin_edges)
    else:
        truncation_index = truncation_index[0, 0]

    bin_middles = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])  # shift to center and remove value at end that shifted beyond bins

    return bin_middles, nontruncated_counts, truncated_counts, truncation_index


def finite_array_powerlaw_exponent(
    arrays: NDArray | list[NDArray],
    variable: str,
    x_sizes: NDArray | list[NDArray] | None = None,
    y_sizes: NDArray | list[NDArray] | None = None,
    bins: int | NDArray = 100,
    min_threshold: float = 10,
    truncation_threshold: float = 0.5,
    min_count_threshold: int = 30,
    return_counts: bool = False
) -> tuple[float, float] | tuple[tuple[float, float], tuple[NDArray, NDArray]]:
    """
    Calculate the power-law exponent for size distributions of structures.

    Calculates the power-law exponent for a list of binary arrays, where 'size' phi
    can be perimeter, area, length, or width::

        n(phi) ∝ phi^{-(1+exponent)}

    Works for binary arrays and also for binary arrays where the data boundary is
    demarcated by nans. This enables the domain boundary to be an arbitrary shape,
    rather than be rectangular.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray
        2-D arrays where objects of interest have value 1, the background has value 0,
        and no data has np.nan. Interior nans are treated like 0's, except the
        perimeter along them is not counted.
    variable : str
        Object attribute to bin by. Options: 'area', 'perimeter', 'nested perimeter',
        'height', 'width'. See Notes for definitions.
    x_sizes : np.ndarray or list, optional
        Pixel sizes in the x direction. If None, assume all pixel dimensions are 1.
        If np.ndarray, use these for each array in 'arrays'. If list, assume
        x_sizes[i] corresponds to arrays[i].
    y_sizes : np.ndarray or list, optional
        Pixel sizes in the y direction. If None, assume all pixel dimensions are 1.
        If np.ndarray, use these for each array in 'arrays'. If list, assume
        y_sizes[i] corresponds to arrays[i].
    bins : int or array-like, default=100
        If int, auto calculate bin locations and make that number of bins.
        If 1-D array: use these as log10(bin edges). They must be uniformly
        logarithmically spaced.
        If using per-array ``x_sizes``/``y_sizes`` lists with very different overall
        scales, prefer explicit ``bins``.
    min_threshold : float, default=10
        Smallest bin edge. If bin edges are passed, this arg is ignored.
    truncation_threshold : float, default=0.5
        Float between 0 and 1. Bins with a larger fraction of truncated objects
        than this are omitted from the regression.
    min_count_threshold : int, default=30
        Omit any bin with counts fewer than this value from the linear regression.
    return_counts : bool, default=False
        If True, also return the bin middles and counts used in the regression.

    Returns
    -------
    exponent : float
        The power-law exponent.
    error : float
        Error estimate (95% confidence interval).
    log_bin_middles : np.ndarray, optional
        Log10 of bin middle values. Only returned if return_counts=True.
    log_counts : np.ndarray, optional
        Log10 of counts used in regression. Only returned if return_counts=True.

    Notes
    -----
    Variable definitions:

    - 'perimeter': Sum of pixel edge lengths between all pixels within a structure and
      neighboring values of 0. Does not include perimeter adjacent to a nan.
      A donut shaped structure returns a single value.
    - 'nested perimeter': Sum of pixel edge lengths between all pixels that are between
      a structure and a neighboring region of 0s. Does not include perimeter adjacent
      to a nan. A donut shaped structure returns two values.
    - 'area': Sum of individual pixel areas constituting the structure.
    - 'length' or 'width': Overall distance between the farthest two points in a
      structure in the x- or y- direction.
    """
    log_bin_middles, nontruncated_counts, truncated_counts, truncation_index = finite_array_size_distribution(
        arrays=arrays,
        variable=variable,
        x_sizes=x_sizes,
        y_sizes=y_sizes,
        bins=bins,
        bin_logs=True,
        min_threshold=min_threshold,
        truncation_threshold=truncation_threshold
    )

    total_good_counts = (truncated_counts + nontruncated_counts)
    total_good_counts[truncation_index:] = np.nan  # remove bins with too many truncated objects

    total_good_counts[total_good_counts < min_count_threshold] = np.nan  # remove bins with too few counts

    if log_bin_middles[total_good_counts.size - 1] - np.log10(min_threshold) < 2:
        warn(f'Power law exponent is being estimated using data spanning only {log_bin_middles[total_good_counts.size - 1] - np.log10(min_threshold):.01f} orders of magnitude')

    log_bin_middles[truncation_index:] = np.nan

    total_good_counts[total_good_counts == 0] = np.nan  # eliminate log of 0 warning

    log_total_good_counts = np.log10(total_good_counts)

    (slope, _), (slope_error, _) = linear_regression(log_bin_middles, log_total_good_counts)

    if return_counts:
        return (-slope, slope_error), (log_bin_middles, log_total_good_counts)
    return -slope, slope_error


def array_size_distribution(
    array: NDArray,
    variable: str = 'area',
    bins: int | NDArray = 30,
    bin_logs: bool = True,
    structure: NDArray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    wrap: str | None = None,
    x_sizes: NDArray | None = None,
    y_sizes: NDArray | None = None
) -> tuple[NDArray, NDArray]:
    """
    Calculate size distribution for a single binary array.

    Given a single binary array, calculate contiguous object sizes and compute
    a size distribution.

    .. warning::
        This function does not account for bias arising from object truncation
        by the domain edge. Prefer ``finite_array_*`` functions for most purposes.

    Parameters
    ----------
    array : np.ndarray
        2-D array where objects of interest have value 1, the background has value 0,
        and no data has np.nan. Nans are treated like 0's, except the perimeter along
        them is not counted.
    variable : str, default='area'
        Object attribute to bin by. Options: 'area', 'perimeter', 'nested perimeter',
        'height', 'width'. See Notes for definitions.
    bins : int or array-like, default=30
        If int, auto calculate bin locations and make that number of bins.
        If 1-D array: use these as bin edges.
    bin_logs : bool, default=True
        If True, bin log10(variable), else bin variable linearly.
    structure : np.ndarray, default=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        3x3 array defining object connectivity.
    wrap : str or None, default=None
        Boundary wrapping options: None, 'sides', 'both'.
        If 'sides', connect structures that span the left/right edge.
        If 'both', connect structures that span all edges.
    x_sizes : np.ndarray, optional
        Pixel sizes in x direction. If None, assume all lengths are 1.
    y_sizes : np.ndarray, optional
        Pixel sizes in y direction. If None, assume all lengths are 1.

    Returns
    -------
    bin_middles : np.ndarray
        Bin middle values. If bin_logs is True, these are log10(bin values).
    counts : np.ndarray
        Counts in each bin.

    Raises
    ------
    ValueError
        If variable is not one of the supported options.

    Notes
    -----
    Variable definitions:

    - 'perimeter': Sum of pixel edge lengths between all pixels within a structure and
      neighboring values of 0. Does not include perimeter adjacent to a nan.
      A donut shaped structure returns a single value.
    - 'nested perimeter': Sum of pixel edge lengths between all pixels that are between
      a structure and a neighboring region of 0s. Does not include perimeter adjacent
      to a nan. A donut shaped structure returns two values.
    - 'area': Sum of individual pixel areas constituting the structure.
    - 'length' or 'width': Overall distance between the farthest two points in a
      structure in the x- or y- direction.
    """
    if x_sizes is None:
        x_sizes = np.ones(array.shape, dtype=np.float32)
    if y_sizes is None:
        y_sizes = np.ones(array.shape, dtype=np.float32)
    if variable in ['area', 'perimeter', 'height', 'width']:
        p, a, h, w = get_structure_props(array, x_sizes, y_sizes, structure, wrap=wrap)
        if variable == 'area':
            to_bin = a
        elif variable == 'perimeter':
            to_bin = p
        elif variable == 'height':
            to_bin = h
        elif variable == 'width':
            to_bin = w
    elif variable == 'nested perimeter':
        to_bin = get_every_boundary_perimeter(array, x_sizes, y_sizes, False)
    else:
        raise ValueError(f'Unsupported variable: {variable}')

    if bin_logs:
        to_bin = np.log10(to_bin)

    if isinstance(bins, int):
        bin_edges = np.linspace(min(to_bin), max(to_bin), bins + 1)
    else:
        bin_edges = bins

    if np.count_nonzero(to_bin > bin_edges[-1]) > 0:
        warn(f'There exist {variable}s outside of bin edges that are being ignored')
    counts, _ = np.histogram(to_bin, bins=bin_edges)

    bin_middles = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])  # shift to center and remove value at end that shifted beyond bins

    return bin_middles, counts
