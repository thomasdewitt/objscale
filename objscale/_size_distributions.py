"""
Functions for calculating size distributions in 2-D domains while taking into account finite size effects.
By Thomas DeWitt (https://github.com/thomasdewitt/)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import warnings
from warnings import warn
from ._object_analysis import (
    label_structures,
    remove_structures_touching_border_nan,
    get_structure_areas,
    get_structure_perimeters,
    get_structure_height_width,
    get_every_boundary_perimeter,
)
from ._utils import linear_regression, encase_in_value

__all__ = [
    'finite_array_size_distribution',
    'finite_array_powerlaw_exponent',
    'array_size_distribution',
]


# Sentinel for "argument not supplied", mirroring the pattern used by
# individual_fractal_dimension in _fractal_dimensions.py. Lets us distinguish
# "user passed min_threshold" from "use the variable-aware default".
_UNSET = object()


def _normalize_variable(variable, stacklevel=3):
    """Map deprecated ``'perimeter'`` to ``'summed perimeter'`` with a warning."""
    if variable == 'perimeter':
        warnings.warn(
            "variable='perimeter' is deprecated, use 'summed perimeter' instead.",
            DeprecationWarning, stacklevel=stacklevel,
        )
        return 'summed perimeter'
    return variable


def _variable_bin_bounds(variable, xs, ys):
    """Physical (lower, upper) auto-bin bounds for one pixel-size grid.

    Bounds are variable-aware and NaN-aware so that, on grids with sub-unity
    pixel sizes, valid objects measured in length units are not silently
    discarded by a domain-AREA upper bound.

    - 'area': [min pixel area, total domain area]
    - 'width': [min x pixel, max row extent (nanmax over rows of nansum x)]
    - 'height': [min y pixel, max column extent (nanmax over cols of nansum y)]
    - 'summed perimeter' / 'nested perimeter':
        [min pixel length, space-filling bound nansum(sqrt(x*y))]
    """
    xy = xs * ys
    if variable == 'area':
        lower = np.nanmin(xy)
        upper = np.nansum(xy)
    elif variable == 'width':
        lower = np.nanmin(xs)
        upper = np.nanmax(np.nansum(xs, axis=1))
    elif variable == 'height':
        lower = np.nanmin(ys)
        upper = np.nanmax(np.nansum(ys, axis=0))
    elif variable in ('summed perimeter', 'nested perimeter'):
        lower = min(np.nanmin(xs), np.nanmin(ys))
        upper = np.nansum(np.sqrt(xy))
    else:
        raise ValueError(f'variable {variable} not supported')
    return float(lower), float(upper)


def _auto_bin_range(variable, x_sizes, y_sizes, n_arrays):
    """Variable-aware auto-bin (lower, upper) across possibly-many grids.

    When ``x_sizes``/``y_sizes`` are lists of per-array grids, the lower bound
    is the min of per-array lower bounds and the upper bound is the max of
    per-array upper bounds, so no array's objects fall outside the range.
    """
    lowers, uppers = [], []
    for i in range(n_arrays):
        xs = x_sizes[i] if isinstance(x_sizes, list) else x_sizes
        ys = y_sizes[i] if isinstance(y_sizes, list) else y_sizes
        lo, hi = _variable_bin_bounds(variable, xs, ys)
        lowers.append(lo)
        uppers.append(hi)
    return min(lowers), max(uppers)


def finite_array_size_distribution(
    arrays: NDArray | list[NDArray],
    variable: str,
    x_sizes: NDArray | list[NDArray] | None = None,
    y_sizes: NDArray | list[NDArray] | None = None,
    bins: int | NDArray = 100,
    bin_logs: bool = True,
    min_threshold: float = _UNSET,
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
        Object attribute to bin by. Options: ``'area'``, ``'summed perimeter'``,
        ``'nested perimeter'``, ``'height'``, ``'width'``. See Notes for
        definitions. ``'perimeter'`` is accepted but deprecated.
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
        The auto-bin range is variable-aware and NaN-aware (using the physical
        pixel-size grids), so length-unit variables are binned over a length
        range rather than the domain area:

        - 'area': [min pixel area, total domain area]
        - 'width': [min x pixel, max row extent]
        - 'height': [min y pixel, max column extent]
        - 'summed perimeter'/'nested perimeter':
          [min pixel length, nansum(sqrt(x_sizes*y_sizes))]

        With per-array ``x_sizes``/``y_sizes`` lists, the range spans all arrays
        (min of lower bounds, max of upper bounds).
    bin_logs : bool, default=True
        If True, bin log10(variable) into logarithmically-spaced bins. If False, bin
        variable into linearly spaced bins.
    min_threshold : float, optional
        Smallest bin edge (the lower edge of the auto-bin range). If not
        provided (or ``None``), defaults to the variable-aware minimum pixel
        scale (see ``bins``). If bin edges are passed explicitly, this arg is ignored.
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

    .. versionchanged:: 2.0.0
        When ``bins`` is an int, the auto-bin range is now variable-aware and
        NaN-aware (see ``bins``): length-unit variables (width/height/summed
        perimeter/nested perimeter) are binned over a length range instead of
        the domain area, so valid objects on grids with sub-unity pixel sizes
        are no longer silently discarded. ``min_threshold`` now defaults to the
        variable-aware minimum pixel scale rather than a fixed value of 10.

    Notes
    -----
    Variable definitions:

    - 'summed perimeter': Sum of pixel edge lengths between all pixels within a
      structure and neighboring values of 0. Does not include perimeter adjacent
      to a nan. A donut shaped structure returns a single value.
    - 'nested perimeter': Sum of pixel edge lengths between all pixels that are
      between a structure and a neighboring region of 0s. Does not include
      perimeter adjacent to a nan. A donut shaped structure returns two values:
      one for the inner circle and one for the outer.
    - 'area': Sum of individual pixel areas constituting the structure.
    - 'length' or 'width': Overall distance between the farthest two points in a
      structure in the x- or y- direction.
    """
    variable = _normalize_variable(variable)
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    if x_sizes is None:
        x_sizes = np.ones(arrays[0].shape, dtype=np.float32)
    if y_sizes is None:
        y_sizes = np.ones(arrays[0].shape, dtype=np.float32)

    if isinstance(bins, int):
        # Variable-aware, NaN-aware auto-bin range using the physical grids.
        # Spans all arrays when x_sizes/y_sizes are per-array lists.
        auto_lower, auto_upper = _auto_bin_range(variable, x_sizes, y_sizes, len(arrays))
        # A space-filling object attains the upper bound exactly (e.g. a full-row
        # object has width == max row extent). Give the top edge a hair of
        # headroom so the float32 measurement, upcast to float64, still lands
        # inside the last (right-inclusive) bin. Immaterial on a log axis.
        auto_upper = auto_upper * (1 + 1e-6)
        lower_edge = (auto_lower if (min_threshold is _UNSET or min_threshold is None)
                      else min_threshold)
        if bin_logs:
            bin_edges = np.linspace(np.log10(lower_edge), np.log10(auto_upper), bins + 1)
        else:
            bin_edges = np.linspace(lower_edge, auto_upper, bins + 1)
    else:
        bin_edges = np.asarray(bins)

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

        if variable in ['summed perimeter','area','height','width']:
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
        # One past the end of the COUNT arrays (which have len(bin_edges)-1
        # entries), so slicing [truncation_index:] is a no-op.
        truncation_index = len(bin_edges) - 1
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
    min_threshold: float = _UNSET,
    truncation_threshold: float = 0.5,
    min_count_threshold: int = 30,
    return_counts: bool = False
) -> float | tuple[float, tuple[NDArray, NDArray]]:
    """
    Calculate the power-law exponent for size distributions of structures.

    Calculates the power-law exponent for a list of binary arrays, where 'size' phi
    can be summed perimeter, area, length, or width::

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
        Object attribute to bin by. Options: ``'area'``, ``'summed perimeter'``,
        ``'nested perimeter'``, ``'height'``, ``'width'``. See Notes for
        definitions. ``'perimeter'`` is accepted but deprecated.
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
        The auto-bin range is variable-aware and NaN-aware; see
        :func:`finite_array_size_distribution`.
        If 1-D array: use these as log10(bin edges). They must be uniformly
        logarithmically spaced.
        If using per-array ``x_sizes``/``y_sizes`` lists with very different overall
        scales, prefer explicit ``bins``.
    min_threshold : float, optional
        Smallest bin edge (lower edge of the auto-bin range). If not provided
        (or ``None``), defaults to the variable-aware minimum pixel scale. If bin edges are
        passed explicitly, this arg is ignored.
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
    counts : tuple of np.ndarray, optional
        ``(log_bin_middles, log_counts)``: log10 of the bin middle values and
        log10 of the counts used in the regression. Only returned if
        ``return_counts=True``, as a single nested tuple
        ``(exponent, (log_bin_middles, log_counts))``.

    .. versionchanged:: 2.0.0
        No longer returns an uncertainty estimate. The previously reported
        error was ``2 x`` the OLS standard error of the log-log fit, which
        assumes independent residuals; the points of a size distribution
        derived from a fractal/multifractal field are strongly correlated
        across scales, so that error is badly miscalibrated (demonstrated by
        bootstrap in the section "Statistical error and parameter uncertainty"
        at https://thomasddewitt.com/thought-cloud/too-many-exponents/index.html).
        Users who need an uncertainty should bootstrap the exponent across
        statistically independent images.

    Notes
    -----
    Variable definitions:

    - 'summed perimeter': Sum of pixel edge lengths between all pixels within a
      structure and neighboring values of 0. Does not include perimeter adjacent
      to a nan. A donut shaped structure returns a single value.
    - 'nested perimeter': Sum of pixel edge lengths between all pixels that are
      between a structure and a neighboring region of 0s. Does not include
      perimeter adjacent to a nan. A donut shaped structure returns two values.
    - 'area': Sum of individual pixel areas constituting the structure.
    - 'length' or 'width': Overall distance between the farthest two points in a
      structure in the x- or y- direction.
    """
    variable = _normalize_variable(variable)
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

    # Orders of magnitude spanned by the bin range (log10 bin middles). Derived
    # from the returned bin middles rather than min_threshold, which may be the
    # variable-aware default sentinel.
    span_decades = log_bin_middles[-1] - log_bin_middles[0]
    if span_decades < 2:
        warn(f'Power law exponent is being estimated using data spanning only {span_decades:.01f} orders of magnitude')

    log_bin_middles[truncation_index:] = np.nan

    total_good_counts[total_good_counts == 0] = np.nan  # eliminate log of 0 warning

    log_total_good_counts = np.log10(total_good_counts)

    (slope, _), _ = linear_regression(log_bin_middles, log_total_good_counts)

    if return_counts:
        return -slope, (log_bin_middles, log_total_good_counts)
    return -slope


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
        Object attribute to bin by. Options: ``'area'``, ``'summed perimeter'``,
        ``'nested perimeter'``, ``'height'``, ``'width'``. See Notes for
        definitions. ``'perimeter'`` is accepted but deprecated.
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

    - 'summed perimeter': Sum of pixel edge lengths between all pixels within a
      structure and neighboring values of 0. Does not include perimeter adjacent
      to a nan. A donut shaped structure returns a single value.
    - 'nested perimeter': Sum of pixel edge lengths between all pixels that are
      between a structure and a neighboring region of 0s. Does not include
      perimeter adjacent to a nan. A donut shaped structure returns two values.
    - 'area': Sum of individual pixel areas constituting the structure.
    - 'length' or 'width': Overall distance between the farthest two points in a
      structure in the x- or y- direction.
    """
    variable = _normalize_variable(variable)
    if x_sizes is None:
        x_sizes = np.ones(array.shape, dtype=np.float32)
    if y_sizes is None:
        y_sizes = np.ones(array.shape, dtype=np.float32)
    if variable in ('area', 'summed perimeter', 'height', 'width'):
        # Pad non-periodic edges with NaN so get_structure_* (which assumes
        # full toroidal periodicity) correctly treats them as domain boundaries.
        if wrap is None:
            array = encase_in_value(array, value=np.nan)
            x_sizes = encase_in_value(x_sizes, value=np.nan)
            y_sizes = encase_in_value(y_sizes, value=np.nan)
        elif wrap == 'sides':
            # Periodic left-right, pad top-bottom only
            array = np.concatenate([np.full((1, array.shape[1]), np.nan), array,
                                    np.full((1, array.shape[1]), np.nan)], axis=0)
            x_sizes = np.concatenate([np.full((1, x_sizes.shape[1]), np.nan), x_sizes,
                                      np.full((1, x_sizes.shape[1]), np.nan)], axis=0)
            y_sizes = np.concatenate([np.full((1, y_sizes.shape[1]), np.nan), y_sizes,
                                      np.full((1, y_sizes.shape[1]), np.nan)], axis=0)
        # wrap='both': fully periodic, no padding needed

        lab, nm, nl = label_structures(array, structure, wrap='both')
        if lab is None:
            to_bin = np.array([], dtype=np.float32)
        elif variable == 'area':
            a = get_structure_areas(lab, nm, nl, x_sizes, y_sizes)
            to_bin = a[a > 0]
        elif variable == 'summed perimeter':
            p = get_structure_perimeters(lab, nm, nl, x_sizes, y_sizes)
            to_bin = p[p > 0]
        elif variable in ('height', 'width'):
            h, w = get_structure_height_width(lab, nm, nl, x_sizes, y_sizes)
            a = get_structure_areas(lab, nm, nl, x_sizes, y_sizes)
            valid = a > 0
            to_bin = h[valid] if variable == 'height' else w[valid]
    elif variable == 'nested perimeter':
        to_bin = get_every_boundary_perimeter(array, x_sizes, y_sizes, wrap=wrap)
    else:
        raise ValueError(f'Unsupported variable: {variable}')

    if not isinstance(bins, int):
        bins = np.asarray(bins)

    if len(to_bin) == 0:
        if isinstance(bins, int):
            return np.zeros(bins), np.zeros(bins)
        else:
            bin_middles = 0.5 * (bins[:-1] + bins[1:])
            return bin_middles, np.zeros(len(bins) - 1)

    if bin_logs:
        to_bin = np.log10(to_bin)

    if isinstance(bins, int):
        bin_edges = np.linspace(min(to_bin), max(to_bin), bins + 1)
    else:
        bin_edges = bins

    if np.count_nonzero(to_bin > bin_edges[-1]) > 0:
        warn(f'There exist {variable}s outside of bin edges that are being ignored')
    counts, _ = np.histogram(to_bin, bins=bin_edges)

    # Per-bin midpoints (edges-based) — correct for both uniform and
    # non-uniform (e.g. log-spaced) custom bin edges.
    bin_middles = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_middles, counts
