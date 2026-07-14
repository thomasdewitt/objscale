from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from warnings import warn
import numba

__all__ = ['linear_regression', 'encase_in_value', 'set_num_threads']


def validate_pixel_sizes(arrays, x_sizes, y_sizes):
    """Validate the x_sizes/y_sizes contract against a list of arrays.

    The contract, shared by every function that accepts per-array pixel-size
    grids (size distributions, individual fractal dimension):

    - ``None``: unit pixels are assumed; no shared grid is required, so arrays
      of differing shapes are allowed.
    - a single ``np.ndarray``: it is applied to every array, so every array
      must share its shape.
    - a ``list``: one grid per array, each matching its array's shape.

    Raises ``ValueError`` with an actionable message on any violation. Does not
    return anything; call it for its side effect before consuming the grids.
    """
    if (x_sizes is None) != (y_sizes is None):
        raise ValueError(
            'x_sizes and y_sizes must both be None (unit pixels) or both '
            'provided; a single None is ambiguous.'
        )
    for name, sizes in (('x_sizes', x_sizes), ('y_sizes', y_sizes)):
        if sizes is None:
            continue
        if isinstance(sizes, list):
            if len(sizes) != len(arrays):
                raise ValueError(
                    f'{name} is a list of length {len(sizes)} but there are '
                    f'{len(arrays)} arrays; when {name} is a list it must have '
                    f'exactly one grid per array.'
                )
            for i, (s, a) in enumerate(zip(sizes, arrays)):
                if s.shape != a.shape:
                    raise ValueError(
                        f'{name}[{i}] has shape {s.shape} but arrays[{i}] has '
                        f'shape {a.shape}; each pixel-size grid must match its '
                        f'array.'
                    )
        else:
            for i, a in enumerate(arrays):
                if sizes.shape != a.shape:
                    raise ValueError(
                        f'{name} is a single grid of shape {sizes.shape} but '
                        f'arrays[{i}] has shape {a.shape}; a single {name} grid '
                        f'requires every array to share that shape. Pass a list '
                        f'of per-array grids, or None for unit pixels, when '
                        f'array shapes differ.'
                    )


def set_num_threads(n: int) -> None:
    """
    Set the number of threads used for parallel computations.

    Controls the number of threads Numba uses for parallel operations
    such as correlation integral calculation and structure property analysis.

    Parameters
    ----------
    n : int
        Number of threads to use. Must be between 1 and the number of
        logical CPU cores available.

    Raises
    ------
    ValueError
        If n is less than 1 or greater than the available CPU count.
    """
    numba.set_num_threads(n)


def linear_regression(
    x: NDArray[np.floating],
    y: NDArray[np.floating]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Perform linear regression and return coefficients with 95% confidence errors.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Dependent variable values.

    Returns
    -------
    coefficients : tuple of float
        (slope, y-intercept) from linear regression.
    errors : tuple of float
        (error_slope, error_y_intercept) for 95% confidence.

    Warning
    -------
    The returned errors are ``2 x`` the OLS standard error of the fit, which
    assumes the residuals of the ``(x, y)`` data points are statistically
    independent. This assumption does **not** hold for scaling functions
    computed from a fractal/multifractal field: the points of such a scaling
    function are strongly correlated across scales, so these errors are badly
    miscalibrated and must **not** be reported as the uncertainty of an
    exponent or dimension estimated from a single field. The only viable way
    to get a trustworthy uncertainty in that case is to bootstrap the estimate
    across statistically independent images. This miscalibration is the reason
    the public exponent/dimension estimators in :mod:`objscale` stopped
    returning an uncertainty in v2.0.0; see the section "Statistical error and
    parameter uncertainty" at
    https://thomasddewitt.com/thought-cloud/too-many-exponents/index.html
    for a bootstrap demonstration.

    Raises
    ------
    TypeError
        If x or y are not numpy arrays.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('x, y, must be of type np.ndarray')
    index = np.isfinite(x) & np.isfinite(y)
    if len(x[index]) < 3:    # "the number of data points must exceed order to scale the covariance matrix"
        warn('Less than 3 points (x,y) are good (not nan), returning nans')
        return (np.nan, np.nan), (np.nan, np.nan)
    try:
        coefficients, cov = np.polyfit(x[index], y[index], 1, cov=True)
        error = np.sqrt(np.diag(cov))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
        warn(f'Linear regression failed: {e!s}')
        return (np.nan, np.nan), (np.nan, np.nan)
    return tuple(coefficients), tuple(2 * error)  # 95% conf interval is 2 times standard error


def encase_in_value(
    array: NDArray,
    value: float = np.nan,
    dtype: np.dtype = np.float32,
    n_deep: int = 1
) -> NDArray:
    """
    Add a border of specified value around a 2-D array.

    Parameters
    ----------
    array : np.ndarray
        2-D input array.
    value : scalar, default=np.nan
        Value to append on the edge.
    dtype : np.dtype, default=np.float32
        Data type of the resulting array.
    n_deep : int, default=1
        Number of layers deep to add around the edge.

    Returns
    -------
    np.ndarray
        Same as input but with a layer 'n_deep' of 'value' all around the edge.
    """
    out_dtype = np.result_type(dtype, array.dtype)
    h, w = array.shape
    result = np.full((h + 2 * n_deep, w + 2 * n_deep), value, dtype=out_dtype)
    result[n_deep:n_deep + h, n_deep:n_deep + w] = array
    return result
