import numpy as np
from warnings import warn


def linear_regression(x, y):
    """
        Return (slope, y-int), (error_slope, error_y_int) for 95% conf
    """
    if type(x) != np.ndarray or type(y) != np.ndarray: raise TypeError('x, y, must be of type np.ndarray')
    index = np.isfinite(x) & np.isfinite(y)
    if len(x[index]) <3:    # "the number of data points must exceed order to scale the covariance matrix"
        warn('Less than 3 points (x,y) are good (not nan), returning nans')
        return (np.nan, np.nan),(np.nan, np.nan)
    try:
        coefficients, cov = np.polyfit(x[index], y[index], 1, cov=True)
        error = np.sqrt(np.diag(cov))
    except Exception as e:
        warn(f'Linear regression failed: {e!s}'
)
        return (np.nan, np.nan),(np.nan, np.nan)
    return coefficients, 2*error  # 95% conf interval is 2 times standard error 


def encase_in_value(array, value=np.nan, dtype=np.float32, n_deep=1):
    """
        Input:
            array: 2-D np.ndarray
            value: value to append on the edge
            dtype: dtype of the resulting array
        Output:
            array: Same as input but with a layer 'n_deep' of 'value' all around the edge: 2-D np.ndarray
    """

    nans_lr = np.empty((array.shape[0],n_deep), dtype=dtype)
    nans_tb = np.empty((n_deep, array.shape[1]+(2*n_deep)), dtype=dtype)  # will be two bigger after first appends
    nans_lr[:], nans_tb[:] = value, value
    array = np.append(nans_lr, array, axis=1)
    array = np.append(array, nans_lr, axis=1)
    array = np.append(nans_tb, array, axis=0)
    array = np.append(array, nans_tb, axis=0)
    return array