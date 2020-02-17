import numpy as np


def make_strictly_decreasing(x_interp, y_interp, prepend_value=3.7):
    """
    Takes a monotonically decreasing array y_interp and makes it strictly decreasing by interpolation over x_interp.

    Arguments:
        x_interp {numpy.ndarray} -- The values to interpolate over.
        y_interp {numpy.ndarray} -- Monotonically decreasing values.

    Keyword Arguments:
        prepend_value {float} -- Value to prepend to y_interp for assesing the difference to the preceding value.
            (default: {3.7})

    Returns:
        numpy.ndarray -- y_interp with interpolated values, where there used to be zero difference to
            the preceding value.
    """
    y_interp_copy = y_interp.copy()
    # Make the tale interpolatable if the last value is not the single minimum.
    if y_interp_copy[-1] >= y_interp_copy[-2]:
        y_interp_copy[-1] -= 0.0001

    # Build a mask for all values, which do not decrease.
    bigger_equal_zero_diff = np.diff(y_interp_copy, prepend=prepend_value) >= 0
    # Replace these values with interpolations based on their border values.
    interp_values = np.interp(
        x_interp[bigger_equal_zero_diff],  # Where to evaluate the interpolation function.
        x_interp[~bigger_equal_zero_diff],  # X values for the interpolation function.
        y_interp_copy[~bigger_equal_zero_diff]  # Y values for the interpolation function.
    )
    y_interp_copy[bigger_equal_zero_diff] = interp_values

    # This has to be given, since interpolation will fail otherwise.
    assert np.all(np.diff(y_interp_copy) < 0), "The result y_copy is not strictly decreasing!"

    return y_interp_copy
