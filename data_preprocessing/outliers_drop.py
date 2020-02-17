import warnings
import numpy as np


class DropCycleException(Exception):
    """
    Used for dropping whole cycles without additional information
    """
    pass


class OutlierException(Exception):
    """
    Used for dropping whole cycles based on detected outliers
    """
    def __init__(self, message, outlier_dict):
        super().__init__(message)
        self.outlier_dict = outlier_dict


def check_for_drop_warning(array_before, array_after, drop_warning_thresh=0.10):
    """
    Checks weather the size of array_after is "drop_warning_thresh"-percent
    smaller than array_before and issues a warning in that case
    """

    try:
        assert float(len(array_before) - len(array_after)) / len(array_before) < drop_warning_thresh, \
            """More than {} percent of values were dropped ({} out of {}).""".format(
                drop_warning_thresh * 100,
                len(array_before) - len(array_after),
                len(array_before))
    except AssertionError as e:
        warnings.warn(str(e))
        # simple_plotly(array_before[-1], V_original=array_before[2])
        # simple_plotly(array_after[-1], V_indexed=array_after[2])
    finally:
        pass


def array_exclude_index(arr, id):
    """
    Returns the given array without the entry at id.
    id can be any valid numpy index
    """

    mask = np.ones_like(arr, bool)
    mask[id] = False
    return arr[mask]


def outlier_dict_without_mask(outlier_dict):
    """
    Modifies an outlier dict for printing purposes by removing the mask

    Arguments:
        outlier_dict {dict} -- Original outliert dictionary.

    Returns:
        dict -- Same outlier dict without the key "outliert_mask"
    """
    outlier_dict_wo_mask = dict()
    for key in outlier_dict.keys():
        outlier_dict_wo_mask[key] = {k: v for k, v in outlier_dict[key].items() if k != "outlier_mask"}
    return outlier_dict_wo_mask


def drop_outliers_starting_left(Qd, T, V, t, verbose, std_threshold):
    """
    Searches for outliers in Qd, T, V and t and drops them one by one starting with the smallest index.
    Outlier indeces are dropped from every array simultaniously, so the sizes still match.
    After the first outliers from every array have been dropped, outliers are computed again, to not drop
    false detections.

    Arguments:
        std_multiple_threshold {int} -- Threshold for the compute_outlier_dict function
        Qd {numpy.ndarray} -- Qd measurements
        T {numpy.ndarray} -- T measurements
        V {numpy.ndarray} -- V measurements
        t {numpy.ndarray} -- t measurements

    Returns:
        tuple of numpy.ndarrays -- All arrays without outliers
    """
    Qd_, T_, V_, t_ = Qd.copy(), T.copy(), V.copy(), t.copy()

    # Initialize and compute outliers
    drop_counter = 0
    outlier_dict = compute_outlier_dict(std_threshold, verbose=False, Qd=Qd_, T=T_, V=V_, t=t_)
    original_outlier_dict = outlier_dict  # copy for debugging und raising OutlierException.

    # Process until no outliers are found.
    while outlier_dict:
        # Get indeces of the left most outlier for every array
        first_outlier_indeces = [np.min(outlier_info["outlier_indeces"]) for outlier_info in outlier_dict.values()]
        # Only consider every index once and make it a list type for numpy indexing in array_exclude_index().
        unique_indeces_to_drop = list(set(first_outlier_indeces))

        # Drop all unique outlier indeces from all arrays.
        Qd_ = array_exclude_index(Qd_, unique_indeces_to_drop)
        T_ = array_exclude_index(T_, unique_indeces_to_drop)
        V_ = array_exclude_index(V_, unique_indeces_to_drop)
        t_ = array_exclude_index(t_, unique_indeces_to_drop)

        drop_counter += len(unique_indeces_to_drop)

        # Recompute outlierts after dropping the unique indeces from all arrays.
        outlier_dict = compute_outlier_dict(std_threshold, verbose=False, Qd=Qd_, T=T_, V=V_, t=t_)

    if drop_counter > 0 and verbose is True:
        print("\tDropped {} outliers in {}".format(drop_counter, list(original_outlier_dict.keys())))
        print("")

    check_for_drop_warning(Qd, Qd_)
    return Qd_, T_, V_, t_


def compute_outlier_dict(std_threshold, verbose, **signals):
    """
    Checks for outliers in all numpy arrays given in *signals* by computing the
    standard deveation of np.diff(). Outliers for every array are defined at
    the indeces, where the np.diff() is bigger than std_threshold times
    the standard deviation.

    Keyword Arguments:
        std_threshold {int} -- Threshold that defines an outlier by multiplying with the
            standard deveation (default: {15})
        verbose {bool} -- If True, prints the values for every found outlier (default: {False})

    Returns:
        dict -- The outliert results taged by the names given in kwargs
    """
    outlier_dict = dict()

    for key, value in signals.items():
        # Caulculate the n-th discrete difference along the given axis,
        # out[i] = a[i+1] - a[i]
        diff_values = np.diff(value)
        # Check the standard deviation in the difference of the values
        # calculated above
        std_diff = diff_values.std()

        # The coefficient of variation (CV=SD/MEAN) is higher than one for most signals.
        # This implies a high frequency of extreme values. In order to filter them out
        # we use our threshold

        # Get the mask for all outliers
        outlier_mask = diff_values > (std_threshold * std_diff)
        # Get the indeces for all outliers
        outlier_indeces = np.argwhere(outlier_mask)

        # Add outlier information to the outlier dict, if an outlier has been found
        if outlier_indeces.size > 0:
            outlier_dict[key] = dict(std_diff=std_diff,
                                     original_values=value[outlier_indeces],
                                     diff_values=diff_values[outlier_indeces],
                                     outlier_indeces=outlier_indeces,
                                     outlier_mask=outlier_mask)

    if verbose and outlier_dict:
        # If outlier_dict has any entries, then print a version without the mask (too big for printing)
        outlier_dict_wo_mask = outlier_dict_without_mask(outlier_dict)  # Generate a smaller dict for better printing
        print("\n############ Found outliers ############")
        print(outlier_dict_wo_mask)
        print("####################################")

    return outlier_dict


def handle_small_Qd_outliers(Qd, t, verbose, std_threshold, Qd_max_outlier):
    """
    Handles specifically small outliers in Qd, which are a result of constant values for a
    small number of measurements before the "outlier". The constant values are imputed by
    linearly interpolating Qd over t, since Qd over t should be linear anyways. This way the
    "outlier" is "neutralized", since there is no "step" left from the constant values to the
    outlier value.

    Arguments:
        std_multiple_threshold {int} -- Threshold to use for the compute_outlier_dict function
        Qd {numpy.ndarray} -- Qd measurements
        t {numpy.ndarray} -- t measurements corresponding to Qd

    Keyword Arguments:
        Qd_max_outlier {float} -- The maximum absolute value for the found outliers in Qd, which get handled
            by this function.
        This is needed only to make the function more specific. (default: {0.06})

    Returns:
        numpy.ndarray -- The interpolated version of Qd.
    """
    Qd_ = Qd.copy()  # Only copy Qd, since it is the only array values are assigned to
    outlier_dict = compute_outlier_dict(std_threshold, verbose=False, Qd=Qd_)

    if outlier_dict.get("Qd"):
        # Get only the indeces of all small outliers
        small_diff_value_mask = outlier_dict["Qd"]["diff_values"] <= Qd_max_outlier
        ids = outlier_dict["Qd"]["outlier_indeces"][small_diff_value_mask]
    else:
        ids = None

    if ids:
        # Interpolate all values before small outliers that stay constant (np.diff == 0)
        for i in ids:
            # Get the last index, where the value of Qd doesn't stay constant before the outlier.
            start_id = int(np.argwhere(np.diff(Qd_[:i]) > 0)[-1])

            # Make a mask for where to interpolate
            interp_mask = np.zeros_like(Qd_, dtype=bool)
            interp_mask[start_id:i] = True
            interp_values = np.interp(
                t[interp_mask],  # Where to evaluate the interpolation function.
                t[~interp_mask],  # X values for the interpolation function.
                Qd_[~interp_mask]  # Y values for the interpolation function.
            )
            # Assign the interpolated values
            Qd_[interp_mask] = interp_values
            if verbose is True:
                print("\tInterpolated small Qd outlier from index {} to {}".format(start_id, i))

    return Qd_


def drop_cycle_big_t_outliers(Qd, T, V, t, verbose, std_threshold, t_diff_outlier_thresh):
    """
    Checks for big outliers in the np.diff() values of t
    If any are found the whole cyce is dropped, with one exception:
    There is only one outlier which lays right after the end of discharging.
    In this case, all measurement values of Qd, T, V and t after this outlier are
    dropped and their values returned

        The end of discharging is defined as a V value below 2.01.

    Arguments:
        outlier_dict {dict} -- Dictionary with outlier information for the whole cycle.
        Qd {numpy.ndarray} -- Qd during discharging
        T {numpy.ndarray} -- T during discharging
        V {numpy.ndarray} -- V during discharging
        t {numpy.ndarray} -- t during discharging
        t_diff_outlier_thresh {int} -- Threshold that defines what a "big" t outlier is (default: {15})

    Raises:
        OutlierException: Will be raised, if the whole cycle is dropped

    Returns:
        Tuple of numpy.ndarray  -- Returns the original values of Qd, T, V and t if no big t outlier is found, or
            a slice of all arrays if the only outlier lays right after the end of discharging.
    """
    outlier_dict = compute_outlier_dict(std_threshold=std_threshold, verbose=verbose, Qd=Qd, T=T, V=V, t=t)
    if outlier_dict.get("t"):  # If any outlier was found in t
        t_outlier_mask = outlier_dict["t"]["diff_values"] > t_diff_outlier_thresh
    else:
        t_outlier_mask = None
    # Take care of the big outliers
    if np.any(t_outlier_mask):
        # Get the indeces 1 before the t outliers
        indeces_before_t_outliers = outlier_dict["t"]["outlier_indeces"][t_outlier_mask] - 1
        # Get the minimum V value right before all t outliers
        V_before_t_outlier = np.min(V[indeces_before_t_outliers])
        # If there is excatly one t outlier right at the end of discharging,
        # drop all values after this index and continue with processing
        if indeces_before_t_outliers.size == 1 and V_before_t_outlier < 2.01:
            i = int(indeces_before_t_outliers) + 1
            return Qd[:i], T[:i], V[:i], t[:i]
        else:
            raise OutlierException("\tDropping cycle based on outliers with np.diff(t) > {} \
                                   with value(s) {}".format(t_diff_outlier_thresh,
                                   list(outlier_dict["t"]["diff_values"][t_outlier_mask])),
                                   outlier_dict)
    else:
        return Qd, T, V, t
