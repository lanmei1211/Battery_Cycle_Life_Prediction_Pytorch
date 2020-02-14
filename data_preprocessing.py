import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from data_preprocessing.pkl_to_dict import pkl_to_dict
from data_preprocessing.outliers_drop import drop_cycle_big_t_outliers, outlier_dict_without_mask
from data_preprocessing.outliers_drop import handle_small_Qd_outliers, drop_outliers_starting_left
from data_preprocessing.outliers_drop import DropCycleException, OutlierException

BREAK_IDX = 100


def filter_by_mask(index_mask, *arrays, drop_warning=False, drop_warning_thresh=0.10):
    """Access multiple elements in arrays by a boolean mask

    Arguments:
        index_mask {1D boolaean array}
            The indexes of the values to extract from the arrays
        arrays {A variable number of 1D arrays}
            The arrays to extract the values from
    Returns:
        tuple -- reindexed numpy arrays from *args in the same order.
    """
    indexed_arrays = [array[index_mask].copy() for array in arrays]
    return tuple(indexed_arrays)


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


def preprocess_cycle(cycle,
                     I_thresh=-3.99,
                     Vdlin_start=3.5,
                     Vdlin_stop=2.0,
                     nr_resample_values=1000,
                     return_original_data=False):
    """
    Processes data (Qd, T, V, t) from one cycle and resamples Qd, T and V to a predefinded dimension.
    discharging_time will be computed based on t and is the only returned feature that is a scalar.

    Arguments:
        cycle {dict} -- One cycle entry from the original data with keys 'I', 'Qd', 'T', 'V', 't'

    Keyword Arguments:
        I_thresh {float} -- Only measurements where the current is smaller than this threshold are chosen
            (default: {-3.99})
        Vdlin_start {float} -- Start value for the resampled V (default: {3.5})
        Vdlin_stop {float} -- Stop value for the resampled V (default: {2.0})
        Vdlin_steps {int} -- Number of steps V, Qd and T are resampled (default: {1000})
        return_original_data {bool} -- Weather the original datapoints, which were used for interpolation,
            shold be returned in the results  (default: {False})

    Returns:
        {dict} -- Dictionary with the resampled (and original) values
    """
    Qd = cycle["Qd"]
    T = cycle["T"]
    V = cycle["V"]
    I = cycle["I"]  # noqa: E741
    t = cycle["t"]

    # Only take the measurements during high current discharging.
    discharge_mask = I < I_thresh
    Qd, T, V, t = filter_by_mask(discharge_mask, Qd, T, V, t)

    # Sort all values after time.
    sorted_indeces = t.argsort()
    Qd, T, V, t = filter_by_mask(sorted_indeces, Qd, T, V, t)

    # Only take timesteps where time is strictly increasing.
    increasing_time_mask = np.diff(t, prepend=0) > 0
    Qd, T, V, t = filter_by_mask(increasing_time_mask, Qd, T, V, t)

    # Drop t outliers
    Qd, T, V, t = drop_cycle_big_t_outliers(Qd, T, V, t, verbose=False, std_threshold=15)

    # Handle Qd outliers
    Qd = handle_small_Qd_outliers(Qd, t, verbose=False, std_threshold=12)

    # Drop Qd outliers
    Qd, T, V, t = drop_outliers_starting_left(Qd, T, V, t, std_threshold=12)

    # Apply savitzky golay filter to V to smooth out the values.
    # This is done in order to not drop too many values in the next processing step (make monotonically decreasing).
    # This way the resulting curves don't become skewed too much in the direction of smaller values.
    savgol_window_length = 25
    if savgol_window_length >= V.size:
        raise DropCycleException("""Dropping cycle with less than {} V values.\nSizes --> Qd:{}, T:{}, V:{}, t:{}"""
                                 .format(savgol_window_length, Qd.size, T.size, V.size, t.size))
    V_savgol = savgol_filter(V, window_length=25, polyorder=2)

    # Only take the measurements, where V is monotonically decreasing (needed for interpolation).
    # This is done by comparing V to the accumulated minimum of V.
    #    accumulated minimum --> (taking always the smallest seen value from V from left to right)
    v_decreasing_mask = V_savgol == np.minimum.accumulate(V_savgol)
    Qd, T, V, t = filter_by_mask(v_decreasing_mask, Qd, T, V_savgol, t, drop_warning=True)

    # Make V_3 strictly decreasing (needed for interpolation).
    V_strict_dec = make_strictly_decreasing(t, V)

    # Calculate discharging time. (Only scalar feature which is returned later)
    discharging_time = t.max() - t.min()
    if discharging_time < 6:
        print("Test")
        raise DropCycleException("Dropping cycle with discharge_time = {}"
                                 .format(discharging_time))

    # Make interpolation function.
    Qd_interp_func = interp1d(
        V_strict_dec[::-1],  # V_strict_dec is inverted because it has to be increasing for interpolation.
        Qd[::-1],  # Qd and T are also inverted, so the correct values line up.
        bounds_error=False,  # Allows the function to be evaluated outside of the range of V_strict_dec.
        fill_value=(Qd[::-1][0], Qd[::-1][-1])  # Values to use, when evaluated outside of V_strict_dec.
    )
    T_interp_func = interp1d(
        V_strict_dec[::-1],
        T[::-1],
        bounds_error=False,
        fill_value=(T[::-1][0], T[::-1][-1])
    )

    # For resampling the decreasing order is chosen again.
    # The order doesn't matter for evaluating Qd_interp_func.
    Vdlin = np.linspace(Vdlin_start, Vdlin_stop, nr_resample_values)

    Qdlin = Qd_interp_func(Vdlin)
    Tdlin = T_interp_func(Vdlin)

    if return_original_data:
        return {
            cst.QDLIN_NAME: Qdlin,
            cst.TDLIN_NAME: Tdlin,
            cst.VDLIN_NAME: Vdlin,
            cst.DISCHARGE_TIME_NAME: discharging_time,
            # Original data used for interpolation.
            "Qd_original_data": Qd,
            "T_original_data": T,
            "V_original_data": V,
            "t_original_data": t
        }
    else:
        return {
            cst.QDLIN_NAME: Qdlin,
            cst.TDLIN_NAME: Tdlin,
            cst.VDLIN_NAME: Vdlin,
            cst.DISCHARGE_TIME_NAME: discharging_time
        }


def preprocess_batch(batch_dict):
    """
    Processes all cycles of every cell in batch_dict and returns the results in the same format.

    Arguments:
        batch_dict {dict} -- Unprocessed batch of cell data.

    Keyword Arguments:
        return_original_data {bool} -- If True, the original data used for interpolation is returned. (default: {False})
        verbose {bool} -- If True prints progress for every cell (default: {False})

    Returns:
        dict -- Results in the same format as batch_dict.
    """
    batch_results = dict()
    cycles_drop_info = dict()

    # for each battery
    for i, cell_key in enumerate(list(batch_dict.keys())):
        if i == BREAK_IDX:
            exit()
        cell = batch_dict[cell_key]
        # print(cell_key, ':', cell["cycle_life"][0][0])
        batch_results[cell_key] = dict(
            # The amount of cycles until 80% of nominal capacity is reached
            cycle_life=cell["cycle_life"][0][0],
            summary={
                'IR': [],
                'QD': [],
                'Remaining_cycles': [],
                'Discharge_time': []
            },
            cycles=dict()
        )
        # go through each cycle
        for cycle_key, cycle in cell["cycles"].items():
            print(cycle_key, ':', cell["cycle_life"][0][0])
            # Cycles zero needs to be droped as they just contain 2 measurements most of the time
            if cycle_key == '0':
                continue
            # Some cells have more cycle measurements than recorded cycle_life
            # The reamining cycles will be dropped
            elif int(cycle_key) > int(cell["cycle_life"][0][0]):
                print("    Cell {} has more cycles than cycle_life ({}): Dropping remaining cycles {} to {}"
                      .format(cell_key,
                              cell["cycle_life"][0][0],
                              cycle_key,
                              max([int(k) for k in cell["cycles"].keys()])))
                break

            # Start processing the cycle
            try:
                cycle_results = preprocess_cycle(cycle)

            except DropCycleException as e:
                print("cell:", cell_key, " cycle:", cycle_key)
                print(e)
                print("")
                # Documenting dropped cell and key
                drop_info = {cell_key: {cycle_key: None}}
                cycles_drop_info.update(drop_info)        
                continue

            except OutlierException as oe:  # Can be raised if preprocess_cycle, if an outlier is found.
                print("cell:", cell_key, " cycle:", cycle_key)
                print(oe)
                print("")
                # Adding outlier dict from Exception to the cycles_drop_info.
                drop_info = {
                    cell_key: {
                        cycle_key: outlier_dict_without_mask(oe.outlier_dict)}}
                cycles_drop_info.update(drop_info)
                continue


def main():
    print('\n\n\tStarted to processing the data...\n')
    batch_dict = pkl_to_dict()

    preprocess_batch(batch_dict)

    #print(batch_dict.keys())

    print("Done!")


if __name__ == "__main__":
    main()
