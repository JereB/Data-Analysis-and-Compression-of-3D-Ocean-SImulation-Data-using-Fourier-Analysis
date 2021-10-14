import numpy as np


def compress_raw_data(data, delta, temporal_changes=True):
    """Accept a numpy array where each row represents one simulation point.
    Delta is the desired accuracy und must be between 0 and 1.
    Temporal changes indicates if the static baseline should be considered when determining delta.
    Return a tuple of
    1. the list of indices
    2. the values of the selected coefficients
    """
    X = np.fft.fft(data)

    return compress_FFT_data(X,delta, ignore_c0=temporal_changes)


def compress_FFT_data(X,delta, ignore_c0=True):
    """Accept a numpy array where each represents FFT coefficients of a single simulation point.
    Delta is the desired accuracy und must be between 0 and 1.
    Temporal changes indicates if the static baseline should be considered when determining delta.
    Return a tuple of
    1. the list of indices
    2. the values of the selected coefficients
    """

    if delta <= 0 or delta >= 1:
        raise Exception("The value for delta must be smaller than 1 and bigger than 0")

    index_values = create_index_list(X, ignore_c0=ignore_c0)

    # part of the original data that is reached with the selected index set
    reached_delta  = 0
    # if c0 should be ignored in the accuracy, it must be added first.
    index_list = [0] if ignore_c0 else []

    for (idx, idx_delta) in index_values:
        index_list.append(idx)
        reached_delta += idx_delta
        if reached_delta > delta:
            break

    # shape of the resulting matrix
    # one row for every point and one column for each coefficient used in the compression
    result_shape = (len(X), len(index_list))

    comp_data = np.zeros(result_shape, dtype="complex128")

    for i,idx in enumerate(index_list):
        # set column i to the values of colum idx of the original coefficients
        comp_data[:,i] = X[:,idx]

    return index_list, comp_data



def create_index_list(X, ignore_c0=True):
    """Take FFT coefficient and return a list of tuples. First part of the tuple is the index,
    second part is their part on the overall changes.
    The tuples are sorted according to their second part
    If ignore_c0 is set to True the zero indexed coefficients will be ignored.
    """

    # gather the sum of the squared absolute for each index
    index_sum = [0] * len(X[0])

    for sim_point in X:
        for idx, coef in enumerate(sim_point):
            if ignore_c0 and idx == 0:
                continue
            else:
                # squared absolute value for coefficients of each index
                index_sum[idx] +=  abs(coef) ** 2

    # the sum of all index_sums
    # this represents all information stored in the system
    all_information = sum(index_sum)

    # normalize the values for index sums, so that they show the fraction of all information that thex belong to
    index_ratio = map(lambda x: x / all_information, index_sum)

    # combine each ratio with the index (done with enumerate)
    # sort the tuples according to their second part
    return sorted(enumerate(index_ratio), key=lambda x: x[1], reverse=True)


def decompress_data(comp_data, index_list, data_steps):
    """Take the compressed matrix, the list of indices used for compression and the count of data steps in the original data.
    The list of indices must be in the same order they are stored in comp_data"""

    result_shape = (len(comp_data), data_steps)
    fft_coefs = np.zeros(result_shape, dtype="complex128")
    for i, idx in enumerate(index_list):
        # set values for the stored indicex
        fft_coefs[:, idx] = comp_data[:,i]

    return np.real(np.fft.ifft(fft_coefs))
