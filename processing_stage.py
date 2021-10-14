from os import path
import sys
import os
import re
from typing import Tuple
import numpy as np
from numpy.lib.shape_base import _put_along_axis_dispatcher
import metos3d.petsc_mod as petsc


def read_all_monitor_data(dir_name: str, file_pattern: str) -> np.array:
    """Read all petsc files in `dir_name` which match the pattern `file_pattern`

    The names of the files will be alphabetically ordered.
    """
    pattern = re.compile(file_pattern)

    l = list()
    # sort all names, to ensure the right order
    names = sorted(list(os.listdir(dir_name)))

    for name in names:
        if pattern.fullmatch(name):
            l.append(petsc.read_PETSc_vec(path.join(dir_name, name)))

    return np.array(l)


def apply_fft(data: np.array) -> np.array:
    """Apply fft to each row of the data array

    It will be assumed, that `data` is a two dimensional array whereby the rows are the series of values for each measurepoint.
    This means each row represents one measurepoint in the simulation.
    The return value is a 2D array with the first dimension beeing the measurement points and the second the fft-coefficients for each point.
    """

    return np.apply_along_axis(np.fft.fft, 1, data)


def to_real_coef(data: np.array) -> Tuple[np.array, np.array]:
    """Extract the real coefficients for cos and sine.

    To properly funtion the input data must be fft-result where the examined data was consisting of real numbers.
    Also the data must be two dimensional, meaning multiple fft-result are examined at once.
    The first array return value are the coefficients for cos, the second those for sin.
    """

    # normalize the input data
    normalized = data / len(data[0])

    def cos_coef(row: np.array) -> np.array:
        return np.array([row[0].real] + [(row[i] + row[-i]).real for i in range(1, len(row)//2)])

    def sin_coef(row):
        return np.array([0] + [(1j * (row[i] - row[-i])).real for i in range(1, len(row)//2)])

    a_k = np.apply_along_axis(cos_coef, 1, normalized)
    b_k = np.apply_along_axis(sin_coef, 1, normalized)

    return a_k, b_k


usage = """Script to save a_k and b_k for each  measuremenpoint when given a year worth of monitor data.
USAGE:
    DATA_DIR FILE_PATTERN SAFE_DIR

    DATA_DIR: path to the directory where all monitoring data is placed

    FILE_PATTERN: regex pattern that matches the monitor data. The files must be alphabetically ordered

    SAVE_DIR: The path where the resulted data will be stored. Data will be stored in numpy format and can be read with `numpy.load`. The files will be:

        data.npy : The matrix of measurements where each row represents the values for one measurepoint

        a_k.npy: matrix of cos-cooefficients. Each row are the coefficients for one measurepoint

        b_k.npy: matrix of cos-cooefficients. Each row are the coefficients for one measurepoint
"""




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(usage)

    _, data_dir, pattern, save_dir = sys.argv

    data = np.transpose(read_all_monitor_data(data_dir, pattern))

    print(f"{len(data[data < 0])} negative data set to zero")
    data[data < 0] = 0

    fft_data = apply_fft(data)

    a_k, b_k = to_real_coef(fft_data)

    np.save(path.join(save_dir, "c.npy"), fft_data)
    np.save(path.join(save_dir, "data.npy"), data)
    np.save(path.join(save_dir, "a_k.npy"), a_k)
    np.save(path.join(save_dir, "b_k.npy"), b_k)
