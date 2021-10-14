import matplotlib.pyplot as plt
from os.path import join
import metos3d.petsc_mod as petsc
import numpy as np


def plot_series(data, steps, save_path, land_sea_mask, layer, mean_values=None):
    stepsize = len(data[0]) // steps
    n1, n2 = np.shape(land_sea_mask.T)
    long, lat = np.meshgrid(np.linspace(-90, 90, n2), np.linspace(0, 360, n1))


    for step in range(steps):

        data = land_sea_mask, data[:, step * stepsize]

        if not data is None:
            data = data / mean_values

        fig, ax = plt.subplots(figsize=(15, 7))
        data_3d, _ , _, _ = petsc.reshape_vector_to_3d(data)

        plot = ax.contourf(lat, long, data_3d[:, :, layer])
        fig.colorbar(plot, ax=ax)
        ax.set_ylabel('latitude')
        ax.set_xlabel('longitude')

        fig.savefig(join(save_path, f"{step}.png"))
        plt.close(fig)
