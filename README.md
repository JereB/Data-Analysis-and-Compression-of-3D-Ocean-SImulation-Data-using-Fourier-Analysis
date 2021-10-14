## Data Analysis on 3D Ocean Simulation Data using Fourier Analysis

This repository belongs the the bachelor thesis with the same name written by Jeremia Böhmig.
The starting parameter for the examined data set as well as the as initial `petsc`-file to describe the model of in the beginning of the year are saved in the folder `initial data`.

The `metos3d` folder contains the `petsc_mod.py` file which was taken from the original Metos3d  repository and contains a function to read `petsc`-files as well as a function to convert the Metos3D data into its 3 dimensional representation. It also contains the `landSeaMask.petsc`-file which describes how the simulation data is structured and is necessary for the conversion to 3D.

The folder `exploration` contains the raw jupyter-notebooks that were created in the exploration phase of this thesis. They are less organized and not suitable for application of other data sets without some tweaking.

The script `processing_stage.py` contains functions to apply FFT on the simulation data. It can also act as an executable and then reads all `petsc`-files in a specified path that match a regex pattern and saves the original data, the FFT coefficients as well as their real counterparts in `.npy`-files that can easily be read with `NumPy`.

The file `plotting` contains a function to generate a series of plots.
This is useful, when trying to express the 

### Requirements

To execute the scripts in this folder `NumPy` and `Matplotlib` must be installed.

Additionally to use the Jupyter-notebooks(`ìpynb`-file extension) one must have juypter installed.



