# Euclidean and Directional Subspace Constrained Mean Shift (SCMS) Algorithms
This repository implements both the classical SCMS algorithm (Ozertem and Erdogmus, 2011) with Euclidean data and our proposed SCMS algorithm under the directional data setting via Python3.

- Paper Reference: 

## Requirements

- Python >= 3.6 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/) (especially the [Basemap](https://matplotlib.org/basemap/) toolkit), [pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/) (A speical function [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) is used to compute the modified Bessel function of the first kind of real order.

## Descriptions

Some high-level descriptions of our Python scripts are as follows:

- **DirSCMS_fun.py**: This script implements the functions of directional KDE and subspace constrained mean shift (SCMS) algorithm with the von Mises kernel.
- **Drawback_Eu.py**: This script contains code for comparing Euclidean KDE with directional KDE as well as comparing Euclidean subspace constrained mean shift (SCMS) with our proposed directional SCMS algorithm on simulated datasets in order to illustrate the drawbacks of Euclidean KDE and SCMS algorithm in handling directional data (Figure B.2 in the paper).
- **Earthquake_Ridges.py**: This script contains code for our applications of Euclidean and directional SCMS algorithms to the earthquake data (Figures C.3 in the paper).
- **Eu_Dir_Ridges.py**: This script contains code for applying Euclidean and directional subspace constrained mean shift (SCMS) algorithm to simulated datasets (Figure 1.1 in the paper).
- **LC_plots.py**: This script contains code for empirically verifying the linear convergence of Euclidean and directional SCMS algorithms (Figures C.1 and C.2 in the paper).
- **SCMS_fun.py**: This script contains code for Euclidean KDE and subspace constrained mean shift (SCMS) algorithm with Gaussian kernel.
- **Utility_fun.py**: This script contains all the utility functions for our experiments.



### Additional Reference
- U. Ozertem and D. Erdogmus (2011). Locally Defined Principal Curves and Surfaces. _Journal of Machine Learning Research_ **12** 1249-1286.
