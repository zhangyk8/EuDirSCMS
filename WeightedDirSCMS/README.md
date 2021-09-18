## Weighted Euclidean and Directional Subspace Constrained Mean Shift (SCMS) Algorithm

This repository contains code to implement the (weighted) Euclidean and directional SCMS algorithm on a sample earthquake dataset. More example codes and details can be found in the Jupyter Notebook [**Weight SCMS on Earthquake Data (An example).ipynb**](https://github.com/zhangyk8/EuDirSCMS/blob/main/WeightedDirSCMS/Weight%20SCMS%20on%20Earthquake%20Data%20(An%20example).ipynb).

### Requirements

- Python >= 3.6 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/) (especially the [Basemap](https://matplotlib.org/basemap/) toolkit), [pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/) (A speical function [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) is used to compute the modified Bessel function of the first kind of real order.
- (Optional) [Ray](https://ray.io/) (Fast and simple distributed computing API for Python and Java)
- We provide an [guideline](https://github.com/zhangyk8/DirMS/blob/main/Install_Basemap_Ubuntu.md) of installing the [Basemap](https://matplotlib.org/basemap/) toolkit on Ubuntu.

### Description
Some high-level descriptions of each python script are as follows:

- **DirSCMS_fun.py**: This script implements the functions of (weighted) directional KDE and subspace constrained mean shift (SCMS) algorithm with the von Mises kernel.
- **MS_SCMS_Ray.py**: This script contains code for the parallel implementations of (weighted) Euclidean/directional mean shift and SCMS algorithms.
- **SCMS_fun.py**: This script contains code for (weighted) Euclidean KDE and subspace constrained mean shift (SCMS) algorithm with Gaussian kernel.
- **Utility_fun.py**: This script contains all the utility functions for our experiments.
