## Euclidean and Directional Subspace Constrained Mean Shift (SCMS) Algorithms
This repository implements both the classical SCMS algorithm (Ozertem and Erdogmus, 2011) with Euclidean data and our proposed SCMS algorithm under the directional data setting via Python3.

- Paper Reference: 

### Requirements

- Python >= 3.6 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/) (especially the [Basemap](https://matplotlib.org/basemap/) toolkit), [pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/) (A speical function [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) is used to compute the modified Bessel function of the first kind of real order.

### Descriptions

Some high-level descriptions of our Python scripts are as follows:

- **DirSCMS_fun.py**: This script implements the functions of directional KDE and subspace constrained mean shift (SCMS) algorithm with the von Mises kernel.
- **Drawback_Eu.py**: This script contains code for comparing Euclidean KDE with directional KDE as well as comparing Euclidean subspace constrained mean shift (SCMS) with our proposed directional SCMS algorithm on simulated datasets in order to illustrate the drawbacks of Euclidean KDE and SCMS algorithm in handling directional data (Figure B.2 in the paper).
- **Earthquake_Ridges.py**: This script contains code for our applications of Euclidean and directional SCMS algorithms to the earthquake data (Figures C.3 in the paper).
- **Eu_Dir_Ridges.py**: This script contains code for applying Euclidean and directional subspace constrained mean shift (SCMS) algorithm to simulated datasets (Figure 1.1 in the paper).
- **LC_plots.py**: This script contains code for empirically verifying the linear convergence of Euclidean and directional SCMS algorithms (Figures C.1 and C.2 in the paper).
- **SCMS_fun.py**: This script contains code for Euclidean KDE and subspace constrained mean shift (SCMS) algorithm with Gaussian kernel.
- **Utility_fun.py**: This script contains all the utility functions for our experiments.

### Euclidean Mean Shift and SCMS Algorithms

Given a random sample <img src="https://latex.codecogs.com/svg.latex?\Large&space;\left\{\mathbf{X}_1,...,\mathbf{X}_n\right\}\in\mathbb{R}^D" />, the (Euclidean) kernel density estimator (KDE) is defined as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{p}_n(\mathbf{x})=\frac{c_{k,D}}{nh^D}\sum_{i=1}^nk\left(\left|\left|\frac{\mathbf{x}-\mathbf{X}_i}{h}\right|\right|_2^2\right)," />

where 
<img src="https://latex.codecogs.com/svg.latex?\large&space;c_{K,D}" /> is a normalizing constant, <img src="https://latex.codecogs.com/svg.latex?\large&space;h" /> is the smoothing bandwidth parameter, and <img src="https://latex.codecogs.com/svg.latex?\large&space;k:[0,\infty)\to[0,\infty)" /> is called the _profile_ of a radially symmetric kernel. Some well-known choices of the profile function are <img src="https://latex.codecogs.com/svg.latex?\large&space;k(x)=\exp(-x/2)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> for the multivariate gaussian kernel, <img src="https://latex.codecogs.com/svg.latex?\large&space;k(x)=(1-x)\cdot\mathbf{1}_{[0,1]}(x)" /> for the Epanechnikov kernel, etc. The Euclidean mean shift algorithm has the following iteration formula:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{\mathbf{x}}^{(t+1)}\gets\frac{\sum_{i=1}^n\mathbf{X}_{i}k'\left(\left|\left|\frac{\widehat{\mathbf{x}}^{(t)}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}{\sum_{i=1}^nk'\left(\left|\left|\frac{\widehat{\mathbf{x}}^{(t)}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}" />

with <img src="https://latex.codecogs.com/svg.latex?\large&space;t=0,1,..." />, and the Euclidean SCMS algorithm iterates the following equation:

<img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{\mathbf{x}}^{(t+1)}\gets\widehat{\mathbf{x}}^{(t)}+\widehat{V}_d(\widehat{\mathbf{x}}^{(t)})\widehat{V}_d(\widehat{\mathbf{x}}^{(t)})^T\left[\frac{\sum_{i=1}^n\mathbf{X}_{i}k'\left(\left|\left|\frac{\widehat{\mathbf{x}}^{(t)}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}{\sum_{i=1}^nk'\left(\left|\left|\frac{\widehat{\mathbf{x}}^{(t)}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}-\widehat{\mathbf{x}}^{(t)}\right]" />

with <img src="https://latex.codecogs.com/svg.latex?\large&space;t=0,1,..." />, where <img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{V}_d(\mathbf{x})=[\widehat{\mathbf{v}}_{d+1}(\mathbf{x}),...,\widehat{\mathbf{v}}_D(\mathbf{x})]" /> has its columns equal to the orthonormal eigenvectors of <img src="https://latex.codecogs.com/svg.latex?\large&space;\nabla\nabla\hat{p}_n(\mathbf{x})" /> associated with its <img src="https://latex.codecogs.com/svg.latex?\large&space;D-d" /> smallest eigenvalues. Here, <img src="https://latex.codecogs.com/svg.latex?\large&space;d" /> is the intrinsic dimension of the estimated Euclidean density ridge.

### Additional Reference
- U. Ozertem and D. Erdogmus (2011). Locally Defined Principal Curves and Surfaces. _Journal of Machine Learning Research_ **12** 1249-1286.
