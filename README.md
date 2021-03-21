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

Given a random sample <img src="https://latex.codecogs.com/svg.latex?\large&space;\left\{\mathbf{X}_1,...,\mathbf{X}_n\right\}\subset\mathbb{R}^D" />, the (Euclidean) kernel density estimator (KDE) is defined as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{p}_n(\mathbf{x})=\frac{c_{k,D}}{nh^D}\sum_{i=1}^nk\left(\left|\left|\frac{\mathbf{x}-\mathbf{X}_i}{h}\right|\right|_2^2\right)," />

where 
<img src="https://latex.codecogs.com/svg.latex?\large&space;c_{K,D}" /> is a normalizing constant, <img src="https://latex.codecogs.com/svg.latex?\large&space;h" /> is the smoothing bandwidth parameter, and <img src="https://latex.codecogs.com/svg.latex?\large&space;k:[0,\infty)\to[0,\infty)" /> is called the _profile_ of a radially symmetric kernel. Some well-known choices of the profile function are <img src="https://latex.codecogs.com/svg.latex?\large&space;k(x)=\exp(-x/2)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> for the multivariate gaussian kernel, <img src="https://latex.codecogs.com/svg.latex?\large&space;k(x)=(1-x)\cdot\mathbf{1}_{[0,1]}(x)" /> for the Epanechnikov kernel, etc. The Euclidean mean shift algorithm has the following iteration formula:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{\mathbf{x}}^{(t+1)}\gets\frac{\sum_{i=1}^n\mathbf{X}_{i}k'\left(\left|\left|\frac{\widehat{\mathbf{x}}^{(t)}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}{\sum_{i=1}^nk'\left(\left|\left|\frac{\widehat{\mathbf{x}}^{(t)}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}" />

with <img src="https://latex.codecogs.com/svg.latex?\large&space;t=0,1,..." />, and the Euclidean SCMS algorithm iterates the following equation:

<img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{\mathbf{x}}^{(t+1)}\gets\widehat{\mathbf{x}}^{(t)}+\widehat{V}_d(\widehat{\mathbf{x}}^{(t)})\widehat{V}_d(\widehat{\mathbf{x}}^{(t)})^T\left[\frac{\sum_{i=1}^n\mathbf{X}_{i}k'\left(\left|\left|\frac{\widehat{\mathbf{x}}^{(t)}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}{\sum_{i=1}^nk'\left(\left|\left|\frac{\widehat{\mathbf{x}}^{(t)}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}-\widehat{\mathbf{x}}^{(t)}\right]" />

with <img src="https://latex.codecogs.com/svg.latex?\large&space;t=0,1,..." />, where <img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{V}_d(\mathbf{x})=[\widehat{\mathbf{v}}_{d+1}(\mathbf{x}),...,\widehat{\mathbf{v}}_D(\mathbf{x})]" /> has its columns equal to the orthonormal eigenvectors of <img src="https://latex.codecogs.com/svg.latex?\large&space;\nabla\nabla\widehat{p}_n(\mathbf{x})" /> associated with its <img src="https://latex.codecogs.com/svg.latex?\large&space;D-d" /> smallest eigenvalues. Here, <img src="https://latex.codecogs.com/svg.latex?\large&space;d" /> is the intrinsic dimension of the estimated Euclidean density ridge.

### Directional Mean Shift and SCMS Algorithms

While the above Euclidean SCMS algorithm has been widely used in various fields, it exhibits some salient drawbacks in dealing with directional data <img src="https://latex.codecogs.com/svg.latex?\large&space;\left\{\mathbf{X}_1,...,\mathbf{X}_n\right\}\subset\Omega_q" />, where <img src="https://latex.codecogs.com/svg.latex?\large&space;\Omega_q=\left\{\mathbf{x}\in\mathbb{R}^{q+1}:||\mathbf{x}||_2=1\right\}\subset\mathbb{R}^{q+1}" />. See **Fig 1** below for the ridge-finding case and our paper for more details. Under the directional data scenario, the directional KDE is formulated as:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{f}_h(\mathbf{x})=\frac{c_{h,q}(L)}{n}\sum_{i=1}^nL\left(\frac{1-\mathbf{x}^T\mathbf{X}_i}{h^2}\right)," />

where <img src="https://latex.codecogs.com/svg.latex?\large&space;L" /> is a directional kernel function, <img src="https://latex.codecogs.com/svg.latex?\large&space;h" /> is the smoothing bandwidth parameter, <img src="https://latex.codecogs.com/svg.latex?\large&space;c_{h,q}\asymp\,h^{-q}" /> is a normalizing constant to ensure that <img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{f}_h" /> is a probability density function. One famous example of the directional kernel function is the so-called von Mises kernel <img src="https://latex.codecogs.com/svg.latex?\large&space;L(r)=e^{-r}" />. Then, the directional mean shift algorithm has a fixed-point iteration formula:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{\underline{\mathbf{x}}}^{(t+1)}=-\frac{\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\mathbf{X}_i^T\widehat{\underline{\mathbf{x}}}^{(t)}}{h^2}\right)}{\left|\left|\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\mathbf{X}_i^T\widehat{\underline{\mathbf{x}}}^{(t)}}{h^2}\right)\right|\right|_2}=\frac{\nabla\widehat{f}_h(\widehat{\underline{\mathbf{x}}}^{(t)})}{\left|\left|\nabla\widehat{f}_h(\widehat{\underline{\mathbf{x}}}^{(t)})\right|\right|_2}" />

with <img src="https://latex.codecogs.com/svg.latex?\large&space;t=0,1,..." />, where <img src="https://latex.codecogs.com/svg.latex?\large&space;\nabla\widehat{f}_h" /> is the total gradient of the directional KDE computed in the ambient Euclidean space  <img src="https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}^{q+1}" />; see Zhang and Chen (2020) for its detailed derivation and [https://github.com/zhangyk8/DirMS](https://github.com/zhangyk8/DirMS) for its Python3 implementation. Our proposed directional SCMS algorithm is built upon the directional mean shift formula and iterates the following procedure:

<img src="https://latex.codecogs.com/svg.latex?\large&space;\underline{\widehat{\mathbf{x}}}^{(t+1)}\gets\underline{\widehat{\mathbf{x}}}^{(t)}+\underline{\widehat{V}}_d(\underline{\widehat{\mathbf{x}}}^{(t)})\underline{\widehat{V}}_d(\underline{\widehat{\mathbf{x}}}^{(t)})^T\cdot\frac{\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\mathbf{X}_i^T\widehat{\underline{\mathbf{x}}}^{(t)}}{h^2}\right)}{\left|\left|\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\mathbf{X}_i^T\widehat{\underline{\mathbf{x}}}^{(t)}}{h^2}\right)\right|\right|_2}" />
and
<img src="https://latex.codecogs.com/svg.latex?\large&space;\underline{\widehat{\mathbf{x}}}^{(t+1)}\gets\frac{\underline{\widehat{\mathbf{x}}}^{(t+1)}}{\left|\left|\underline{\widehat{\mathbf{x}}}^{(t+1)}\right|\right|_2}" />

with <img src="https://latex.codecogs.com/svg.latex?\large&space;t=0,1,..." />, where <img src="https://latex.codecogs.com/svg.latex?\large&space;\underline{\widehat{V}}_d(\mathbf{x})=[\underline{\widehat{\mathbf{v}}}_{d+1}(\mathbf{x}),...,\underline{\widehat{\mathbf{v}}}_{q+1}(\mathbf{x})]" /> has its columns equal to the orthonormal eigenvectors of the (estimated) Riemannian Hessian <img src="https://latex.codecogs.com/svg.latex?\large&space;\mathcal{H}\widehat{f}_h(\mathbf{x})" /> associated with its <img src="https://latex.codecogs.com/svg.latex?\large&space;q-d" /> smallest eigenvalues with the tangent space <img src="https://latex.codecogs.com/svg.latex?\large&space;T_{\mathbf{x}}" />. Here, <img src="https://latex.codecogs.com/svg.latex?\large&space;d" /> is the intrinsic dimension of the estimated directional density ridge as a submanifold of the unit hypersphere <img src="https://latex.codecogs.com/svg.latex?\large&space;\Omega_q\subset\mathbb{R}^{q+1}" />.

The implementations of Euclidean and directional SCMS algorithms are encapsulated into two Python function called `SCMS_KDE` in the script **SCMS_fun.py** and `SCMS_DirKDE` in the script **DirSCMS_fun.py**, respectively. However, in our applications of these two algorithms, we use their log-density versions `SCMS_Log_KDE` and `SCMS_Log_DirKDE` in the corresponding script. The input arguments for the functions `SCMS_Log_KDE` and `SCMS_Log_DirKDE` are essentially the same; thus, we combine the descriptions of their arguments as follows:
`def SCMS_Log_KDE(x, data, h=None)`
 
`def SCMS_Log_DirKDE(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000, stop_cri='proj_grad')`
- Parameters:
    - mesh_0: a (m,D)-array
          ---- The Euclidean coordinates of m directional initial points in the D-dimensional Euclidean space.
    - data: a (n,D)-array
          ---- The Euclidean coordinates of n directional data sample points in the D-dimensional Euclidean space.
    - d: int
          ---- The order of the density ridge. (Default: d=1.)
    - h: float
          ---- The bandwidth parameter. (Default: h=None. Then a rule of thumb for directional KDEs with the von Mises kernel in Garcia-Portugues (2013) is applied.)
    - eps: float
          ---- The precision parameter. (Default: eps=1e-7.)
    - max_iter: int
          ---- The maximum number of iterations for the directional SCMS algorithm on each initial point. (Default: max_iter=1000.)
    - stop_cri: string ('proj_grad'/'pts_diff')
          ---- The indicator of which stopping criteria that will be used to terminate the SCMS algorithm. (When stop_cri='pts_diff', the errors between two consecutive iteration points need to be smaller than 'eps' for terminating the algorithm. When stop_cri='proj_grad' or others, the projected/principal gradient of the current point need to be smaller than 'eps' for terminating the algorithm.) (Default: stop_cri='proj_grad'.)
    
- Return:
    - SCMS_path: (m,D,T)-array
          ---- The entire iterative SCMS sequence for each initial point.


### Additional Reference
- U. Ozertem and D. Erdogmus (2011). Locally Defined Principal Curves and Surfaces. _Journal of Machine Learning Research_ **12** 1249-1286.
- Y. Zhang and Y.-C. Chen (2020). Kernel Smoothing, Mean Shift, and Their Learning Theory with Directional Data. _arXiv preprint arXiv:2010.13523_.
