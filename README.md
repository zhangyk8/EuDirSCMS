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

While the above Euclidean SCMS algorithm has been widely used in various fields, it exhibits some salient drawbacks in dealing with directional data <img src="https://latex.codecogs.com/svg.latex?\large&space;\left\{\mathbf{X}_1,...,\mathbf{X}_n\right\}\subset\Omega_q" />, where <img src="https://latex.codecogs.com/svg.latex?\large&space;\Omega_q=\left\{\mathbf{x}\in\mathbb{R}^{q+1}:||\mathbf{x}||_2=1\right\}\subset\mathbb{R}^{q+1}" />; see **Fig 1** below for the ridge-finding case and our paper for more details. Under the directional data scenario, the directional KDE is formulated as:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{f}_h(\mathbf{x})=\frac{c_{h,q}(L)}{n}\sum_{i=1}^nL\left(\frac{1-\mathbf{x}^T\mathbf{X}_i}{h^2}\right)," />

where <img src="https://latex.codecogs.com/svg.latex?\large&space;L" /> is a directional kernel function, <img src="https://latex.codecogs.com/svg.latex?\large&space;h" /> is the smoothing bandwidth parameter, <img src="https://latex.codecogs.com/svg.latex?\large&space;c_{h,q}\asymp\,h^{-q}" /> is a normalizing constant to ensure that <img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{f}_h" /> is a probability density function. One famous example of the directional kernel function is the so-called von Mises kernel <img src="https://latex.codecogs.com/svg.latex?\large&space;L(r)=e^{-r}" />. Then, the directional mean shift algorithm has a fixed-point iteration formula:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{\underline{\mathbf{x}}}^{(t+1)}=-\frac{\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\mathbf{X}_i^T\widehat{\underline{\mathbf{x}}}^{(t)}}{h^2}\right)}{\left|\left|\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\mathbf{X}_i^T\widehat{\underline{\mathbf{x}}}^{(t)}}{h^2}\right)\right|\right|_2}=\frac{\nabla\widehat{f}_h(\widehat{\underline{\mathbf{x}}}^{(t)})}{\left|\left|\nabla\widehat{f}_h(\widehat{\underline{\mathbf{x}}}^{(t)})\right|\right|_2}" />

with <img src="https://latex.codecogs.com/svg.latex?\large&space;t=0,1,..." />, where <img src="https://latex.codecogs.com/svg.latex?\large&space;\nabla\widehat{f}_h" /> is the total gradient of the directional KDE computed in the ambient Euclidean space  <img src="https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}^{q+1}" />; see Zhang and Chen (2020) for its detailed derivation and [https://github.com/zhangyk8/DirMS](https://github.com/zhangyk8/DirMS) for its Python3 implementation. Our proposed directional SCMS algorithm is built upon the directional mean shift formula and iterates the following procedure:

<img src="https://latex.codecogs.com/svg.latex?\large&space;\underline{\widehat{\mathbf{x}}}^{(t+1)}\gets\underline{\widehat{\mathbf{x}}}^{(t)}+\underline{\widehat{V}}_d(\underline{\widehat{\mathbf{x}}}^{(t)})\underline{\widehat{V}}_d(\underline{\widehat{\mathbf{x}}}^{(t)})^T\cdot\frac{\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\mathbf{X}_i^T\widehat{\underline{\mathbf{x}}}^{(t)}}{h^2}\right)}{\left|\left|\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\mathbf{X}_i^T\widehat{\underline{\mathbf{x}}}^{(t)}}{h^2}\right)\right|\right|_2}\quad\quad\text{and}\\\quad\underline{\widehat{\mathbf{x}}}^{(t+1)}\gets\frac{\underline{\widehat{\mathbf{x}}}^{(t+1)}}{\left|\left|\underline{\widehat{\mathbf{x}}}^{(t+1)}\right|\right|_2}" />

with <img src="https://latex.codecogs.com/svg.latex?\large&space;t=0,1,..." />, where <img src="https://latex.codecogs.com/svg.latex?\large&space;\underline{\widehat{V}}_d(\mathbf{x})=[\underline{\widehat{\mathbf{v}}}_{d+1}(\mathbf{x}),...,\underline{\widehat{\mathbf{v}}}_{q+1}(\mathbf{x})]" /> has its columns equal to the orthonormal eigenvectors of the (estimated) Riemannian Hessian <img src="https://latex.codecogs.com/svg.latex?\large&space;\mathcal{H}\widehat{f}_h(\mathbf{x})" /> associated with its <img src="https://latex.codecogs.com/svg.latex?\large&space;q-d" /> smallest eigenvalues with the tangent space <img src="https://latex.codecogs.com/svg.latex?\large&space;T_{\mathbf{x}}" />. Here, <img src="https://latex.codecogs.com/svg.latex?\large&space;d" /> is the intrinsic dimension of the estimated directional density ridge as a submanifold of the unit hypersphere <img src="https://latex.codecogs.com/svg.latex?\large&space;\Omega_q\subset\mathbb{R}^{q+1}" />.

The implementations of Euclidean and directional SCMS algorithms are encapsulated into two Python function called `SCMS_KDE` in the script **SCMS_fun.py** and `SCMS_DirKDE` in the script **DirSCMS_fun.py**, respectively. However, in our applications of these two algorithms, we use their log-density versions `SCMS_Log_KDE` and `SCMS_Log_DirKDE` in the corresponding scripts. The input arguments for the functions `SCMS_Log_KDE` and `SCMS_Log_DirKDE` are essentially the same; thus, we combine the descriptions of their arguments as follows:
`def SCMS_Log_KDE(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000, stop_cri='proj_grad')`
 
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
          ---- The indicator of which stopping criteria that will be used to terminate the SCMS algorithm. (When stop_cri='pts_diff', the errors between two consecutive iteration points need to be smaller than 'eps' for terminating the algorithm. When stop_cri='proj_grad' or others, the projected/principal (Riemannian) gradient of the current point need to be smaller than 'eps' for terminating the algorithm.) (Default: stop_cri='proj_grad'.)
    
- Return:
    - SCMS_path: (m,D,T)-array
          ---- The entire iterative SCMS sequence for each initial point.

Example code:
```bash
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from Utility_fun import cart2sph, Cir_Sph_samp
from SCMS_fun import KDE, SCMS_Log_KDE
from DirSCMS_fun import DirKDE, SCMS_Log_DirKDE

np.random.seed(111)  ## Set an arbitrary seed for reproducibility
## Sampling the points on a circle that crosses through the north and south poles
cir_samp = Cir_Sph_samp(1000, lat_c=0, sigma=0.2, pv_ax=np.array([1,0,0]))
lon_c, lat_c, r = cart2sph(*cir_samp.T)
cir_samp_ang = np.concatenate((lon_c.reshape(len(lon_c),1), 
                               lat_c.reshape(len(lat_c),1)), axis=1)
    
## Denoising step
bw_Dir = None
d_hat2_Dir = DirKDE(cir_samp, cir_samp, h=bw_Dir)
tau = 0.1
print('Removing the data points whose directional KDE values are below '
      +str(tau)+' of the maximum density.')
cir_samp_thres = cir_samp[d_hat2_Dir >= tau*max(d_hat2_Dir),:]
print('Ratio of the numbers of data points after and before the denoising '\
      'step: ' + str(cir_samp_thres.shape[0]/cir_samp.shape[0]) + '.\n')
bw_Eu = None
d_hat2_Eu = KDE(cir_samp_ang, cir_samp_ang, h=bw_Eu)
tau = 0.1
print('Removing the data points whose Euclidean KDE values are below '\
      +str(tau)+' of the maximum density.')
cir_samp_ang_thres = cir_samp_ang[d_hat2_Eu >= tau*max(d_hat2_Eu),:]
print('Ratio of the numbers of data points after and before the denoising '\
      'step: ' + str(cir_samp_ang_thres.shape[0]/cir_samp_ang.shape[0]) + '.\n')

## Apply the directional and Euclidean SCMS algorithms
SCMS_Dir_log2 = SCMS_Log_DirKDE(cir_samp_thres, cir_samp, d=1, h=bw_Dir, 
                                eps=1e-7, max_iter=5000)
Dir_ridge_log2 = SCMS_Dir_log2[:,:,SCMS_Dir_log2.shape[2]-1]
    
SCMS_Eu_log2 = SCMS_Log_KDE(cir_samp_ang_thres, cir_samp_ang, d=1, h=bw_Eu, 
                            eps=1e-7, max_iter=5000)
Eu_ridge_log2 = SCMS_Eu_log2[:,:,SCMS_Eu_log2.shape[2]-1]
    
fig = plt.figure(figsize=(14,8))
lon_t = np.concatenate([90*np.ones(50,), -90*np.ones(50,)])
lat_t = np.concatenate([np.linspace(-90, 90, 50), np.linspace(90, -90, 50)])
lon_c, lat_c, r = cart2sph(*cir_samp.T)
lon_r_Eu = Eu_ridge_log2[:,0]
lat_r_Eu = Eu_ridge_log2[:,1]
lon_r_Dir, lat_r_Dir, r = cart2sph(*Dir_ridge_log2.T)
# Set up map projection
m1 = Basemap(projection='hammer', llcrnrlon=-180, urcrnrlon=180,
             llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0)
# Draw lat/lon grid lines every 30 degrees.
m1.drawmeridians(np.arange(-180, 180, 30))
m1.drawparallels(np.arange(-90, 90, 30))
# Compute native map projection coordinates of lat/lon grid.
x, y = m1(lon_c, lat_c)
x_t, y_t = m1(lon_t, lat_t)
x_Eu, y_Eu = m1(lon_r_Eu, lat_r_Eu)
x_Dir, y_Dir = m1(lon_r_Dir, lat_r_Dir)
# Scatter plots over the map.
cs = m1.scatter(x, y, facecolors='none', edgecolors='black', s=20)
cs = m1.plot(x_t, y_t, color='blue', linewidth=4, alpha=0.5)
cs = m1.scatter(x_Eu, y_Eu, color='darkgreen', s=25, alpha=1)
cs = m1.scatter(x_Dir, y_Dir, color='red', s=35, alpha=0.7)
fig.savefig('./Figures/Output.png')
```

<p align="center">
<img src="https://github.com/zhangyk8/EuDirSCMS/blob/main/Figures/Output.png" style="zoom:60%" />
 <br><B>Fig 1. </B>An illustration of Euclidean and directional SCMS algorithms applied to a simulated dataset with an underlying circular structure on the sphere. (Here, The red points represent the estimated directional ridge identified by our directional SCMS algorithm. The green points indicate the estimated ridge obtained by the Euclidean SCMS algorithm. And the blue curve exhibits the true circular structure.)
 </p>

### Additional Reference
- U. Ozertem and D. Erdogmus (2011). Locally Defined Principal Curves and Surfaces. _Journal of Machine Learning Research_ **12** 1249-1286.
- Y. Zhang and Y.-C. Chen (2020). Kernel Smoothing, Mean Shift, and Their Learning Theory with Directional Data. _arXiv preprint arXiv:2010.13523_.
- E. Garcı́a-Portugués (2013). Exact risk improvement of bandwidth selectors for kernel density estimation with directional data. _Electronic Journal of Statistics_ **7** 1655–1685.
