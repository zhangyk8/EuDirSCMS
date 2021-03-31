#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: March 31, 2021

Description: This script contains code for empirically verifying the linear 
convergence of Euclidean and directional SCMS algorithms (Figures C.1 and C.2 
in the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from mpl_toolkits.basemap import Basemap
from Utility_fun import cart2sph, sph2cart, Cir_Sph_samp, Gauss_Mix, Eu_Ring_Data, vMF_samp_mix
from SCMS_fun import KDE, SCMS_Log_KDE
from DirSCMS_fun import DirKDE, SCMS_Log_DirKDE
from matplotlib import rc


if __name__ == "__main__":
    ## Set up the parameters for the Gaussian mixture model
    mu2 = np.array([[1,1], [-1,-1]])
    cov2 = np.zeros((2,2,2))
    cov2[:,:,0] = np.diag([1/4,1/4])
    cov2[:,:,1] = np.array([[1/2,1/4], [1/4,1/2]])
    prob2 = [0.4, 0.6]

    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    Gau_data = Gauss_Mix(1000, mu=mu2, cov=cov2, prob=prob2)
    
    ## Estimate the densities on query points
    n_x = 100
    n_y = 100
    x = np.linspace(-3.5, 3.5, n_x)
    y = np.linspace(-3.5, 3.5, n_y)
    X, Y = np.meshgrid(x, y)
    query_pts = np.concatenate((X.reshape(n_x*n_y, 1), Y.reshape(n_x*n_y, 1)), axis=1)
    d_hat1 = KDE(query_pts, Gau_data, h=None)
    Z1 = d_hat1.reshape(n_x, n_y)
    
    ## Apply Euclidean SCMS algorithm to the denoised Gaussian mixture synthetic dataset
    SCMS_log_path = SCMS_Log_KDE(Gau_data, Gau_data, d=1, h=None, 
                                 eps=1e-9, max_iter=5000, stop_cri='proj_grad')
    ridge_pts2 = SCMS_log_path[:,:,SCMS_log_path.shape[2]-1]
    
    # Choose two points for plotting their SCMS trajectories and linear convergence trends
    pt1 = np.array([[-1.5, -2.5]])
    p1_SCMS_path = SCMS_Log_KDE(pt1, Gau_data, d=1, h=None, eps=1e-9, 
                                max_iter=5000, stop_cri='proj_grad')
    p1_SCMS_path = np.concatenate(p1_SCMS_path, axis=0).T
    pt2 = np.array([[0, 1]])
    p2_SCMS_path = SCMS_Log_KDE(pt2, Gau_data, d=1, h=None, eps=1e-9, 
                                max_iter=5000, stop_cri='proj_grad')
    p2_SCMS_path = np.concatenate(p2_SCMS_path, axis=0).T
    
    print("Generating the LC plots for Euclidean SCMS algorithms on the "\
          "Gaussian mixture synthetic dataset. \n")
    
    fig = plt.figure(figsize=(6,6))
    X_r = ridge_pts2[:,0]
    Y_r = ridge_pts2[:,1]
    X_p1 = p1_SCMS_path[:,0]
    Y_p1 = p1_SCMS_path[:,1]
    X_p2 = p2_SCMS_path[:,0]
    Y_p2 = p2_SCMS_path[:,1]
    plt.contourf(X, Y, Z1, cmap='viridis')
    plt.scatter(X_r, Y_r, s=40, color='red')
    plt.scatter(X_p1, Y_p1, s=30, color='black')
    plt.scatter(X_p2, Y_p2, facecolors='none', edgecolors='cyan', s=30)
    plt.axis('off')
    fig.savefig('./Figures/LC_pts_path.pdf')
    
    rc('text', usetex=True)   ## Use the latex in labels or axes
    plt.rcParams.update({'font.size': 30})  ## Change the font sizes of ouput figures
    
    fig = plt.figure(figsize=(11,8))
    iter_step = np.array(range(p2_SCMS_path.shape[0]-1))+1
    err = []
    for i in iter_step:
        final_pt = p2_SCMS_path[p2_SCMS_path.shape[0]-1,:]
        err.append(LA.norm(p2_SCMS_path[i-1,:]-final_pt))
    plt.plot(iter_step, np.log(err), linewidth=3)
    plt.xlabel(r'Number of iterations (t)')
    plt.ylabel(r'$\log\left(||\widehat{x}^{(t)}-\widehat{x}^*||_2\right)$')
    fig.savefig('./Figures/LC_dist_to_limit.pdf')
    
    fig = plt.figure(figsize=(12,8))
    iter_step = np.array(range(p2_SCMS_path.shape[0]-1))+1
    err2 = []
    ridge_pts_new = np.concatenate((ridge_pts2, 
                                    p2_SCMS_path[p2_SCMS_path.shape[0]-1,:].reshape(1,-1)), 
                                   axis=0)
    for i in iter_step:
        curr_pt = p2_SCMS_path[i-1,:]
        err2.append(min(np.sqrt(np.sum((curr_pt - ridge_pts_new)**2, axis=1))))
    plt.plot(iter_step, np.log(err2), linewidth=3)
    plt.xlabel(r'Number of iterations (t)')
    plt.ylabel(r'$\log\left(d_E(\widehat{x}^{(t)},\widehat{R}_d)\right)$')
    fig.savefig('./Figures/LC_dist_to_ridge.pdf')
    
    print("Save the plots as 'LC_pts_path.pdf', 'LC_dist_to_limit.pdf', "\
          "and 'LC_dist_to_ridge.pdf'.\n\n")
    
    
    ## Simulate data points for a half circle
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    radius = 2
    ring_Eu = Eu_Ring_Data(N=1000, R=radius, sigma=0.3, half=True)
    curr_bw = None
    
    ## Estimate the density values on query points
    n_x = 100
    n_y = 100
    x = np.linspace(-3, 3, n_x)
    y = np.linspace(-1, 4, n_y)
    X2, Y2 = np.meshgrid(x, y)
    query_pts2 = np.concatenate((X2.reshape(n_x*n_y, 1), 
                                 Y2.reshape(n_x*n_y, 1)), axis=1)
    d_hat2 = KDE(query_pts2, ring_Eu, h=curr_bw)
    Z2 = d_hat2.reshape(n_x, n_y)
    
    ## Apply Euclidean SCMS algorithm to the half-circle simulated dataset
    SCMS_Eu_log2 = SCMS_Log_KDE(ring_Eu, ring_Eu, d=1, h=curr_bw, 
                                eps=1e-9, max_iter=5000)
    Eu_ridge_log2 = SCMS_Eu_log2[:,:,SCMS_Eu_log2.shape[2]-1]
    
    ## Choose two points for plot their SCMS trajectories and linear convergence trends
    pt3 = np.array([[0, 0]])
    p3_SCMS_path = SCMS_Log_KDE(pt3, ring_Eu, d=1, h=None, eps=1e-9, 
                                max_iter=5000, stop_cri='proj_grad')
    p3_SCMS_path = np.concatenate(p3_SCMS_path, axis=0).T
    
    pt4 = np.array([[2, 3]])
    p4_SCMS_path = SCMS_Log_KDE(pt4, ring_Eu, d=1, h=None, eps=1e-9, 
                                max_iter=5000, stop_cri='proj_grad')
    p4_SCMS_path = np.concatenate(p4_SCMS_path, axis=0).T
    
    print("Generating the LC plots for Euclidean SCMS algorithms on the "\
          "half-circle simulated dataset. \n")
    
    fig = plt.figure(figsize=(6,6))
    X_r = Eu_ridge_log2[:,0]
    Y_r = Eu_ridge_log2[:,1]
    X_p3 = p3_SCMS_path[:,0]
    Y_p3 = p3_SCMS_path[:,1]
    X_p4 = p4_SCMS_path[:,0]
    Y_p4 = p4_SCMS_path[:,1]
    plt.contourf(X2, Y2, Z2, cmap='viridis')
    plt.scatter(X_r, Y_r, s=40, color='red')
    plt.scatter(X_p3, Y_p3, facecolors='none', edgecolors='cyan', s=30)
    plt.scatter(X_p4, Y_p4, s=30, color='black')
    plt.axis('equal')
    plt.axis('off')
    fig.savefig('./Figures/LC_pts_path_ring.pdf')
    
    fig = plt.figure(figsize=(11,8))
    iter_step = np.array(range(p3_SCMS_path.shape[0]-1))+1
    err = []
    final_pt = p3_SCMS_path[p3_SCMS_path.shape[0]-1,:]
    for i in iter_step:
        err.append(LA.norm(p3_SCMS_path[i-1,:]-final_pt))
    plt.plot(iter_step, np.log(err), linewidth=3)
    plt.xlabel(r'Number of iterations (t)')
    plt.ylabel(r'$\log\left(||\widehat{x}^{(t)}-\widehat{x}^*||_2\right)$')
    fig.savefig('./Figures/LC_dist_to_limit_ring.pdf')
    
    fig = plt.figure(figsize = (13,8))
    iter_step = np.array(range(p3_SCMS_path.shape[0]-1))+1
    err2 = []
    ## Include the limit point of the SCMS sequence initialized from our chosen point 
    ## into the collection of ridge points 
    ridge_pts_new = np.concatenate((Eu_ridge_log2, 
                                    p3_SCMS_path[p3_SCMS_path.shape[0]-1,:].reshape(1,-1)), 
                                   axis=0)
    for i in iter_step:
        curr_pt = p3_SCMS_path[i-1,:]
        err2.append(min(np.sqrt(np.sum((curr_pt - ridge_pts_new)**2, axis=1))))
    plt.plot(iter_step, np.log(err2), linewidth=3)
    plt.xlabel('Number of iterations (t)')
    plt.ylabel(r'$\log\left(d_E(\widehat{x}^{(t)},\widehat{R}_d) \right)$')
    fig.savefig('./Figures/LC_dist_to_ridge_ring.pdf')
    
    print("Save the plots as 'LC_pts_path_ring.pdf', 'LC_dist_to_limit_ring.pdf', "\
          "and 'LC_dist_to_ridge_ring.pdf'.\n\n")
    
    
    ## Simulate data points from a von Mises-Fisher (vMF) mixture model
    mu2 = np.array([[0,0,1], [1,0,0]])
    kappa2 = [10.0, 10.0]
    prob2 = [0.4, 0.6]
    np.random.seed(101)  ## Set an arbitrary seed for reproducibility
    vMF_data2 = vMF_samp_mix(1000, mu=mu2, kappa=kappa2, prob=prob2)
    curr_bw = None
    
    ## Estimate the directional densities on query points
    nrows, ncols = (90, 180)
    lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), 
                           np.linspace(-90, 90, nrows))
    xg, yg, zg = sph2cart(lon, lat)
    query_points = np.concatenate((xg.reshape(nrows*ncols, 1), 
                                   yg.reshape(nrows*ncols, 1),
                                   zg.reshape(nrows*ncols, 1)), axis=1)
    d_hat2 = DirKDE(query_points, vMF_data2).reshape(nrows, ncols)
    
    ## Apply our directional SCMS algorithm to synthetic vMF-distributed data points
    SCMS_path2_log = SCMS_Log_DirKDE(vMF_data2, vMF_data2, d=1, h=None, 
                                     eps=1e-9, max_iter=5000)
    vMF_Ridge2 = SCMS_path2_log[:,:,SCMS_path2_log.shape[2]-1]
    
    ## Pick two points for plotting their SCMS trajectories on the sphere and 
    ## linear convergence trends
    pt3 = np.array([-30, 30])
    pt3 = np.array(sph2cart(*pt3)).reshape(1,-1)
    p3_SCMS_path = SCMS_Log_DirKDE(pt3, vMF_data2, d=1, h=None, eps=1e-9, 
                                   max_iter=5000)
    p3_SCMS_path = np.concatenate(p3_SCMS_path, axis=0).T
    
    pt4 = np.array([-60, 60])
    pt4 = np.array(sph2cart(*pt4)).reshape(1,-1)
    p4_SCMS_path = SCMS_Log_DirKDE(pt4, vMF_data2, d=1, h=None, eps=1e-9, 
                                   max_iter=5000)
    p4_SCMS_path = np.concatenate(p4_SCMS_path, axis=0).T
    
    print("Generating the LC plots for directional SCMS algorithms on the "\
          "vMF-distributed dataset. \n")
    
    fig = plt.figure(figsize=(6,6))
    lon_r, lat_r, R = cart2sph(*vMF_Ridge2.T)
    lon3, lat3, R = cart2sph(*p3_SCMS_path.T)
    lon4, lat4, R = cart2sph(*p4_SCMS_path.T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
    # Draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    # Compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x_r, y_r = m2(lon_r, lat_r)
    x3, y3 = m2(lon3, lat3)
    x4, y4 = m2(lon4, lat4)
    # Contour data over the map.
    cs = m2.contourf(x, y, d_hat2)
    cs = m2.scatter(x_r, y_r, color='red', s=40)
    cs = m2.scatter(x3, y3, facecolors='none', edgecolors='cyan', s=30)
    cs = m2.scatter(x4, y4, color='black', s=30)
    fig.savefig('./Figures/LC_pts_path_vMF.pdf')
    
    fig = plt.figure(figsize=(13,8))
    iter_step = np.array(range(p3_SCMS_path.shape[0]-1))+1
    err2 = []
    for i in iter_step:
        final_pt = p3_SCMS_path[p3_SCMS_path.shape[0]-1,:]
        err2.append(np.arccos(np.dot(p3_SCMS_path[i-1,:], final_pt)))
    plt.plot(iter_step, np.log(np.array(err2)), linewidth=3)
    plt.xlabel(r'Number of iterations (t)')
    plt.ylabel(r'$\log\left(d_g(\widehat{\underline{x}}^{(t)},\widehat{\underline{x}}^*)\right)$')
    fig.savefig('./Figures/LC_dist_to_limit_vMF.pdf')
    
    fig = plt.figure(figsize=(13,8))
    iter_step = np.array(range(p3_SCMS_path.shape[0]-1))+1
    err2 = []
    DirRidge_new = np.concatenate((vMF_Ridge2, 
                                   p3_SCMS_path[p3_SCMS_path.shape[0]-1,:].reshape(1,-1)), 
                                  axis=0)
    for i in iter_step:
        curr_pt = p3_SCMS_path[i-1,:]
        err2.append(min(np.arccos(np.dot(DirRidge_new, curr_pt))))
    plt.plot(iter_step, np.log(np.array(err2)), linewidth=3)
    plt.xlabel(r'Number of iterations (t)')
    plt.ylabel(r'$\log\left(d_g(\widehat{\underline{x}}^{(t)},\widehat{\underline{R}}_d)\right)$')
    fig.savefig('./Figures/LC_dist_to_ridge_vMF.pdf')
    
    print("Save the plots as 'LC_pts_path_vMF.pdf', 'LC_dist_to_limit_vMF.pdf', "\
          "and 'LC_dist_to_ridge_vMF.pdf'.\n\n")
    
    ## Generate data points from a circle on the sphere (with additive noises)
    np.random.seed(111)  ## Set an arbitrary seed for reproducibility
    cir_samp = Cir_Sph_samp(1000, lat_c=0, sigma=0.2, pv_ax=np.array([1,0,0]))
    
    ## Denoising step
    bw_Dir = None
    d_hat1_Dir = DirKDE(cir_samp, cir_samp, h=bw_Dir)
    tau = 0.1
    print('Removing the data points whose directional density values are below '\
          +str(tau)+' of the maximum density.')
    cir_samp_thres = cir_samp[d_hat1_Dir >= tau*max(d_hat1_Dir),:]
    print('Ratio of the numbers of data points after and before the denoising '\
          'step: ' + str(cir_samp_thres.shape[0]/cir_samp.shape[0]) + '.\n')
    
    ## Estimate the directional density values on query points
    nrows, ncols = (90, 180)
    lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), 
                           np.linspace(-90, 90, nrows))
    xg, yg, zg = sph2cart(lon, lat)
    query_points = np.concatenate((xg.reshape(nrows*ncols, 1), 
                                   yg.reshape(nrows*ncols, 1),
                                   zg.reshape(nrows*ncols, 1)), axis=1)
    d_hat1 = DirKDE(query_points, cir_samp, h=bw_Dir).reshape(nrows, ncols)
    
    ## Apply our directional SCMS algorithm to the simulated dataset
    SCMS_Dir_log1 = SCMS_Log_DirKDE(cir_samp_thres, cir_samp, d=1, h=bw_Dir, 
                                    eps=1e-9, max_iter=5000)
    DirRidge1 = SCMS_Dir_log1[:,:,SCMS_Dir_log1.shape[2]-1]
    
    ## Take two points for plotting their directional SCMS trajectories 
    ## and linear convergence trends
    pt1 = np.array([-20, 10])
    pt1 = np.array(sph2cart(*pt1)).reshape(1,-1)
    p1_SCMS_path = SCMS_Log_DirKDE(pt1, cir_samp, d=1, h=None, eps=1e-9, 
                                   max_iter=5000)
    p1_SCMS_path = np.concatenate(p1_SCMS_path, axis=0).T
    
    pt2 = np.array([20, 10])
    pt2 = np.array(sph2cart(*pt2)).reshape(1,-1)
    p2_SCMS_path = SCMS_Log_DirKDE(pt2, cir_samp, d=1, h=None, eps=1e-9, 
                                   max_iter=5000)
    p2_SCMS_path = np.concatenate(p2_SCMS_path, axis=0).T
    
    print("Generating the LC plots for directional SCMS algorithms on the "\
          "circular dataset on the sphere. \n")
    
    fig = plt.figure(figsize=(6,6))
    lon_r, lat_r, R = cart2sph(*DirRidge1.T)
    lon1, lat1, R = cart2sph(*p1_SCMS_path.T)
    lon2, lat2, R = cart2sph(*p2_SCMS_path.T)
    m2 = Basemap(projection='ortho', lat_0=40, lon_0=10)
    # Draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    # Compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x_r, y_r = m2(lon_r, lat_r)
    x1, y1 = m2(lon1, lat1)
    x2, y2 = m2(lon2, lat2)
    # Contour data over the map.
    cs = m2.contourf(x, y, d_hat1)
    cs = m2.scatter(x_r, y_r, color='red', s=40)
    cs = m2.scatter(x1, y1, facecolors='none', edgecolors='cyan', s=30)
    cs = m2.scatter(x2, y2, color='black', s=30)
    fig.savefig('./Figures/LC_pts_path_circle_Dir.pdf')
    
    fig = plt.figure(figsize=(13,8))
    iter_step = np.array(range(p1_SCMS_path.shape[0]-1))+1
    err2 = []
    for i in iter_step:
        final_pt = p1_SCMS_path[p1_SCMS_path.shape[0]-1,:]
        err2.append(np.arccos(np.dot(p1_SCMS_path[i-1,:], final_pt)))
    plt.plot(iter_step, np.log(np.array(err2)), linewidth=3)
    plt.xlabel(r'Number of iterations (t)')
    plt.ylabel(r'$\log\left(d_g(\widehat{\underline{x}}^{(t)},\widehat{x}^*)\right)$')
    fig.savefig('./Figures/LC_dist_to_limit_cicle_Dir.pdf')
    
    fig = plt.figure(figsize=(13,8))
    iter_step = np.array(range(p1_SCMS_path.shape[0]-1))+1
    err2 = []
    DirRidge_new = np.concatenate((DirRidge1, 
                                   p1_SCMS_path[p1_SCMS_path.shape[0]-1,:].reshape(1,-1)), 
                                  axis=0)
    for i in iter_step:
        curr_pt = p1_SCMS_path[i-1,:]
        err2.append(min(np.arccos(np.dot(DirRidge_new, curr_pt))))
    plt.plot(iter_step, np.log(np.array(err2)), linewidth=3)
    plt.xlabel(r'Number of iterations (t)')
    plt.ylabel(r'$\log\left(d_g(\widehat{\underline{x}}^{(t)},\widehat{\underline{R}}_d)\right)$')
    fig.savefig('./Figures/LC_dist_to_ridge_circle_Dir.pdf')
    
    print("Save the plots as 'LC_pts_path_circle_Dir.pdf', "\
          "'LC_dist_to_limit_circle_Dir.pdf', "\
          "and 'LC_dist_to_ridge_circle_Dir.pdf'.\n\n")