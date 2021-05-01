#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: March 31, 2021

Description: This script contains code for comparing Euclidean KDE with 
directional KDE as well as comparing Euclidean subspace constrained mean shift 
(SCMS) with our proposed directional SCMS algorithm on simulated datasets in 
order to illustrate the drawbacks of Euclidean KDE and SCMS algorithm in 
handling directional data (Figure 4 in the arxiv version of the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from Utility_fun import cart2sph, sph2cart, Cir_Sph_samp
from SCMS_fun import KDE, SCMS_Log_KDE
from DirSCMS_fun import DirKDE, SCMS_Log_DirKDE

if __name__ == "__main__":
    np.random.seed(111)  # Set an arbitrary seed for reproducibility
    ## Sampling the points on a circle that crosses through the north and south poles
    cir_samp = Cir_Sph_samp(1000, lat_c=0, sigma=0.2, pv_ax=np.array([1,0,0]))
    lon_c, lat_c, r = cart2sph(*cir_samp.T)
    cir_samp_ang = np.concatenate((lon_c.reshape(len(lon_c),1), 
                                   lat_c.reshape(len(lat_c),1)), axis=1)
    bw_Dir = None
    bw_Eu = None
    
    
    ## Estimate the directional and Euclidean densities on query points
    nrows, ncols = (90, 180)
    lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), 
                           np.linspace(-90, 90, nrows))
    xg, yg, zg = sph2cart(lon, lat)
    query_points = np.concatenate((xg.reshape(nrows*ncols, 1), 
                                   yg.reshape(nrows*ncols, 1),
                                   zg.reshape(nrows*ncols, 1)), axis=1)
    d_hat2 = DirKDE(query_points, cir_samp, h=bw_Dir).reshape(nrows, ncols)
    
    query_points_ang = np.concatenate((lon.reshape(nrows*ncols, 1), 
                                       lat.reshape(nrows*ncols, 1)), axis=1)
    d_hat2_Eu = KDE(query_points_ang, cir_samp_ang, h=bw_Eu).reshape(nrows, ncols)

    ## Apply the directional and Euclidean SCMS algorithms
    SCMS_Dir_log2 = SCMS_Log_DirKDE(cir_samp, cir_samp, d=1, h=bw_Dir, 
                                    eps=1e-7, max_iter=5000)
    Dir_ridge_log2 = SCMS_Dir_log2[:,:,SCMS_Dir_log2.shape[2]-1]
    
    SCMS_Eu_log2 = SCMS_Log_KDE(cir_samp_ang, cir_samp_ang, d=1, h=bw_Eu, 
                                eps=1e-7, max_iter=5000)
    Eu_ridge_log2 = SCMS_Eu_log2[:,:,SCMS_Eu_log2.shape[2]-1]
    
    print("Generating the plots for Euclidean and directional SCMS algorithms "\
          "on the synthetic dataset. \n")
    
    fig = plt.figure(figsize=(14,8))
    Eu_step = 0
    lon1 = SCMS_Eu_log2[:,0,Eu_step]
    lat1 = SCMS_Eu_log2[:,1,Eu_step]
    m2 = Basemap(projection='hammer', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0)
    # Draw lat/lon grid lines every 30 degrees.
    # m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    # Compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x1, y1 = m2(lon1, lat1)
    # Contour data over the map.
    cs = m2.contourf(x, y, d_hat2_Eu)
    cs = m2.scatter(x1, y1, color='red', s=30)
    fig.savefig('./Figures/EuSCMS_ring_Step0_Eu_noise_hammer.pdf')
    
    fig = plt.figure(figsize=(14,8))
    Eu_step = SCMS_Eu_log2.shape[2] - 1
    lon1 = SCMS_Eu_log2[:,0,Eu_step]
    lat1 = SCMS_Eu_log2[:,1,Eu_step]
    m2 = Basemap(projection='hammer', lat_0=30, lon_0=0)
    # Draw lat/lon grid lines every 30 degrees.
    # m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    # Compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x1, y1 = m2(lon1, lat1)
    # Contour data over the map.
    cs = m2.contourf(x, y, d_hat2_Eu)
    cs = m2.scatter(x1, y1, color='red', s=30)
    fig.savefig('./Figures/EuSCMS_ring_Step_conv_Eu_noise_hammer.pdf')
    
    fig = plt.figure(figsize=(6,6))
    Eu_final_step = SCMS_Eu_log2.shape[2] - 1
    lon6 = SCMS_Eu_log2[:,0,Eu_final_step]
    lat6 = SCMS_Eu_log2[:,1,Eu_final_step]
    m2 = Basemap(projection='ortho', lat_0=40, lon_0=0)
    # Draw lat/lon grid lines every 30 degrees.
    # m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-80.,81.,20.))
    # Compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x6, y6 = m2(lon6, lat6)
    # Contour data over the map.
    cs = m2.contourf(x, y, d_hat2_Eu)
    cs = m2.scatter(x6, y6, color='red', s=30)
    fig.savefig('./Figures/EuSCMS_ring_Step_conv_ortho.pdf')
    
    fig = plt.figure(figsize=(14,8))
    curr_step = 0
    lon2, lat2, R = cart2sph(*SCMS_Dir_log2[:,:,curr_step].T)
    m2 = Basemap(projection='hammer', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0)
    # Draw lat/lon grid lines every 30 degrees.
    # m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    # Compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x2, y2 = m2(lon2, lat2)
    # Contour data over the map.
    cs = m2.contourf(x, y, d_hat2)
    cs = m2.scatter(x2, y2, color='red', s=30)
    fig.savefig('./Figures/DirSCMS_ring_Step0_Eu_noise_hammer.pdf')
    
    fig = plt.figure(figsize=(14,8))
    curr_step = SCMS_Dir_log2.shape[2]-1
    lon2, lat2, R = cart2sph(*SCMS_Dir_log2[:,:,curr_step].T)
    m2 = Basemap(projection='hammer', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0)
    # Draw lat/lon grid lines every 30 degrees.
    # m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    # Compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x2, y2 = m2(lon2, lat2)
    # Contour data over the map.
    cs = m2.contourf(x, y, d_hat2)
    cs = m2.scatter(x2, y2, color='red', s=30)
    fig.savefig('./Figures/DirSCMS_ring_Step_conv_Eu_noise_hammer.pdf')
    
    fig = plt.figure(figsize=(6,6))
    curr_step = SCMS_Dir_log2.shape[2]-1
    lon5, lat5, R = cart2sph(*SCMS_Dir_log2[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=40, lon_0=0)
    # Draw lat/lon grid lines every 30 degrees.
    # m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    # Compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x5, y5 = m2(lon5, lat5)
    # Contour data over the map.
    cs = m2.contourf(x, y, d_hat2)
    cs = m2.scatter(x5, y5, color='red', s=30)
    fig.savefig('./Figures/DirSCMS_ring_Step_conv_ortho.pdf')
    
    print("Save the plots as 'EuSCMS_ring_Step0_Eu_noise_hammer.pdf', "\
          "'EuSCMS_ring_Step_conv_Eu_noise_hammer.pdf', "\
          "'SCMS_ring_Step_conv_Eu_noise_north.pdf', "\
          "'DirSCMS_ring_Step0_Eu_noise_hammer.pdf', "\
          "'DirSCMS_ring_Step_conv_Eu_noise_hammer.pdf', and "\
          "'SCMS_ring_Step_conv_Eu_noise_ortho.pdf'.\n\n")