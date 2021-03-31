#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: March 31, 2021

Description: This script contains code for applying Euclidean and directional 
subspace constrained mean shift (SCMS) algorithm to simulated datasets 
(Figure 1.1 in the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from Utility_fun import cart2sph, Eu_Ring_Data, Cir_Sph_samp
from SCMS_fun import SCMS_Log_KDE
from DirSCMS_fun import SCMS_Log_DirKDE


if __name__ == "__main__":
    np.random.seed(111)  ## Set an arbitrary seed for reproducibility
    radius = 2
    ring_Eu = Eu_Ring_Data(N=1000, R=radius, sigma=0.2)
    curr_bw = 0.35
    
    ## Apply the Euclidean SCMS
    SCMS_Eu_log1 = SCMS_Log_KDE(ring_Eu, ring_Eu, d=1, h=curr_bw, 
                                eps=1e-10, max_iter=5000)
    Eu_ridge_log1 = SCMS_Eu_log1[:,:,SCMS_Eu_log1.shape[2]-1]
    
    
    print("Generating the plot of Euclidean SCMS algorithm on the simulated "\
          "ring dataset. \n")
    # Generating the figures
    fig = plt.figure(figsize=(6,6))
    theta = np.linspace(0, 2*np.pi, 50)
    plt.scatter(ring_Eu[:,0], ring_Eu[:,1], facecolors='none', 
                edgecolors='black', s=7)
    plt.plot(radius*np.cos(theta), radius*np.sin(theta), color='blue', 
             linewidth=2)
    plt.scatter(Eu_ridge_log1[:,0], Eu_ridge_log1[:,1], color='red', s=10)
    plt.axis('equal')
    plt.axis('off')
    fig.savefig('./Figures/Ring_data_Eu.pdf')
    
    print("Save the plot as 'Ring_data_Eu.pdf'.\n\n")
    
    
    np.random.seed(111)  ## Set an arbitrary seed for reproducibility
    ## Sampling the points on a circle that crosses through the north and south poles
    cir_samp = Cir_Sph_samp(1000, lat_c=0, sigma=0.2, pv_ax=np.array([1,0,0]))
    lon_c, lat_c, r = cart2sph(*cir_samp.T)
    cir_samp_ang = np.concatenate((lon_c.reshape(len(lon_c),1), 
                                   lat_c.reshape(len(lat_c),1)), axis=1)
    bw_Dir = None
    bw_Eu = None

    ## Apply the directional and Euclidean SCMS algorithms
    SCMS_Dir_log2 = SCMS_Log_DirKDE(cir_samp, cir_samp, d=1, h=bw_Dir, 
                                    eps=1e-7, max_iter=5000)
    Dir_ridge_log2 = SCMS_Dir_log2[:,:,SCMS_Dir_log2.shape[2]-1]
    
    SCMS_Eu_log2 = SCMS_Log_KDE(cir_samp_ang, cir_samp_ang, d=1, h=bw_Eu, 
                                eps=1e-7, max_iter=5000)
    Eu_ridge_log2 = SCMS_Eu_log2[:,:,SCMS_Eu_log2.shape[2]-1]
    
    print("Generating the plot of Euclidean and directional SCMS algorithms on "\
          "the simulated ring dataset on the sphere. \n")
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
    fig.savefig('./Figures/Ring_on_Sphere_hammer.pdf')
    
    print("Save the plot as 'Ring_on_Sphere_hammer.pdf'.\n\n")