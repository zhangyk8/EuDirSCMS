#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: November 14, 2021

Description: This script contains code for comparing Euclidean KDE with 
directional KDE as well as comparing Euclidean subspace constrained mean shift 
(SCMS) with our proposed directional SCMS algorithm on simulated datasets in 
order to illustrate the drawbacks of Euclidean KDE and SCMS algorithm in 
handling directional data (Figures 9 and 10 in the paper).

Warnings: Due to repeated experiments in the comparisons, this script takes 
around 2 hours to run on my laptop with 8 CPU cores.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from Utility_fun import cart2sph, sph2cart, Cir_Sph_samp
from SCMS_fun import KDE, SCMS_Log_KDE
from DirSCMS_fun import DirKDE, SCMS_Log_DirKDE
import pandas as pd
import time
from numpy import linalg as LA

def ProjDist_Dir(x, Fila):
    x = x.values
    if np.isnan(np.arccos(np.dot(Fila, x))).any():
        print(x)
    return np.min(np.arccos(np.dot(Fila, x)))

if __name__ == "__main__":
    np.random.seed(111)  # Set an arbitrary seed for reproducibility
    # Sampling the points on a circle that crosses through the north and south poles
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
    
    # Vary the maximum latitudes attained by the true circular structures and 
    # repeat the above sampling schemes. Apply Euclidean and directional SCMS
    # algorithms on each simulated data and compute their top 10% distance errors.
    lat_h = 90
    lat_l = 45
    N = 1000
    top_p = 0.1  # Computing the top 10% average distance errors
    topN = int(N*top_p)   ## Top N distance values for computing the average
    bw_Dir = None
    bw_Eu = None
    sigma = 0.2  # Standard deviation of the additive Gaussian noises
    K = 20   # Number of repeated steps

    lat_val = np.linspace(lat_l, lat_h, 10)
    Eu_err = np.zeros((K, len(lat_val)))
    Dir_err = np.zeros((K, len(lat_val)))
    Eu_err3 = np.zeros((K, len(lat_val)))
    
    Eu_step = np.zeros((K, len(lat_val)))
    Dir_step = np.zeros((K, len(lat_val)))
    Eu_step3 = np.zeros((K, len(lat_val)))
    
    Eu_time = np.zeros((K, len(lat_val)))
    Dir_time = np.zeros((K, len(lat_val)))
    Eu_time3 = np.zeros((K, len(lat_val)))
    
    print("Varing the latitudes of the true circular structures...\n")
    np.random.seed(111)  ## Set an arbitrary seed for reproducibility
    for i in range(len(lat_val)):
        # Sample points on the true circular structure (i.e., data points without noises)
        cir_true = Cir_Sph_samp(N, lat_c=90-lat_val[i], sigma=0, pv_ax=np.array([1,0,0]))
        
        for k in range(K):
            # Sample points on a circle parallel to the great circle that crosses
            # the North and South Poles
            cir_samp = Cir_Sph_samp(N, lat_c=90-lat_val[i], sigma=sigma, 
                                    pv_ax=np.array([1,0,0]))
            lon_c, lat_c, r = cart2sph(*cir_samp.T)
            cir_samp_ang = np.concatenate((lon_c.reshape(len(lon_c),1), 
                                           lat_c.reshape(len(lat_c),1)), axis=1)
        
            # Compute the directional ridges
            start = time.time()
            DirSCMS_log = SCMS_Log_DirKDE(cir_samp, cir_samp, d=1, h=bw_Dir, 
                                          eps=1e-7, max_iter=5000)
            DirRidge_log = DirSCMS_log[:,:,DirSCMS_log.shape[2]-1]
            Dir_time[k,i] = time.time() - start
            Dir_step[k,i] = DirSCMS_log.shape[2]-1
    
            # Compute the Euclidean ridges
            start = time.time()
            EuSCMS_log = SCMS_Log_KDE(cir_samp_ang, cir_samp_ang, d=1, h=bw_Eu, 
                                      eps=1e-7, max_iter=5000)
            EuRidge_log = EuSCMS_log[:,:,EuSCMS_log.shape[2]-1]
            Eu_time[k,i] = time.time() - start
            Eu_step[k,i] = EuSCMS_log.shape[2]-1
    
            # Compute the Euclidean ridges in R^3
            start = time.time()
            EuSCMS_log3 = SCMS_Log_KDE(cir_samp, cir_samp, d=1, h=bw_Eu, eps=1e-7, 
                                       max_iter=5000)
            EuRidge_log3 = EuSCMS_log3[:,:,EuSCMS_log3.shape[2]-1]
            Eu_time3[k,i] = time.time() - start
            Eu_step3[k,i] = EuSCMS_log3.shape[2]-1
        
            # Compute the projected distance errors
            xg, yg, zg = sph2cart(EuRidge_log[:,0], EuRidge_log[:,1])
            EuRidge_log_cart = np.concatenate((xg.reshape(-1, 1), 
                                               yg.reshape(-1, 1),
                                               zg.reshape(-1, 1)), axis=1)
            DistErr_Eu = pd.DataFrame(EuRidge_log_cart).\
                apply(lambda x: ProjDist_Dir(x, Fila=cir_true), axis=1)
            DistErr_Dir = pd.DataFrame(DirRidge_log).\
                apply(lambda x: ProjDist_Dir(x, Fila=cir_true), axis=1)
            # Standardize the points on Euclidean ridges in R^3 back to Omega_2
            EuRidge_log3 = EuRidge_log3/LA.norm(EuRidge_log3, axis=1).reshape(-1,1)
            DistErr_Eu3 = pd.DataFrame(EuRidge_log3).\
                apply(lambda x: ProjDist_Dir(x, Fila=cir_true), axis=1)
            
            Eu_err[k,i] = np.mean(DistErr_Eu)
            Dir_err[k,i] = np.mean(DistErr_Dir)
            Eu_err3[k,i] = np.mean(DistErr_Eu3)
        
    plt.rcParams.update({'font.size': 15})  # Change the font sizes of ouput figures
    fig = plt.figure(figsize=(8,6))
    plt.scatter(lat_val, np.mean(Eu_err, axis=0))
    plt.scatter(lat_val, np.mean(Dir_err, axis=0))
    plt.scatter(lat_val, np.mean(Eu_err3, axis=0))
    plt.errorbar(lat_val, np.mean(Eu_err, axis=0), yerr=np.std(Eu_err, axis=0), 
                 capsize=5, elinewidth=1, label='Euclidean SCMS (Angular)')
    plt.errorbar(lat_val, np.mean(Dir_err, axis=0), yerr=np.std(Dir_err, axis=0), 
                 capsize=5, elinewidth=1, label='Directional SCMS on $\Omega_2$')
    plt.errorbar(lat_val, np.mean(Eu_err3, axis=0), yerr=np.std(Eu_err3, axis=0), 
                 capsize=5, elinewidth=1, label='Euclidean SCMS in $R^3$ (Cartesian)')
    plt.xlabel('Latitudes')
    plt.ylabel('Average geodesic distance errors')
    plt.xticks(np.linspace(45, 90, 10))
    plt.legend()
    plt.tight_layout()
    fig.savefig('./Figures/Estimate_err_DirEu.pdf')
    
    fig = plt.figure(figsize=(8,6))
    plt.errorbar(lat_val, np.mean(Eu_step, axis=0), yerr=np.std(Eu_step, axis=0), 
                 capsize=5, elinewidth=1, label='Euclidean SCMS (Angular)')
    plt.errorbar(lat_val, np.mean(Dir_step, axis=0), yerr=np.std(Dir_step, axis=0), 
                 capsize=5, elinewidth=1, label='Directional SCMS on $\Omega_2$')
    plt.errorbar(lat_val, np.mean(Eu_step3, axis=0), yerr=np.std(Eu_step3, axis=0), 
                 capsize=5, elinewidth=1, label='Euclidean SCMS in $R^3$ (Cartesian)')
    plt.plot(lat_val, np.mean(Eu_step, axis=0))
    plt.plot(lat_val, np.mean(Dir_step, axis=0))
    plt.plot(lat_val, np.mean(Eu_step3, axis=0))
    plt.xlabel('Latitudes')
    plt.ylabel('Number of iterative steps')
    plt.xticks(np.linspace(45, 90, 10))
    plt.legend()
    plt.tight_layout()
    fig.savefig('./Figures/Estimate_step_DirEu.pdf')
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(lat_val, np.mean(Eu_time, axis=0))
    plt.scatter(lat_val, np.mean(Dir_time, axis=0))
    plt.scatter(lat_val, np.mean(Eu_time3, axis=0))
    plt.errorbar(lat_val, np.mean(Eu_time, axis=0), yerr=np.std(Eu_time, axis=0), 
                 capsize=5, elinewidth=1, label='Euclidean SCMS (Angular)')
    plt.errorbar(lat_val, np.mean(Dir_time, axis=0), yerr=np.std(Dir_time, axis=0), 
                 capsize=5, elinewidth=1, label='Directional SCMS on $\Omega_2$')
    plt.errorbar(lat_val, np.mean(Eu_time3, axis=0), yerr=np.std(Eu_time3, axis=0), 
                 capsize=5, elinewidth=1, label='Euclidean SCMS in $R^3$ (Cartesian)')
    plt.xlabel('Latitudes')
    plt.ylabel('Running time in seconds')
    plt.xticks(np.linspace(45, 90, 10))
    plt.legend()
    plt.tight_layout()
    fig.savefig('./Figures/Estimate_time_DirEu.pdf')
    
    fig = plt.figure(figsize=(6,6))
    # set up map projection
    m1 = Basemap(projection='ortho', lat_0=30, lon_0=70)
    # draw lat/lon grid lines every 30 degrees.
    m1.drawmeridians(np.arange(0, 360, 30))
    m1.drawparallels(np.arange(-90, 90, 30))
    for lat in np.linspace(45, 90, 6):
        cir_true = Cir_Sph_samp(3000, lat_c=90-lat, sigma=0, pv_ax=np.array([1,0,0]))
        lon_t, lat_t, r = cart2sph(*cir_true.T)
        x_t, y_t = m1(lon_t, lat_t)
        cs = m1.scatter(x_t, y_t, color='blue', s=2)
    x1, y1 = m1(-6,36)
    plt.annotate('$45^{\circ}$', xy=(x1, y1), xycoords='data', xytext=(x1, y1), 
                 textcoords='data', size=13)
    x2, y2 = m1(180, 90)
    plt.annotate('$90^{\circ}$', xy=(x2, y2), xycoords='data', xytext=(x2, y2), 
                 textcoords='data', size=14)
    fig.savefig('./Figures/pall_cir.pdf')
    print("Save the plots as 'Estimate_err_DirEu.pdf', 'Estimate_step_DirEu.pdf', "\
          "'Estimate_time_DirEu.pdf', and 'pall_cir.pdf'\n\n")