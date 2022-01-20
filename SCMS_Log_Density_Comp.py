#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: January 11, 2022

Description: This script contains code for empirically demonstrating that the 
(Euclidean/directional) SCMS algorithms with the logarithm of the estimated 
densities are faster than their counterparts with the original densities 
(Figure 7 in the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Utility_fun import cart2sph, sph2cart, Cir_Sph_samp, Gauss_Mix, Eu_Ring_Data, vMF_samp_mix
from SCMS_fun import KDE, SCMS_KDE, SCMS_Log_KDE
from DirSCMS_fun import DirKDE, SCMS_DirKDE, SCMS_Log_DirKDE
import time
import seaborn as sns

if __name__ == "__main__":
    ## Set up the parameters for the Gaussian mixture model
    mu2 = np.array([[1,1], [-1,-1]])
    cov2 = np.zeros((2,2,2))
    cov2[:,:,0] = np.diag([1/4,1/4])
    cov2[:,:,1] = np.array([[1/2,1/4], [1/4,1/2]])
    prob2 = [0.4, 0.6]
    
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    EuSCMS_t = []
    EuSCMS_log_t = []
    N = 20
    for k in range(N):
        Gau_data = Gauss_Mix(1000, mu=mu2, cov=cov2, prob=prob2)
    
        ## Denoising step
        Eu_bw=None
        d_que1 = KDE(Gau_data, Gau_data, h=Eu_bw)
        tau = 0.25
        print('Removing the data points whose density values are below '+str(tau)\
              +' of the maximum density.')
        Gau_data_thres = Gau_data[d_que1 >= tau*max(d_que1),:]
        print('Ratio of the numbers of data points after and before the denoising '\
              'step: ' + str(Gau_data_thres.shape[0]/Gau_data.shape[0]) + '.\n')
        
        ## Estimate the densities on query points
        n_x = 100
        n_y = 100
        x = np.linspace(-3.5, 3.5, n_x)
        y = np.linspace(-3.5, 3.5, n_y)
        X, Y = np.meshgrid(x, y)
        query_pts = np.concatenate((X.reshape(n_x*n_y, 1), Y.reshape(n_x*n_y, 1)), axis=1)
        d_hat1 = KDE(query_pts, Gau_data, h=None)
        Z1 = d_hat1.reshape(n_x, n_y)
        
        ## Apply Euclidean SCMS algorithm with log-density to the denoised Gaussian mixture synthetic dataset
        start = time.time()
        SCMS_path = SCMS_KDE(Gau_data_thres, Gau_data, d=1, h=None, 
                             eps=1e-9, max_iter=10000, stop_cri='proj_grad')
        ridge_pts1 = SCMS_path[:,:,SCMS_path.shape[2]-1]
        EuSCMS_t.append(time.time() - start)
        
        ## Apply Euclidean SCMS algorithm to the denoised Gaussian mixture synthetic dataset
        start = time.time()
        SCMS_log_path = SCMS_Log_KDE(Gau_data_thres, Gau_data, d=1, h=None, 
                                     eps=1e-9, max_iter=10000, stop_cri='proj_grad')
        ridge_pts2 = SCMS_log_path[:,:,SCMS_log_path.shape[2]-1]
        EuSCMS_log_t.append(time.time() - start)
        
    
    ## Simulate data points for a half circle
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    radius = 2
    N = 20
    EuSCMS_t2 = []
    EuSCMS_log_t2 = []
    for k in range(N):
        ring_Eu = Eu_Ring_Data(N=1000, R=radius, sigma=0.3, half=True)
    
        ## Denoising step
        curr_bw = None
        d_hat_Eu = KDE(ring_Eu, ring_Eu, h=curr_bw)
        tau = 0.25
        print('Removing the data points whose density values are below '\
              +str(tau)+' of the maximum density.')
        ring_Eu_thres = ring_Eu[d_hat_Eu >= tau*max(d_hat_Eu),:]
        print('Ratio of the numbers of data points after and before the denoising '\
              'step: ' + str(ring_Eu_thres.shape[0]/ring_Eu.shape[0]) + '.\n')
        
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
        start = time.time()
        SCMS_Eu2 = SCMS_KDE(ring_Eu_thres, ring_Eu, d=1, h=curr_bw, 
                            eps=1e-9, max_iter=10000)
        Eu_ridge2 = SCMS_Eu2[:,:,SCMS_Eu2.shape[2]-1]
        EuSCMS_t2.append(time.time() - start)
        
        ## Apply Euclidean SCMS algorithm with log-density to the half-circle simulated dataset
        start = time.time()
        SCMS_Eu_log2 = SCMS_Log_KDE(ring_Eu_thres, ring_Eu, d=1, h=curr_bw, 
                                    eps=1e-9, max_iter=10000)
        Eu_ridge_log2 = SCMS_Eu_log2[:,:,SCMS_Eu_log2.shape[2]-1]
        EuSCMS_log_t2.append(time.time() - start)
        
        
    ## Simulate data points from a von Mises-Fisher (vMF) mixture model
    mu2 = np.array([[0,0,1], [1,0,0]])
    kappa2 = [10.0, 10.0]
    prob2 = [0.4, 0.6]
    np.random.seed(101)  ## Set an arbitrary seed for reproducibility
    N = 20
    DirSCMS_t = []
    DirSCMS_log_t = []
    for k in range(N):
        vMF_data2 = vMF_samp_mix(1000, mu=mu2, kappa=kappa2, prob=prob2)
        
        ## Denoising step
        curr_bw = None
        d_que1 = DirKDE(vMF_data2, vMF_data2, h=curr_bw)
        tau = 0.1
        print('Removing the data points whose directional density values are below '\
              +str(tau)+' of the maximum density.')
        qpts_thres = vMF_data2[d_que1 >= tau*max(d_que1),:]
        print('Ratio of the numbers of data points after and before the denoising '\
              'step: ' + str(qpts_thres.shape[0]/vMF_data2.shape[0]) + '.\n')
        
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
        start = time.time()
        SCMS_path = SCMS_DirKDE(qpts_thres, vMF_data2, d=1, h=None, 
                                eps=1e-9, max_iter=10000)
        vMF_Ridge = SCMS_path[:,:,SCMS_path.shape[2]-1]
        DirSCMS_t.append(time.time() - start)
        
        ## Apply our directional SCMS algorithm with log densities to synthetic vMF-distributed data points
        start = time.time()
        SCMS_path_log = SCMS_Log_DirKDE(qpts_thres, vMF_data2, d=1, h=None, 
                                        eps=1e-9, max_iter=10000)
        vMF_Ridge2 = SCMS_path_log[:,:,SCMS_path_log.shape[2]-1]
        DirSCMS_log_t.append(time.time() - start)
        
        
    ## Generate data points from a circle on the sphere (with additive noises)
    np.random.seed(111)  ## Set an arbitrary seed for reproducibility
    N = 20
    DirSCMS_t2 = []
    DirSCMS_log_t2 = []
    for k in range(N):
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
        start = time.time()
        SCMS_Dir2 = SCMS_DirKDE(cir_samp_thres, cir_samp, d=1, h=bw_Dir, 
                                eps=1e-9, max_iter=10000)
        DirRidge1 = SCMS_Dir2[:,:,SCMS_Dir2.shape[2]-1]
        DirSCMS_t2.append(time.time() - start)
        
        ## Apply our directional SCMS algorithm with log densities to the simulated dataset
        start = time.time()
        SCMS_Dir_log2 = SCMS_Log_DirKDE(cir_samp_thres, cir_samp, d=1, h=bw_Dir, 
                                       eps=1e-9, max_iter=10000)
        DirRidge2 = SCMS_Dir_log2[:,:,SCMS_Dir_log2.shape[2]-1]
        DirSCMS_log_t2.append(time.time() - start)
        
    
    print("Generating the boxplots of running time comparisons between the SCMS "\
          "algorithms with the original densities and log-densities. \n")
    
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(10, 7))
    SCMS_comp = pd.DataFrame({"Running Time in Seconds": EuSCMS_t + EuSCMS_t2 + EuSCMS_log_t + EuSCMS_log_t2,
                              "Algorithms": list(np.repeat(['Euclidean SCMS with $\hat{p}_n(\mathbf{x})$', 
                                                       'Euclidean SCMS with $\log\hat{p}_n(\mathbf{x})$'], 2*N)), 
                              "Simulated Datasets": list(np.repeat(['Gaussian Mixture', 'Half Circle'], N))*2})
    # create grouped boxplot 
    sns.boxplot(x="Simulated Datasets", y="Running Time in Seconds", 
                hue="Algorithms", data=SCMS_comp, palette='Set2')
    plt.legend()
    fig.tight_layout()
    plt.savefig('Figures/EuSCMS_log_comp.pdf')
    
    
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(10, 7))
    DirSCMS_comp = pd.DataFrame({"Running Time in Seconds": DirSCMS_t + DirSCMS_t2 + DirSCMS_log_t + DirSCMS_log_t2,
                                 "Algorithms": list(np.repeat(['Directional SCMS with $\hat{f}_h(\mathbf{x})$', 
                                                'Directional SCMS with $\log\hat{f}_h(\mathbf{x})$'], 2*N)), 
                                 "Simulated Datasets": list(np.repeat(['vMF Mixture', 'Circle on $\Omega_2$'], N))*2})
    sns.boxplot(x="Simulated Datasets", y="Running Time in Seconds", 
                hue="Algorithms", data=DirSCMS_comp, palette='Set3')
    plt.legend()
    fig.tight_layout()
    plt.savefig('Figures/DirSCMS_log_comp.pdf')
    
    print("Save the plots as 'EuSCMS_log_comp.pdf' and 'DirSCMS_log_comp.pdf'.\n\n")