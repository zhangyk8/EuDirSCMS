#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: April 8, 2021

Description: This script contains code for our applications of Euclidean and 
directional SCMS algorithms to the earthquake data (Figures C.3 in the paper).
This script takes only several minutes to run on a laptop with 8 CPU cores.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pandas as pd
from mpl_toolkits.basemap import Basemap
from Utility_fun import cart2sph, sph2cart
from SCMS_fun import KDE
from DirSCMS_fun import DirKDE
# from SCMS_fun import SCMS_Log_KDE
# from DirSCMS_fun import SCMS_Log_DirKDE
from MS_SCMS_Ray import SCMS_Log_KDE_Fs, SCMS_Log_DirKDE_Fs
import ray

if __name__ == "__main__":
    ## Load the earthquake data
    Earthquakes = pd.read_csv('Data/Earthquakes_20201001-20210331.csv')
    X, Y, Z = sph2cart(*Earthquakes[['longitude', 'latitude']].values.T)
    EQ_cart = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)], 
                              axis=1)
    EQ_ang = Earthquakes[['longitude', 'latitude']].values
    
    ## Estimate the density values on query points via Euclidean and 
    ## directional KDEs
    nrows, ncols = (90, 180)
    lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), 
                           np.linspace(-90, 90, nrows))
    xg, yg, zg = sph2cart(lon, lat)
    query_points = np.concatenate((xg.reshape(nrows*ncols, 1), 
                                   yg.reshape(nrows*ncols, 1),
                                   zg.reshape(nrows*ncols, 1)), axis=1)
    bw_Dir = 0.1
    d_EQ = DirKDE(query_points, EQ_cart, h=bw_Dir).reshape(nrows, ncols)

    query_points_ang = np.concatenate((lon.reshape(nrows*ncols, 1), 
                                       lat.reshape(nrows*ncols, 1)), axis=1)
    bw_Eu = 7.0
    d_EQ_Eu = KDE(query_points_ang, EQ_ang, h=bw_Eu).reshape(nrows, ncols)
    
    ## Sample 5000 points uniformly on the sphere as the mesh points
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    mesh_pts = np.random.multivariate_normal(mean=[0,0,0], cov=np.eye(3), 
                                             size=5000)
    mesh_pts = mesh_pts/LA.norm(mesh_pts, axis=1).reshape(-1,1)
    lon_m, lat_m, R = cart2sph(*mesh_pts.T)
    mesh_pts_ang = np.concatenate((lon_m.reshape(-1, 1), 
                                    lat_m.reshape(-1, 1)), axis=1)
    
    ## Apply directional and Euclidean SCMS algorithms to the earthquake data
    '''
    SCMS_path_Dir = SCMS_Log_DirKDE(EQ_cart, EQ_cart, d=1, h=bw_Dir, eps=1e-9, 
                                    max_iter=5000)
    SCMS_Eu_Log_EQ = SCMS_Log_KDE(EQ_ang, EQ_ang, d=1, h=bw_Eu, eps=1e-9, 
                                  max_iter=5000)
    '''
    ray.init()
    chunksize = 10
    num_p = mesh_pts.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(SCMS_Log_DirKDE_Fs.remote(mesh_pts[i:(i+chunksize)], 
                                                    EQ_cart, d=1, h=bw_Dir, 
                                                    eps=1e-9, max_iter=5000))
    DirSCMS_pts = ray.get(result_ids)
    DirSCMS_pts = np.concatenate(DirSCMS_pts, axis=0)
    ray.shutdown()
    
    ray.init()
    chunksize = 10
    num_p = mesh_pts_ang.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(SCMS_Log_KDE_Fs.remote(mesh_pts_ang[i:(i+chunksize)], 
                                                 EQ_ang, d=1, h=bw_Eu, eps=1e-9, 
                                                 max_iter=5000))
    EuSCMS_pts = ray.get(result_ids)
    EuSCMS_pts = np.concatenate(EuSCMS_pts, axis=0)
    ray.shutdown()
    
    print("Generating the plots for Euclidean and directional density ridges"\
          " on the earthquake data. \n")
    plt.rcParams.update({'font.size': 13})  # Change the font sizes of ouput figures
    fig = plt.figure(figsize=(14,8))
    lon3 = EuSCMS_pts[:,0]
    lat3 = EuSCMS_pts[:,1]
    m2 = Basemap(projection='robin', lon_0=0, resolution='c')
    # Draw coastlines, country boundaries, fill continents.
    m2.drawcoastlines(linewidth=0.25)
    m2.drawcountries(linewidth=0.25)
    m2.etopo(scale=0.5, alpha=0.07)
    # Draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x3, y3 = m2(lon3, lat3)
    x, y = m2(lon, lat)
    cs = m2.scatter(x3, y3, color='red', s=40, marker='D')
    # cs = m2.contour(x, y, d_EQ_Eu, linewidths=3, cmap='hsv')
    plt.title('Earthquake Ridges on the World Map (h=7.0)')
    fig.savefig('./Figures/Earth_Ridges_Eu_SCMS.pdf')
    
    fig = plt.figure(figsize=(14,8))
    lon1, lat1, R = cart2sph(*DirSCMS_pts.T)
    m1 = Basemap(projection='robin', lon_0=0, resolution='c')
    # Draw coastlines, country boundaries, fill continents.
    m1.drawcoastlines(linewidth=0.25)
    m1.drawcountries(linewidth=0.25)
    m1.etopo(scale=0.5, alpha=0.07)
    # Draw lat/lon grid lines every 30 degrees.
    m1.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m1.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x1, y1 = m1(lon1, lat1)
    x, y = m1(lon, lat)
    cs = m1.scatter(x1, y1, color='red', s=40, marker='D')
    # cs = m1.contour(x, y, d_EQ, linewidths=3, cmap='hsv')
    plt.title('Earthquake Ridges on the World Map (h=0.1)')
    fig.savefig('./Figures/Earth_Ridges_Dir_SCMS.pdf')
    
    print("Save the plots as 'Earth_Ridges_Eu_SCMS.pdf' "\
          "and 'Earth_Ridges_Dir_SCMS.pdf'.\n\n")