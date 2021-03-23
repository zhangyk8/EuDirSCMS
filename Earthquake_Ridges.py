#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: March 20, 2021

Description: This script contains code for our applications of Euclidean and 
directional SCMS algorithms to the earthquake data (Figures C.3 in the paper).
This script may take more than half an hour to execute, depending on the 
implementation environment.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from Utility_fun import cart2sph, sph2cart
from SCMS_fun import KDE, SCMS_Log_KDE
from DirSCMS_fun import DirKDE, SCMS_Log_DirKDE

if __name__ == "__main__":
    ## Load the earthquake data
    Earthquakes = pd.read_csv('Data/Earthquakes_20201001-20210228.csv')
    X, Y, Z = sph2cart(*Earthquakes[['longitude', 'latitude']].values.T)
    EQ_cart = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)], 
                              axis=1)
    EQ_ang = Earthquakes[['longitude', 'latitude']].values
    
    ## Estimate the density values on query points via Euclidean and 
    ## directional KDEs
    nrows, ncols = (90, 180)
    lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), np.linspace(-90, 90, nrows))
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
    
    ## Apply directional and Euclidean SCMS algorithms to the earthquake data
    SCMS_path_Dir = SCMS_Log_DirKDE(EQ_cart, EQ_cart, d=1, h=0.1, eps=1e-7, 
                                    max_iter=5000)
    SCMS_Eu_Log_EQ = SCMS_Log_KDE(EQ_ang, EQ_ang, d=1, h=7, eps=1e-7, 
                                  max_iter=5000)
    
    print("Generating the plots for Euclidean and directional density ridges"\
          " on the earthquake data. \n")
    plt.rcParams.update({'font.size': 20})  ## Change the font sizes of ouput figures
    fig = plt.figure(figsize=(14,8))
    curr_step = SCMS_Eu_Log_EQ.shape[2]-1
    lon3 = SCMS_Eu_Log_EQ[:,0,curr_step]
    lat3 = SCMS_Eu_Log_EQ[:,1,curr_step]
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,\
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
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
    cs = m2.contour(x, y, d_EQ_Eu, linewidths=3, cmap='hsv')
    plt.title('Earthquake Ridges on the World Map ' + '('\
                + str(curr_step)+ ' Step, h=7.0)')
    fig.savefig('./Figures/Earth_Ridges_Eu_SCMS.pdf')
    
    fig = plt.figure(figsize=(14,8))
    curr_step = SCMS_path_Dir.shape[2]-1
    lon1, lat1, R = cart2sph(*SCMS_path_Dir[:,:,curr_step].T)
    m1 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,\
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
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
    cs = m1.contour(x, y, d_EQ, linewidths=3, cmap='hsv')
    plt.title('Earthquake Ridges on the World Map ' + '('\
              + str(curr_step)+ ' Step, h=0.1)')
    fig.savefig('./Figures/Earth_Ridges_Dir_SCMS.pdf')
    
    print("Save the plots as 'Earth_Ridges_Eu_SCMS.pdf' "\
          "and 'Earth_Ridges_Dir_SCMS.pdf'.\n\n")
