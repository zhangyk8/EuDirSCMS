#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: March 20, 2021

Description: This script contains all the utility functions for our experiments.
"""

import numpy as np
import scipy.special as sp


## Converting Euclidean coordinates to Spherical coordinate and vice versa
def cart2sph(x, y, z):
    '''
    Converting the Euclidean coordinate of a data point in R^3 to its Spherical 
    coordinates.
    
    @ Parameters:
        x, y, z -- Euclidean coordinate in R^3 of a data point.
    @ Returns:
        theta -- Longitude (ranging from -180 degree to 180 degree).
        phi -- Latitude (ranging from -90 degree to 90 degree).
        r -- Radial distance from the origin to the data point.
    '''
    dxy = np.sqrt(x**2 + y**2)
    r = np.sqrt(dxy**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, dxy)
    theta, phi = np.rad2deg([theta, phi])
    return theta, phi, r

def sph2cart(theta, phi, r=1):
    '''
    Converting the Euclidean coordinate of a data point in R^3 to its Spherical 
    coordinates.
    
    @ Parameters:
        theta -- Longitude (ranging from -180 degree to 180 degree).
        phi -- Latitude (ranging from -90 degree to 90 degree).
        r -- Radial distance from the origin to the data point (Default: r=1).
    @ Returns:
        x, y, z -- Euclidean coordinate in R^3 of a data point.
    '''
    theta, phi = np.deg2rad([theta, phi])
    z = r * np.sin(phi)
    rcosphi = r * np.cos(phi)
    x = rcosphi * np.cos(theta)
    y = rcosphi * np.sin(theta)
    return x, y, z


def Eu_Ring_Data(N=200, R=2, sigma=0.05, half=False):
    '''
    Generating data points from a (half) circle with additive Gaussian noises
    
    Parameters:
        N: int
            The number of randomly generated data points.
    
        R: float
            The radius of the circle (true manifold structure).
       
        sigma: float
            The standard deviation of Gaussian noises.
        
        half: bool
            An indicator of whether the hidden manifold is a half or full circle
    
    Return:
        pts_noise: (N,2)-array
            The 2-dim coordinates of N simulated data points.
    '''
    if half:
        ang = np.random.rand(N,1)*np.pi
    else:
        ang = np.random.rand(N,1)*2*np.pi
    pts_c = np.concatenate([R*np.cos(ang), R*np.sin(ang)], axis=1)
    pts_noise = pts_c + sigma * np.random.randn(pts_c.shape[0], pts_c.shape[1])
    return pts_noise


def Cir_Sph_samp(N, lat_c=60, sigma=0.01, pv_ax=np.array([0,0,1])):
    '''
    Generating data points from a circle on the unit sphere with additive Gaussian 
    noises to their Cartesian coordinates plus L2 normalizations
    
    Parameter:
        N: int
            The number of randomly generated data points.
            
        lat_c: float (range: 0-90)
            The latitude of the circle with respect to the pivotal axis.
            
        sigma: float
            The standard deviation of Gaussian noises.
    
        pv_ax: (3,)-array
            The pivotal axis of the circle on the sphere from which the data 
            points are generated (plus noises).
            
    Return:
        pts_c_noise: (N,3)-array
            The Cartesian coordinates of N simulated data points.
    
    '''
    ## Random longitudes with range (-180, 180)
    lon_c = np.random.rand(N,)*360-180
    lat_c = np.ones((N,))*lat_c
    x_c, y_c, z_c = sph2cart(lon_c, lat_c)

    pts_c = np.concatenate((x_c.reshape(len(x_c), 1), 
                            y_c.reshape(len(y_c), 1),
                            z_c.reshape(len(z_c), 1)), axis=1)
    ## Add Gaussian noises
    pts_c_noise = pts_c + sigma * np.random.randn(pts_c.shape[0], pts_c.shape[1])
    ## Standardize the noisy points
    pts_c_noise = pts_c_noise/np.sqrt(np.sum(pts_c_noise**2, axis=1)).reshape(N,1)
    
    ## Rotate the data samples accordingly
    mu_c = np.array([[0,0,1]])
    R = 2*np.dot(pv_ax.reshape(3,1)+mu_c.T, pv_ax.reshape(1,3)+mu_c)/\
        np.sum((mu_c+pv_ax.reshape(1,3))**2, axis=1) - np.identity(3)
    pts_c_noise = np.dot(R, pts_c_noise.T).T
    return pts_c_noise


def Gauss_Mix(n, mu=np.array([[1,1]]), cov=np.diag([1,1]).reshape(2,2,1), prob=[1.0]):
    '''
    Generating data points from a Gaussian mixture model
    
    Parameters:
        n: int
            The number of randomly generated data points.
    
        mu: (m,d)-array
            The means of the Gaussian mixture model with m components.
       
        cov: (d,d,m)-array
            The (d,d)-covariance matrices of the Gaussian mixture model with 
            m components.
            
        prob: a list of floats with length m.
    
    Return:
        data_ps: (n,d)-array
            The Cartesian coordinates of N simulated data points.
    '''
    m = len(prob)   ## The number of mixtures
    d = mu.shape[1]  ## Dimension of the data
    assert (cov.shape[2] == len(prob)), "'cov.shape[2]' and 'len(prob)' "\
    "should be equal."
    inds = np.random.choice(list(range(m)), n, replace=True, 
                            p=np.array(prob)/sum(prob))
    data_ps = np.zeros((n,d))
    for i in range(m):
        data_ps[inds == i,:] = np.random.multivariate_normal(mu[i,:], cov[:,:,i], 
                                                             size=sum(inds == i))
    return data_ps


def vMF_density(x, mu=np.array([[0,0,1]]), kappa=[1.0], prob=[1.0]):
    '''
    q-dimensional von-Mises Fisher density function or its mixture.
    
    Parameters:
        x: (n,d)-array
            The Eulidean coordinates of n query points on a unit hypersphere, 
            where d=q+1 is the Euclidean dimension of data.
    
        mu: a (m,d)-array
            The Euclidean coordinates of the m mean directions for a mixture of 
            von-Mises Fisher densities. (Default: mu=np.array([[0,0,1]]).)
       
        kappa: a list of floats with length m
            The concentration parameters for the mixture of von-Mises Fisher \
            densities. (Default: kappa=[1.0])
       
        prob: a list of floats with length m
            The mixture probabilities. (Default: prob=[1.0])
            
    Return:
        A (n,)-array with the corresponding density value on each query point.
    '''
    assert (mu.shape[1] == x.shape[1] and mu.shape[0] == len(prob)), \
    "The parameter 'x' and mu' should be a (n,d)-array and (m,d)-array, respectively, \
    and 'prob' should be a list of length m."
    assert (len(kappa) == len(prob)), "The parameters 'kappa' and 'prob' should \
    be of the same length."
    d = x.shape[1]   ## Euclidean dimension of the data
    prob = np.array(prob).reshape(len(prob), 1)
    kappa = np.array(kappa)
    dens = kappa**(d/2-1)*np.exp(kappa*np.dot(x, mu.T))/((2*np.pi)**(d/2)*sp.iv(d/2-1, kappa))
    return np.dot(dens, prob)


def vMF_samp(n, mu=np.array([0,0,1]), kappa=1):
    '''
    Randomly sampling data points from a q-dimensional von-Mises Fisher density
    
    Parameters:
        n: int
            The number of sampling random data points.
        
        mu: (d, )-array
            The Euclidean coordinate of the mean directions of the q-dim vMF
            density, where d=q+1. (Default: mu=np.array([0,0,1]).)
            
        kappa: float
            The concentration parameter of the vMF density.
    
    Return:
        data_ps: (n, d)-array
            The Euclidean coordinates of the randomly sampled points from the vMF density.
    '''
    d = len(mu)   ## Euclidean dimension of the data
    data_ps = np.zeros((n,d))
    ## Sample points from standard normal and then standardize them
    sam_can = np.random.multivariate_normal(mean=np.zeros((d,)), cov=np.identity(d), size=n)
    dist_sam = np.sqrt(np.sum(sam_can**2, axis=1)).reshape(n,1)
    sam_can = sam_can/dist_sam

    unif_sam = np.random.uniform(0, 1, n)
    ## Reject some inadequate data points  
    ## (When the uniform proposal density is used, the normalizing constant in 
    ## front of the vMF density has no effects in rejection sampling.)
    mu = mu.reshape(d,1)
    sams = sam_can[unif_sam < np.exp(kappa*(np.dot(sam_can, mu)-1))[:,0],:]
    cnt = sams.shape[0]
    data_ps[:cnt,:] = sams
    while cnt < n:
        can_p = np.random.multivariate_normal(mean=np.zeros((d,)), cov=np.identity(d), size=1)
        can_p = can_p/np.sqrt(np.sum(can_p**2))
        unif_p = np.random.uniform(0, 1, 1)
        if np.exp(kappa*(np.dot(can_p, mu)-1)) > unif_p:
            data_ps[cnt,:] = can_p
            cnt += 1
    return data_ps


def vMF_samp_mix(n, mu=np.array([[0,0,1]]), kappa=[1.0], prob=[1.0]):
    '''
    Randomly sampling data points from a mixture of q-dimensional von-Mises Fisher densities.
    
    Parameters:
        n: int
            The number of sampling random data points.
    
        mu: a (m,d)-array
            The Euclidean coordinates of the m mean directions for a mixture of 
            von-Mises Fisher densities. (Default: mu=np.array([[0,0,1]]).)
       
        kappa: a list of floats with length m
            The concentration parameters for the mixture of von-Mises Fisher \
            densities. (Default: kappa=[1.0])
       
        prob: a list of floats with length m
            The mixture probabilities. (Default: prob=[1.0])
            
    Return:
        data_ps: (n, d)-array
            The Euclidean coordinates of the randomly sampled points from the vMF mixture.
    '''
    m = len(prob)   ## The number of mixtures
    d = mu.shape[1]  ## Euclidean dimension of the data
    assert (len(kappa) == len(prob)), "The parameters 'kappa' and 'prob' should be of the same length."
    inds = np.random.choice(list(range(m)), n, replace=True, p=np.array(prob)/sum(prob))
    data_ps = np.zeros((n,d))
    for i in range(m):
        data_ps[inds == i,:] = vMF_samp(sum(inds == i), mu=mu[i,:], kappa=kappa[i])
    return data_ps


def Unique_Modes(can_modes, tol=1e-4):
    '''
    A helper function: Group the output mesh points from any mode-seeking algorithm 
    into distinct modes and output the corresponding labels for mesh points.
    
    Parameter:
        can_modes: (N,d)-array
            The output d-dimensional mesh points from any mode-seeking algorithm.
            
        tol: float
            The tolerance level for pairwise distances between mesh points 
            (Any pair of mesh points with distance less than this value will be 
            grouped into the same cluster).
    Return: 
        1) A (m,d) array with the coordinates of m distinct modes. 
        2) A (N, ) array with integer labels specifying the affiliation of each mesh point.
'''
    n_modes = can_modes.shape[0]   ## The number of candidate modes
    d = can_modes.shape[1]    ## The dimension of (candidate) modes
    modes_ind = [0]   ## Candidate list of unique modes
    labels = np.empty([n_modes, ], dtype=int)
    labels[0] = 0
    curr_lb = 0   ## The current label indicator
    
    for i in range(1, n_modes):
        flag = None   ## Indicate whether index i should be added to the candidate list of unique modes
        for j in modes_ind:
            if 1-np.dot(can_modes[i,:].reshape(1,d), can_modes[j,:].reshape(d,1)) <= tol:
                flag = labels[j]  # The mode has been existing
        if flag is None:
            curr_lb += 1
            modes_ind.append(i)
            labels[i] = curr_lb
        else:
            labels[i] = flag
    
    return can_modes[modes_ind,:], labels