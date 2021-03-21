#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: February 23, 2021

Description: This script contains code for directional KDE and subspace 
constrained mean shift (SCMS) algorithms with the von Mises kernel.
"""

import numpy as np
from numpy import linalg as LA
import scipy.special as sp


def DirKDE(x, data, h=None):
    '''
    q-dim directional KDE with the von Mises Kernel
    
    Parameters:
        x: (m,d)-array
            The Eulidean coordinates of m query points on a unit hypersphere, 
            where d=q+1 is the Euclidean dimension of data
    
        data: (n,d)-array
            The Euclidean coordinates of n directional random sample points in 
            the d-dimensional Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then a rule of thumb for 
            directional KDEs with the von Mises kernel in Garcia-Portugues (2013)
            is applied.)
    
    Return:
        f_hat: (m,)-array
            The corresponding directinal density estimates at m query points.
    '''
    n = data.shape[0]  ## Number of data points
    d = data.shape[1]  ## Euclidean Dimension of the data

    ## Rule of thumb for directional KDE
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (d - R_bar ** 2) / (1 - R_bar ** 2)
        if d == 3:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv(d / 2 - 1, kap_hat)**2) / \
                 (n * kap_hat ** (d / 2) * (2 * (d - 1) * sp.iv(d/2, 2*kap_hat) + \
                                  (d+1) * kap_hat * sp.iv(d/2+1, 2*kap_hat)))) ** (1/(d + 3))
    print("The current bandwidth is " + str(h) + ".\n")
    
    if d == 3:
        f_hat = np.mean(np.exp((np.dot(x, data.T)-1)/(h**2))/(2*np.pi\
                        *(1-np.exp(-2/h**2))*h**2), axis=1)
    else:
        f_hat = np.mean(np.exp(np.dot(x, data.T)/(h**2))/((2*np.pi)**(d/2)*\
                           sp.iv(d/2-1, 1/(h**2))*h**(d-2)), axis=1)
    return f_hat


def SCMS_DirKDE(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000):
    '''
    Directional Subspace Constrained Mean Shift Algorithm with von-Mises kernel 
    
    Parameters:
        mesh_0: a (m,D)-array
            The Euclidean coordinates of m directional initial points in the 
            D-dimensional Euclidean space.
    
        data: a (n,D)-array
            The Euclidean coordinates of n directional data sample points in the 
            D-dimensional Euclidean space.
       
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: float
            The bandwidth parameter. (Default: h=None. Then a rule of thumb for 
            directional KDEs with the von Mises kernel in Garcia-Portugues (2013)
            is applied.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the directional SCMS algorithm 
            on each initial point. (Default: max_iter=1000.)
    '''
    
    n = data.shape[0]  ## Number of data points
    D = data.shape[1]  ## Euclidean dimension of data points
    ## Rule of thumb for directional KDE
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (D - R_bar ** 2) / (1 - R_bar ** 2)
        if D == 3:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv(D / 2 - 1, kap_hat)**2) / \
                 (n * kap_hat ** (D / 2) * (2 * (D - 1) * sp.iv(D/2, 2*kap_hat) + \
                                  (D+1) * kap_hat * sp.iv(D/2+1, 2*kap_hat)))) ** (1/(D + 3))
    print("The current bandwidth is " + str(h) + ".\n")
    
    SCMS_path = np.zeros((mesh_0.shape[0], D, max_iter))
    SCMS_path[:,:,0] = mesh_0
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))  
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The directional SCMS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1].reshape(D, 1)
                ## Compute the Hessian matrix
                Hess = np.exp(-1/(h**2))*np.dot(data.T, data*np.exp(np.dot(data, x_pts)/(h**2)))/(h**2) \
                    - np.exp(-1/(h**2))*np.diag(np.sum(np.dot(data, x_pts) \
                                 * np.exp(np.dot(data, x_pts)/(h**2))) * np.ones(len(x_pts),))
                x_pts = x_pts.reshape(len(x_pts), 1)
                proj_mat = np.diag(np.ones(x_pts.shape[0],)) - np.dot(x_pts, x_pts.T)
                Hess = np.dot(np.dot(proj_mat, Hess), proj_mat)
                w, v = LA.eig(Hess)
                ## Obtain the eigenpairs within the tangent space
                tang_eig_v = v[:, (abs(np.dot(x_pts.T, v)) < 1e-8)[0,:]]
                tang_eig_w = w[(abs(np.dot(x_pts.T, v)) < 1e-8)[0,:]]
                V_d = tang_eig_v[:, np.argsort(tang_eig_w)[:(x_pts.shape[0]-1-d)]]
                vtot_grad = np.sum(data*np.exp(np.dot(data, x_pts)/(h**2)), axis=0) \
                            * np.exp(-1/(h**2))
                ## Iterative vector for the directional mean shift algorithm
                ms_v = vtot_grad / LA.norm(vtot_grad)
                ## Subspace constrained gradient and mean shift vector
                SCMS_grad = np.dot(V_d, np.dot(V_d.T, vtot_grad))
                SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                ## SCMS update
                x_new = SCMS_v + x_pts.reshape(x_pts.shape[0], )
                x_new = x_new / LA.norm(x_new)
                if LA.norm(SCMS_grad) < eps:
                    conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    
    if t >= max_iter-1:
        print('The directional SCMS algorithm reaches the maximum number of '\
               'iterations,'+str(max_iter)+', and has not yet converged.')
    return SCMS_path[:,:,:t]


def SCMS_Log_DirKDE(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000):
    '''
    Directional Subspace Constrained Mean Shift algorithm with log density and 
    von-Mises kernel
    
    Parameters:
        mesh_0: a (m,D)-array
            The Euclidean coordinates of m directional initial points in the 
            D-dimensional Euclidean space.
    
        data: a (n,D)-array
            The Euclidean coordinates of n directional data sample points in the 
            D-dimensional Euclidean space.
       
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: float
            The bandwidth parameter. (Default: h=None. Then a rule of thumb for 
            directional KDEs with the von Mises kernel in Garcia-Portugues (2013)
            is applied.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the directional SCMS algorithm 
            on each initial point. (Default: max_iter=1000.)
    '''
    
    n = data.shape[0]  ## Number of data points
    D = data.shape[1]  ## Euclidean dimension of data points
    ## Rule of thumb for directional KDE
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (D - R_bar ** 2) / (1 - R_bar ** 2)
        if D == 3:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv(D / 2 - 1, kap_hat)**2) / \
                 (n * kap_hat ** (D / 2) * (2 * (D - 1) * sp.iv(D/2, 2*kap_hat) + \
                                  (D+1) * kap_hat * sp.iv(D/2+1, 2*kap_hat)))) ** (1/(D + 3))
    print("The current bandwidth is " + str(h) + ".\n")
    
    SCMS_path = np.zeros((mesh_0.shape[0], D, max_iter))
    SCMS_path[:,:,0] = mesh_0
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))  
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The directional SCMS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                x_pts = x_pts.reshape(len(x_pts), 1)
                ## Compute the directional KDE up to a constant
                den_prop = np.sum(np.exp(np.dot(data, x_pts)/(h**2)))
                ## Compute the total gradient of the log density
                vtot_Log_grad = np.sum(data*np.exp(np.dot(data, x_pts)/(h**2)), axis=0) \
                                / ((h**2)*den_prop)
                ## Compute the Hessian of the log density 
                Log_Hess = np.dot(data.T, data*np.exp(np.dot(data, x_pts)/(h**2)))/((h**4)*den_prop) \
                           - np.dot(vtot_Log_grad.reshape(D,1), vtot_Log_grad.reshape(1,D)) \
                           - np.diag(np.sum(np.dot(data, x_pts) * np.exp(np.dot(data, x_pts)/(h**2))) \
                                     * np.ones(D,))/((h**2)*den_prop)
                proj_mat = np.diag(np.ones(D,)) - np.dot(x_pts, x_pts.T)
                Log_Hess = np.dot(np.dot(proj_mat, Log_Hess), proj_mat)
                w, v = LA.eig(Log_Hess)
                ## Obtain the eigenpairs inside the tangent space
                tang_eig_v = v[:, (abs(np.dot(x_pts.T, v)) < 1e-8)[0,:]]
                tang_eig_w = w[(abs(np.dot(x_pts.T, v)) < 1e-8)[0,:]]
                V_d = tang_eig_v[:, np.argsort(tang_eig_w)[:(x_pts.shape[0]-1-d)]]
                ## Iterative vector for the directional mean shift algorithm
                ms_v = vtot_Log_grad / LA.norm(vtot_Log_grad)
                ## Subspace constrained gradient and mean shift vector
                SCMS_grad = np.dot(V_d, np.dot(V_d.T, vtot_Log_grad))
                SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                ## SCMS update
                x_new = SCMS_v + x_pts.reshape(x_pts.shape[0], )
                x_new = x_new / LA.norm(x_new)
                if LA.norm(SCMS_grad) < eps:
                    conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    
    if t >= max_iter-1:
        print('The directional SCMS algorithm reaches the maximum number of '\
              'iterations,'+str(max_iter)+', and has not yet converged.')
    return SCMS_path[:,:,:t]


def SCMS_DirKDE_org(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000):
    '''
    Directional Subspace Constrained Mean Shift algorithm with von-Mises kernel 
    using the original directional mean shift vector in its iterative steps
    
    Parameters:
        mesh_0: a (m,D)-array
            The Euclidean coordinates of m directional initial points in the 
            D-dimensional Euclidean space.
    
        data: a (n,D)-array
            The Euclidean coordinates of n directional data sample points in the 
            D-dimensional Euclidean space.
       
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: float
            The bandwidth parameter. (Default: h=None. Then a rule of thumb for 
            directional KDEs with the von Mises kernel in Garcia-Portugues (2013)
            is applied.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the directional SCMS algorithm 
            on each initial point. (Default: max_iter=1000.)
    '''
    
    n = data.shape[0]  ## Number of data points
    D = data.shape[1]  ## Euclidean dimension of data points
    ## Rule of thumb for directional KDE
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (D - R_bar ** 2) / (1 - R_bar ** 2)
        if D == 3:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv(D / 2 - 1, kap_hat)**2) / \
                 (n * kap_hat ** (D / 2) * (2 * (D - 1) * sp.iv(D/2, 2*kap_hat) + \
                                  (D+1) * kap_hat * sp.iv(D/2+1, 2*kap_hat)))) ** (1/(D + 3))
        print("The current bandwidth is " + str(h) + ".\n")
    
    SCMS_path = np.zeros((mesh_0.shape[0], D, max_iter))
    SCMS_path[:,:,0] = mesh_0
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))  
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The directional SCMS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1].reshape(D, 1)
                ## Compute the Hessian matrix
                Hess = np.exp(-1/(h**2))*np.dot(data.T, data*np.exp(np.dot(data, x_pts)/(h**2)))/(h**2) \
                    - np.exp(-1/(h**2))*np.diag(np.sum(np.dot(data, x_pts) \
                                 * np.exp(np.dot(data, x_pts)/(h**2))) * np.ones(len(x_pts),))
                x_pts = x_pts.reshape(len(x_pts), 1)
                proj_mat = np.diag(np.ones(x_pts.shape[0],)) - np.dot(x_pts, x_pts.T)
                Hess = np.dot(np.dot(proj_mat, Hess), proj_mat)
                w, v = LA.eig(Hess)
                ## Obtain the eigenpairs inside the tangent space
                tang_eig_v = v[:, (abs(np.dot(x_pts.T, v)) < 1e-8)[0,:]]
                tang_eig_w = w[(abs(np.dot(x_pts.T, v)) < 1e-8)[0,:]]
                V_d = tang_eig_v[:, np.argsort(tang_eig_w)[:(x_pts.shape[0]-1-d)]]
                vtot_grad = np.sum(data*np.exp(np.dot(data, x_pts)/(h**2)), axis=0) * np.exp(-1/(h**2))
                ## Mean Shift vector
                ms_v = np.sum(data*np.exp(np.dot(data, x_pts)/(h**2)), axis=0) \
                       / np.sum(np.exp(np.dot(data, x_pts)/(h**2)))
                ## Subspace constrained gradient and mean shift vector
                SCMS_grad = np.dot(V_d, np.dot(V_d.T, vtot_grad))
                SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                ## SCMS update
                x_new = SCMS_v + x_pts.reshape(x_pts.shape[0], )
                x_new = x_new / LA.norm(x_new)
                if LA.norm(SCMS_grad) < eps:
                    conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    
    if t >= max_iter-1:
        print('The directional SCMS algorithm reaches the maximum number of '\
              'iterations,'+str(max_iter)+', and has not yet converged.')
    return SCMS_path[:,:,:t]


def SCMS_Log_DirKDE_org(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000):
    '''
    Subspace Constrained Mean Shift algorithm with the log density and von-Mises kernel for q-dimensional directional data
    
    Parameters
    --------
    mesh_0: a (m,3)-array
       The coordinates of m initial points.
    
    data: a (n,3)-array
       The coordinates of n directional data sample points.
       
    d: int
       The order of the density ridge.
       
    h: float
       The bandwidth parameter.
       
    eps: float
       The precision parameter.
       
    max_iter: int
       The maximum number of iterations for the SCMS algorithm.
    '''
    
    n = data.shape[0]  ## Number of data points
    D = data.shape[1]  ## Euclidean dimension of data points
    ## Rule of thumb for directional KDE
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (D - R_bar ** 2) / (1 - R_bar ** 2)
        if D == 3:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv(D / 2 - 1, kap_hat)**2) / \
                 (n * kap_hat ** (D / 2) * (2 * (D - 1) * sp.iv(D/2, 2*kap_hat) + \
                                  (D+1) * kap_hat * sp.iv(D/2+1, 2*kap_hat)))) ** (1/(D + 3))
        print("The current bandwidth is " + str(h) + ".\n")
    
    SCMS_path = np.zeros((mesh_0.shape[0], D, max_iter))
    SCMS_path[:,:,0] = mesh_0
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))  
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The directional SCMS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                x_pts = x_pts.reshape(len(x_pts), 1)
                ## Compute the directional KDE up to a constant
                den_prop = np.sum(np.exp(np.dot(data, x_pts)/(h**2)))
                ## Compute the total gradient of the log density
                vtot_Log_grad = np.sum(data*np.exp(np.dot(data, x_pts)/(h**2)), axis=0) \
                                / ((h**2)*den_prop)
                ## Compute the Hessian of the log density 
                Log_Hess = np.dot(data.T, data*np.exp(np.dot(data, x_pts)/(h**2)))/((h**4)*den_prop) \
                           - np.dot(vtot_Log_grad.reshape(D,1), vtot_Log_grad.reshape(1,D)) \
                           - np.diag(np.sum(np.dot(data, x_pts) * np.exp(np.dot(data, x_pts)/(h**2))) \
                                     * np.ones(D,))/((h**2)*den_prop)
                proj_mat = np.diag(np.ones(D,)) - np.dot(x_pts, x_pts.T)
                Log_Hess = np.dot(np.dot(proj_mat, Log_Hess), proj_mat)
                w, v = LA.eig(Log_Hess)
                ## Obtain the eigenpairs inside the tangent space
                tang_eig_v = v[:, (abs(np.dot(x_pts.T, v)) < 1e-8)[0,:]]
                tang_eig_w = w[(abs(np.dot(x_pts.T, v)) < 1e-8)[0,:]]
                V_d = tang_eig_v[:, np.argsort(tang_eig_w)[:(x_pts.shape[0]-1-d)]]
                ## Mean Shift vector
                ms_v = np.sum(data*np.exp(np.dot(data, x_pts)/(h**2)), axis=0) \
                       / np.sum(np.exp(np.dot(data, x_pts)/(h**2)))
                ## Subspace constrained gradient and mean shift vector
                SCMS_grad = np.dot(V_d, np.dot(V_d.T, vtot_Log_grad))
                SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                ## SCMS update
                x_new = SCMS_v + x_pts.reshape(x_pts.shape[0], )
                x_new = x_new / LA.norm(x_new)
                if LA.norm(SCMS_grad) < eps:
                    conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    
    if t >= max_iter-1:
        print('The directional SCMS algorithm reaches the maximum number of '
              'iterations,'+str(max_iter)+', and has not yet converged.')
    return SCMS_path[:,:,:t]
