#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: December 24, 2021

Description: This script contains the code for plotting the contour lines and 
(principal/subspace constrained) gradient flows of the example function 
(Figure 2 in the paper).
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def p(x,y):
    return np.exp(-(x**2 + y**2-1)**2)/(1/2*(np.pi)**(3/2)*(1 + math.erf(1)))

if __name__ == "__main__":
    # Create the mesh points for contour lines
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = p(X, Y)
    
    plt.figure(figsize=(7,5))
    theta = np.linspace(-np.pi, np.pi, 50)
    plt.plot(np.cos(theta), np.sin(theta), color='black', linewidth=3, label='Ridge')
    plt.contour(X, Y, Z, 8, cmap='YlOrRd')
    plt.colorbar()
    theta2 = np.linspace(-np.pi, np.pi, 15)
    for i in range(len(theta2)-2):
        pt = np.linspace((2-np.sqrt(2))/2, (2+np.sqrt(2))/2, 30)
        plt.plot(pt*np.cos(theta2[i]), pt*np.sin(theta2[i]), color='darkgreen', 
                 linestyle='dashed', alpha=0.7, linewidth=2)
    plt.plot(pt*np.cos(theta2[i+1]), pt*np.sin(theta2[i+1]), color='darkgreen', 
             linestyle='dashed', alpha=0.7, linewidth=2, 
             label='(Principal) gradient directions')
    plt.axis('equal')
    plt.legend()
    plt.ylim(-2,2)
    plt.xlim(-2,2)
    plt.tight_layout()
    plt.savefig('./Figures/Density_Example.pdf')