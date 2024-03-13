# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:27:30 2024

@author: ds
"""

import matplotlib.pyplot as plt

from skimage.exposure import equalize_hist

import numpy as np

def subset_components(stack,
                      drop_list=None):
    
    out = None
    
    names = []
    
    drop_list = [11,
                 13,
                 14,
                 20,
                 21,
                 26,
                 28,
                 31,
                 34,
                 37]
    
    dims = stack.shape[-1]
    
    for i in range(dims):
        if not i in drop_list:
            if out is None:
                out = stack[:,:,i]
            else:
                out = np.dstack((out,stack[:,:,i]))
            names.append(i+1)
    
    return out, names
    

def false_colour(stack,
                 r,
                 g,
                 b,
                 equalize=True):
    if equalize is True:
        composite = np.dstack((equalize_hist(stack[:,:,r]),
                               equalize_hist(stack[:,:,g]),
                               equalize_hist(stack[:,:,b])))
    else:    
        composite = np.dstack((stack[:,:,r],
                               stack[:,:,g],
                               stack[:,:,b]))


    plt.imshow(composite)
    plt.axis('off')
    plt.show()
    

def get_minmaxpca(pca):
    
    max_eg = np.argmax(pca[1].components_,axis=1)
    min_eg = np.argmin(pca[1].components_,axis=1)
    
    return np.column_stack((max_eg,min_eg))

def mulit_plot(stack,
               rows_cols = (3,4),
               name = 'PCA',
               equalize = True,
               names = None,
               ):
    
    rows,cols = rows_cols
    
    dims = rows*cols
    
    if dims>stack.shape[-1]:
        raise Exception('Too many plots for data')
    
    figs,axs = plt.subplots(rows,
                            cols, 
                            figsize=(10,12))
    
    plt.subplots_adjust(hspace=0.05, wspace=0.05) 
    
    axs = axs.flatten()
    
    for i in range(dims):
        if equalize is True:
            image = equalize_hist(stack[:,:,i])
        else:
            image = stack[:,:,i]
            
        axs[i].imshow(image, cmap='viridis')
        axs[i].axis('off')
        
        if names is None:
            b_name = [i+1]
        else:
            b_name = names[i]

        axs[i].text(0,
                    0, 
                    f'{name} {b_name}', 
                    fontsize=10,
                    color='black',
                    ha='center')
        
    #plt.tight_layout()
    plt.show()