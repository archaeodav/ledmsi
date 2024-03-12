# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:27:30 2024

@author: ds
"""

import matplotlib.pyplot as plt

from skimage.exposure import equalize_hist

def subset_components(stack,
                      ):
    
    pass


def mulit_plot(stack,
               rows_cols = (3,4),
               name = 'PCA',
               equalize = True
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

        axs[i].text(0,
                    0, 
                    f'{name} {i+1}', 
                    fontsize=10,
                    color='black',
                    ha='center')
        
    #plt.tight_layout()
    plt.show()