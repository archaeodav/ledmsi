# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:41:21 2023

@author: ds
"""

from processing import HSVimage
from DataHandler import ImageDict

import matplotlib.pyplot as plt
import os

import numpy as np

from skimage.exposure import equalize_hist


def rgb_flou_stack(indir):
    fstack = {}
    
    for f in os.listdir(indir):
        if f.endswith('jpg'):
            h = HSVimage(os.path.join(indir,f))
            him = np.rot90(h.h_diff(), k=3)
            wl = f.split('_')[-1].split('.')[0]
            fstack[wl]=him
            
    return fstack

def f_plot(indict,
           rows_cols = (4,4),
           name = 'PCA',
           equalize = True,
           names = None,
           ):
    
    rows,cols = rows_cols
    
    dims = rows*cols
    
    names = list(indict.keys())
    
    if dims>len(indict.keys()):
        raise Exception('Too many plots for data')
    
    figs,axs = plt.subplots(rows,
                            cols, 
                            figsize=(10,12))
    
    plt.subplots_adjust(hspace=0.05, wspace=0.05) 
    
    axs = axs.flatten()
    
    for i in range(dims):
        k = names[i]
        
        if equalize is True:
            image = equalize_hist(indict[k])
        else:
            image = indict[k]
            
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