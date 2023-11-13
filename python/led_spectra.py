# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:37:19 2023

@author: ds
"""

import os
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import peak_widths



def spectra(file):
    #data = np.loadtxt(file, delimiter=',',skiprows=2)
    data = []
    averaged = []
    with open(file,'r')as ifile:
        next(ifile)
        next(ifile)
        for row in ifile:
            #print (row)
            d = np.array(row.strip('\n').rstrip(',').split(','),
                         dtype=float)
            
            a = np.average(d[1:])
            data.append(d)
            averaged.append(a)
    data = np.array(data)    
    
    norm = (averaged-np.min(averaged))/(np.max(averaged)-np.min(averaged))
        
    return data, averaged, norm

def get_fwhm(data):
   peak = [np.argmax(data)]
   
   fwhm = peak_widths(data,peak)
   
   return fwhm


def load_data(indir=r"C:\Users\ds\OneDrive - Moesgaard Museum\Dokumenter\GitHub\ledmsi\led_spectra\final"):
    data = {}

    x = None
    for file in os.listdir(indir):
        if file.endswith('.csv'):
            d,a,n = spectra(os.path.join(indir,file))
            data[file.split('.')[0]]=n
            print (file,get_fwhm(n)[0])
            if x is None:
                x = d[:,0]
    print (data.keys())
    wavelengths = data.keys()
    

    return wavelengths, x, data


def plot(wl,
         x,
         data):
    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    wl=data.keys()
    
    # Plot on each subplot
    i = 0
    for w in wl:
        axes[i].plot(x, data[w])
        axes[i].set_title(w)
        #axes[i].legend()
        axes[i].set_xlabel('nm')
        axes[i].set_xticks([400,500,600,700,800,900])
        i+=1

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
