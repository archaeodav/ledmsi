# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:37:19 2023

@author: ds
"""

import os
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import peak_widths

from scipy.interpolate import make_interp_spline

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

def get_fwhm(data,
             ht = 0.5):
   peak = [np.argmax(data)]
   
   fwhm = peak_widths(data,peak,rel_height=ht)
   
   return fwhm

def extract_peaks(data,
                  wl,
                  exclude = ['940nm','850nm','White']):
    
    out = {}
    
    for k in data.keys():
        if not k in exclude:
            out[k]={}
            peak = get_fwhm(data[k],ht=0.97)
            
            y =data[k][int(peak[2]):int(peak[3])]
            x = wl[int(peak[2]):int(peak[3])]
            
            x_smooth = np.linspace(x.min(), x.max(), 40)
            spl = make_interp_spline(x, y)
            
            out[k]['em']=spl(x_smooth)
            out[k]['x']=x_smooth
            
    return out
    
    
    

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
         data,
         rgb):
    
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(14, 25))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    wl=data.keys()
    
    # Plot on each subplot
    i = 0
    for w in wl:
        if not w == '940nm':
            #axes[i].plot(rgb[0], rgb[1], label='Red', color='red')
            axes[i].fill_between(rgb[0], rgb[1], color='lightcoral',alpha=0.6)
    
            #axes[i].plot(rgb[0], rgb[2], label='Green', color='green')
            axes[i].fill_between(rgb[0], rgb[2], color='lightgreen',alpha=0.6)
    
            #axes[i].plot(rgb[0], rgb[3], label='Blue', color='blue')
            axes[i].fill_between(rgb[0], rgb[3], color='lightblue',alpha=0.6)
            
            y = data[w]
            
            x_smooth = np.linspace(x.min(), x.max(), 150)
            spl = make_interp_spline(x, y,k=3)
            
            
            axes[i].plot(x_smooth, spl(x_smooth),color='black')
            
            axes[i].set_xlim(left=350)
            
            axes[i].set_title(w,fontsize=18)
            #axes[i].legend()
            axes[i].set_xlabel('nm',fontsize=18)
            axes[i].set_ylabel('DN',fontsize=18)
            axes[i].set_xticks([400,500,600,700,800,900])
            axes[i].tick_params(axis='both', which='major', labelsize=16)  

            i+=1

    plt.subplots_adjust(hspace=0.15, wspace=0.08)
    

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()
    
    # Show the plot
    #plt.show()
    plt.savefig(r'C:\Users\ds\OneDrive - Moesgaard Museum\titan\article\figures\working\led_spectra.png', dpi=300)
    
def rgb_plot(peaks,
             file=r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\Rasperberry Pi IMX477R.csv"):
    
    data = np.loadtxt(file,skiprows=1,delimiter=',')
    print (data.shape)
    
    x = data[:,0]
    
    
    data = data[:,1:]
    print (data.shape) 
    
    data = (data-np.min(data))/(np.max(data)-np.min(data))
    print (data)
    
    r = data[:,0]
    
    g1 = data[:,1]
    
    b = data[:,2]
    
    g2 = data[:,3]
    
    plt.plot(x, r, label='Red', color='red')
    plt.fill_between(x, r, color='lightcoral',alpha=0.6)

    plt.plot(x, g1, label='Green', color='green')
    plt.fill_between(x, g1, color='lightgreen',alpha=0.6)

    plt.plot(x, b, label='Blue', color='blue')
    plt.fill_between(x, b, color='lightblue',alpha=0.6)
    
    '''for p in peaks:
        plt.plot(peaks[p]['x'],peaks[p]['em'],color='black')'''
        
    plt.xlim(left=300,right=800)
    
    plt.ylim(bottom=0)

        

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Relative sensitivty (DN)')
    plt.title('Sony IMX477R')
    
    plt.legend()

    # Displaying the plot
    
    plt.tight_layout()
    
    #plt.show()
    plt.savefig(r'C:\Users\ds\OneDrive - Moesgaard Museum\titan\article\figures\working\camera_sensitivity.png', dpi=300)
    
    return x,r,g1,b