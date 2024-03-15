# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 08:16:52 2024

@author: ds
"""

import numpy as np

import os

import rawpy

from DataHandler import ImageDict

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import fastica

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler

from skimage import io

from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage.color import rgb2gray

from skimage.exposure import equalize_hist

from skimage.filters import threshold_otsu
from skimage.filters import threshold_mean
from skimage.filters import threshold_li
from skimage.measure import pearson_corr_coeff

from skimage.draw import polygon2mask

import json

from processing import RGBimage




class FluoStack(ImageDict):
    def __init__(self,
                 odir,
                 fname):
        
       super().__init__(odir, fname)
       
       self.load_imdict()
       
   
    def gen_fluo_stack_np(self,
                          save_stacks=True,
                          rotate=3):
         
         '''
         Method converts images to a numpy array and saves them as a *.npy file,
         saves a pointer to this array in the image dict
         
         Parameters
         -------
         save_stack : bool
             save the image stack to disk?
             
         Returns
         -------
         ndarray
         
         '''
         
         hstack = None
         sstack = None
         vstack = None
         
         hdiff = None
         

         
         for wl in self.wl_ordered:
             
             print (self.odir, self.fname)

             image = '%s_%s.dng' %(self.fname,wl)
             
             image = os.path.join(self.odir,self.fname,image)
             
             im = RGBimage(image).image
             
             im = self.rescale(im)
             
             hsvim = rgb2hsv(im)
             
             him = hsvim[:,:,0] 
             sim = hsvim[:,:,1]
             vim = hsvim[:,:,2]
             
             himdiff = self.h_diff(him)
             
             if hstack is None:
                 hstack = him
                 
             else:
                 hstack = np.dstack((hstack,him))
                 
             if sstack is None:
                 sstack = sim
                 
             else:
                 sstack = np.dstack((sstack,sim))
                 
             if vstack is None:
                 vstack = vim
                 
             else:
                 vstack = np.dstack((vstack,vim))
                 
             if hdiff is None:
                 hdiff = himdiff
                
             else:
                 hdiff = np.dstack((hdiff,himdiff))
                 
                 
          
         if rotate > 0:
             hstack = np.rot90(hstack,rotate)
             sstack = np.rot90(sstack,rotate)
             vstack = np.rot90(vstack,rotate)
             hdiff = np.rot90(hdiff,rotate)

          
         if save_stacks is True:
            self.save_stack(hstack, 'hue')
            self.save_stack(sstack, 'sat')
            self.save_stack(sstack, 'val')
            self.save_stack(hdiff, 'hue_diff')
              
              
         return hstack,vstack,sstack,hdiff
     
    def save_stack(self,stack,stack_name):
         npy = '%s_%s%s' %(self.fname,stack_name,'.npy')
         np.save(os.path.join(self.img_dir,npy),stack)
         
    def mean_hue(self,h):

         mean = np.mean(h)
         
         return mean
     
    def h_diff(self,
               h,
               calib_image = None):
         
         
        
         c_h = self.mean_hue(h)
         
         #atan2(sin(x-y), cos(x-y))
         
         diff = np.arctan2(np.sin(h-c_h),np.cos(h-c_h))
         
         return np.abs(diff)

    def reshape_stack(self,
                      stack):
        
        reshaped = np.reshape(stack,
                              ((stack.shape[0]*stack.shape[1]),
                              stack.shape[2]))
        
        return reshaped
    
    def reshaped_to_rast(self,
                         stack,
                         dims,
                         reshaped):
        
        rast = np.reshape(reshaped,
                          (stack.shape[0],
                           stack.shape[1],
                           dims))
        
        return rast
    
    def rescale(self,
                rgb_image):
        
        #print (rgb_image.shape)
        
        
        #reshaped = self.reshape_stack(rgb_image)
        
        #print (reshaped.shape)
        
        
        out = None
        
        for band in range(rgb_image.shape[-1]):
            
            scaler = MinMaxScaler(feature_range=(0,255))
            data = scaler.fit_transform(rgb_image[:,:,band])
            
            if out is None:
                out = data
                
            else:
                out = np.dstack((out,data))
        
        #out = self.reshaped_to_rast(rgb_image, 3, out)
        
        return out
