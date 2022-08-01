#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:00:10 2022

@author: dav
"""


import numpy as np

import os

import rawpy

from DataHandler import ImageDict

from sklearn.decomposition import PCA

from skimage.color import rgb2hsv


class RGBimage():
    def __init__(self,
                 image):
        
        self.image = None
        
        if type(image) is str:
            self.image = self.convert_raw(image)
            
        elif type(image) is np.ndarry:
            self.image = image
            
        
    def convert_raw(self,
                    image):
     
         '''
         Method converts 12bit DNG to 16 bit linear tiff.
         
         NOTE: Uses the rawpy library which seems flaky on M1 OSX. As a reuslt we'll
         look at alternatives
         
         Parameters
         -------
         image : str
             path to image
         wl : str
             wavelength designation string
         
         Returns
         -------
         ndarray
         '''
         
         with rawpy.imread(image) as raw:
             rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
             
         
         return rgb
     
class HSVimage():     

    def __init__(self,
                 rgb_array):
        
        self.hsv = self.tohsv(rgb_array)
        
        self.mean_hue = self.meanhsv()
     
    def tohsv(self,
               image):
         
         """
         Method converts image to Hue saturation Value
         
         Parameters
         -------
         image : np.ndarray rgb image
        

         Returns
         -------
         ndarray.

         """
     
         hsv = rgb2hsv(image)
         
         return hsv
     
    def meanhsv(self):

         mean = np.mean(self.hsv[:,:,0])
         
         return mean
     
    def h_diff(self,
               calib_image = None):
         
         if not image is None:
             h = self.hsv(image)
             
         elif not hsv is None:
             h = hsv
             
         else:
              raise Exception('Specify either an RGB or HSV array')
         
         if not calib_imag is None:
             c_h = self.meanhsv(rgb = calib_image)
         else:
             c_h = self.meanhsv(hsv=h)
         
         #atan2(sin(x-y), cos(x-y))
         
         diff = np.arctan2(np.sin(h-c_h),np.cos(h-c_h))
         
         return diff
     
    def h_treshold(self,
                    hsv,
                    method = 'otsu',
                    threshold = None):
         
         h = hsv[:,:,0]
         
         if method == 'otsu':
             pass
         
         elif threshold is not None:
             
             
             
     
    def h_to_rgb(self,
                  hsv_image,
                  threshold):
         pass
     
        
        

class ArrayHandler(ImageDict):
    def __init__(self,
                 odir,
                 fname):
        
        
        super().__init__(odir, fname)
        
        
                
    def gen_image_stack_np(self,
                           save_stack=True):
         
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
         
         stack = None
         
         
         
         for wl in self.wl_ordered:
             im = self.convert_raw(self.im_dict['Images_DNG'][wl],wl)
         
             if stack is None:
                 stack = im
                 
             else:
                 stack = np.dstack((stack,im))
                 
         if save_stack is True:
             npy = self.fname+'.npy'
             np.save(os.path.join(self.img_dir,npy),stack)
             self.im_dict['Numpy_stack']=npy
             self.save_dict()
              
              
         return stack
             

    def load_image_stack(self):
         
         
         '''
         Method loads numpy image stack.
         
         Parameters
         -------
         None
             
             
         Returns
         -------
         ndarray
         
         '''
         npy_file = os.path.join(self.img_dir,self.im_dict['Numpy_stack'])
         stack = np.load(npy_file)
         
         return stack
     
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
                              
     
        
    def animate(self,
                stack = None):
        """
        Method animates the image stack
        """
        pass         

    def monochrome(self):
        """
        Method creates a stack of average luminance from the RGB channels in the
        image stack
        

        Returns
        -------
        None.

        """
        
        pass
    
    def get_im_dims(self,
                    stack):
        x,y = stack.shape()[0:2]
        
        return x,y
        
    
    def stack_pca(self,
                  stack,
                  n_comp = 9):
        
    
        
        pca = PCA(n_components=n_comp)
        
        x = self.reshape_stack(stack)
        
        pca.fit(x)
        
        predict = pca.transform(x)
        
        out = self.reshaped_to_rast(stack,n_comp,predict)
        
        return out
    
    

class Results(ArrayHandler):
    def __init__(self,
                 odir,
                 fname):
        super().__init__(odir, fname)
        
    def make_plot(self):
        pass
    
    def plot_h_diff(self):
        pass
        
    def plot_pca(self):
        pass
    
    def save_plot(self):
        pass
    
    
    
    