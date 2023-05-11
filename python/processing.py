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
from sklearn.decomposition import KernelPCA


from skimage import io

from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage.color import rgb2gray

from skimage.exposure import equalize_hist

from skimage.filters import threshold_otsu
from skimage.filters import threshold_mean
from skimage.filters import threshold_li

class RGBimage():
    def __init__(self,
                 image):
        
        self.image = None
        
        if type(image) is str:
            if image.endswith('.dng'):
                self.image = self.convert_raw(image)
            elif image.endswith('.jpg') or image.endswith('.tif'):
                self.image = io.imread(image)
                
            
        elif type(image) is np.ndarray:
            self.image = image
            
        print (self.image.shape)
            
        
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
        
        self.rgb = RGBimage(rgb_array).image
        
        self.hsv = self.tohsv(self.rgb)
        
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
         
         print (image)
     
         hsv = rgb2hsv(image)
         
         return hsv
     
    def meanhsv(self):

         mean = np.mean(self.hsv[:,:,0])
         
         return mean
     
    def h_diff(self,
               calib_image = None):
         
         
         if not calib_image is None:
             calib_image = RGBimage(calib_image).image 
             c_h = HSVimage(calib_image).mean_hue
             
         else:
             c_h = self.mean_hue
         
         #atan2(sin(x-y), cos(x-y))
         
         diff = np.arctan2(np.sin(self.hsv[:,:,0]-c_h),np.cos(self.hsv[:,:,0]-c_h))
         
         return np.abs(diff)
     
    def h_threshold(self,
                    h,
                    method = 'otsu',
                    threshold = None):
        
        if method == 'otsu':
             threshold = threshold_otsu(h)
             
        elif method == 'mean':
            threshold = threshold_mean(h)
            
        elif method == 'li':
            threshold = threshold_li(h)
            
        elif threshold is not None:
             threshold = threshold
             
        return threshold
     
    def h_to_rgb(self,
                  hsv_image,
                  threshold):
        
        rgb = hsv2rgb(hsv_image)
        
        return rgb
    
    def fluo_image(self):
        
        # desaturate rgb
        desaturated = rgb2gray(self.rgb)
        
        
        diff = self.h_diff()
        
        thresh = self.h_threshold(diff)
        
        fl  = diff > thresh
        '''
        gr = desaturated
        
        gg = desaturated
        
        gb = desaturated
        '''
        
        g = np.zeros(self.rgb.shape,dtype=np.uint8)
        
        gr = g[:,:,0]
        gg = g[:,:,1]
        gb = g[:,:,2]
        
        r = self.rgb[:,:,0]
        
        g = self.rgb[:,:,1]
        
        b = self.rgb[:,:,2]
        
        gr[fl]=r[fl]
        
        gg[fl]=g[fl]
        
        gb[fl]=b[fl]
        
        fluo = np.dstack((gr,gg,gb))
        
        return fluo
     
class FluoStack():
    def __init__(self,
                 indir):
        
        flou_dict = []
        
        for file in indir:
            band_name = file.split('_')[-1].split('.')[0]
        

class ArrayHandler(ImageDict):
    def __init__(self,
                 odir,
                 fname):
        
        
        super().__init__(odir, fname)
        
        self.load_imdict()
        
        
        
                
    def gen_image_stack_np(self,
                           save_stack=True,
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
         
         stack = None
         
         
         
         for wl in self.wl_ordered:
             
             print (self.odir, self.fname)
            
              
            
             image = '%s_%s.dng' %(self.fname,wl)
             
             image = os.path.join(self.odir,self.fname,image)
            
             #im = self.convert_raw(self.im_dict['Images_DNG'][wl],wl)
             
             print (image)
             
             im = RGBimage(image).image
             
             if stack is None:
                 stack = im
                 
             else:
                 stack = np.dstack((stack,im))
          
         if rotate > 0:
             stack = np.rot90(stack,rotate)
          
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

    def monochrome(self,
                   stack):
        """
        Method creates a stack of average luminance from the RGB channels in the
        image stack
        

        Returns
        -------
        None.

        """
        
        mono = np.average(stack,axis = 2)
        
        return mono
    
    def get_im_dims(self,
                    stack):
        x,y = stack.shape()[0:2]
        
        return x,y
    
    def compositor(self,
                   stack,
                   bands):
        
        out  = None
        
        for b in bands:
            band =  equalize_hist(stack[:,:,b])
            
            if out is None:
                out = band
            else:
                out = np.dstack((out,band))
                
        out = self.monochrome(out)
        
        return out
     
        
    def fluo_comp(self,
                  stack,
                  uv_r=[45,42],
                  b_g=[37,34,31],
                  g_rb=[26,22],
                  r_all=[19,16,13],
                  ir_all=[10,11,7]):
       
        nir_comp = self.compositor(stack,ir_all)
        r_comp = self.compositor(stack,r_all)
        g_comp = self.compositor(stack,g_rb)
        b_comp = self.compositor(stack,b_g)
        uv_comp = self.compositor(stack,uv_r)
        
        return nir_comp,r_comp,g_comp,b_comp,uv_comp
        
    
    
    def bin_refl_composites(self,
                            stack,
                            nir=[3,5,6,8,9,],
                            r=[12,15,18,21],
                            g=[25,28,22],
                            b=[32,35,38,41],
                            uv=[44,47]):
        
        nir_comp = self.compositor(stack,nir)
        r_comp = self.compositor(stack,r)
        g_comp = self.compositor(stack,g)
        b_comp = self.compositor(stack,b)
        uv_comp = self.compositor(stack,uv)
        
        return nir_comp,r_comp,g_comp,b_comp,uv_comp
    
    def stack_pca(self,
                  stack,
                  n_comp = 25):
        
    
        
        pca = PCA(n_components=n_comp)
        
        x = self.reshape_stack(stack)
        
        pca.fit(x)
        
        predict = pca.transform(x)
        
        cov = pca.get_covariance()
        
        out = self.reshaped_to_rast(stack,n_comp,predict)
        
        return out, cov
    
    def stack_kpca(self,
                   stack,
                   n_comp = 25):
        
    
        
        pca = KernelPCA(n_components=n_comp)
        
        x = self.reshape_stack(stack)
        
        pca.fit(x)
        
        predict = pca.transform(x)
        
        cov = pca.get_covariance()
        
        out = self.reshaped_to_rast(stack,n_comp,predict)
        
        return out, cov
    
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
    
    
    
    
    