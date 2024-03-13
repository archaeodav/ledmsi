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

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import fastica

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
             #rgb = raw.postprocess(gamma=(2.2,4.5), output_bps=16)
         
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
            
             #im = self.convert_raw(self.im_dict['Images_DNG'][wl],wl)
             
             print (image)
             
             im = RGBimage(image).image
             
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
                 
         print(stack.shape)
          
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
    
    def stack_ica(self,
                  stack,
                  n_components=None):
        
        if n_components is None:
            n_components = stack.shape[-1]
        
        
        K, W, S = fastica(self.reshape_stack(stack),
                          n_components=n_components,
                          whiten='unit-variance')
        
        im = self.reshaped_to_rast(stack, 
                                   n_components,
                                   S)
        
        return im, K, W
    
    def stack_pca(self,
                  stack,
                  n_components=None):
        
        if n_components is None:
            n_components = stack.shape[-1]
        
        pca = PCA(n_components=n_components)
        
        x = self.reshape_stack(stack)
        
        pca.fit(x)
        
        predict = pca.transform(x)
        
        cov = pca.get_covariance()
        
        out = self.reshaped_to_rast(stack,n_components,predict)
        
        return out, pca
    
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
    
    
    def pearsons(self,
                 stack):
        
        bands = stack.shape[2]
        
        pearsons_matrix_s=None
        pearsons_matrix_p=None
        
        for i in range(bands):
            s = []
            p = []
            for j in range(bands):
                pcc = pearson_corr_coeff(stack[:,:,i], stack[:,:,j])
                s.append(pcc[0])
                p.append[pcc[1]]
            s = np.ndarray(s)
            p = np.ndarray(p)
            
            if pearsons_matrix_s is None:
                pearsons_matrix_s = s
            else:
                pearsons_matrix_s = np.vstack((pearsons_matrix_s,s))
                
            
            if pearsons_matrix_p is None:
                pearsons_matrix_p = p
            else:
                pearsons_matrix_p = np.vstack((pearsons_matrix_p,p))
                
        return np.dstack((pearsons_matrix_s,pearsons_matrix_s))
    
    #def plot_pca_cov()

class SampleMasks():
    def __init__(self):
        self.masks = {}
        self.im_dir = None
        self.samples = {}
        
    def load_masks_from_json(self,
                             json_file,
                             im_dir,
                             img_col_key='_via_img_metadata'):
        
        json_dict = None
        
        self.im_dir = im_dir
        
        with open(json_file,'r') as infile:
            json_dict = json.load(infile)
            
            
        for k in json_dict[img_col_key].keys():
            img_key = k.split('.')[0]+'.jpeg'
            regions = json_dict[img_col_key][k]['regions']
            print (regions)
            
            if not k in self.masks:
                self.masks[img_key]={}
            
            for r in regions:
               
                
                geom_type = r['shape_attributes']['name']
                rclass = r['region_attributes']['Type'].strip('\n')
                
                print(rclass)
                
                if geom_type == 'rect':
                    x = r['shape_attributes']['x']
                    y = r['shape_attributes']['y']
                    w = r['shape_attributes']['width']
                    h = r['shape_attributes']['height']
                    
                    '''poly = [(x,y),
                            (x+w,y),
                            (x+w,y+h),
                            (x,y+h)]'''
    
                    poly = [(y,x),
                            (y,x+w),
                            (y+h,x+w),
                            (y+h,x)]
                    
                elif geom_type == 'polygon':
                    poly = []
                    
                    x = r['shape_attributes']['all_points_x']
                    y = r['shape_attributes']['all_points_y']
                    
                    for i in range(0,len(x)-1):
                        poly.append((y[i],x[i]))
                
                    print('poly')
                
                else:
                    print ('ELSE')
                    
                if not rclass+'_polys' in self.masks[img_key]:
                    self.masks[img_key][rclass+'_polys']=[]
                
                
                self.masks[img_key][rclass+'_polys'].append(poly)
                
                
                
    def sample_masks(self,
                     mask):
        
        for img in os.listdir(self.im_dir):
            print (img)
            if img.endswith('jpeg'):
                
                im = os.path.join(self.im_dir,img)
                dims = io.imread(im).shape
                
                mask_keys = list(self.masks[img].keys())
                print (mask_keys)
                
                for c in mask_keys:
                    individual_masks = []
                    mask_class = c.split('_')[0]
                    mask = np.zeros(dims[0:2],dtype=bool)
                    for poly in self.masks[img][c]:
                        individual_masks.append(polygon2mask(dims[0:2],
                                                poly))
                    for m in individual_masks:
                        print(m.shape)
                        mask = mask+m
                    self.masks[img][mask_class]=mask
                    print (c,mask.shape)
                    
    
    def sampler(self,
                image_mask_names, #list of tuples containg mask name, path to image 
                ):
        for m, image in image_mask_names:
            mask = os.path.split(m)[-1]
            if image.endswith('npy'):
                img = np.load(image)
            else:
                img = io.imread(image)
                    
            for mclass in self.masks[mask]:
                if not mclass.endswith('_polys'):
                    sample = img[self.masks[mask][mclass]]
                    if not mclass in self.samples:
                        self.samples[mclass]=sample
                        
                    else:
                        self.samples[mclass]=np.vstack((self.samples[mclass],
                                                        sample))
        return self.samples
    
    def sample_prep(self,
                    classes=[],
                    n_samples = 10000,
                    feature_names=["R White",
                                   "G White",
                                   "B White"
                                   "R 940nm",
                                   "G 940nm",
                                   "B 940nm",
                                   "R 850nm",
                                   "G 850nm",
                                   "B 850nm",
                                   "R 740nm",
                                   "G 740nm",
                                   "B 740nm",
                                   "R 660nm",
                                   "G 660nm",
                                   "B 660nm",
                                   "R 630nm",
                                   "G 630nm",
                                   "B 630nm",
                                   "R 605nm",
                                   "G 605nm",
                                   "B 605nm",
                                   "R 590nm",
                                   "G 590nm",
                                   "B 590nm",
                                   "R 525nm",
                                   "G 525nm",
                                   "B 525nm",
                                   "R 505nm",
                                   "G 505nm",
                                   "B 505nm",
                                   "R 480nm",
                                   "G 480nm",
                                   "B 480nm",
                                   "R 470nm",
                                   "G 470nm",
                                   "B 470nm"
                                   "R 450nm",
                                   "G 450nm",
                                   "B 450nm",
                                   "R 410nm",
                                   "G 410nm",
                                   "B 410nm",
                                   "R 395nm",
                                   "G 395nm",
                                   "B 395nm",
                                   "R 365nm",
                                   "G 365nm",
                                   "B 365nm"]):
        
        data = None
        
        target = None
        
        feature_names = feature_names
        
        if len(classes)==0:
            classes = self.samples.keys()
            
        target_no = 0
        
        classes = np.array(classes)
        feature_names = np.array(feature_names)
            
        for c in classes:
            if n_samples is not None:
                rng = np.random.default_rng()
                sample = rng.choice(self.samples[c],
                                    n_samples,
                                    replace = False)
                
            else:
                sample = self.samples[c]
                
            t = np.full(sample.shape[0],
                        target_no,
                        dtype=int)
            if data is None:
                data = sample
                print (data.shape)
                target = t
            
            else:
                data = np.vstack((data,sample))
                print ('t',t.shape,'target',target.shape)
                target = np.concatenate((target,t),axis=None)
                
                
            target_no +=1
                
        return data,target,feature_names,classes
                

    def hists(self):
        pass
    
    def lda_plot(self):
        pass
        
    def hist_plot(self):
        pass
    
    def test(self):
        self.load_masks_from_json(r"C:\Users\ds\Downloads\Titan(2).json",
                               r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks")
        self.sample_masks(r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks")
        self.sampler([(r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks\watts_no_filter_1_comp_rir_gg_buv.jpeg",r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\watts_no_filter_1\watts_no_filter_1.npy"),
                   (r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks\watts_no_filter_4_comp_rir_gg_buv.jpeg",r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\watts_no_filter_4\watts_no_filter_4.npy"),
                   (r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks\watts_no_filter_6_comp_rir_gg_buv.jpeg",r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\watts_no_filter_6\watts_no_filter_6.npy")])
        
    def rgb_test(self):
        self.load_masks_from_json(r"C:\Users\ds\Downloads\Titan(2).json",
                               r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks")
        self.sample_masks(r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks")
        self.sampler([(r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks\watts_no_filter_1_comp_rir_gg_buv.jpeg",r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\watts_no_filter_1_comp_rr_gg_bb.tif"),
                   (r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks\watts_no_filter_4_comp_rir_gg_buv.jpeg",r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\watts_no_filter_4_comp_rr_gg_bb.tif"),
                   (r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks\watts_no_filter_6_comp_rir_gg_buv.jpeg",r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\watts_no_filter_6_comp_rr_gg_bb.tif")])
        
    def fluo_test(self):
        self.load_masks_from_json(r"C:\Users\ds\Downloads\Titan(2).json",
                               r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks")
        self.sample_masks(r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks")
        self.sampler([(r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks\watts_no_filter_1_comp_rir_gg_buv.jpeg",r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\watts_no_filter_1\watts_no_filter_1_hue_diff.npy"),
                   (r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks\watts_no_filter_4_comp_rir_gg_buv.jpeg",r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\watts_no_filter_4\watts_no_filter_4_hue_diff.npy"),
                   (r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\masks\watts_no_filter_6_comp_rir_gg_buv.jpeg",r"C:\Users\ds\OneDrive - Moesgaard Museum\titan\sm\No_filter\watts_no_filter_6\watts_no_filter_6_hue_diff.npy")])
        
        
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


def run_comps(indir):
    for d in os.listdir(indir):
        print (d)
        if os.path.isdir(os.path.join(indir,d)):
            a = ArrayHandler(indir, d)
            print(a)
            stack = a.gen_image_stack_np(save_stack=True)
            print(stack)
            
            refl = a.bin_refl_composites(stack,r=[15,18,21])
            fluo = a.fluo_comp(stack)
            
            reflc=np.dstack((equalize_hist(refl[1]),
                            equalize_hist(refl[2]),
                            equalize_hist(refl[3])))
            
            uvirc=np.dstack((equalize_hist(refl[0]),
                            equalize_hist(refl[2]),
                            equalize_hist(refl[4])))
            
            fluoc=np.dstack((equalize_hist(fluo[0]),
                            equalize_hist(refl[2]),
                            equalize_hist(fluo[4])))
            
            print (reflc,uvirc,fluoc)
            
            io.imsave(os.path.join(indir,d+'_comp_rr_gg_bb.tif'), 
                      reflc)
            
            io.imsave(os.path.join(indir,d+'_comp_rir_gg_buv.tif'), 
                      uvirc)
            
            io.imsave(os.path.join(indir,d+'_comp_rrirf_gg_bbuvf.tif'), 
                      fluoc)
            
            del(a)