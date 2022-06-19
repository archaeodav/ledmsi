# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:21:39 2022

@author: adav
"""

import os

import json

import numpy as np

import glob

#import rawpy


#import board_control


class ImagingSystem():
    def __init__(self,
                 system_definition=None,
                 calibration=None):
        
        '''Init method loads the imaging system defintion, and the most recent 
        calibration file for the system'''
        
        # if no defintion is provided load a default from the current directory
        if system_definition is None:
            system_definition = os.path.join(os.path.dirname(__file__),'system_definition.json')
            
        # if no calibration exists load the most recent from the default calibration directory
        if calibration is None:
            calib_dir = os.path.join(os.path.join(os.path.dirname(__file__),'calibrations'))
            
            if os.path.exists(calib_dir):
            
                files = glob.glob(os.path.join(calib_dir,'*.json'))
            
                calibration = max(files, key=os.path.getctime)
                
                self.cal = self.load_defs(calibration)
                
            else:
                self.cal = None
            
        self.sys_def = self.load_defs(system_definition)
        
        
        self.wl_ordered = self.sysdef['OrderedWavelengths']
        
        
        
    def load_defs(self,
                  json_file):
        
        # method to load JSON files
        with open(json_file, 'r') as infile:
            odict = json.load(infile)
            
        return odict

class ImageDict(ImagingSystem):
    def __init__(self,
                 odir,
                 fname_root,
                 defs = None,
                 calib = None):
        
        '''
        Method initiates to the ImageDict object. First intites the system 
        defintion as we'll need this to set up the data structure, then sets up 
        the output directories and filenames
        
        Parameters
        -------
        odir: str
            path of output directory
        fname_root: str
            file name for image stack. uses this to create the directory to 
            contain the image stack if it doesn't exist'
        defs : str
            layer defintion json file
        calib: str
            calibration directory file
            
        Returns
        -------
        None
        
        '''
        
        #intitiate ImagingSystem object                
        super().__init__(defs,calib)
        
        self.odir = odir
        
        self.fname = fname_root
        
        self.img_dir = os.path.join(self.odir,self.fname)
        
        if not os.path.exists(self.fname):
            os.mkdir(self.img_dir)
            
        self.im_dict = {}
        
        self.imdict_json = os.path.join(self.img_dir,self.fname+'.json')
        
    def load_imdict(self):
        '''
        Method loads im_dict from disk.
        
        Parameters
        -------
        None
            
        Returns
        -------
        None 
        '''
        
        with open(self.imdict_json, 'r') as infile:
            self.im_dict = json.load(infile)
    
    def save_dict(self):
        '''
        Method saves im_dict to a json file.
        
        Parameters
        -------
        None
            
        Returns
        -------
        None
        
        '''
        
        with open(self.imdict_json, 'w') as ofile:
            json.dump(self.im_dict,
                      ofile,
                      sort_keys=True,
                      indent=4,
                      ensure_ascii=False)
            
            ofile.close()
    
            
    def init_image_stack(self):
        
        '''
        Method initiates a new im_dict 
        
        Parameters
        -------
        None
            
        Returns
        -------
        None
        
        '''
        
        self.im_dict['Image directory']=self.img_dir
        self.im_dict['Calibration']=self.cal
        self.im_dict['Images_DNG']={}
        self.im_dict['Images_JPG']={}
        
    
    def image_data(self,
                   image_name,
                   image_wl):        
        
        '''
        Method appends image to the ImageDict object.
        
        Parameters
        -------
        image_name: str
            filename of the saved image
        image_wl: str
            wavelength designation of the image
            
        Returns
        -------
        None
        
        '''
        
        im_name_root = image_name.split('.')[0]
        
        
        self.im_dict['Images_DNG'][image_wl]=im_name_root+'.dng'
        
        self.im_dict['Images_JPG'][image_wl]=im_name_root+'.jpg'
    
   
    
    """   
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
             
             
        return stack
            
            
    
    def convert_raw(self,
                    image,
                    wl):
    
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
    """