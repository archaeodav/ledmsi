#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 20:33:33 2022

@author: dav
"""

import os
import DataHandler
import board_control

import json

import datetime

from PIL import Image, ExifTags

import sys




class CameraControl():
    ''' class to control the camera'''
    def __init__(self,
                 sys_def=None,
                 calib = None,
                 tempdir = None):
        
        
        self.sys_def = sys_def
        
        self.calib = calib
        
        self.tempdir = tempdir
    
    def init_imgsys(self):
        '''
        method initiates an ImagingSystem object from DataHandler.
        
        This approach is chosen rather than inheriting ImagingSystem or in the
        init method of this class becasue we need to be able to redefine this 
        object after calibration.
        
        
        Paramaters
        -------
        None
        
        Returns
        -------
        None
        
        '''
        
        
        system = DataHandler.ImagingSystem()
        
        self.sys_def = system.sys_def
        
        self.calib =  system.cal
        
        self.ordered = system.wl_ordered
        
        self.lights = board_control.LightArray()
        
    
    
    def preview(self):
        '''
        method turns on the white led and provides a preview image to adjust 
        focus, framing etc
        
        Paramaters
        -------
        
        Returns
        -------
        
        '''
        pass
    
    def calibrate(self,
                  calib_dir = None,
                  exposure_factor = 4,
                  uv_closest = '365'):
        """
        Method performs a calibration and records this in a JSON file.
        Calibration uses auto exposure for each LED wavelength and records the 
        value to file
        
        It's presumed the target will be a white reflector, so the exposure 
        will be increased using the exposure factor
        
        The UV-C & UV-B flourescence leds are outside the camera sensitivity and 
        present a problem as we can't use the camera to perform the calibration. 
        We'll approximate exposure time for these using the output of the shortest 
        visible wavelength and a power factor based on the number of LEDs * wattage
        
        I anticipate using a UV sensor to do this, but this will take a bit of 
        tuning
        
        Paramaters
        -------
        calib_dir : str
            directory to contain the calibration file
            
        exposure_factor : float
            exposure correction 

        Returns
        -------
        str
            calibration file path.

        """
        
        self.calib = {}
        
        uv_later = []
        
        #for wavelength in ordered wavlengths
        for wl in self.ordered:
            if self.sys_def[wl]['method']=='camera':
                
                self.lights.light_on(self.sys_def[wl]['pin'])
                
                oname = os.path.join(self.tempdir,wl+'.jpg')
                camera_command = 'libcamera-still -n -r --metering average --gain 1 -o %s' %(oname)
                
                os.system(camera_command)
                
                self.lights.lights_off()
                
                img = Image.open(oname)
                
                calib_exp_time = (img._getexif()[33434]*1000000)*exposure_factor
                
                print (img._getexif()[33434]*1000000)
                print (wl,calib_exp_time)
                
                self.calib[wl]=int(calib_exp_time)
                
            else:
                if self.sys_def[wl]['method']=='uv':
                    uv_later.append(wl)
                
        for wl in uv_later:
            self.calib[wl]=self.calib[uv_closest]*9
            
                
        # log all this to JSON
        
        if calib_dir is None:
            calib_dir = os.path.join(os.path.dirname(__file__),
                                     'calibrations')
            
            if not os.path.exists(calib_dir):
                os.mkdir(calib_dir)
                
        fname = os.path.join(calib_dir,'calib_'+self.timestring()+'.json')
        
        with open(fname, 'w') as ofile:
            json.dump(self.calib,
                      ofile,
                      sort_keys=True,
                      indent=4,
                      ensure_ascii=False)
            
            ofile.close()
            
            
                
        # reload the system def
        self.init_imgsys()
        
        
    
    
    def acquire_stack(self,
                      odir,
                      fname,
                      auto_id = True):
        #for wl in wavelengths ordered
        
        if auto_id is True:
            ids = os.listdir(os.path.join(odir,fname))
            ids_int = []
            for i in ids:
                try:
                    n = int(i.split('.')[0].split('_')[-1])
                    ids_int.append(n)
                except(ValueError):
                    ids_int.append(-1)
                    
            last_id = max(ids_int)
            
            if not last_id == -1:
                fname= '%s_%s' %(fname,last_id)
            else:
                fname = '%s_0' %(fname)
                        
        
        odata = DataHandler.ImageDict(odir,fname)
        
        odata.init_image_stack()
        
        for wl in self.ordered:
            
            im_name = '%s_%s.jpg' %(fname,wl)
            
            oname = os.path.join(odir,fname,im_name)
            
            #TODO call board control from __init__ rather than here- it
            # takes a few seconds to initiate the connection
            
            
            self.lights.light_on(self.sys_def[wl]['pin'])
            
               
            camera_command = 'libcamera-still -n -r --shutter %s --gain 1 --immediate -o %s' %(str(self.calib[wl]),oname)
            
            os.system(camera_command)
            
            self.lights.lights_off()
            
            odata.image_data(fname,wl)
           
        odata.save_dict()
                
    
    def take_still(self):
        pass
    
    def timestring(self,
                   long = False):
        
        '''
        method returns current date / time as a string for making unique 
        filenames
        
        Paramaters
        -------
        long : bool
            if long include microseconds, else just seconds
        
        Returns
        -------
        str
            datetime string
        
        '''
        
        current_time = datetime.datetime.now()
        
        if long is True:
            ts = current_time.strftime("%Y-%m-%d-%H%M%S-%f")
    
        else:
            ts = current_time.strftime("%Y-%m-%d-%H%M%S")
        
        return ts
        
def main():
    #TODO behaviour for buttons
    # First button calls acquire stack
    # second button does calibration
    
    #TODO toggle switch behaviour
    
    pass
    
    
if __name__ == ('__main__'):
    c = CameraControl(tempdir="/home/dav/temp")
    
    c.init_imgsys()
    
    method = sys.argv[1]
    
    if method == '--c':
        c.calibrate()
        
    elif method == '--s':
        odir = sys.argv[2]
        
        fname = sys.argv[3]
        
        c.acquire_stack(odir, fname)
    
    