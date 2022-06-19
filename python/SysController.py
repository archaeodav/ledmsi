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
        
        
        system = DataHandler.ImagingSystem(self.sys_def,
                                           self.calib)
        
        self.sys_def = system.sysdef
        
        self.calib =  system.calib
        
        self.ordered = system.wl_ordered
        
    
    
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
                lights = board_control.LightArray()
                lights.light_on(self.sys_def[wl]['pin'])
                
                oname = os.path.join(self.tempdir,wl+'.jpg')
                camera_command = 'libcamera-still -n -r --metering average --gain 1 -o %s' %(oname)
                
                os.system(camera_command)
                
                lights.lights_off()
                
                img = Image.open(oname)
                
                calib_exp_time = (img._getexif()[33434]*1000000)*exposure_factor
                
                self.calib[wl]=calib_exp_time
                
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
                      fname):
        #for wl in wavelengths ordered
        
        odata = DataHandler.ImagingSystem(odir, fname)
        
        odata.init_image_stack()
        
        for wl in self.ordered:
            
            oname = '%s_%s.jpg' %(fname,wl)
            
            lights = board_control.LightArray()
            lights.light_on(self.sys_def[wl]['pin'])
            
               
            camera_command = 'libcamera-still -n -r --shutter %s --gain 1 -o %s' %(str(self.calib[wl]),oname)
            
            os.system(camera_command)
            
            lights.lights_off()
            
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
    main()
    