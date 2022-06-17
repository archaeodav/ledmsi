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




class CameraControl():
    ''' class to control the camera'''
    def __init__(self,
                 sys_def=None,
                 calib = None):
        
        
        self.sys_def = sys_def
        
        self.calib = calib
        
    
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
        
        self.layer_def = system.sysdef
        
        self.calib =  system.calib
        
    
    
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
                  exposure_factor = 4):
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
        
        #for wavelength in ordered wavlengths
            # get method
            # if method is camera
                # take a photo
                #get the exposure 
            # else
                # append to a list of UV leds
                # loop through after the others and calculate using closest wl 
                
        # log all this to JSON
        
        if calib_dir is None:
            calib_dir = os.path.join(os.path.dirname(__file__),
                                     'calibrations')
                
        # reload the system def
        self.init_imgsys()
        
        pass
    
    
    def acquire_stack(self):
        #for wl in wavelengths ordered
        
            # get and switch LED
            #get exposure from calib
            
            #set exposure and take photo
            #pass to Data
        
        pass
    
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
    
    
    
    
    
if __name__ == ('__main__'):
    main()
    