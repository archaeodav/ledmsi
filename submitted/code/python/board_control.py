#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:47:58 2022

@author: dav
"""

import pyfirmata
import json
import time
import os

class LightArray():
    ''' Class controls light array '''
    def __init__(self,
                 device='/dev/ttyUSB0', #location of the device
                 array_definition=None #array definition dict
                 ):
        '''

        Parameters
        ----------
        device : string, optional
            DESCRIPTION. The default is '/dev/ttyUSB0'.
            array_definition : str, optional
            DESCRIPTION. The default is None
        Returns
        -------
        None.

        '''

        #set the board
        self.board = pyfirmata.ArduinoMega(device)



        # define array defintion dict
        self.array_definition = array_definition

        self.set_all_high()

        # if there is none load default file from the script directory
        '''if self.array_definition is None:
            self.load_array_def()'''

    def set_all_high(self,
                     r=(20,50)):
        '''
        Turns off all LEDS by setting all output pins to high

        Parameters
        ----------
        r : tuple, optional
            DESCRIPTION. Range of pins to set to high. The default is (20,50).

        Returns
        -------
        None.

        '''

        for i in range(r[0],r[1]):
            self.board.digital[i].write(1)

    def load_array_def(self,
                       infile=None):
        '''
        Loads array definition file

        Parameters
        ----------
        infile : str, optional
            DESCRIPTION. Path to array definition json. The default is None.

        Returns
        -------
        None.

        '''


        if infile is None:
            infile = os.path.join(os.path.dirname(__file__),'array_definition.json')

        with open(infile,'r') as i:
            self.array_definition = json.load(i)

    def test(self,
             pins = [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]):
        '''
        Test method that turns on each LED for 1 second

        Parameters
        ----------
        pins : list, optional
            DESCRIPTION. The default is [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45].

        Returns
        -------
        None.

        '''

        for i in pins:
            print (i)
            self.light_on(i)
            time.sleep(1)
            self.lights_off()

    def light_on(self,
                 pin):
        '''
        turn led on by setting aclled pin to low

        Parameters
        ----------
        pin : int
            DESCRIPTION. Which Pin?

        Returns
        -------
        None.

        '''
        #method to turn a light on
        self.board.digital[pin].write(0)

    def lights_off(self):
        '''
        Turns all leds off by calling self.set_all_high()

        Returns
        -------
        None.

        '''
        self.set_all_high()






