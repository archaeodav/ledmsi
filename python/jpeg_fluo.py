# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:41:21 2023

@author: ds
"""

from processing import HSVimage
from DataHandler import ImageDict

import matplotlib.pyplot as plt
import os

import numpy as np

def rgb_flou_stack(indir):
    fstack = {}
    
    for f in os.listdir(indir):
        if f.endswith('jpg'):
            h = HSVimage(os.path.join(indir,f))
            him = np.rot90(h.h_diff(), k=3)
            wl = f.split('_')[-1].split('.')[0]
            fstack[wl]=him
            
    return fstack

def fplot(indict):
    for k in indict.keys():
        
