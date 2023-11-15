# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:51:01 2023

@author: ds
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis

def lda(data,
        target,
        feature_names,
        classes):
    
    clf = LinearDiscriminantAnalysis(n_components=2)
    
    #clf.fit(data,target)
    
    lda_x = clf.fit(data,target).transform(data)
    
    colors = ["navy", "turquoise", "darkorange","yellowgreen"]
    
    plt.figure()
    for color, i, target_name in zip(colors,
                                     list(range(0,classes.shape[0])),
                                     classes):
        print (color,i,target_name)
        
        plt.scatter(lda_x[target == i, 0], 
                    lda_x[target == i, 1], 
                    alpha=0.8, 
                    color=color, 
                    label=target_name,
                    s=1)
        plt.legend(loc="best", shadow=False, scatterpoints=2)
        plt.title("LDA full dataset")

    plt.show()
    

def rgb_lda(data,
            target,
            feature_names,
            classes):
    
    clf = LinearDiscriminantAnalysis(n_components=2)
    
    #clf.fit(data,target)
    
    lda_x = clf.fit(data,target).transform(data)
    
    colors = ["navy", "turquoise", "darkorange","yellowgreen"]
    
    plt.figure()
    for color, i, target_name in zip(colors,
                                     list(range(0,classes.shape[0])),
                                     classes):
        print (color,i,target_name)
        
        plt.scatter(lda_x[target == i, 0], 
                    lda_x[target == i, 1], 
                    alpha=0.8, 
                    color=color, 
                    label=target_name,
                    s=1)
        plt.legend(loc="best", shadow=False, scatterpoints=2)
        plt.title("LDA RGB dataset")

    plt.show()
     