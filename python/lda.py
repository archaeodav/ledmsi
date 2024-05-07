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
    
    clf = LinearDiscriminantAnalysis(n_components=3)
    
    #clf.fit(data,target)
    
    lda_x = clf.fit(data,target).transform(data)
    
    
    
    return lda_x



    
    
def plot_both(full_samples,
              rgb_samples):
    
    rgb_lda = lda(rgb_samples[0],
                  rgb_samples[1],
                  rgb_samples[2],
                  rgb_samples[3])
    full_lda = lda(full_samples[0],
                   full_samples[1],
                   full_samples[2],
                   full_samples[3],)
    
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    colors = ["navy", "turquoise", "darkorange","yellowgreen"]
    
    classes = rgb_samples[3]

    for color, i, target_name in zip(colors,
                                     list(range(0,classes.shape[0])),
                                     classes):
        target = rgb_samples[1]

        
        
        axs[0].scatter(rgb_lda[target == i, 0], 
                      rgb_lda[target == i, 1], 
                      alpha=0.8, 
                      color=color, 
                      label=target_name,
                      s=1)
        axs[0].legend(loc="best", shadow=False, scatterpoints=6)
        axs[0].set_title("LDA RGB dataset", fontsize=18)
        axs[0].set_xlabel('First LDA Component',fontsize=14)
        axs[0].set_ylabel('Second LDA Component',fontsize=14)
        axs[0].tick_params(axis='both', which='major', labelsize=14)  
        
    classes = full_samples[3]
    for color, i, target_name in zip(colors,
                                     list(range(0,classes.shape[0])),
                                     classes):
        target = full_samples[1]
        
        
        axs[1].scatter(full_lda[target == i, 0], 
                      full_lda[target == i, 1], 
                      alpha=0.8, 
                      color=color, 
                      label=target_name,
                      s=1)
        axs[1].legend(loc="best", shadow=False, scatterpoints=6)
        axs[1].set_title("LDA Full Dataset", fontsize=18)
        axs[1].set_xlabel('First LDA Component',fontsize=14)
        axs[1].set_ylabel('Second LDA Component',fontsize=14)
        axs[1].tick_params(axis='both', which='major', labelsize=14)  
        
        
    plt.subplots_adjust(hspace=0.25, wspace=0.08)
    

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()
    
    # Show the plot
    #plt.show()
    plt.savefig(r'C:\Users\ds\OneDrive - Moesgaard Museum\titan\article\figures\working\lda.png', dpi=300)
    
    
    
    
'''def lda(data,
        target,
        feature_names,
        classes):
    
    clf = LinearDiscriminantAnalysis(n_components=3)
    
    #clf.fit(data,target)
    
    lda_x = clf.fit(data,target).transform(data)
    
    #colors = ["navy", "turquoise", "darkorange","yellowgreen","hotpink","blueviolet"]
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

    plt.show()'''