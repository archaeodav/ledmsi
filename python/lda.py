# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:51:01 2023

@author: ds
"""

import processing

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis

from pathlib import Path

root_dir = Path.cwd().parents[1]

outdir = root_dir / 'output'

samples = root_dir / 'data' / 'images' / 'samples' / 'Titan_samples.json'

maskdir = root_dir / 'data' / 'images' / 'masks'


def lda(data,
        target,
        feature_names,
        classes):
    '''
    Performs linear discriminant analysis using skimage

    Parameters
    ----------
    data : ndarray
        data
    target : ndarray
        class values for data.
    feature_names : ndarray
        array of strings. names of bands.
    classes : ndarray
        array of strings. class names

    Returns
    -------
    lda_x : ld
        DESCRIPTION.

    '''

    clf = LinearDiscriminantAnalysis(n_components=3)

    #clf.fit(data,target)

    lda_x = clf.fit(data,target).transform(data)



    return lda_x

def full():
    '''
    samples full image stack

    Returns
    -------
    smp : tuple of ndarray
        data, targets, fature names and class names for LDA.

    '''

    s = processing.SampleMasks()


    s.load_masks_from_json(samples,
                           maskdir)
    s.sample_masks(maskdir)
    s.sampler([(maskdir / "watts_no_filter_1_comp_rir_gg_buv.jpeg", maskdir / "watts_no_filter_1.npy"),
               (maskdir / "watts_no_filter_4_comp_rir_gg_buv.jpeg", maskdir / "watts_no_filter_4.npy"),
               (maskdir / "watts_no_filter_6_comp_rir_gg_buv.jpeg", maskdir / "watts_no_filter_6.npy")])
    smp = s.sample_prep()

    return smp

def rgb():
    '''
    samples rgb composite image stack

    Returns
    -------
    smp : tuple of ndarray
        data, targets, fature names and class names for LDA.

    '''

    s = processing.SampleMasks()


    s.load_masks_from_json(samples,
                           maskdir)
    s.sample_masks(maskdir)
    s.sampler([(maskdir / "watts_no_filter_1_comp_rir_gg_buv.jpeg", maskdir / "watts_no_filter_1_comp_rr_gg_bb.tif"),
               (maskdir / "watts_no_filter_4_comp_rir_gg_buv.jpeg", maskdir / "watts_no_filter_4_comp_rr_gg_bb.tif"),
               (maskdir / "watts_no_filter_6_comp_rir_gg_buv.jpeg", maskdir / "watts_no_filter_6_comp_rr_gg_bb.tif")])
    smp = s.sample_prep()

    return smp


def plot_both(full_samples,
              rgb_samples):
    '''
    performs LDS and plots rgb and multispectral image stacks. writes to output dir

    Parameters
    ----------
    full_samples : tuple of ndarray
        full samples. Output of full()
    rgb_samples : tuple of ndarray
        rgb samples. output of rgb()

    Returns
    -------
    None.

    '''

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
    plt.savefig(outdir / 'lda.png', dpi=300)

if __name__ == '__main__':
    r = rgb()
    f = full()

    plot_both(f,r)

