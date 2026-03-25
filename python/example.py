# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:37:17 2025

@author: ds
"""

import processing

import DngFLuo

from pathlib import Path

import matplotlib.pyplot as plt

from skimage.exposure import equalize_hist

from skimage.exposure import rescale_intensity

from skimage.io import imsave

import numpy as np

import sys


root_dir = Path.cwd().parents[1]

def stdminmax(arr, n):
    st = np.std(arr)*n
    print(st)
    mean = np.mean(arr)
    print(mean)
    vmin = mean-st
    print(vmin)
    vmax = mean+st
    print(vmax)
    return vmin,vmax



def subset_components(stack,
                      drop_list=None):
    '''
    Subsets bands

    Parameters
    ----------
    stack : TYPE
        DESCRIPTION.
    drop_list : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    out : TYPE
        DESCRIPTION.
    names : TYPE
        DESCRIPTION.

    '''

    out = None

    names = []

    #drop_list = [2,6,8,13,14,16,17,20,24,25,33,37,0,12,38]

    dims = stack.shape[-1]

    for i in range(dims):
        if not i in drop_list:
            if out is None:
                out = stack[:,:,i]
            else:
                out = np.dstack((out,stack[:,:,i]))
            names.append(i+1)

    return out, names



def false_colour(stack,
                 r,
                 g,
                 b,
                 equalize=True,
                 plot=False,
                 name = ''):
    '''
    Function plots false colour image from image stack

    Parameters
    ----------
    stack : TYPE ndarray
        DESCRIPTION. image stak
    r : TYPE int
        DESCRIPTION. band to display in red channel
    g : TYPE int
        DESCRIPTION. band to display green channel
    b : TYPE int
        DESCRIPTION. band to display blue channel
    equalize : TYPE bool, optional
        DESCRIPTION. Eqaualize historgam for each band.The default is True.
    plot : TYPE bool, optional
        DESCRIPTION. plot image? The default is False.
    name : TYPE str, optional
        DESCRIPTION file name prefix. The default is ''.

    Returns
    -------
    composite : TYPE
        DESCRIPTION.

    '''

    if equalize is True:
        composite = np.dstack((equalize_hist(stack[:,:,r]),
                               equalize_hist(stack[:,:,g]),
                               equalize_hist(stack[:,:,b])))
    else:
        composite = np.dstack((stack[:,:,r],
                               stack[:,:,g],
                               stack[:,:,b]))

    '''composite = rescale_intensity(composite,
                                  in_range='image',
                                  out_range=(0,255))'''

    if plot is True:
        plt.imshow(composite)
        plt.axis('off')
        #plt.show()

        f = '%s_%s_%s_%s.png' % (name,r,g,b)

        fpath = root_dir / 'output' / f

        plt.savefig(fpath)

        plt.close()


    return composite



def single_im(im,
              name,
              stdev= 0.5):
    '''
    Method performs stretch to supplied number of standard deviations and saves 
    it

    Parameters
    ----------
    im : np.ndarry
        DESCRIPTION. The image
    name : str
        DESCRIPTION. Name to save it as
    stdev : float or int, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    None.

    '''

    vmin,vmax = stdminmax(im,
                          stdev)

    plt.imshow(im,
               vmin=vmin,
               vmax=vmax)

    plt.axis('off')
    plt.show()

    f = '%s.png' % (name)

    fpath = root_dir / 'output' / f

    plt.savefig(fpath)

    plt.close()


def multi_plot(stack,
               rows_cols = (3,4),
               name = '',
               equalize = True,
               names = None,
               figsize=(9,15),
               text_offset=(300,-50),
               fontsize=12,
               outfile=None,
               dpi = 300,
               wspace = 0.01,
               hspace = 0.35,
               fname='test.png'):
    '''
    Plots a grid of sub plot images

    Parameters
    ----------
    stack : TYPE
        DESCRIPTION.
    rows_cols : TYPE tup, optional
        DESCRIPTION number of rows and columns. The default is (3,4).
    name : TYPE str, optional
        DESCRIPTION. prefix for names The default is ''.
    equalize : TYPE, optional
        DESCRIPTION. The default is True.
    names : TYPE, optional
        DESCRIPTION. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is (9,15).
    text_offset : TYPE, optional
        DESCRIPTION. The default is (300,-50).
    fontsize : TYPE, optional
        DESCRIPTION. The default is 12.
    outfile : TYPE, optional
        DESCRIPTION. The default is None.
    dpi : TYPE, optional
        DESCRIPTION. The default is 300.
    wspace : TYPE, optional
        DESCRIPTION. The default is 0.01.
    hspace : TYPE, optional
        DESCRIPTION. The default is 0.35.
    fname : TYPE, optional
        DESCRIPTION. The default is 'test.png'.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    rows,cols = rows_cols

    dims = rows*cols

    if dims>stack.shape[-1]:
        raise Exception('Too many plots for data')

    figs,axs = plt.subplots(rows,
                            cols,
                            figsize=figsize,
                            gridspec_kw={'wspace': wspace, 'hspace':hspace})

    #plt.subplots_adjust(hspace=0.15, wspace=0.08)

    axs = axs.flatten()

    for i in range(dims):
        if equalize is True:
            image = equalize_hist(stack[:,:,i])
        else:
            image = stack[:,:,i]

        axs[i].imshow(image, cmap='viridis')
        axs[i].axis('off')

        if names is None:
            b_name = i+1
        else:
            b_name = names[i]


        axs[i].text(text_offset[0],
                    text_offset[1],
                    f'{name} {b_name}',
                    fontsize=fontsize,
                    color='black',
                    ha='left')

    #plt.tight_layout()
    #plt.show()
    plt.savefig(root_dir / 'output' / fname, dpi=300)
    #plt.close()


def plot_all_bands(stack):
    '''
    plot raw luminace for all bands in stack. saves to output

    Parameters
    ----------
    stack : TYPE np array
        DESCRIPTION.

    Returns
    -------
    None.

    '''

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
                   "B 365nm"]

    multi_plot(stack,
               name='',
               rows_cols=(9,5),
               names=feature_names,
               figsize=(5,14),
               text_offset=(940,-50),
               fontsize=8,
               fname = 'Fig6_all_bands_raw.png')


def load_stack(path):
    '''
    load stack from dng images

    Parameters
    ----------
    path : TYPE pathlike
        DESCRIPTION. path to directory

    Returns
    -------
    a : TYPE ArrayHandler object
        DESCRIPTION.
    stack: TYPE ndarray
        DESCRIPTION. multispectral image stack

    '''

    indir = Path(path).parent

    d = Path(path).name

    a = processing.ArrayHandler(indir, d)

    stack = a.gen_image_stack_np()

    stack = a.denoise(stack)

    return a,stack[:,:,3:]

def run_pca(a,
            stack,
            save_np = False):
    '''
    run PCA on stack and saves multi plots and false colour composite

    Parameters
    ----------
    a : TYPE ArrayHandler object
        DESCRIPTION.
    stack : TYPE ndarray
        DESCRIPTION. image stack

    Returns
    -------
    None.

    '''

    pca = a.stack_pca(stack, n_components=20)

    multi_plot(pca[0],
               name='PCA',
               rows_cols=(5,4),
               figsize=(5,7.5),
               text_offset=(440,-50),
               fontsize=8,
               fname = 'Fig7_pca_multi.png')

    #false_colour(pca[0], 0, 1, 2, name='PCA')
    
    if save_np is True:
        np.save(root_dir / 'output' / 'PCA_Image.npy',
                pca[0])

def run_ica(a,
            stack,
            save_np=False):
    '''
    run ICA on stack and saves multi plots and false colour composite

    Parameters
    ----------
    a : TYPE ArrayHandler object
        DESCRIPTION.
    stack : TYPE ndarray
        DESCRIPTION. image stack

    Returns
    -------
    None.

    '''



    ica = a.stack_ica(stack, n_components=None)


    multi_plot(ica[0],
               name='ICA',
               rows_cols=(7,6),
               figsize=(9,15),
               text_offset=(440,-50),
               fontsize=8,
               fname = 'Fig9_ica_multi.png')

    #false_colour(ica[0], 0, 1, 6, name='ICA')
    
    if save_np:
        np.save(root_dir / 'output' / 'ICA_Image.npy',
                ica[0])

    return ica

def run_fica(a,stack):
    '''
    run ICA on flourescence stack and saves multi plots and false colour 
    composite

    Parameters
    ----------
    a : TYPE ArrayHandler object
        DESCRIPTION.
    stack : TYPE ndarray
        DESCRIPTION. image stack

    Returns
    -------
    None.

    '''



    ica = a.stack_ica(stack, n_components=None)


    multi_plot(ica[0],
               name='ICA',
               rows_cols=(5,3),
               figsize=(9,15),
               text_offset=(440,-50),
               fontsize=8,
               fname = 'F_ica_multi.png')

    #false_colour(ica[0], 0, 1, 6, name='ICA')

    np.save(root_dir / 'output' / 'ICA_Image.npy',
            ica[0])

    return ica


if __name__ == '__main__':

    if sys.argv[1]=='folder':

        data = root_dir / 'data' / 'images' / 'full_images' / 'watts_no_filter_6'

        s = load_stack(data)

        plot_all_bands(s[1])

        run_pca(s[0],s[1])

        run_ica(s[0],s[1])

    elif sys.argv[1]=='paper':
        data = root_dir / 'data' / 'images' / 'subsets' / 'watts_6_crop.npy'

        full_data = root_dir / 'data' / 'images' / 'full_images' / 'watts_no_filter_6'

        print ('Loading data')

        fs = load_stack(full_data)

        s = np.load(data)

        print ('Plotting raw per band luminance')
        plot_all_bands(s)

        print ('running PCA')
        pca = run_pca(fs[0],s)

        false_colour(pca[0], 0, 1, 2, name='Fig8a_PCA', plot = True)
        false_colour(pca[0], 3, 4, 5, name='Fig8b_PCA', plot = True)
        false_colour(pca[0], 6, 7, 8, name='Fig8c_PCA', plot = True)
        false_colour(pca[0], 9, 10, 11, name='Fig8d_PCA', plot = True)

        
        print ('running ICA')
        ica = run_ica(fs[0],s)

        print ('plotting ICA')
        false_colour(ica[0], 6, 3, 9, name='Fig10l_ICA', plot=True)
        false_colour(ica[0], 17, 9, 40, name='Fig10r_ICA', plot=True)

    elif sys.argv[1]=='fluo':
        hdata = root_dir / 'data' / 'images' / 'full_images' / 'watts_no_filter_2'
        h = DngFLuo.FluoStack(hdata.parent,hdata.name)


        ah_init = root_dir / 'data' / 'images' / 'full_images' / 'watts_no_filter_2'
        a = processing.ArrayHandler(Path(ah_init).parent,
                                    Path(ah_init).name)

        h_stack = h.gen_fluo_stack_np()

        ica = run_fica(h,a.denoise(h_stack[3]))
        #ica = run_fica(h_stack[3])

        single_im(h_stack[3][:,:,14], 'fig13_lowerleftt_365nm_huediff',stdev=1)

        full_data = root_dir / 'data' / 'images' / 'full_images' / 'watts_no_filter_2'
        fs = load_stack(full_data)
        single_im(fs[1][:,:,-2], 'fig13_upperright_365nm_green')

