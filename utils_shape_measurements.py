#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:22:48 2018

@author: rararipe
"""

import numpy as np
import galsim
import psf_toolkit as tk
import matplotlib.pyplot as plt
from sf_tools.image.shape import Ellipticity
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import os
from os import listdir
from os.path import isfile, join
from scipy.interpolate import interp1d
from matplotlib import cm

plt.rcParams['figure.autolayout'] = False
plt.rcParams['axes.labelsize'] = 18 
plt.rcParams['axes.titlesize'] = 20 
plt.rcParams['font.size'] = 16 
plt.rcParams['lines.linewidth'] = 1.3 # he has 2.0 here but I like 1.3 better
plt.rcParams['lines.markersize'] = 8 
plt.rcParams['legend.fontsize'] = 14 
plt.rcParams['text.usetex'] = True 
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.serif'] = 'cm'


def interp_SEDs(lbdas,sed_path,plot_path,angs_unit=True,interpolate_ponctual=False):
    nlbdas = len(lbdas)
    SED_names = [f for f in listdir(sed_path) if isfile(join(sed_path, f))]
    SED_names.remove('.DS_Store')
    
    cmap = cm.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 0.9, len(SED_names))]
    
    # prepare interpolated SED array
    lmin, lmax = 550, 900 # VIS
    
    
    SEDs = []
    for galtype,col in zip(SED_names,colors):
    
        # Read SED 
        sed = np.loadtxt(sed_path+galtype)
        if angs_unit:
            sed[:,0]/=10.0 # and convert to nm
        
        # Linearly interpolate
        interp = interp1d(sed[:,0], sed[:,1])
        
        if interpolate_ponctual:
            this_sed = np.array([interp(lbda) for lbda in lbdas])

            # Keep only Euclid VIS band
            sed = sed[(lmin<=sed[:,0]) & (sed[:,0]<=lmax)]
        
        else:
            
            # Keep only Euclid VIS band
            sed = sed[(lmin<=sed[:,0]) & (sed[:,0]<=lmax)] 
            
            #  Take the mean of spectrum chunk in front
            this_sed = []
            for v in range(nlbdas-1):
                if v == 0:
                    lbdamin = lbdas[0]
                else:
                    lbdamin = lbdas[v-1]
                lbdamax = lbdas[v+1]
                sed_chunk = sed[(lbdamin<=sed[:,0]) & (sed[:,0]<lbdamax)]
                this_sed.append(np.mean(sed_chunk[:,1]))
            this_sed.append(interp(lbdas[-1]))
            
        SEDs.append(this_sed)   
    
        # Plot it
        plt.plot(lbdas, this_sed, c=col)
        plt.plot(sed[:,0], sed[:,1], '.', c=col)
        
        
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.legend(loc=5, bbox_to_anchor=(1.26,.5))
    plt.savefig(plot_path+'SEDs.pdf', bbox_inches='tight')
    
    #Pass to lambdaRCA format
    SEDs = np.array(SEDs)
    SEDs = SEDs.swapaxes(0,1)
    # Normalize..
    SEDs /= np.sum(abs(SEDs),axis=0)
    
    return SEDs



def rbf_components(rep_train, pos_train, pos_test, n_neighbors=15):
    n_components, ntrain = rep_train.shape
    ntest = pos_test.shape[0]
    rep_test = np.empty((n_components, ntest))
    for i, pos in enumerate(pos_test):
        # determine neighbors
        nbs, pos_nbs = tk.return_neighbors(pos, rep_train.T, pos_train, n_neighbors)
        # train RBF and interpolate for each component
        for j in range(n_components):
            rbfi = Rbf(pos_nbs[:,0], pos_nbs[:,1], nbs[:,j], function='thin_plate')
            rep_test[j,i] = rbfi(pos[0], pos[1])
    return rep_test
    
    
    
def computeHSMshapes(PSFs,pixel_scale): 
   
    # Convert to galsim images 
    PSFs_galS = []
    for im in PSFs:
        PSFs_galS.append(galsim.Image(im, scale=pixel_scale))

    # Compute HSM shapes
    moms_list = [galsim.hsm.FindAdaptiveMom(psf) for psf in PSFs_galS] 
    
    shapes = np.array([[moms.observed_shape.g1, moms.observed_shape.g2,
                        2.*moms.moments_sigma**2] for moms in moms_list])



    return shapes


def paulin(gal_size, trupsf_shape, estpsf_shape):
    """Computes Paulin predicted bias values.
    
    Assumes last two inputs are length-3 arrays containing (e1,e2,R^2).
    
    Returns:
    m, c
    m the multiplicative term and c the additive one."""
    deltapsf = estpsf_shape - trupsf_shape
    m = 1. + deltapsf[2] / gal_size
    c = -(trupsf_shape[2]/gal_size*deltapsf[:2] + 
           deltapsf[2]/gal_size*trupsf_shape[:2])
    return m, c


def paulin_predict(tru_shapes, rec_shapes, R2s):
    
    
    paulin_allPositions = [[paulin(siz,tru_shap,rca_shap) for
              tru_shap,rca_shap in zip(tru_shapes,rec_shapes)] for siz in R2s]

    
#    paulin_allPositions = [[paulin(siz,tru_shap,rec_shap) for siz in R2s] for
#              tru_shap,rec_shap in zip(tru_shapes,rec_shapes)]
    
    m_mean = [np.mean([r[0] for r in rcap])-1 for rcap in paulin_allPositions]
    m_std = [np.std([r[0] for r in rcap]) for rcap in paulin_allPositions]
    
    c1_mean = [np.mean([r[1][0] for r in rcap]) for rcap in paulin_allPositions]
    c1_std = [np.std([r[1][0] for r in rcap]) for rcap in paulin_allPositions]
    
    c2_mean = [np.mean([r[1][1] for r in rcap]) for rcap in paulin_allPositions]
    c2_std = [np.std([r[1][1] for r in rcap]) for rcap in paulin_allPositions]
  
    return m_mean,m_std,c1_mean,c1_std,c2_mean,c2_std










   
    
    
    