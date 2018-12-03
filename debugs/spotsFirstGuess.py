#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:55:46 2018

@author: rararipe
"""

import numpy as np
import sys
sys.path.append('../')
import psf_toolkit as tk
import psf_learning_utils as psflu
import utils

from sf_tools.image.shape import Ellipticity




def Spots(shap,nb_obj,nb_comp_chrom,sr_first_guesses):
    in_fact = 1.2
    out_fact = 0.6
    p = shap[0]*shap[1]
    nb_atoms = 2
    D_stack = np.zeros((p,nb_atoms,nb_comp_chrom))
    
    for i in range(nb_comp_chrom):
        Ys = np.zeros((p,nb_atoms)) 
        guess = sr_first_guesses[i]
        
        
        #insert black spots
        samp = 4.
        sig = .75 * 12 / 0.1 / samp
        Ell_guess = Ellipticity(guess, sigma=sig)
        centroid = np.round(int(Ell_guess.centroid))
        
        zIn = utils.clipped_zoom(guess,in_fact)
        
        #            zOut = utils.clipped_zoom(guess,out_fact)
        zOut = guess
        #            zOut_shift = utils.shift_y_to_grid_x(zOut,zIn)
        zIn_shift = utils.shift_y_to_grid_x(zIn,zOut)
        # zOut = guess.reshape(-1)
        
        
        
        
        Ys[:,0] = np.copy(zIn_shift.reshape(-1) / np.sum(zIn_shift)) #normalize the total of mass in each line
        Ys[:,1] = np.copy(zOut.reshape(-1)/ np.sum(zOut))
        
        D_stack[:,0,i] = np.copy(Ys[:,0])
        D_stack[:,1,i] = np.copy(Ys[:,1])
        
    
    

first_guesses = np.load('/Users/rararipe/Documents/Data/QuickestGenerator/FirstGuess_500/first_guesses.npy') 
D_stack = Spots((42,42),80,3,sr_first_guesses=first_guesses)