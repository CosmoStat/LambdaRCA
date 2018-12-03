#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 01:27:32 2018

@author: rararipe
"""

import numpy as np
import sys
sys.path.append('../')
import psf_toolkit as tk
import psf_learning_utils as psflu
import utils
import transforms
import linear
from sf_tools.image.shape import Ellipticity

def Red(shap,nb_obj,nb_comp_chrom,sr_first_guesses):
    in_fact = 1.2
    out_fact = 0.5
    p = shap[0]*shap[1]
    nb_atoms = 2
    D_stack = np.zeros((p,nb_atoms,nb_comp_chrom))
    nb_comp_chrom=3
    
    for i in range(nb_comp_chrom):
        Ys = np.zeros((p,nb_atoms)) 
        guess = sr_first_guesses[i]

        zIn = utils.clipped_zoom(guess,in_fact) 
        tk.plot_func(zIn,title="before")
        zOut = utils.clipped_zoom(guess,out_fact)
        
        samp = 4.
        sig = .75 * 12 / 0.1 / samp
        Ell_zIn = Ellipticity(zIn, sigma=sig)
        cent = Ell_zIn.centroid
        r = 7.0
        nb_pixels_border = 3
        avg_pixel = 0.0
        for m in range(nb_pixels_border):
            for j in range(nb_pixels_border):
                avg_pixel += zIn[m,j] + zIn[-m,-j]
        avg_pixel /= nb_pixels_border*nb_pixels_border*2 
        mask = np.zeros((shap[0],shap[1]))
        for row in range(shap[0]):
            for col in range(shap[1]):
                if ((row-cent[0])**2 + (col-cent[1])**2) >=r and zIn[row,col]  > 70*avg_pixel:
                    mask[row,col] =1.0
                    zIn[row,col] *=2.0
        tk.plot_func(mask,title="mask")   
        tk.plot_func(zIn,title="later")            
        zIn_shift = utils.shift_y_to_grid_x(zIn,zOut)
        # zOut = guess.reshape(-1)
 
        Ys[:,0] = np.copy(zIn_shift.reshape(-1) / np.sum(zIn_shift)) #normalize the total of mass in each line
        Ys[:,1] = np.copy(zOut.reshape(-1)/ np.sum(zOut))
        
        D_stack[:,0,i] = np.copy(Ys[:,0])
        D_stack[:,1,i] = np.copy(Ys[:,1])
        print "dictxs ",i
        tk.plot_func(D_stack[:,0,i])
        tk.plot_func(D_stack[:,1,i])
        
    for i in range(D_stack.shape[-1]):
        print "===============comp " + str(i)
        for at in range(2):
            print "atom "+ str(at)
            tk.plot_func(D_stack[:,at,i])
        
    return D_stack
        

def RedWavelet(shap,nb_obj,nb_comp_chrom,linear_transf,sr_first_guesses):
    in_fact = 1.1
    out_fact = 0.5
    p = shap[0]*shap[1]
    nb_atoms = 2
    D_stack = np.zeros((p,nb_atoms,nb_comp_chrom))
    
    for i in range(nb_comp_chrom):
        Ys = np.zeros((p,nb_atoms)) 
        guess = sr_first_guesses[i]
        
        
       
        
        zIn = utils.clipped_zoom(guess,in_fact) 
        zOut = utils.clipped_zoom(guess,out_fact)
#        zOut = guess
        #            zOut_shift = utils.shift_y_to_grid_x(zOut,zIn)
        
        
        zIn_transf = linear_transf.op_single(zIn)
        #Make coarse scale more important
        zIn_transf[-1,:] *=1.5
        tk.plot_func(zIn_transf[-1,:])
        zIn_rec = linear_transf.adj_op_single(zIn_transf)
        
        zIn_shift = utils.shift_y_to_grid_x(zIn_rec,zOut)
        # zOut = guess.reshape(-1)
        
        
        
        
        Ys[:,0] = np.copy(zIn_shift.reshape(-1) / np.sum(zIn_shift)) #normalize the total of mass in each line
        Ys[:,1] = np.copy(zOut.reshape(-1)/ np.sum(zOut))
        
        D_stack[:,0,i] = np.copy(Ys[:,0])
        D_stack[:,1,i] = np.copy(Ys[:,1])
        
    return D_stack
    

first_guesses = np.load('/Users/rararipe/Documents/Data/QuickestGenerator/FirstGuess_500/first_guesses.npy') 


Wavelet_transf_dict = transforms.dict_wavelet_transform((42,42),3)
Lin_comb_wavelet_dict = linear.lin_comb(Wavelet_transf_dict)


#D_stack = Red((42,42),80,3,Lin_comb_wavelet_dict,sr_first_guesses=first_guesses)

D_stack = Red((42,42),80,3,sr_first_guesses=first_guesses)
for i in range(D_stack.shape[-1]):
    print "===============comp " + str(i)
    for at in range(2):
        print "atom "+ str(at)
        tk.plot_func(D_stack[:,at,i])
    
