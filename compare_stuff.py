#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:49:39 2018

@author: rararipe
"""

import numpy as np
import psf_toolkit as tk
import matplotlib.pyplot as plt

data_path_8wvl =  '/Users/rararipe/Documents/Data/QuickestGenerator/trueSEDs/42x42pixels_8lbdas80pos/'
data_path_10wvl = '/Users/rararipe/Documents/Data/QuickestGenerator/trueSEDs/42x42pixels_10lbdas80pos/'
load_path_8wvl = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_8lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it443dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4_newData_GETBACK/result/'
load_path_10wvl = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_10lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it443dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4_newData_GETBACK/result/'


lbdas_8wvl = np.load(data_path_8wvl + 'lbdas.npy')
lbdas_10wvl =np.load(data_path_10wvl + 'lbdas.npy')
MSE_rel_nor_alternates_8wvl = np.load(load_path_8wvl+'MSE_rel_nor_alternates.npy')
MSE_rel_nor_alternates_10wvl = np.load(load_path_10wvl+'MSE_rel_nor_alternates.npy')
nb_altiter = MSE_rel_nor_alternates_8wvl.shape[0]

for it in range(nb_altiter):
    fig = plt.figure()
    plt.errorbar(lbdas_8wvl,np.mean(abs(MSE_rel_nor_alternates_8wvl[it,:,:]),axis= 0),yerr=np.std(abs(MSE_rel_nor_alternates_8wvl[it,:,:]),axis= 0)  ,label='8 bins', color='darkorchid',capsize=3, alpha=.7)
    plt.errorbar(lbdas_10wvl,np.mean(abs(MSE_rel_nor_alternates_10wvl[it,:,:]), axis= 0),yerr = np.std(abs(MSE_rel_nor_alternates_10wvl[it,:,:]), axis= 0), label='10 bins', color='red',capsize=3, alpha=.7)
    plt.xlabel(r'$\lambda$ (nm)')
    plt.ylabel(r' NMSE')
    plt.title(r'Iter {}'.format(it))
    plt.legend()
    plt.show()
    plt.close()
#for wvl in range(nb_wvl):
#    fig = plt.figure()
#    plt.plot(range(MSE_rel_nor_alternates.shape[0]), MSE_rel_nor_alternates[:,:,wvl]) # <iter,objs, wvls>
#    plt.title(r'MSE at wvl {} across alternate scheme'.format(wvl))
#    plt.show()
#    fig = plt.figure()
#    plt.plot(range(nb_obj), MSE_rel_nor[-1,:,wvl] )
#    plt.xlabel(r'Objects')
#    plt.ylabel(r'relative MSE')
#    plt.title('Final')