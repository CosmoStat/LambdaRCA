#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:59:14 2018

@author: rararipe
"""

import numpy as np
import psf_toolkit as tk
import matplotlib.pyplot as plt
from matplotlib import cm


load_path = []
wvl_bins = []
load_path.append('/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_8NEWlbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4_RIGHTTRAIN/result/')
wvl_bins.append(8)
load_path.append('/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_10lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4/result/')
wvl_bins.append(10)
load_path.append('/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_12lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4/result/')
wvl_bins.append(12)
load_path.append('/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_14lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4/result/')
wvl_bins.append(14)


all_lbdas = np.load('/Users/rararipe/Documents/Data/QuickestGenerator/full_res_70lbdas_80train300test/train/full_res_gt/all_lbdas.npy')

MSE_rel_nor_alternates_list = []
MSE_rel_nor_stars_2euclidres_list = []
for i in range(len(wvl_bins)):
    MSE_rel_nor_alternates_list.append(np.load(load_path[i]+'MSE_rel_nor_alternates.npy'))
    MSE_rel_nor_stars_2euclidres_list.append(np.load(load_path[i]+'MSE_rel_nor_stars_2euclidres.npy'))


nb_altiter = MSE_rel_nor_alternates_list[0].shape[0]
cmap=cm.get_cmap('Spectral')
color_indx_pos = np.linspace(0,1,4)
#color_indx_pos = 1.0/(4-1)*np.array(range(4))




for it in range(nb_altiter):
    fig = plt.figure()
    for i in range(len(wvl_bins)):
        plt.errorbar(all_lbdas,np.mean(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0) ,label=r'{} bins'.format(wvl_bins[i]), color=cmap(color_indx_pos[i]),capsize=3)
    plt.xlabel(r'$\lambda$ (nm)')
    plt.ylabel(r' mean(NMSE)')
    plt.title(r'Iter {}'.format(it))
    plt.legend()
    plt.show()
    plt.close()

print "============"
for it in range(nb_altiter):
    fig = plt.figure()
    for i in range(len(wvl_bins)):
        plt.errorbar(all_lbdas,np.std(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0) ,label=r'{} bins'.format(wvl_bins[i]), color=cmap(color_indx_pos[i]),capsize=3)
    plt.xlabel(r'$\lambda$ (nm)')
    plt.ylabel(r' std(NMSE)')
    plt.title(r'Iter {}'.format(it))
    plt.legend()
    plt.show()
    plt.close()    

    
print "============"

for it in range(MSE_rel_nor_stars_2euclidres_list[0].shape[0]):
    fig = plt.figure()
    plt.errorbar(wvl_bins,[np.mean(abs(MSE_rel_nor_stars_2euclidres_list[j][it,:])) for j in range(len(wvl_bins))],
                           yerr=[np.std(abs(MSE_rel_nor_stars_2euclidres_list[j][it,:])) for j in range(len(wvl_bins))],
                           color='darkorchid',capsize=3)
    plt.xlabel(r'$\lambda$ bins')
    plt.ylabel(r'Integrated NMSE')
    plt.title(r'Iter {}'.format(it))
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
    
    
#    np.std(abs(MSE_rel_nor_alternates_8wvl[it,:,:]),axis= 0) 