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




temp = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_6lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4/result/'
load_path.append(temp)
wvl_bins.append(6)
temp = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_8NEWlbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4_RIGHTTRAIN/result/'
load_path.append(temp)
wvl_bins.append(8)
temp = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_10lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4/result/'
load_path.append(temp)
wvl_bins.append(10)
temp = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_12lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4/result/'
load_path.append(temp)
wvl_bins.append(12)
temp = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_14lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4/result/'
load_path.append(temp)
wvl_bins.append(14)

plot_path = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/Plot_compare_bins'
save_plot = True


all_lbdas = np.load('/Users/rararipe/Documents/Data/QuickestGenerator/full_res_70lbdas_80train300test/train/full_res_gt/all_lbdas.npy')

#%%
# Thomas Tram's plot parameters
plt.rcParams['figure.autolayout'] = False
plt.rcParams['axes.labelsize'] = 22 
plt.rcParams['axes.titlesize'] = 22 
plt.rcParams['legend.fontsize'] = 15 
plt.rcParams['font.size'] = 16 
plt.rcParams['lines.linewidth'] = 1.3 # he has 2.0 here but I like 1.3 better
plt.rcParams['lines.markersize'] = 9.0 
plt.rcParams['text.usetex'] = True 
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.serif'] = 'cm'
# plus bigger pics!
plt.rcParams['figure.figsize'] = 10, 7.5

#%%


MSE_rel_nor_alternates_list = []
MSE_rel_nor_stars_2euclidres_list = []
for i in range(len(wvl_bins)):
    MSE_rel_nor_alternates_list.append(np.load(load_path[i]+'MSE_rel_nor_alternates.npy'))
    MSE_rel_nor_stars_2euclidres_list.append(np.load(load_path[i]+'MSE_rel_nor_stars_2euclidres.npy'))

nb_altiter = MSE_rel_nor_alternates_list[0].shape[0]
cmap=cm.get_cmap('Spectral')
color_indx_pos = np.linspace(0,1,5)
#color_indx_pos = 1.0/(4-1)*np.array(range(4))




for it in range(nb_altiter):
    fig = plt.figure()
    for i in range(len(wvl_bins)):
        plt.errorbar(all_lbdas,np.mean(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0) ,label=r'{} bins'.format(wvl_bins[i]), color=cmap(color_indx_pos[i]),capsize=3)
    plt.xlabel(r'$\lambda$ (nm)')
    plt.ylabel(r' mean(NMSE)')
    plt.title(r'Iter {}'.format(it))
    plt.legend()
    if save_plot:
        plt.savefig(plot_path+'meanNMSEiter{}.eps'.format(it),format='eps',dpi=1200)
        plt.savefig(plot_path+'meanNMSEiter{}.jpg'.format(it),format='jpeg',dpi=1200)
        
    plt.show()
    plt.close()

#%%
for it in range(nb_altiter):
    fig = plt.figure()
    for i in range(len(wvl_bins)):
        plt.errorbar(all_lbdas,np.std(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0) ,label=r'{} bins'.format(wvl_bins[i]), color=cmap(color_indx_pos[i]),capsize=3)
    plt.xlabel(r'$\lambda$ (nm)')
    plt.ylabel(r' std(NMSE)')
    plt.title(r'Iter {}'.format(it))
    plt.legend()
    if save_plot:
        plt.savefig(plot_path+'stdNMSEiter{}.eps'.format(it),format='eps',dpi=1200)
        plt.savefig(plot_path+'stdNMSEiter{}.jpg'.format(it),format='jpeg',dpi=1200)
    plt.show()
    plt.close()    

#%%
for it in range(nb_altiter):
    fig = plt.figure()
    for i in range(len(wvl_bins)):
        plt.fill_between(all_lbdas,np.mean(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0)-np.std(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0),np.mean(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0)+np.std(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0),color=cmap(color_indx_pos[i]),alpha=0.5,label=r'{} bins'.format(wvl_bins[i]))
#        plt.errorbar(all_lbdas,np.mean(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0),yerr=np.std(abs(MSE_rel_nor_alternates_list[i][it,:,:]),axis= 0) ,label=r'{} bins'.format(wvl_bins[i]), color=cmap(color_indx_pos[i]),capsize=3)
    plt.xlabel(r'$\lambda$ (nm)')
    plt.ylabel(r' NMSE')
    plt.title(r'Iter {}'.format(it))
    plt.legend()
    if save_plot:
        plt.savefig(plot_path+'filledNMSEiter{}.eps'.format(it),format='eps',dpi=1200)
        plt.savefig(plot_path+'filledNMSEiter{}.jpg'.format(it),format='jpeg',dpi=1200)
    plt.show()
    plt.close()  

#%%

for it in range(MSE_rel_nor_stars_2euclidres_list[0].shape[0]):
    fig = plt.figure()
    plt.errorbar(wvl_bins,[np.mean(abs(MSE_rel_nor_stars_2euclidres_list[j][it,:])) for j in range(len(wvl_bins))],
                           yerr=[np.std(abs(MSE_rel_nor_stars_2euclidres_list[j][it,:])) for j in range(len(wvl_bins))],
                           color='darkorchid',capsize=3)
    plt.xlabel(r'$\lambda$ bins')
    plt.ylabel(r'Integrated NMSE')
    plt.title(r'Iter {}'.format(it))
    if save_plot:
        plt.savefig(plot_path+'integratedNMSE_vsbins_iter{}.eps'.format(it),format='eps',dpi=1200)
        plt.savefig(plot_path+'integratedNMSE_vsbins_iter{}.jpg'.format(it),format='jpeg',dpi=1200)
    plt.show()
    plt.close()    
    


