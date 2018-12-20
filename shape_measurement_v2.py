#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:38:15 2018

@author: rararipe
"""

import numpy as np
import psf_toolkit as tk
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from tqdm import tqdm
from matplotlib import cm
import matplotlib
import colors_plot as cp
import random
import copy
import utils_shape_measurements as utils



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

# Paramenters

pixel_scale = 0.0083333333333
twiceEuclid = 0.05
plot_each = False
plot_paulin_testGal_all = True
plot_shape_measurements_all = True
save_stuff = False
root_path = '/Users/rararipe/Documents/Data/'

# PSFs paths
lbdaRCA_path = root_path+ 'GradientDescent_output/trueSEDs/42x42pixels_14lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4/result/'
data_path = root_path+'QuickestGenerator/full_res_70lbdas_80train300test/train/PickleSEDs/Interp_14wvls/'
gtobs_path = root_path+'QuickestGenerator/full_res_70lbdas_80train300test/train/PickleSEDs/'
full_res_gt_path = root_path+'QuickestGenerator/full_res_70lbdas_80train300test/train/full_res_gt/'
test_data_path = root_path+'QuickestGenerator/full_res_70lbdas_80train300test/test/'
RCA_path = root_path+'RCA/trueSEDs/42x42pixels_Pickles70lbdasStars80pos/'
nb_pos_test = 300 
nb_pos = 80

# Paulin paramenters
sizmin = .1#.04
sizmax = .5#.9
nb_points = 30#300
# R ("size") to R^2
sizes = np.linspace(sizmin, sizmax, nb_points)
R2s = sizes**2


# gal SEDs paths
sed_path = root_path+'/galsim/galaxySEDs/'

# Plot path (change later)
lbda_interp_path = lbdaRCA_path + "Interpolation/"
rca_interp_path = RCA_path + "Interpolation/"
results_path = lbdaRCA_path + "Shape_measurements/"
plot_path = results_path + 'Plots/'


if not os.path.exists(plot_path):
    os.makedirs(plot_path)        
 
if not os.path.exists(lbda_interp_path):
    os.makedirs(lbda_interp_path) 
if not os.path.exists(rca_interp_path):
    os.makedirs(rca_interp_path)  

#%% Get galaxies SEDs

# Load lbdas used in lbdaRCA
all_lbdas = np.load(gtobs_path+'all_lbdas.npy') 
      
galSEDs = utils.interp_SEDs(all_lbdas,sed_path,plot_path) # <nb_wvl, nb_gal>

fov = np.load(gtobs_path+'fov.npy') 
fov_test = np.load(test_data_path+'fov.npy') 
np.save(results_path+'fov.npy',fov)
np.save(results_path+'fov_test.npy',fov_test)    
#%% Load all PSFs, barycenters, components and weights
        
PSFs_integrated_fr = np.load(gtobs_path+'stars_fullres_gt.npy')
PSFs_integrated_fr /= np.sum(abs(PSFs_integrated_fr), axis=(0,1))
# HSM format
PSFs_integrated_fr = PSFs_integrated_fr.swapaxes(2,1).swapaxes(1,0)

fov_test = np.load(test_data_path+'fov.npy')
fov_train = np.load(gtobs_path+'fov.npy')


# and "experimental" PSFs, starting with "true" ones...
truPSFs = np.load(gtobs_path+'PSFs_2euclidrec.npy')

truPSFs /= np.sum(abs(truPSFs), axis=(0,1))
truPSFs = truPSFs.swapaxes(2,3)
truPSFs_integrated = np.load(gtobs_path+'stars_2euclidrec_gt.npy')
truPSFs_integrated /= np.sum(abs(truPSFs_integrated), axis=(0,1))
W,_,_ = truPSFs_integrated.shape
# HSM format
truPSFs_integrated = truPSFs_integrated.swapaxes(2,1).swapaxes(1,0)

# ... lbdaRCA ones...
lbdaPSFs = np.load(lbdaRCA_path+'psf_est.npy')
lbdaPSFs /= np.sum(abs(lbdaPSFs),axis=(0,1))
lbdaPSFs_integrated = np.load(lbdaRCA_path+'stars_est_2euclidres.npy')
lbdaPSFs_integrated /= np.sum(abs(lbdaPSFs_integrated),axis=(0,1))
barycenters = np.load(lbdaRCA_path+'barycenters_wvlAll.npy')
A_train_lbdaRCA = np.load(lbdaRCA_path+'A.npy')
# HSM format
lbdaPSFs_integrated = lbdaPSFs_integrated.swapaxes(2,1).swapaxes(1,0)


# ... And RCA ones
rcaPSFs = fits.getdata(RCA_path+'est_psf.fits')
rcaPSFs /= np.sum(abs(rcaPSFs),axis=(0,1))
components_rca = fits.getdata(RCA_path+'components.fits')
A_train_rca = fits.getdata(RCA_path+'A.fits')
# HSM format
rcaPSFs = rcaPSFs.swapaxes(2,1).swapaxes(1,0)

#%% Interpolations that don't depend on SED

# RCA: test positions
A_test_rca = utils.rbf_components(A_train_rca, fov_train, fov_test) # get A coefficients at galaxy positions
rcaPSFs_test = components_rca.dot(A_test_rca).swapaxes(2,1).swapaxes(1,0) # reconstruct PSF at galaxy positions
rcaPSFs_test /= np.sum(abs(rcaPSFs_test),axis=(1,2)).reshape(rcaPSFs_test.shape[0],1,1)

# lbdaRCA: A for test positions
A_test_lbdaRCA = utils.rbf_components(A_train_lbdaRCA, fov_train, fov_test) # get A coefficients at galaxy positions


# Save interpolations
np.save(rca_interp_path+'rcaPSFs_test.npy',rcaPSFs_test)
np.save(lbda_interp_path+'A_test_lbdaRCA.npy',A_test_lbdaRCA)
#%% Shapes that don't depend on SED

full_shapes = utils.computeHSMshapes_stack(PSFs_integrated_fr,pixel_scale) # obs: format of PSFs has to be <nb_obj, pixel, pixel>
truInt_shapes = utils.computeHSMshapes_stack(truPSFs_integrated,twiceEuclid)
rca_shapes = utils.computeHSMshapes_stack(rcaPSFs,twiceEuclid)
rca_shapes_test = utils.computeHSMshapes_stack(rcaPSFs_test,twiceEuclid)
lbdaInt_shapes = utils.computeHSMshapes_stack(lbdaPSFs_integrated,twiceEuclid)

full_R_pixel = full_shapes[:,2]
truInt_R_pixel = truInt_shapes[:,2]
rca_R_pixel = rca_shapes[:,2]
rca_R_pixel_test = rca_shapes_test[:,2]
lbdaInt_R_pixel = lbdaInt_shapes[:,2]

#Convert all to arcsec^2  
full_shapes[:,2] *= pixel_scale**2
truInt_shapes[:,2]*= twiceEuclid**2
rca_shapes[:,2]*= twiceEuclid**2
rca_shapes_test[:,2]*= twiceEuclid**2
lbdaInt_shapes[:,2]*= twiceEuclid**2

# Paulin predictions
paulin_preds_rca = utils.paulin_predict_stack(truInt_shapes, rca_shapes, R2s)
paulin_preds_lbda = utils.paulin_predict_stack(truInt_shapes, lbdaInt_shapes, R2s)

#%% Delete heavy stuff
del PSFs_integrated_fr
del truPSFs_integrated
del rcaPSFs
del rcaPSFs_test
del lbdaPSFs_integrated
#%% Thinking differently


full_shapes_testgal = [] 
truInt_shapes_testgal = []
lbdaInt_shapes_testgal = []
full_R_pixels_testgal = []
truInt_R_pixels_testgal = []
lbdaInt_R_pixels_testgal = []
paulin_preds_rca_testgal = []
paulin_preds_lbda_testgal = []
for pos in tqdm(range(nb_pos_test)):
    full_shapes_testgal_thrgals = [] 
    truInt_shapes_testgal_thrgals = []
    lbdaInt_shapes_testgal_thrgals = []
    full_R_pixels_testgal_thrgals = []
    truInt_R_pixels_testgal_thrgals = []
    lbdaInt_R_pixels_testgal_thrgals = []
    paulin_pred_rca_testgal_thrgals = []
    paulin_pred_lbda_testgal_thrgals = []
    for gal_i in range(2): # galSEDs.shape[-1]
       
        PSFs_full_res_test_gt = np.load(test_data_path+'PSF{}.npy'.format(pos))
        PSFs_full_res_test_gt /= np.sum(abs(PSFs_full_res_test_gt),axis=(1,2)).reshape((PSFs_full_res_test_gt.shape[0],1,1))
        star_testGal_fr_gt = np.sum(PSFs_full_res_test_gt * galSEDs[:,gal_i].reshape(-1,1,1), axis=0)
        
           
        star_2euclidrec_testGal_gt = tk.decimate(star_testGal_fr_gt, downsamp=6)
        star_2euclidrec_testGal_gt = star_2euclidrec_testGal_gt[21:21+42,21:21+42]
        
        star_testGal_fr_gt /= np.sum(abs(star_testGal_fr_gt))
        star_2euclidrec_testGal_gt /= np.sum(abs(star_2euclidrec_testGal_gt))
   
        # lbdaRCA galSED integration
        components_gal = barycenters.dot(galSEDs[:,gal_i]).reshape((W,W,barycenters.shape[1]))

        # lbdaRCA: test positions and galSED
        lbdaPSF_testGal = components_gal.dot(A_test_lbdaRCA[:,pos]) 
        lbdaPSF_testGal /= np.sum(abs(lbdaPSF_testGal))


        # Compute HSM shapes
        full_shape_testgal = utils.computeHSMshapes(star_testGal_fr_gt,pixel_scale)
        del star_testGal_fr_gt
        truInt_shape_testgal = utils.computeHSMshapes(star_2euclidrec_testGal_gt,pixel_scale)
        del star_2euclidrec_testGal_gt        
        lbdaInt_shape_testgal = utils.computeHSMshapes(lbdaPSF_testGal,pixel_scale)
        del lbdaPSF_testGal
        
        full_R_pixels_testgal_thrgals.append(full_shape_testgal[2])
        truInt_R_pixels_testgal_thrgals.append(truInt_shape_testgal[2])
        lbdaInt_R_pixels_testgal_thrgals.append(lbdaInt_shape_testgal[2])
          
        
        full_shape_testgal[2] *= pixel_scale**2
        truInt_shape_testgal[2] *= twiceEuclid**2
        lbdaInt_shape_testgal[2] *= twiceEuclid**2
        
        # Compute Paulin prediction
#        raise ValueError('Ops')
        # DEBUG
#        estpsf_shape = rca_shapes_test[pos]
#        trupsf_shape = truInt_shape_testgal
#        gal_size = R2s[0]
#        
#        deltapsf = np.array(estpsf_shape) - np.array(trupsf_shape)
#        m = 1. + deltapsf[2] / gal_size
#        c = -(trupsf_shape[2]/gal_size*deltapsf[:2] + 
#               deltapsf[2]/gal_size*trupsf_shape[:2])
        #DEBUG

        paulin_pred_rca_testgal_thrgals.append([utils.paulin(siz,truInt_shape_testgal,rca_shapes_test[pos],flat=True) for siz in R2s])
        paulin_pred_lbda_testgal_thrgals.append([utils.paulin(siz,truInt_shape_testgal,lbdaInt_shape_testgal,flat=True) for siz in R2s])
        
        

        full_shapes_testgal_thrgals.append(full_shape_testgal)
        truInt_shapes_testgal_thrgals.append(truInt_shape_testgal)
        lbdaInt_shapes_testgal_thrgals.append(lbdaInt_shape_testgal)
        

    paulin_preds_rca_testgal.append(paulin_pred_rca_testgal_thrgals)
    paulin_preds_lbda_testgal.append(paulin_pred_lbda_testgal_thrgals)

    full_shapes_testgal.append(full_shapes_testgal_thrgals)
    truInt_shapes_testgal.append(truInt_shapes_testgal_thrgals)
    lbdaInt_shapes_testgal.append(lbdaInt_shapes_testgal_thrgals)
    full_R_pixels_testgal.append(full_R_pixels_testgal_thrgals)
    truInt_R_pixels_testgal.append(truInt_R_pixels_testgal_thrgals)
    lbdaInt_R_pixels_testgal.append(lbdaInt_R_pixels_testgal_thrgals)



full_shapes_testgal = np.array(full_shapes_testgal)
truInt_shapes_testgal = np.array(truInt_shapes_testgal)
lbdaInt_shapes_testgal = np.array(lbdaInt_shapes_testgal)
full_R_pixels_testgal = np.array(full_R_pixels_testgal)
truInt_R_pixels_testgal = np.array(truInt_R_pixels_testgal)
lbdaInt_R_pixels_testgal = np.array(lbdaInt_R_pixels_testgal) 

paulin_preds_rca_testgal = np.array(paulin_preds_rca_testgal)
paulin_preds_lbda_testgal = np.array(paulin_preds_lbda_testgal)

#%% Paulin stats
paulin_stats_rca = np.zeros((6,len(R2s))) # <m_mean,m_std,c1_mean,c1_std,c2_mean,c2_std>

paulin_stats_rca[0,:],paulin_stats_rca[1,:],paulin_stats_rca[2,:],paulin_stats_rca[3,:],paulin_stats_rca[4,:],paulin_stats_rca[5,:] = utils.paulin_stats(paulin_preds_rca)

paulin_stats_lbda = np.zeros((6,len(R2s))) # <m_mean,m_std,c1_mean,c1_std,c2_mean,c2_std>

paulin_stats_lbda[0,:],paulin_stats_lbda[1,:],paulin_stats_lbda[2,:],paulin_stats_lbda[3,:],paulin_stats_lbda[4,:],paulin_stats_lbda[5,:] = utils.paulin_stats(paulin_preds_lbda)


temp = np.zeros((6,len(R2s))) # <m_mean,m_std,c1_mean,c1_std,c2_mean,c2_std>

paulin_stats_rca_testgal_listPos = []
for pos in range(paulin_preds_rca_testgal.shape[0]): 
    temp[0,:],temp[1,:],temp[2,:],temp[3,:],\
    temp[4,:],temp[5,:] = utils.paulin_stats(paulin_preds_rca_testgal[pos].swapaxes(0,1),flat=True) # make sure format is <30 points of R,  gal or pos, , 3 shapes>    
    paulin_stats_rca_testgal_listPos.append(np.copy(temp))   
paulin_stats_rca_testgal_listPos = np.array(paulin_stats_rca_testgal_listPos)

paulin_stats_rca_testgal_listGals = []
for gal_i in range(paulin_preds_rca_testgal.shape[1]):
    temp[0,:],temp[1,:],temp[2,:],temp[3,:],\
    temp[4,:],temp[5,:] = utils.paulin_stats(paulin_preds_rca_testgal[:,gal_i,:].swapaxes(0,1),flat=True)    
    paulin_stats_rca_testgal_listGals.append(np.copy(temp))    
paulin_stats_rca_testgal_listGals = np.array(paulin_stats_rca_testgal_listGals)

paulin_stats_lbda_testgal_listPos = []
for pos in range(paulin_preds_lbda_testgal.shape[0]): 
    temp[0,:],temp[1,:],temp[2,:],temp[3,:],\
    temp[4,:],temp[5,:] = utils.paulin_stats(paulin_preds_lbda_testgal[pos].swapaxes(0,1),flat=True)    
    paulin_stats_lbda_testgal_listPos.append(np.copy(temp))   
paulin_stats_lbda_testgal_listPos = np.array(paulin_stats_lbda_testgal_listPos)

paulin_stats_lbda_testgal_listGals = []
for gal_i in range(paulin_preds_lbda_testgal.shape[1]):
    temp[0,:],temp[1,:],temp[2,:],temp[3,:],\
    temp[4,:],temp[5,:] = utils.paulin_stats(paulin_preds_lbda_testgal[:,gal_i,:].swapaxes(0,1),flat=True)    
    paulin_stats_lbda_testgal_listGals.append(np.copy(temp))    
paulin_stats_lbda_testgal_listGals = np.array(paulin_stats_lbda_testgal_listGals)

#%% Paulin stats stacking positions and galaxies as a single mix of datapoints

data_stack_lbda = np.zeros((len(R2s),paulin_preds_lbda_testgal.shape[0]*paulin_preds_lbda_testgal.shape[1],3))# <30 points of R,all, 3 shapes> 
for r in range(len(R2s)):
    for s in range(3):
       data_stack_lbda[r,:,s]  = paulin_preds_lbda_testgal[:,:,r,s].reshape(-1) # <pos, gals, Rs, shapes>

paulin_stats_lbda_testgal_All = np.zeros((6,len(R2s))) # <m_mean,m_std,c1_mean,c1_std,c2_mean,c2_std>
paulin_stats_lbda_testgal_All[0,:],paulin_stats_lbda_testgal_All[1,:],paulin_stats_lbda_testgal_All[2,:],paulin_stats_lbda_testgal_All[3,:],\
    paulin_stats_lbda_testgal_All[4,:],paulin_stats_lbda_testgal_All[5,:] = utils.paulin_stats(data_stack_lbda,flat=True) 
 
#%%
data_stack_rca = np.zeros((len(R2s),paulin_preds_rca_testgal.shape[0]*paulin_preds_rca_testgal.shape[1],3))# <30 points of R,all, 3 shapes> 
for r in range(len(R2s)):
    for s in range(3):
       data_stack_rca[r,:,s]  = paulin_preds_rca_testgal[:,:,r,s].reshape(-1) # <pos, gals, Rs, shapes>

paulin_stats_rca_testgal_All = np.zeros((6,len(R2s))) # <m_mean,m_std,c1_mean,c1_std,c2_mean,c2_std>
paulin_stats_rca_testgal_All[0,:],paulin_stats_rca_testgal_All[1,:],paulin_stats_rca_testgal_All[2,:],paulin_stats_rca_testgal_All[3,:],\
    paulin_stats_rca_testgal_All[4,:],paulin_stats_rca_testgal_All[5,:] = utils.paulin_stats(data_stack_rca,flat=True) 

#%% Save work
np.save(results_path+'full_R_pixel.npy',full_R_pixel)
np.save(results_path+'truInt_R_pixel.npy',truInt_R_pixel)
np.save(results_path+'rca_R_pixel.npy',rca_R_pixel)
np.save(results_path+'rca_R_pixel_test.npy',rca_R_pixel_test)
np.save(results_path+'lbdaInt_R_pixel.npy',lbdaInt_R_pixel)

np.save(results_path+'full_shapes.npy',full_shapes)
np.save(results_path+'truInt_shapes.npy',truInt_shapes)
np.save(results_path+'rca_shapes.npy',rca_shapes)
np.save(results_path+'rca_shapes_test.npy',rca_shapes_test)
np.save(results_path+'lbdaInt_shapes.npy',lbdaInt_shapes)


#np.save(results_path+'paulin_preds_rca.npy',paulin_preds_rca)
#np.save(results_path+'paulin_preds_lbda.npy',paulin_preds_lbda)

np.save(results_path+'full_shapes_testgal.npy',full_shapes_testgal)
np.save(results_path+'truInt_shapes_testgal.npy',truInt_shapes_testgal)
np.save(results_path+'lbdaInt_shapes_testgal.npy',lbdaInt_shapes_testgal)

np.save(results_path+'full_R_pixels_testgal.npy',full_R_pixels_testgal)
np.save(results_path+'truInt_R_pixels_testgal.npy',truInt_R_pixels_testgal)
np.save(results_path+'lbdaInt_R_pixels_testgal.npy',lbdaInt_R_pixels_testgal)

np.save(results_path+'paulin_pred_rca_testgal_thrgals.npy',paulin_pred_rca_testgal_thrgals)
np.save(results_path+'paulin_pred_lbda_testgal_thrgals.npy',paulin_pred_lbda_testgal_thrgals)

np.save(results_path+'paulin_stats_rca_testgal_listPos.npy',paulin_stats_rca_testgal_listPos)
np.save(results_path+'paulin_stats_rca_testgal_listGals.npy',paulin_stats_rca_testgal_listGals)
np.save(results_path+'paulin_stats_lbda_testgal_listPos.npy',paulin_stats_lbda_testgal_listPos)
np.save(results_path+'paulin_stats_lbda_testgal_listGals.npy',paulin_stats_lbda_testgal_listGals)
np.save(results_path+'paulin_stats_rca.npy',paulin_stats_rca)
np.save(results_path+'paulin_stats_lbda.npy',paulin_stats_lbda)
np.save(results_path+'paulin_stats_lbda_testgal_All.npy',paulin_stats_lbda_testgal_All)
np.save(results_path+'paulin_stats_rca_testgal_All.npy',paulin_stats_rca_testgal_All)

np.save(results_path+'sizes.npy',sizes)

