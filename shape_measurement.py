#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:24:59 2018

@author: rararipe
"""


import numpy as np
import psf_toolkit as tk
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from tqdm import tqdm


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
root_path = '/Users/rararipe/Documents/Data/'

# PSFs paths
data_path = root_path+'QuickestGenerator/trueSEDs/42x42pixels_8lbdas80pos/'
RCA_path = root_path+'RCA/trueSEDs/42x42pixels_8lbdas80pos/'
lbdaRCA_path = root_path+ 'GradientDescent_output/trueSEDs/42x42pixels_8lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_dict_coef_sigmaEqualsLinTrace_alpha1pBeta0p1_binAradialSR_3it743dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4_newData/result/'
fullres_path = lbdaRCA_path+'full_res/'

# gal SEDs paths
sed_path = root_path+'/galsim/galaxySEDs/'

# Plot path (change later)
plot_path = '/Users/rararipe/Documents/Data/Plots'



if not os.path.exists(fullres_path):
        os.makedirs(fullres_path)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)        
 


#%% Get galaxies SEDs

# Load lbdas used in lbdaRCA
lbdas = np.load(data_path+'lbdas.npy') 
      
galSEDs = utils.interp_SEDs(lbdas,sed_path,plot_path) # <nb_wvl, nb_gal>

      
#%% Load all PSFs, barycenters, components and weights
        
# full res PSFs

PSFs_fr = np.load(data_path+'PSFs_full_res.npy')
PSFs_fr /= np.sum(abs(PSFs_fr), axis=(0,1))
PSFs_fr_W, _ ,nb_wvl , nb_obj  = PSFs_fr.shape

PSFs_integrated_fr = np.load(data_path+'stars_fullres_gt.npy')
PSFs_integrated_fr /= np.sum(abs(PSFs_integrated_fr), axis=(1,2)).reshape((PSFs_integrated_fr.shape[0],1,1))

PSFs_full_res_test_gt = np.load(data_path+'PSFs_full_res_test.npy')
PSFs_full_res_test_gt /= np.sum(abs(PSFs_full_res_test_gt),axis=(0,1))
PSFs_full_res_test_gt =  PSFs_full_res_test_gt.swapaxes(3,2).swapaxes(2,1).swapaxes(1,0) 


fov_test = np.load(data_path+'fov_test.npy')
fov_train = np.load(data_path+'fov.npy')


# and "experimental" PSFs, starting with "true" ones...
truPSFs = np.load(data_path+'PSFs_2euclidrec.npy')

truPSFs /= np.sum(abs(truPSFs), axis=(0,1))
truPSFs = truPSFs.swapaxes(2,3)
truPSFs_integrated = np.load(data_path+'stars_2euclidrec_gt.npy')
truPSFs_integrated /= np.sum(abs(truPSFs_integrated), axis=(0,1))
W,_,_ = truPSFs_integrated.shape

# ... lbdaRCA ones...
lbdaPSFs = np.load(lbdaRCA_path+'psf_est.npy')
lbdaPSFs /= np.sum(abs(lbdaPSFs),axis=(0,1))
lbdaPSFs_integrated = np.load(lbdaRCA_path+'stars_est_2euclidres.npy')
lbdaPSFs_integrated /= np.sum(abs(lbdaPSFs_integrated),axis=(0,1))
barycenters = np.load(lbdaRCA_path+'barycenters.npy')
A_train_lbdaRCA = np.load(lbdaRCA_path+'A.npy')



# ... And RCA ones
rcaPSFs = fits.getdata(RCA_path+'est_psf.fits')
rcaPSFs /= np.sum(abs(rcaPSFs),axis=(0,1))
components_rca = fits.getdata(RCA_path+'components.fits')
A_train_rca = fits.getdata(RCA_path+'A.fits')


#%% Interpolate PSFs with test positions and galSED

paulin_rca_testgal_m_mean_list = []
paulin_rca_testgal_m_std_list = []
paulin_rca_testgal_c1_mean_list = []
paulin_rca_testgal_c1_std_list = []
paulin_rca_testgal_c2_mean_list = []
paulin_rca_testgal_c2_std_list = []

paulin_lbda_testgal_m_mean_list = []
paulin_lbda_testgal_m_std_list = []
paulin_lbda_testgal_c1_mean_list = []
paulin_lbda_testgal_c1_std_list = []
paulin_lbda_testgal_c2_mean_list = []
paulin_lbda_testgal_c2_std_list = []

print "Interpolation done."
for gal_i in tqdm(range(galSEDs.shape[-1])):
    # Ground truth: test positions and galSED
    stars_testGal_fr_gt = PSFs_full_res_test_gt.dot(galSEDs[:,gal_i])
    
    stars_2euclidrec_testGal_gt = np.array([tk.decimate(star, downsamp=6) for star in stars_testGal_fr_gt])
    stars_2euclidrec_testGal_gt = stars_2euclidrec_testGal_gt[:,21:21+42,21:21+42]
    
    stars_testGal_fr_gt /= np.sum(abs(stars_testGal_fr_gt),axis=(1,2)).reshape(stars_testGal_fr_gt.shape[0],1,1) 
    stars_2euclidrec_testGal_gt /= np.sum(abs(stars_2euclidrec_testGal_gt),axis=(1,2)).reshape(stars_2euclidrec_testGal_gt.shape[0],1,1)
    
    # Ground truth: train positions and galSED
    stars_trainGal_fr_gt = (PSFs_fr.swapaxes(3,2).swapaxes(2,1).swapaxes(1,0)).dot(galSEDs[:,gal_i])
    
    stars_2euclidrec_trainGal_gt = np.array([tk.decimate(star, downsamp=6) for star in stars_trainGal_fr_gt])
    stars_2euclidrec_trainGal_gt = stars_2euclidrec_trainGal_gt[:,21:21+42,21:21+42]
    
    stars_trainGal_fr_gt /= np.sum(abs(stars_trainGal_fr_gt),axis=(1,2)).reshape(stars_trainGal_fr_gt.shape[0],1,1) 
    stars_2euclidrec_trainGal_gt /= np.sum(abs(stars_2euclidrec_trainGal_gt),axis=(1,2)).reshape(stars_2euclidrec_trainGal_gt.shape[0],1,1) 
    
    
    
    # lbdaRCA galSED integration
    components_gal = barycenters.dot(galSEDs[:,gal_i]).reshape((W,W,barycenters.shape[1]))
    
    
    # lbdaRCA: test positions and galSED
    A_test_lbdaRCA = utils.rbf_components(A_train_lbdaRCA, fov_train, fov_test) # get A coefficients at galaxy positions
    lbdaPSFs_testGal = components_gal.dot(A_test_lbdaRCA).swapaxes(2,1).swapaxes(1,0) # reconstruct PSF at galaxy positions
    lbdaPSFs_testGal /= np.sum(abs(lbdaPSFs_testGal),axis=(1,2)).reshape(lbdaPSFs_testGal.shape[0],1,1)
    
    # lbdaRCA: train positions and galSED
    lbdaPSFsgal_train = components_gal.dot(A_train_lbdaRCA).swapaxes(2,1).swapaxes(1,0) #construct PSFs using components built with gal SEDs
    lbdaPSFsgal_train /= np.sum(abs(lbdaPSFsgal_train),axis=(1,2)).reshape(lbdaPSFsgal_train.shape[0],1,1)
    
    
    # RCA: test positions
    A_test_rca = utils.rbf_components(A_train_rca, fov_train, fov_test) # get A coefficients at galaxy positions
    rcaPSFs_test = components_rca.dot(A_test_rca).swapaxes(2,1).swapaxes(1,0) # reconstruct PSF at galaxy positions
    rcaPSFs_test /= np.sum(abs(rcaPSFs_test),axis=(1,2)).reshape(rcaPSFs_test.shape[0],1,1)
    
    
    #%%  Compute HSM shapes
    
    full_shapes = utils.computeHSMshapes(PSFs_integrated_fr,pixel_scale) # obs: format of PSFs has to be <nb_obj, pixel, pixel>
    full_shapes_testgal = utils.computeHSMshapes(stars_testGal_fr_gt,pixel_scale)
    full_shapes_traingal = utils.computeHSMshapes(stars_trainGal_fr_gt,pixel_scale)
    
    truInt_shapes = utils.computeHSMshapes(truPSFs_integrated.swapaxes(2,1).swapaxes(1,0),pixel_scale)
    truInt_shapes_testgal = utils.computeHSMshapes(stars_2euclidrec_testGal_gt,pixel_scale)
    truInt_shapes_traingal = utils.computeHSMshapes(stars_2euclidrec_trainGal_gt,pixel_scale)
    
    lbdaInt_shapes = utils.computeHSMshapes(lbdaPSFs_integrated.swapaxes(2,1).swapaxes(1,0),pixel_scale)
    lbdaInt_shapes_testgal = utils.computeHSMshapes(lbdaPSFs_testGal,pixel_scale)
    lbdaInt_shapes_traingal = utils.computeHSMshapes(lbdaPSFsgal_train,pixel_scale)
    
    rca_shapes = utils.computeHSMshapes(rcaPSFs.swapaxes(2,1).swapaxes(1,0),pixel_scale)
    rca_shapes_test = utils.computeHSMshapes(rcaPSFs_test,pixel_scale)
    
    
    
    
    
    #%% Plot shapes
    if plot_each:
        #=== True shapes: full vs 2euclid resolution
        fig, ax = plt.subplots()
        plt.plot(full_shapes[:,0],truInt_shapes[:,0],'.',label=r'$e_1$')
        plt.plot(full_shapes[:,1],truInt_shapes[:,1],'.',label=r'$e_2$')
        plt.legend(loc=0)
        plt.xlabel('PSF ellipticity measured at full resolution')
        plt.ylabel('`Known\' PSF ellipticity')
        plt.title(r'PSF ellipticity with star SEDs at star positions')
        plt.legend(loc=0)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        plt.plot(lims, lims, 'k--')
        
        
        fig, ax = plt.subplots()
        plt.plot(full_shapes_testgal[:,0],truInt_shapes_testgal[:,0],'.',label=r'$e_1$')
        plt.plot(full_shapes_testgal[:,1],truInt_shapes_testgal[:,1],'.',label=r'$e_2$')
        plt.legend(loc=0)
        plt.xlabel('PSF ellipticity measured at full resolution')
        plt.ylabel('`Known\' PSF ellipticity')
        plt.title(r'PSF ellipticity with galaxy SED at galaxy positions')
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        plt.plot(lims, lims, 'k--')
        
        
        fig, ax = plt.subplots()
        plt.plot(full_shapes_traingal[:,0],truInt_shapes_traingal[:,0],'.',label=r'$e_1$')
        plt.plot(full_shapes_traingal[:,1],truInt_shapes_traingal[:,1],'.',label=r'$e_2$')
        plt.legend(loc=0)
        plt.xlabel('PSF ellipticity measured at full resolution')
        plt.ylabel('`Known\' PSF ellipticity')
        plt.title(r'PSF ellipticity with galaxy SED at star positions')
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        plt.plot(lims, lims, 'k--')
        
        
        #=== Recovered shapes
        fig, ax = plt.subplots()
        plt.plot(truInt_shapes[:,0],lbdaInt_shapes[:,0],'o',label=r'lbdaRCA $e_1$')
        plt.plot(truInt_shapes[:,1],lbdaInt_shapes[:,1],'o',label=r'lbdaRCA $e_2$')
        plt.plot(truInt_shapes[:,0],rca_shapes[:,0],'o',label=r'RCA $e_1$')
        plt.plot(truInt_shapes[:,1],rca_shapes[:,1],'o',label=r'RCA $e_2$')
        plt.xlabel('`Known\' PSF ellipticity')
        plt.ylabel('Recovered PSF ellipticity')
        plt.title(r'PSF ellipticity with star SEDs at star positions')
        plt.legend(loc=0)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        plt.plot(lims, lims, 'k--')
        
        fig, ax = plt.subplots()
        plt.plot(truInt_shapes_testgal[:,0],lbdaInt_shapes_testgal[:,0],'o',label=r'lbdaRCA $e_1$')
        plt.plot(truInt_shapes_testgal[:,1],lbdaInt_shapes_testgal[:,1],'o',label=r'lbdaRCA $e_2$')
        plt.plot(truInt_shapes_testgal[:,0],rca_shapes_test[:,0],'o',label=r'RCA $e_1$')
        plt.plot(truInt_shapes_testgal[:,1],rca_shapes_test[:,1],'o',label=r'RCA $e_2$')
        plt.xlabel('`Known\' PSF ellipticity')
        plt.ylabel('Recovered PSF ellipticity')
        plt.title(r'PSF ellipticity with galaxy SED at galaxy positions')
        plt.legend(loc=0)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        plt.plot(lims, lims, 'k--')
        
        
        fig, ax = plt.subplots()
        plt.plot(truInt_shapes_traingal[:,0],lbdaInt_shapes_traingal[:,0],'o',label=r'lbdaRCA $e_1$')
        plt.plot(truInt_shapes_traingal[:,1],lbdaInt_shapes_traingal[:,1],'o',label=r'lbdaRCA $e_2$')
        plt.plot(truInt_shapes_traingal[:,0],rca_shapes[:,0],'o',label=r'RCA $e_1$')
        plt.plot(truInt_shapes_traingal[:,1],rca_shapes[:,1],'o',label=r'RCA $e_2$')
        plt.xlabel('`Known\' PSF ellipticity')
        plt.ylabel('Recovered PSF ellipticity')
        plt.title(r'PSF ellipticity with galaxy SED at star positions')
        plt.legend(loc=0)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        plt.plot(lims, lims, 'k--')
    
    
    #%% Plot pixel errors
    if plot_each:
        plt.hist(truInt_shapes[:,0]-rca_shapes[:,0], bins=11, label='RCA', color='darkorchid', alpha=.6)
        plt.hist(truInt_shapes[:,0]-lbdaInt_shapes[:,0], bins=11, label='lbdaRCA', color='red', alpha=.2)
        plt.xlim(-.06,.06)
        plt.xlabel(r'$e_1^\mathrm{true} - e_1^\mathrm{est}$')
        plt.title('PSF errors with star SEDs at star positions')
        plt.legend()
        plt.show()
        plt.close()
        
        plt.hist(truInt_shapes_testgal[:,0]-rca_shapes_test[:,0], bins=11, label='RCA', color='darkorchid', alpha=.6)
        plt.hist(truInt_shapes_testgal[:,0]-lbdaInt_shapes_testgal[:,0], bins=11, label='lbdaRCA', color='red', alpha=.2)
        plt.xlim(-.06,.06)
        plt.xlabel(r'$e_1^\mathrm{true} - e_1^\mathrm{est}$')
        plt.title('PSF errors with galaxy SEDs at galaxy positions')
        plt.legend()
        plt.show()
        plt.close()
        
        
        plt.hist(truInt_shapes_traingal[:,0]-rca_shapes[:,0], bins=11, label='RCA', color='darkorchid', alpha=.6)
        plt.hist(truInt_shapes_traingal[:,0]-lbdaInt_shapes_traingal[:,0], bins=11, label='lbdaRCA', color='red', alpha=.2)
        plt.xlim(-.06,.06)
        plt.xlabel(r'$e_1^\mathrm{true} - e_1^\mathrm{est}$')
        plt.title('PSF errors with galaxy SEDs at star positions')
        plt.legend()
        plt.show()
        plt.close()
    
        #========= \
        
        plt.hist(truInt_shapes[:,1]-rca_shapes[:,1], bins=11, label='RCA', color='darkorchid', alpha=.6)
        plt.hist(truInt_shapes[:,1]-lbdaInt_shapes[:,1], bins=11, label='lbdaRCA', color='red', alpha=.2)
        #plt.xlim(-.4,.4)
        plt.xlabel(r'$e_2^\mathrm{true} - e_2^\mathrm{est}$')
        plt.title('PSF errors with star SEDs at star positions')
        plt.legend()
        plt.show()
        plt.close()
        
        plt.hist(truInt_shapes_testgal[:,1]-rca_shapes_test[:,1], bins=11, label='RCA', color='darkorchid', alpha=.6)
        plt.hist(truInt_shapes_testgal[:,1]-lbdaInt_shapes_testgal[:,1], bins=11, label='lbdaRCA', color='red', alpha=.2)
        #plt.xlim(-.4,.4)
        plt.xlabel(r'$e_2^\mathrm{true} - e_2^\mathrm{est}$')
        plt.title('PSF errors with galaxy SEDs at galaxy positions')
        plt.legend()
        plt.show()
        plt.close()
        
        
        plt.hist(truInt_shapes_traingal[:,1]-rca_shapes[:,1], bins=11, label='RCA', color='darkorchid', alpha=.6)
        plt.hist(truInt_shapes_traingal[:,1]-lbdaInt_shapes_traingal[:,1], bins=11, label='lbdaRCA', color='red', alpha=.2)
        #plt.xlim(-.4,.4)
        plt.xlabel(r'$e_2^\mathrm{true} - e_2^\mathrm{est}$')
        plt.title('PSF errors with galaxy SEDs at star positions')
        plt.legend()
        plt.show()
        plt.close()
        
        #========= \
        
        plt.hist(truInt_shapes[:,2]-rca_shapes[:,2], bins=11, label='RCA', color='darkorchid', alpha=.6)
        plt.hist(truInt_shapes[:,2]-lbdaInt_shapes[:,2], bins=11, label='lbdaRCA', color='red', alpha=.2)
        #plt.xlim(-.4,.4)
        plt.xlabel(r'$R^2_\mathrm{true} - R^2_\mathrm{est}$')
        plt.title('PSF errors with star SEDs at star positions')
        plt.legend()
        plt.show()
        plt.close()
        
        plt.hist(truInt_shapes_testgal[:,2]-rca_shapes_test[:,2], bins=11, label='RCA', color='darkorchid', alpha=.6)
        plt.hist(truInt_shapes_testgal[:,2]-lbdaInt_shapes_testgal[:,2], bins=11, label='lbdaRCA', color='red', alpha=.2)
        #plt.xlim(-.4,.4)
        plt.xlabel(r'$R^2_\mathrm{true} - R^2_\mathrm{est}$')
        plt.title('PSF errors with galaxy SEDs at galaxy positions')
        plt.legend()
        plt.show()
        plt.close()
        
        plt.hist(truInt_shapes_traingal[:,2]-rca_shapes[:,2], bins=11, label='RCA', color='darkorchid', alpha=.6)
        plt.hist(truInt_shapes_traingal[:,2]-lbdaInt_shapes_traingal[:,2], bins=11, label='lbdaRCA', color='red', alpha=.2)
        #plt.xlim(-.4,.4)
        plt.xlabel(r'$R^2_\mathrm{true} - R^2_\mathrm{est}$')
        plt.title('PSF errors with galaxy SEDs at star positions')
        plt.legend()
        plt.show()
        plt.close()
    
    #%% Convert all to arcsec^2           
    
    full_shapes[:,2] *= pixel_scale**2
    full_shapes_testgal[:,2] *= pixel_scale**2
    full_shapes_traingal[:,2] *= pixel_scale**2
    
    truInt_shapes[:,2] *= twiceEuclid**2
    truInt_shapes_testgal[:,2] *= twiceEuclid**2
    truInt_shapes_traingal[:,2] *= twiceEuclid**2
    
    rca_shapes[:,2] *= twiceEuclid**2
    rca_shapes_test[:,2] *= twiceEuclid**2
    
    lbdaInt_shapes[:,2] *= twiceEuclid**2
    lbdaInt_shapes_testgal[:,2] *= twiceEuclid**2
    lbdaInt_shapes_traingal[:,2] *= twiceEuclid**2
    
    #%% Plot R
    if plot_each:
        fig, ax = plt.subplots()
        plt.plot(full_shapes[:,2],truInt_shapes[:,2],'o',c='k')#0/05 for trueInt_shapes
        plt.plot(full_shapes[:,2],rca_shapes[:,2],'o',c='r')
        plt.plot(full_shapes[:,2],lbdaInt_shapes[:,2],'o',c='darkorchid')
        plt.title("PSF errors with star SEDs at star positions")
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        plt.plot(lims, lims, 'k--')
        
        fig, ax = plt.subplots()
        plt.plot(full_shapes_testgal[:,2],truInt_shapes_testgal[:,2],'o',c='k')#0/05 for trueInt_shapes
        plt.plot(full_shapes_testgal[:,2],rca_shapes_test[:,2],'o',c='r')
        plt.plot(full_shapes_testgal[:,2],lbdaInt_shapes_testgal[:,2],'o',c='darkorchid')
        plt.title("PSF errors with galaxy SEDs at galaxy positions")
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        plt.plot(lims, lims, 'k--')
        
        fig, ax = plt.subplots()
        plt.plot(full_shapes_traingal[:,2],truInt_shapes_traingal[:,2],'o',c='k')#0/05 for trueInt_shapes
        plt.plot(full_shapes_traingal[:,2],rca_shapes[:,2],'o',c='r')
        plt.plot(full_shapes_traingal[:,2],lbdaInt_shapes_traingal[:,2],'o',c='darkorchid')
        plt.title("PSF errors with galaxy SEDs at star positions")
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        plt.plot(lims, lims, 'k--')
    
    #%% Compute Paulin predictions
    sizmin = .1#.04
    sizmax = .5#.9
    nb_points = 30#300
    
    # R ("size") to R^2
    sizes = np.linspace(sizmin, sizmax, nb_points)
    R2s = sizes**2
    
    paulin_rca_m_mean, paulin_rca_m_std, paulin_rca_c1_mean, \
    paulin_rca_c1_std,paulin_rca_c2_mean, paulin_rca_c2_std  = utils.paulin_predict(truInt_shapes, rca_shapes, R2s)
    
    paulin_rca_testgal_m_mean ,paulin_rca_testgal_m_std,paulin_rca_testgal_c1_mean,\
    paulin_rca_testgal_c1_std,paulin_rca_testgal_c2_mean,paulin_rca_testgal_c2_std =  utils.paulin_predict(truInt_shapes_testgal, rca_shapes_test, R2s)
    
    paulin_rca_traingal_m_mean,paulin_rca_traingal_m_std,paulin_rca_traingal_c1_mean,\
    paulin_rca_traingal_c1_std,paulin_rca_traingal_c2_mean,paulin_rca_traingal_c2_std  =  utils.paulin_predict(truInt_shapes_traingal, rca_shapes, R2s)
    
    #===
    
    paulin_lbda_m_mean,paulin_lbda_m_std,paulin_lbda_c1_mean,\
    paulin_lbda_c1_std,paulin_lbda_c2_mean,paulin_lbda_c2_std  =  utils.paulin_predict(truInt_shapes, lbdaInt_shapes, R2s)
    
    
    paulin_lbda_testgal_m_mean,paulin_lbda_testgal_m_std,paulin_lbda_testgal_c1_mean,\
    paulin_lbda_testgal_c1_std,paulin_lbda_testgal_c2_mean,paulin_lbda_testgal_c2_std  =  utils.paulin_predict(truInt_shapes_testgal, lbdaInt_shapes_testgal, R2s)
    
    
    paulin_lbda_traingal_m_mean,paulin_lbda_traingal_m_std,paulin_lbda_traingal_c1_mean,\
    paulin_lbda_traingal_c1_std,paulin_lbda_traingal_c2_mean,paulin_lbda_traingal_c2_std  =  utils.paulin_predict(truInt_shapes_traingal, lbdaInt_shapes_traingal, R2s)
    
    
    
    #===
    
    paulin_rca_testgal_m_mean_list.append(paulin_rca_testgal_m_mean)
    paulin_rca_testgal_m_std_list.append(paulin_rca_testgal_m_std)
    paulin_rca_testgal_c1_mean_list.append(paulin_rca_testgal_c1_mean)
    paulin_rca_testgal_c1_std_list.append(paulin_rca_testgal_c1_std)
    paulin_rca_testgal_c2_mean_list.append(paulin_rca_testgal_c2_mean)
    paulin_rca_testgal_c2_std_list.append(paulin_rca_testgal_c2_std)
    
    paulin_lbda_testgal_m_mean_list.append(paulin_lbda_testgal_m_mean)
    paulin_lbda_testgal_m_std_list.append(paulin_lbda_testgal_m_std)
    paulin_lbda_testgal_c1_mean_list.append(paulin_lbda_testgal_c1_mean)
    paulin_lbda_testgal_c1_std_list.append(paulin_lbda_testgal_c1_std)
    paulin_lbda_testgal_c2_mean_list.append(paulin_lbda_testgal_c2_mean)
    paulin_lbda_testgal_c2_std_list.append(paulin_lbda_testgal_c2_std)
    
    #%% Plot Paulin
    #plt.rcParams['figure.figsize'] = 8,5#12,7.9
    leg = ['RCA', 'lbdaRCA']
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['axes.labelsize'] = 17 
    plt.rcParams['axes.titlesize'] = 17 
    plt.rcParams['font.size'] = 22
    plt.rcParams['lines.linewidth'] = 1.3 # he has 2.0 here but I like 1.3 better
    plt.rcParams['lines.markersize'] = 8 
    plt.rcParams['legend.fontsize'] = 17
    plt.rcParams['text.usetex'] = True 
    plt.rcParams['font.family'] = 'serif' 
    plt.rcParams['font.serif'] = 'cm'
    
    if plot_each:
        
        
        # Multiplicative bias
        
        plt.errorbar(sizes, paulin_rca_m_mean, yerr=paulin_rca_m_std, c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_m_mean, yerr=paulin_lbda_m_std, c='red', alpha=.7, capsize=5)
        plt.xlabel(r'Galaxy size $R$ (arcsec)')
        plt.ylabel(r'Paulin predicted $m$')
        
        plt.legend(leg)
        plt.title('Multiplicative bias: with star SEDs at star positions')
        plt.show()
        
        plt.errorbar(sizes, paulin_rca_testgal_m_mean, yerr=paulin_rca_testgal_m_std, c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_m_mean, yerr=paulin_lbda_testgal_m_std, c='red', alpha=.7, capsize=5)
        plt.xlabel(r'Galaxy size $R$ (arcsec)')
        plt.ylabel(r'Paulin predicted $m$')
        
        plt.legend(leg)
        plt.title('Multiplicative bias: with galaxy SEDs at galaxy positions')
        plt.show()
        
        plt.errorbar(sizes, paulin_rca_traingal_m_mean, yerr=paulin_rca_traingal_m_std, c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_traingal_m_mean, yerr=paulin_lbda_traingal_m_std, c='red', alpha=.7, capsize=5)
        plt.xlabel(r'Galaxy size $R$ (arcsec)')
        plt.ylabel(r'Paulin predicted $m$')
        
        plt.legend(leg)
        plt.title('Multiplicative bias: with galaxy SEDs at star positions')
        plt.show()
        
        
        # Additive bias
        
        plt.errorbar(sizes, paulin_rca_c1_mean, yerr=paulin_rca_c1_std, c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_c1_mean, yerr=paulin_lbda_c1_std, c='red', alpha=.7, capsize=5)
        
        plt.xlabel(r'Galaxy size $R$ (arcsec)')
        plt.ylabel(r'Paulin predicted $c_1$')
        plt.legend(leg)
        plt.xlim(.075,0.525)
        plt.plot([0,0.6], [0,0], 'k--')
        plt.title('Additive bias (1st component): with star SEDs at star positions')
        plt.show()
        
        
        plt.errorbar(sizes, paulin_rca_testgal_c1_mean, yerr=paulin_rca_testgal_c1_std, c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_c1_mean, yerr=paulin_lbda_testgal_c1_std, c='red', alpha=.7, capsize=5)
        
        plt.xlabel(r'Galaxy size $R$ (arcsec)')
        plt.ylabel(r'Paulin predicted $c_1$')
        plt.legend(leg)
        plt.xlim(.075,0.525)
        plt.plot([0,0.6], [0,0], 'k--')
        plt.title('Additive bias (1st component): with galaxy SEDs at galaxy positions')
        plt.show()
        
        plt.errorbar(sizes, paulin_rca_traingal_c1_mean, yerr=paulin_rca_traingal_c1_std, c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_traingal_c1_mean, yerr=paulin_lbda_traingal_c1_std, c='red', alpha=.7, capsize=5)
        
        
        plt.xlabel(r'Galaxy size $R$ (arcsec)')
        plt.ylabel(r'Paulin predicted $c_1$')
        plt.legend(leg)
        plt.xlim(.075,0.525)
        plt.plot([0,0.6], [0,0], 'k--')
        plt.title('Additive bias (1st component): with galaxy SEDs at star positions')
        plt.show()
        
        
        #==
        
        plt.errorbar(sizes, paulin_rca_c2_mean, yerr=paulin_rca_c2_std, c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_c2_mean, yerr=paulin_lbda_c2_std, c='red', alpha=.7, capsize=5)
        
        plt.xlabel(r'Galaxy size $R$ (arcsec)')
        plt.ylabel(r'Paulin predicted $c_2$')
        plt.legend(leg)
        plt.xlim(.075,0.525)
        plt.plot([0,0.6], [0,0], 'k--')
        plt.title('Additive bias (2nd component): with star SEDs at star positions')
        plt.show()
        
        plt.errorbar(sizes, paulin_rca_testgal_c2_mean, yerr=paulin_rca_testgal_c2_std, c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_c2_mean, yerr=paulin_lbda_testgal_c2_std, c='red', alpha=.7, capsize=5)
        
        plt.xlabel(r'Galaxy size $R$ (arcsec)')
        plt.ylabel(r'Paulin predicted $c_2$')
        plt.legend(leg)
        plt.xlim(.075,0.525)
        plt.plot([0,0.6], [0,0], 'k--')
        plt.title('Additive bias (2nd component): with galaxy SEDs at galaxy positions')
        plt.show()
        
        plt.errorbar(sizes, paulin_rca_traingal_c2_mean, yerr=paulin_rca_traingal_c2_std, c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_traingal_c2_mean, yerr=paulin_lbda_traingal_c2_std, c='red', alpha=.7, capsize=5)
        
        plt.xlabel(r'Galaxy size $R$ (arcsec)')
        plt.ylabel(r'Paulin predicted $c_2$')
        plt.legend(leg)
        plt.xlim(.075,0.525)
        plt.plot([0,0.6], [0,0], 'k--')
        plt.title('Additive bias (2nd component): with galaxy SEDs at star positions')
        plt.show()
        
 

#%%                  
if plot_paulin_testGal_all:
    
    for gal in range(len(paulin_rca_testgal_m_mean_list)):
        plt.errorbar(sizes, paulin_rca_testgal_m_mean_list[gal], yerr=paulin_rca_testgal_m_std_list[gal], c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_m_mean_list[gal], yerr=paulin_lbda_testgal_m_std_list[gal], c='red', alpha=.7, capsize=5)
    plt.xlabel(r'Galaxy size $R$ (arcsec)')
    plt.ylabel(r'Paulin predicted $m$')     
    plt.legend(leg)
    plt.title('Multiplicative bias: with galaxy SEDs at galaxy positions')
    plt.show()
    
    for gal in range(len(paulin_rca_testgal_m_mean_list)):
        plt.errorbar(sizes, paulin_rca_testgal_c1_mean_list[gal], yerr=paulin_rca_testgal_c1_std_list[gal], c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_c1_mean_list[gal], yerr=paulin_lbda_testgal_c1_std_list[gal], c='red', alpha=.7, capsize=5)
        
    plt.xlabel(r'Galaxy size $R$ (arcsec)')
    plt.ylabel(r'Paulin predicted $c_1$')
    plt.legend(leg)
    plt.xlim(.075,0.525)
    plt.plot([0,0.6], [0,0], 'k--')
    plt.title('Additive bias (1st component): with galaxy SEDs at galaxy positions')
    plt.show()
    
    for gal in range(len(paulin_rca_testgal_m_mean_list)):
        plt.errorbar(sizes, paulin_rca_testgal_c2_mean_list[gal], yerr=paulin_rca_testgal_c2_std_list[gal], c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_c2_mean_list[gal], yerr=paulin_lbda_testgal_c2_std_list[gal], c='red', alpha=.7, capsize=5)
        
    plt.xlabel(r'Galaxy size $R$ (arcsec)')
    plt.ylabel(r'Paulin predicted $c_2$')
    plt.legend(leg)
    plt.xlim(.075,0.525)
    plt.plot([0,0.6], [0,0], 'k--')
    plt.title('Additive bias (2nd component): with galaxy SEDs at galaxy positions')
    plt.show() 
    
    
#%%
if plot_paulin_testGal_all:    
    for gal in range(len(paulin_rca_testgal_m_mean_list)):
        plt.errorbar(sizes, paulin_rca_testgal_m_mean_list[gal], c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_m_mean_list[gal], c='red', alpha=.7, capsize=5)
    plt.xlabel(r'Galaxy size $R$ (arcsec)')
    plt.ylabel(r'Paulin predicted $m$')     
    plt.legend(leg)
    plt.title('Multiplicative bias: with galaxy SEDs at galaxy positions MEANS')
    plt.show()
    
    for gal in range(len(paulin_rca_testgal_m_mean_list)):
        plt.errorbar(sizes, paulin_rca_testgal_m_std_list[gal], c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_m_std_list[gal], c='red', alpha=.7, capsize=5)
    plt.xlabel(r'Galaxy size $R$ (arcsec)')
    plt.ylabel(r'Paulin predicted $m$')     
    plt.legend(leg)
    plt.title('Multiplicative bias: with galaxy SEDs at galaxy positions STDS')
    plt.show()
    
    
    for gal in range(len(paulin_rca_testgal_m_mean_list)):
        plt.errorbar(sizes, paulin_rca_testgal_c1_mean_list[gal], c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_c1_mean_list[gal], c='red', alpha=.7, capsize=5)
        
    plt.xlabel(r'Galaxy size $R$ (arcsec)')
    plt.ylabel(r'Paulin predicted $c_1$')
    plt.legend(leg)
    plt.xlim(.075,0.525)
    plt.plot([0,0.6], [0,0], 'k--')
    plt.title('Additive bias (1st component): with galaxy SEDs at galaxy positions MEANS')
    plt.show()
    
    for gal in range(len(paulin_rca_testgal_m_mean_list)):
        plt.errorbar(sizes, paulin_rca_testgal_c1_std_list[gal], c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_c1_std_list[gal], c='red', alpha=.7, capsize=5)
        
    plt.xlabel(r'Galaxy size $R$ (arcsec)')
    plt.ylabel(r'Paulin predicted $c_1$')
    plt.legend(leg)
    plt.xlim(.075,0.525)
    plt.plot([0,0.6], [0,0], 'k--')
    plt.title('Additive bias (1st component): with galaxy SEDs at galaxy positions STDS')
    plt.show()
    
    for gal in range(len(paulin_rca_testgal_m_mean_list)):
        plt.errorbar(sizes, paulin_rca_testgal_c2_mean_list[gal], c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_c2_mean_list[gal], c='red', alpha=.7, capsize=5)
        
    plt.xlabel(r'Galaxy size $R$ (arcsec)')
    plt.ylabel(r'Paulin predicted $c_2$')
    plt.legend(leg)
    plt.xlim(.075,0.525)
    plt.plot([0,0.6], [0,0], 'k--')
    plt.title('Additive bias (2nd component): with galaxy SEDs at galaxy positions MEANS')
    plt.show()     
    
    for gal in range(len(paulin_rca_testgal_m_mean_list)):
        plt.errorbar(sizes, paulin_rca_testgal_c2_std_list[gal], c='darkorchid', lw=2, capsize=5)
        plt.errorbar(sizes, paulin_lbda_testgal_c2_std_list[gal], c='red', alpha=.7, capsize=5)
        
    plt.xlabel(r'Galaxy size $R$ (arcsec)')
    plt.ylabel(r'Paulin predicted $c_2$')
    plt.legend(leg)
    plt.xlim(.075,0.525)
    plt.plot([0,0.6], [0,0], 'k--')
    plt.title('Additive bias (2nd component): with galaxy SEDs at galaxy positions STDS')
    plt.show()     
    
    
    