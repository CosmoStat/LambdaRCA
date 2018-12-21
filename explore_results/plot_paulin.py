#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 09:52:47 2018

@author: rararipe
"""
import numpy as np
import psf_toolkit as tk
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import os


#%% Load data
load_path = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/ULTIMATE_8wvl/result/Shape_measurements/'
#load_path = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_12lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4/result/Shape_measurements/'
plot_path = load_path + 'Plots/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)       
ID = 0
save_plot = True


full_R_pixel = np.load(load_path+'full_R_pixel.npy')
truInt_R_pixel = np.load(load_path+'truInt_R_pixel.npy')
rca_R_pixel = np.load(load_path+'rca_R_pixel.npy')
rca_R_pixel_test = np.load(load_path+'rca_R_pixel_test.npy')
lbdaInt_R_pixel = np.load(load_path+'lbdaInt_R_pixel.npy')

full_shapes = np.load(load_path+'full_shapes.npy')
truInt_shapes = np.load(load_path+'truInt_shapes.npy')
rca_shapes = np.load(load_path+'rca_shapes.npy')
rca_shapes_test = np.load(load_path+'rca_shapes_test.npy')
lbdaInt_shapes = np.load(load_path+'lbdaInt_shapes.npy')

full_shapes_testgal = np.load(load_path+'full_shapes_testgal.npy')
truInt_shapes_testgal = np.load(load_path+'truInt_shapes_testgal.npy')
lbdaInt_shapes_testgal = np.load(load_path+'lbdaInt_shapes_testgal.npy')

full_R_pixels_testgal = np.load(load_path+'full_R_pixels_testgal.npy')
truInt_R_pixels_testgal = np.load(load_path+'truInt_R_pixels_testgal.npy')
lbdaInt_R_pixels_testgal = np.load(load_path+'lbdaInt_R_pixels_testgal.npy')

paulin_stats_rca_testgal_listPos = np.load(load_path+'paulin_stats_rca_testgal_listPos.npy')
paulin_stats_rca_testgal_listGals = np.load(load_path+'paulin_stats_rca_testgal_listGals.npy')
paulin_stats_lbda_testgal_listPos = np.load(load_path+'paulin_stats_lbda_testgal_listPos.npy')
paulin_stats_lbda_testgal_listGals = np.load(load_path+'paulin_stats_lbda_testgal_listGals.npy')

paulin_stats_rca = np.load(load_path+'paulin_stats_rca.npy')
paulin_stats_lbda = np.load(load_path+'paulin_stats_lbda.npy')
sizes = np.load(load_path+'sizes.npy')

paulin_stats_lbda_testgal_All = np.load(load_path+'paulin_stats_lbda_testgal_All.npy')
paulin_stats_rca_testgal_All = np.load(load_path+'paulin_stats_rca_testgal_All.npy')

fov = np.load(load_path+'fov.npy')
fov_test = np.load(load_path+'fov_test.npy')
#%% Plot shapes

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

fig, ax = plt.subplots()
plt.plot(full_shapes[:,0],truInt_shapes[:,0],'o',markeredgecolor='midnightblue',label=r'$e_1$')
plt.plot(full_shapes[:,1],truInt_shapes[:,1],'o',markeredgecolor='midnightblue',label=r'$e_2$')
plt.legend(loc=0)
plt.xlabel('PSF ellipticity measured at full resolution')
plt.ylabel('`Known\' PSF ellipticity')
#plt.title(r'PSF ellipticity with star SEDs at star positions')
plt.legend(loc=0)
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')

if save_plot:
    plt.savefig(plot_path+'truee_vs_knowe_scatter_starSEDs_starPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'truee_vs_knowe_scatter_starSEDs_starPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
#%%
cmap_1=cm.winter 
cmap_2 = cm.spring 
cmap_3 = cm.Greys
cmap_4 = cm.get_cmap('Spectral')
color_indx = 1.0/(full_shapes_testgal.shape[1]-1)*np.array(range(full_shapes_testgal.shape[1])) # for galaxies
color_indx_2 = 1.0/(full_shapes_testgal.shape[0]-1)*np.array(range(full_shapes_testgal.shape[0])) # for positions

gals_chosen = np.random.permutation(range(full_shapes_testgal.shape[1]))[:100]
color_index_gc = 1.0/((len(gals_chosen)-1))*np.array(range(len(gals_chosen))) # for galaxies
color_index_gc = np.random.permutation(color_index_gc)
c_gc = np.arange(1,len(gals_chosen)+1)

pos_chosen = np.random.permutation(range(full_shapes_testgal.shape[0]))[:100]
color_index_pc = 1.0/((len(pos_chosen)-1))*np.array(range(len(pos_chosen))) # for galaxies
color_index_pc = np.random.permutation(color_index_pc)
c_pc = np.arange(1,len(pos_chosen)+1)

#%%
fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(gals_chosen))
dummie_cax = ax.scatter(c_gc, c_gc, c=c_gc, cmap=this_cmap)
ax.cla()
leg = [r'$e_1$',r'$e_2$']
for i in range(len(gals_chosen)):
        gal_i = gals_chosen[i]
        plt.plot(full_shapes_testgal[pos_chosen,gal_i,0],truInt_shapes_testgal[pos_chosen,gal_i,0],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_gc[i]),alpha = 0.7)
        plt.plot(full_shapes_testgal[pos_chosen,gal_i,1],truInt_shapes_testgal[pos_chosen,gal_i,1],'*',markeredgecolor='midnightblue',markersize=17.0,c=this_cmap(color_index_gc[i]),alpha = 0.7)
plt.legend(leg)
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Galaxies')
plt.xlabel('PSF ellipticity measured at full resolution')
plt.ylabel('`Known\' PSF ellipticity')
#plt.title(r'PSF ellipticity with galaxy SED at galaxy positions')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'truee_vs_knowe_scatter_cbGalaxies.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'truee_vs_knowe_scatter_cbGalaxies.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show() 
plt.close()

fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(pos_chosen))
dummie_cax = ax.scatter(c_pc, c_pc, c=c_pc, cmap=this_cmap)
ax.cla()
leg = [r'$e_1$',r'$e_2$']
for i in range(len(pos_chosen)):
        pos = pos_chosen[i]
        plt.plot(full_shapes_testgal[pos,gals_chosen,0],truInt_shapes_testgal[pos,gals_chosen,0],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_pc[i]),alpha = 0.7)
        plt.plot(full_shapes_testgal[pos,gals_chosen,1],truInt_shapes_testgal[pos,gals_chosen,1],'*',markeredgecolor='midnightblue',markersize=17.0,c=this_cmap(color_index_pc[i]),alpha = 0.7)
plt.legend(leg)
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Positions')
plt.xlabel('PSF ellipticity measured at full resolution')
plt.ylabel('`Known\' PSF ellipticity')
#plt.title(r'PSF ellipticity with galaxy SED at galaxy positions')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'truee_vs_knowe_scatter_cbPositions.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'truee_vs_knowe_scatter_cbPositions.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show() 
plt.close()

#fig, ax = plt.subplots()
#for i in range(len(pos_chosen)):
#    pos = pos_chosen[i]
#    plt.plot(fov_test[pos,0],fov_test[pos,1], 'o', markeredgecolor='midnightblue',c=this_cmap(color_index_pc[i]),alpha = 1.0)
#plt.xlabel('$x$')
#plt.ylabel('$y$')
#plt.title(r'FOV')
#plt.show() 
#plt.close()
#%%
#=== Recovered shapes
fig, ax = plt.subplots()
plt.plot(truInt_shapes[:,0],lbdaInt_shapes[:,0],'o',markeredgecolor='midnightblue',label=r'$\lambda$RCA $e_1$',c='blue')
plt.plot(truInt_shapes[:,1],lbdaInt_shapes[:,1],'*',markeredgecolor='midnightblue',markersize=15.0,label=r'$\lambda$RCA $e_2$',c='blue')
plt.plot(truInt_shapes[:,0],rca_shapes[:,0],'o',markeredgecolor='midnightblue',label=r'RCA $e_1$',c='red')
plt.plot(truInt_shapes[:,1],rca_shapes[:,1],'*',markeredgecolor='midnightblue',markersize=15.0,label=r'RCA $e_2$',c='red')
plt.xlabel('`Known\' PSF ellipticity')
plt.ylabel('Recovered PSF ellipticity')
#plt.title(r'PSF ellipticity with star SEDs at star positions')
plt.legend(loc=0)
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'rca_vs_lambda_elip_scatter_starSEDs_starPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_vs_lambda_elip_scatter_starSEDs_starPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
#%%    
fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(gals_chosen))
dummie_cax = ax.scatter(c_gc, c_gc, c=c_gc, cmap=this_cmap)
ax.cla()
for i in range(len(gals_chosen)):
    gal_i = gals_chosen[i]
    plt.plot(truInt_shapes_testgal[pos_chosen,gal_i,0],lbdaInt_shapes_testgal[pos_chosen,gal_i,0],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_gc[i]))

plt.xlabel('`Known\' PSF ellipticity $e_1$')
plt.ylabel('Recovered PSF ellipticity $e_1$')
plt.title(r'$\lambda$RCA')
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Galaxies')
#plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'lambda_el1_cbGalaxies.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'lambda_el1_cbGalaxies.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(gals_chosen))
dummie_cax = ax.scatter(c_pc, c_pc, c=c_pc, cmap=this_cmap)
ax.cla()
for i in range(len(pos_chosen)):
    pos = pos_chosen[i]
    plt.plot(truInt_shapes_testgal[pos,gals_chosen,0],lbdaInt_shapes_testgal[pos,gals_chosen,0],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_pc[i]))

plt.xlabel('`Known\' PSF ellipticity $e_1$')
plt.ylabel('Recovered PSF ellipticity $e_1$')
plt.title(r'$\lambda$RCA')
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Positions')
#plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'lambda_e1_cbPositions.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'lambda_e1_cbPositions.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
#%%
fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(gals_chosen))
dummie_cax = ax.scatter(c_gc, c_gc, c=c_gc, cmap=this_cmap)
ax.cla()
for i in range(len(gals_chosen)):
    gal_i = gals_chosen[i]
    plt.plot(truInt_shapes_testgal[pos_chosen,gal_i,1],lbdaInt_shapes_testgal[pos_chosen,gal_i,1],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_gc[i]))

plt.xlabel('`Known\' PSF ellipticity $e_2$')
plt.ylabel('Recovered PSF ellipticity $e_2$')
plt.title(r'$\lambda$RCA')
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Galaxies')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'lambda_el2_cbGalaxies.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'lambda_el2_cbGalaxies.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(pos_chosen))
dummie_cax = ax.scatter(c_pc, c_pc, c=c_pc, cmap=this_cmap)
ax.cla()
for i in range(len(pos_chosen)):
    pos = pos_chosen[i]
    plt.plot(truInt_shapes_testgal[pos,gals_chosen,1],lbdaInt_shapes_testgal[pos,gals_chosen,1],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_pc[i]))

plt.xlabel('`Known\' PSF ellipticity $e_2$')
plt.ylabel('Recovered PSF ellipticity $e_2$')
plt.title(r'$\lambda$RCA')
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Positions')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'lambda_el2_cbPositions.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'lambda_el2_cbPositions.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
#%%
fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(gals_chosen))
dummie_cax = ax.scatter(c_gc, c_gc, c=c_gc, cmap=this_cmap)
ax.cla()
for i in range(len(gals_chosen)):
    gal_i = gals_chosen[i]
    plt.plot(truInt_shapes_testgal[pos_chosen,gal_i,0],rca_shapes_test[pos_chosen,0],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_gc[i]))
plt.xlabel('`Known\' PSF ellipticity $e_1$')
plt.ylabel('Recovered PSF ellipticity $e_1$')
plt.title(r'RCA')
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Galaxies')
#plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'rca_el1_cbGalaxies.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_el1_cbGalaxies.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()


fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(pos_chosen))
dummie_cax = ax.scatter(c_pc, c_pc, c=c_pc, cmap=this_cmap)
ax.cla()
for i in range(len(pos_chosen)):
    pos = pos_chosen[i]
    plt.plot(truInt_shapes_testgal[pos,gals_chosen,0].reshape(-1),np.repeat(rca_shapes_test[pos,0],len(gals_chosen)),'o',markeredgecolor='midnightblue',c=this_cmap(color_index_pc[i]))
plt.xlabel('`Known\' PSF ellipticity $e_1$')
plt.ylabel('Recovered PSF ellipticity $e_1$')
plt.title(r'RCA')
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Positions')
#plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'rca_el1_cbPositions.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_el1_cbPositions.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
#%%
fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(gals_chosen))
dummie_cax = ax.scatter(c_gc, c_gc, c=c_gc, cmap=this_cmap)
ax.cla()
for i in range(len(gals_chosen)):
    gal_i = gals_chosen[i]
    plt.plot(truInt_shapes_testgal[pos_chosen,gal_i,1],rca_shapes_test[pos_chosen,1],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_gc[i]))
plt.xlabel('`Known\' PSF ellipticity $e_2$')
plt.ylabel('Recovered PSF ellipticity $e_2$')
plt.title(r'RCA')
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Galaxies')
#plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'rca_el2_cbGalaxies.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_el2_cbGalaxies.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(pos_chosen))
dummie_cax = ax.scatter(c_pc, c_pc, c=c_pc, cmap=this_cmap)
ax.cla()
for i in range(len(pos_chosen)):
    pos = pos_chosen[i]
    plt.plot(truInt_shapes_testgal[pos,gals_chosen,1].reshape(-1),np.repeat(rca_shapes_test[pos,1],len(gals_chosen)),'o',markeredgecolor='midnightblue',c=this_cmap(color_index_pc[i]))
plt.xlabel('`Known\' PSF ellipticity $e_2$')
plt.ylabel('Recovered PSF ellipticity $e_2$')
plt.title(r'RCA')
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Positions')
#plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'rca_el2_cbPositions.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_el2_cbPositions.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
#%%
fig, ax = plt.subplots()
plt.plot(full_shapes[:,2],truInt_shapes[:,2],'o',markeredgecolor='midnightblue',c='k')#0/05 for trueInt_shapes
plt.plot(full_shapes[:,2],rca_shapes[:,2],'o',markeredgecolor='midnightblue',label=r'RCA',c='r')
plt.plot(full_shapes[:,2],lbdaInt_shapes[:,2],'o',markeredgecolor='midnightblue',label=r'$\lambda$RCA',c='blue')
#plt.title("PSF sizes with star SEDs at star positions")
plt.xlabel('`Known\' PSF size $R$')
plt.ylabel('Recovered PSF size $R$')
plt.legend()
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
lims = [np.min([ax.get_xlim(), ax.get_ylim()]), 0.006]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_true_R_scatter_starSEDs_starPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_true_R_scatter_starSEDs_starPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1
plt.show()
plt.close()

#%%    
leg = ['true','RCA', '$\lambda$RCA']
fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(gals_chosen))
grey_cmap = cm.get_cmap('Greys',len(gals_chosen))
dummie_cax = ax.scatter(c_gc, c_gc, c=c_gc, cmap=this_cmap)
ax.cla()
for i in range(len(gals_chosen)):
    gal_i = gals_chosen[i]
    plt.plot(full_shapes_testgal[pos_chosen,gal_i,2],truInt_shapes_testgal[pos_chosen,gal_i,2],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_gc[i]))#0/05 for trueInt_shapes
    plt.plot(full_shapes_testgal[pos_chosen,gal_i,2],rca_shapes_test[pos_chosen,2],'v',markeredgecolor='midnightblue',c=this_cmap(color_index_gc[i]))
    plt.plot(full_shapes_testgal[pos_chosen,gal_i,2],lbdaInt_shapes_testgal[pos_chosen,gal_i,2],'*',markersize=15,markeredgecolor='midnightblue',c=this_cmap(color_index_gc[i]))
#plt.title("PSF sizes with galaxy SEDs at galaxy positions")
plt.xlabel('`Known\' PSF size $R$')
plt.ylabel('Recovered PSF size $R$')
plt.legend(leg) #,loc='lower left', bbox_to_anchor=(1, 0.5)

cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Galaxies')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
lims = [np.min([ax.get_xlim(), ax.get_ylim()]), 0.007]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_true_R_cbGalaxies_galSEDs_galPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_true_R_cbGalaxies_galSEDs_galPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()   



fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',len(pos_chosen))
grey_cmap = cm.get_cmap('Greys',len(pos_chosen))
dummie_cax = ax.scatter(c_pc, c_pc, c=c_pc, cmap=this_cmap)
ax.cla()
for i in range(len(pos_chosen)):
    pos = pos_chosen[i]
    plt.plot(full_shapes_testgal[pos,gals_chosen,2],truInt_shapes_testgal[pos,gals_chosen,2],'o',markeredgecolor='midnightblue',c=this_cmap(color_index_pc[i]),alpha=0.7)#0/05 for trueInt_shapes
    plt.plot(full_shapes_testgal[pos,gals_chosen,2],[rca_shapes_test[pos,2] for j in range(len(gals_chosen))],'v',markeredgecolor='midnightblue',c=this_cmap(color_index_pc[i]))
    plt.plot(full_shapes_testgal[pos,gals_chosen,2],lbdaInt_shapes_testgal[pos,gals_chosen,2],'*',markersize=15,markeredgecolor='midnightblue',c=this_cmap(color_index_pc[i]))
#plt.title("PSF errors with galaxy SEDs at galaxy positions")
plt.legend(leg) #,loc='lower left', bbox_to_anchor=(1, 0.5)
plt.xlabel('`Known\' PSF size $R$')
plt.ylabel('Recovered PSF size $R$')

cb=fig.colorbar(dummie_cax)
cb.set_ticks([])
cb.set_label(r'Positions')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
lims = [np.min([ax.get_xlim(), ax.get_ylim()]), 0.007]
plt.plot(lims, lims, 'k--')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_true_R_cbPositions_galSEDs_galPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_true_R_cbPositions_galSEDs_galPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()  
plt.close()


#%% Plot errors histograms 

fig, ax = plt.subplots()
plt.hist(truInt_shapes[:,0]-rca_shapes[:,0], bins=11, label='RCA', color='darkorchid', alpha=.6)
plt.hist(truInt_shapes[:,0]-lbdaInt_shapes[:,0], bins=11, label=r'$\lambda$RCA', color='red', alpha=.2)
plt.xlim(-.06,.06)
plt.xlabel(r'$e_1^\mathrm{true} - e_1^\mathrm{est}$')
plt.ylabel(r'Counts')
#plt.title('PSF errors with star SEDs at star positions')
plt.legend()
if save_plot:
    plt.savefig(plot_path+'rca_lambda_histerror_e1_starSEDs_starPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_histerror_e1_starSEDs_starPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%%
fig, ax = plt.subplots()
plt.hist(truInt_shapes_testgal[:,:,0].reshape(-1)-np.repeat(rca_shapes_test[:,0],truInt_shapes_testgal.shape[1]), bins=11, label='RCA', color='darkorchid', alpha=.6)
plt.hist(truInt_shapes_testgal[:,:,0].reshape(-1)-lbdaInt_shapes_testgal[:,:,0].reshape(-1), bins=11, label=r'$\lambda$RCA',color='red', alpha=.2)
plt.xlim(-.06,.06)
plt.xlabel(r'$e_1^\mathrm{true} - e_1^\mathrm{est}$')
plt.ylabel(r'Counts')
#plt.title('PSF errors with galaxy SEDs at galaxy positions')
plt.legend()
if save_plot:
    plt.savefig(plot_path+'rca_lambda_histerror_e1_galSEDs_galPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_histerror_e1_galSEDs_galPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
  
#%%      
fig, ax = plt.subplots()        
plt.hist(truInt_shapes[:,1]-rca_shapes[:,1], bins=11, label='RCA', color='darkorchid', alpha=.6)
plt.hist(truInt_shapes[:,1]-lbdaInt_shapes[:,1], bins=11, label=r'$\lambda$RCA', color='red', alpha=.2)
plt.xlabel(r'$e_2^\mathrm{true} - e_2^\mathrm{est}$')
plt.ylabel(r'Counts')
#plt.title('PSF errors with star SEDs at star positions')
plt.legend()
if save_plot:
    plt.savefig(plot_path+'rca_lambda_histerror_e2_starSEDs_starPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_histerror_e2_starSEDs_starPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%%
fig, ax = plt.subplots()
plt.hist(truInt_shapes_testgal[:,:,1].reshape(-1)-np.repeat(rca_shapes_test[:,1],truInt_shapes_testgal.shape[1]), bins=11, label='RCA', color='darkorchid', alpha=.6)
plt.hist(truInt_shapes_testgal[:,:,1].reshape(-1)-lbdaInt_shapes_testgal[:,:,1].reshape(-1), bins=11, label=r'$\lambda$RCA',color='red', alpha=.2)
#plt.xlim(-.4,.4)
plt.xlabel(r'$e_2^\mathrm{true} - e_2^\mathrm{est}$')
plt.ylabel(r'Counts')
#plt.title('PSF errors with galaxy a SED at galaxy positions')
plt.legend()
if save_plot:
    plt.savefig(plot_path+'rca_lambda_histerror_e2_galSEDs_galPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_histerror_e2_galSEDs_galPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%%
      
fig, ax = plt.subplots()       
plt.hist(truInt_R_pixel-rca_R_pixel, bins=11, label='RCA', color='darkorchid', alpha=.6)
plt.hist(truInt_R_pixel-lbdaInt_R_pixel, bins=11, label=r'$\lambda$RCA', color='red', alpha=.2)
plt.xlabel(r'$R^2_\mathrm{true} - R^2_\mathrm{est}$')
plt.ylabel(r'Counts')
#plt.title('PSF errors with star SEDs at star positions')
plt.legend()
if save_plot:
    plt.savefig(plot_path+'rca_lambda_histerror_R2_starSEDs_starPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_histerror_R2_starSEDs_starPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
#%%
fig, ax = plt.subplots()
plt.hist(truInt_R_pixels_testgal[:,:].reshape(-1)-np.repeat(rca_R_pixel_test,truInt_R_pixels_testgal.shape[1]), bins=11, label='RCA', color='darkorchid', alpha=.6)
plt.hist(truInt_R_pixels_testgal[:,:].reshape(-1)-lbdaInt_R_pixels_testgal[:,:].reshape(-1), bins=11, label=r'$\lambda$RCA', color='red', alpha=.2)
#plt.xlim(-.4,.4)
plt.xlabel(r'$R^2_\mathrm{true} - R^2_\mathrm{est}$')
plt.ylabel(r'Counts')
plt.title('PSF errors with a galaxy SED at galaxy positions')
plt.legend()
if save_plot:
    plt.savefig(plot_path+'rca_lambda_histerror_R2_galSEDs_galPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_histerror_R2_galSEDs_galPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%% Plot Paulin
leg = [r'RCA', r'$\lambda$RCA']

#%%
# Multiplicative bias
plt.errorbar(sizes,paulin_stats_rca[0], yerr=paulin_stats_rca[1], c='darkorchid', lw=2, capsize=5)
plt.errorbar(sizes, paulin_stats_lbda[0], yerr=paulin_stats_lbda[1], c='red', alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $m$')
plt.legend(leg)
plt.title('Multiplicative bias')# Every position has one single SED. Errorbar position.
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_m_starSEDs_starPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_m_starSEDs_starPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1
plt.show()
plt.close()
#%%
import random
cmap_1=cm.spring 
cmap_2 = cm.winter 
color_indx_pos = 1.0/(paulin_stats_rca_testgal_listPos.shape[0]-1)*np.array(range(paulin_stats_rca_testgal_listPos.shape[0]))
random.shuffle(color_indx_pos) 
color_indx_gal = 1.0/(paulin_stats_rca_testgal_listGals.shape[0]-1)*np.array(range(paulin_stats_rca_testgal_listGals.shape[0]))
random.shuffle(color_indx_gal) 
   
fig, ax = plt.subplots()
for pos in range(paulin_stats_rca_testgal_listPos.shape[0]):
    plt.errorbar(sizes, paulin_stats_rca_testgal_listPos[pos,0,:],yerr=paulin_stats_rca_testgal_listPos[pos,1,:], c=cmap_1(color_indx_pos[pos]), lw=2, capsize=5)
    plt.errorbar(sizes, paulin_stats_lbda_testgal_listPos[pos,0,:],yerr=paulin_stats_lbda_testgal_listPos[pos,1,:], c=cmap_2(color_indx_pos[pos]), alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $m$')     
plt.legend(leg)
plt.title(r'Multiplicative bias: with galaxy SEDs at galaxy positions.  Errorbars in SEDs.')
plt.show()
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_m_cbSEDs.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_m_cbSEDs.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.close()

#%% down + up m SEDs
rca_pos_idx_m_max = np.argmax((paulin_stats_rca_testgal_listPos[:,0,0]))
rca_pos_idx_m_min = np.argmin((paulin_stats_rca_testgal_listPos[:,0,0]))

lbda_pos_idx_m_max = np.argmax((paulin_stats_lbda_testgal_listPos[:,0,0]))
lbda_pos_idx_m_min = np.argmin((paulin_stats_lbda_testgal_listPos[:,0,0]))


cindx = 1.0/(3-1)*np.array(range(3)) # for galaxies

fig, ax = plt.subplots()
plt.errorbar(sizes, paulin_stats_rca_testgal_listPos[rca_pos_idx_m_max,0,:],yerr=paulin_stats_rca_testgal_listPos[rca_pos_idx_m_max,1,:], c='red', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_lbda_testgal_listPos[lbda_pos_idx_m_max,0,:],yerr=paulin_stats_lbda_testgal_listPos[lbda_pos_idx_m_max,1,:],c='orchid', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_rca_testgal_listPos[rca_pos_idx_m_min,0,:],paulin_stats_rca_testgal_listPos[rca_pos_idx_m_min,1,:], c='firebrick', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_lbda_testgal_listPos[lbda_pos_idx_m_min,0,:],yerr=paulin_stats_lbda_testgal_listPos[lbda_pos_idx_m_min,1,:], c='darkorchid', lw=2, capsize=5,elinewidth=1.2)

plt.legend(leg)

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $m$')

plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Multiplicative bias')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_m_cbSEDs_downup.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_m_cbSEDs_downup.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%% 
fig, ax = plt.subplots()
for gal_i in range(paulin_stats_rca_testgal_listGals.shape[0]):
    plt.errorbar(sizes, paulin_stats_rca_testgal_listGals[gal_i,0,:],yerr=paulin_stats_rca_testgal_listGals[gal_i,1,:], c=cmap_1(color_indx_gal[gal_i]), lw=2, capsize=5)
    plt.errorbar(sizes, paulin_stats_lbda_testgal_listGals[gal_i,0,:],yerr=paulin_stats_lbda_testgal_listGals[gal_i,1,:], c=cmap_2(color_indx_gal[gal_i]), alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $m$')     
plt.legend(leg)
plt.title(r'Multiplicative bias')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_m_cbPositions.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_m_cbPositions.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%% down + up m positions
rca_gal_idx_m_max = np.argmax((paulin_stats_rca_testgal_listGals[:,0,0]))
rca_gal_idx_m_min = np.argmin((paulin_stats_rca_testgal_listGals[:,0,0]))

lbda_gal_idx_m_max = np.argmax((paulin_stats_lbda_testgal_listGals[:,0,0]))
lbda_gal_idx_m_min = np.argmin((paulin_stats_lbda_testgal_listGals[:,0,0]))


cindx = 1.0/(3-1)*np.array(range(3)) # for galaxies

fig, ax = plt.subplots()
plt.errorbar(sizes, paulin_stats_rca_testgal_listGals[rca_gal_idx_m_max,0,:],yerr=paulin_stats_rca_testgal_listGals[rca_gal_idx_m_max,1,:], c='red', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_max,0,:],yerr=paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_max,1,:],c='orchid', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_rca_testgal_listGals[rca_gal_idx_m_min,0,:],paulin_stats_rca_testgal_listGals[rca_gal_idx_m_min,1,:], c='firebrick', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_min,0,:],yerr=paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_min,1,:], c='darkorchid', lw=2, capsize=5,elinewidth=1.2)


#plt.fill_between(sizes,paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_max,0,:]-paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_max,1,:],paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_max,0,:]+paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_max,1,:],color='dodgerblue',alpha=0.3,label=r'$\lambda$RCA')
#plt.fill_between(sizes,paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_min,0,:]-paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_min,1,:],paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_min,0,:]+paulin_stats_lbda_testgal_listGals[lbda_gal_idx_m_min,1,:],color='mediumblue',alpha=0.3)
plt.legend(leg)

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $m$')

plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Multiplicative bias')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_m_cbPositions_downup.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_m_cbPositions_downup.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%%
# Additive bias c1
plt.errorbar(sizes,paulin_stats_rca[2], yerr=paulin_stats_rca[3], c='darkorchid', lw=2, capsize=5)
plt.errorbar(sizes, paulin_stats_lbda[2], yerr=paulin_stats_lbda[3], c='red', alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_1$')
plt.legend(leg)
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (1st component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c1_starSEDs_starPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c1_starSEDs_starPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()    
plt.close()
#%%
fig, ax = plt.subplots()
for pos in range(paulin_stats_rca_testgal_listPos.shape[0]):
    plt.errorbar(sizes, paulin_stats_rca_testgal_listPos[pos,2,:],yerr=paulin_stats_rca_testgal_listPos[pos,3,:], c=cmap_1(color_indx_pos[pos]), lw=2, capsize=5)
    plt.errorbar(sizes, paulin_stats_lbda_testgal_listPos[pos,2,:],yerr=paulin_stats_lbda_testgal_listPos[pos,3,:], c=cmap_2(color_indx_pos[pos]), alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_1$')
plt.legend(leg)
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (1st component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbSEDs.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbSEDs.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%% down + up c1
rca_pos_idx_c1_max = np.argmax((paulin_stats_rca_testgal_listPos[:,2,0]))
rca_pos_idx_c1_min = np.argmin((paulin_stats_rca_testgal_listPos[:,2,0]))

lbda_pos_idx_c1_max = np.argmax((paulin_stats_lbda_testgal_listPos[:,2,0]))
lbda_pos_idx_c1_min = np.argmin((paulin_stats_lbda_testgal_listPos[:,2,0]))


cindx = 1.0/(3-1)*np.array(range(3)) 
fig, ax = plt.subplots()
plt.errorbar(sizes, paulin_stats_rca_testgal_listPos[rca_pos_idx_c1_max,2,:],yerr=paulin_stats_rca_testgal_listPos[rca_pos_idx_c1_max,3,:], c='firebrick', lw=2, capsize=5,elinewidth=1.2,label=r'RCA')
plt.errorbar(sizes, paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_max,2,:],c='aqua', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_rca_testgal_listPos[rca_pos_idx_c1_min,2,:],paulin_stats_rca_testgal_listPos[rca_pos_idx_c1_min,3,:], c='red', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_min,2,:], c='mediumblue', lw=2, capsize=5,elinewidth=1.2)

plt.fill_between(sizes,paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_max,2,:]-paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_max,3,:],paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_max,2,:]+paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_max,3,:],color='dodgerblue',alpha=0.3,label=r'$\lambda$RCA')
plt.fill_between(sizes,paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_min,2,:]-paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_min,3,:],paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_min,2,:]+paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c1_min,3,:],color='mediumblue',alpha=0.3)
plt.legend()

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_1$')

plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (1st component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbSEDs_downup.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbSEDs_downup.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
#%%
fig, ax = plt.subplots()
plt.plot(sizes, np.mean(paulin_stats_rca_testgal_listPos[:,2,:],axis=0), c='red', lw=2)
plt.plot(sizes, np.mean(paulin_stats_rca_testgal_listPos[:,3,:],axis=0),'--k' ,c='red',lw=2)
plt.plot(sizes, np.mean(paulin_stats_lbda_testgal_listPos[:,2,:],axis=0), c='darkorchid', lw=2)
plt.plot(sizes, np.mean(paulin_stats_lbda_testgal_listPos[:,3,:],axis=0),'--k', c='darkorchid',lw=2)

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_1$')
plt.legend([r'RCA mean',r'RCA std',r'$\lambda$RCA mean',r'$\lambda$RCA std'])
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (1st component): with galaxy SEDs at galaxy positions. Errorbars in SEDs.')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbSEDs_meanstd.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbSEDs_meanstd.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1
plt.show()
plt.close()
#%%
fig, ax = plt.subplots()
for gal_i in range(paulin_stats_rca_testgal_listGals.shape[0]):#paulin_stats_rca_testgal_listGals.shape[0]
    plt.errorbar(sizes, paulin_stats_rca_testgal_listGals[gal_i,2,:],yerr=paulin_stats_rca_testgal_listGals[gal_i,3,:], c=cmap_1(color_indx_gal[gal_i]), lw=2, capsize=5)
    plt.errorbar(sizes, paulin_stats_lbda_testgal_listGals[gal_i,2,:],yerr=paulin_stats_lbda_testgal_listGals[gal_i,3,:], c=cmap_2(color_indx_gal[gal_i]), alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_1$')
plt.legend(leg)
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (1st component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%% down + up c1
rca_gal_idx_c1_max = np.argmax((paulin_stats_rca_testgal_listGals[:,2,0]))
rca_gal_idx_c1_min = np.argmin((paulin_stats_rca_testgal_listGals[:,2,0]))

lbda_gal_idx_c1_max = np.argmax((paulin_stats_lbda_testgal_listGals[:,2,0]))
lbda_gal_idx_c1_min = np.argmin((paulin_stats_lbda_testgal_listGals[:,2,0]))


cindx = 1.0/(3-1)*np.array(range(3)) 
fig, ax = plt.subplots()
plt.errorbar(sizes, paulin_stats_rca_testgal_listGals[rca_gal_idx_c1_max,2,:],yerr=paulin_stats_rca_testgal_listGals[rca_gal_idx_c1_max,3,:], c='firebrick', lw=2, capsize=5,elinewidth=1.2,label=r'RCA')
plt.errorbar(sizes, paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_max,2,:],c='aqua', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_rca_testgal_listGals[rca_gal_idx_c1_min,2,:],paulin_stats_rca_testgal_listGals[rca_gal_idx_c1_min,3,:], c='red', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_min,2,:], c='mediumblue', lw=2, capsize=5,elinewidth=1.2)

plt.fill_between(sizes,paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_max,2,:]-paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_max,3,:],paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_max,2,:]+paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_max,3,:],color='dodgerblue',alpha=0.3,label=r'$\lambda$RCA')
plt.fill_between(sizes,paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_min,2,:]-paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_min,3,:],paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_min,2,:]+paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c1_min,3,:],color='mediumblue',alpha=0.3)
plt.legend()

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_1$')

plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (1st component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbPos_downup.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbPos_downup.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()



#%%
fig, ax = plt.subplots()
plt.plot(sizes, np.mean(paulin_stats_rca_testgal_listGals[:,2,:],axis=0), c='red', lw=2)
plt.plot(sizes, np.mean(paulin_stats_rca_testgal_listGals[:,3,:],axis=0),'--k' ,c='red',lw=2)
plt.plot(sizes, np.mean(paulin_stats_lbda_testgal_listGals[:,2,:],axis=0), c='darkorchid', lw=2)
plt.plot(sizes, np.mean(paulin_stats_lbda_testgal_listGals[:,3,:],axis=0),'--k', c='darkorchid',lw=2)

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_1$')
plt.legend([r'RCA mean',r'RCA std',r'$\lambda$RCA mean',r'$\lambda$RCA std'])
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (1st component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbPos_meanstd.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c1_cbPos_meanstd.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()
#%%

# Additive bias c2
plt.errorbar(sizes,paulin_stats_rca[4], yerr=paulin_stats_rca[5], c='darkorchid', lw=2, capsize=5)
plt.errorbar(sizes, paulin_stats_lbda[4], yerr=paulin_stats_lbda[5], c='red', alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_2$')
plt.legend(leg)
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (2nd component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c2_starSEDs_starPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c2_starSEDs_starPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()    
plt.close()
#%%
fig, ax = plt.subplots()
for pos in range(paulin_stats_rca_testgal_listPos.shape[0]):
    plt.errorbar(sizes, paulin_stats_rca_testgal_listPos[pos,4,:],yerr=paulin_stats_rca_testgal_listPos[pos,5,:], c=cmap_1(color_indx_pos[pos]), lw=2, capsize=5)
    plt.errorbar(sizes, paulin_stats_lbda_testgal_listPos[pos,4,:],yerr=paulin_stats_lbda_testgal_listPos[pos,5,:], c=cmap_2(color_indx_pos[pos]), alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_2$')
plt.legend(leg)
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (2nd component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbSEDs.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbSEDs.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%% Down + up c2
rca_pos_idx_c2_max = np.argmax(abs(paulin_stats_rca_testgal_listPos[:,4,0]))
rca_pos_idx_c2_min = np.argmin(abs(paulin_stats_rca_testgal_listPos[:,4,0]))

lbda_pos_idx_c2_max = np.argmax(abs(paulin_stats_lbda_testgal_listPos[:,4,0]))
lbda_pos_idx_c2_min = np.argmin(abs(paulin_stats_lbda_testgal_listPos[:,4,0]))


cindx = 1.0/(3-1)*np.array(range(3)) # for galaxies
fig, ax = plt.subplots()
plt.errorbar(sizes, paulin_stats_rca_testgal_listPos[rca_pos_idx_c2_max,4,:],yerr=paulin_stats_rca_testgal_listPos[rca_pos_idx_c2_max,5,:], c='firebrick', lw=2, capsize=5,elinewidth=1.2,label=r'RCA')
plt.errorbar(sizes, paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_max,4,:],c='aqua', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_rca_testgal_listPos[rca_pos_idx_c2_min,4,:],paulin_stats_rca_testgal_listPos[rca_pos_idx_c2_min,5,:], c='red', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_min,4,:], c='mediumblue', lw=2, capsize=5,elinewidth=1.2)

plt.fill_between(sizes,paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_max,4,:]-paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_max,5,:],paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_max,4,:]+paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_max,5,:],color='dodgerblue',alpha=0.3,label=r'$\lambda$RCA')
plt.fill_between(sizes,paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_min,4,:]-paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_min,5,:],paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_min,4,:]+paulin_stats_lbda_testgal_listPos[lbda_pos_idx_c2_min,5,:],color='mediumblue',alpha=0.3)
plt.legend()

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_2$')

plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (2nd component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbSEDs_downup.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbSEDs_downup.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%%
fig, ax = plt.subplots()
plt.plot(sizes, np.mean(paulin_stats_rca_testgal_listPos[:,4,:],axis=0), c='red', lw=2)
plt.plot(sizes, np.mean(paulin_stats_rca_testgal_listPos[:,5,:],axis=0),'--k' ,c='red',lw=2)
plt.plot(sizes, np.mean(paulin_stats_lbda_testgal_listPos[:,4,:],axis=0), c='darkorchid', lw=2)
plt.plot(sizes, np.mean(paulin_stats_lbda_testgal_listPos[:,5,:],axis=0),'--k', c='darkorchid',lw=2)

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_2$')
plt.legend([r'RCA mean',r'RCA std',r'$\lambda$RCA mean',r'$\lambda$RCA std'])
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (2nd component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbSEDs_meanstd.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbSEDs_meanstd.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()




#%%

fig, ax = plt.subplots()
for gal_i in range(paulin_stats_rca_testgal_listGals.shape[0]):
    plt.errorbar(sizes, paulin_stats_rca_testgal_listGals[gal_i,4,:],yerr=paulin_stats_rca_testgal_listGals[gal_i,5,:], c=cmap_1(color_indx_gal[gal_i]), lw=2, capsize=5)
    plt.errorbar(sizes, paulin_stats_lbda_testgal_listGals[gal_i,4,:],yerr=paulin_stats_lbda_testgal_listGals[gal_i,5,:], c=cmap_2(color_indx_gal[gal_i]), alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_2$')
plt.legend(leg)
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (2nd component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()

#%% down + up c2
rca_gal_idx_c2_max = np.argmax(abs(paulin_stats_rca_testgal_listGals[:,4,0]))
rca_gal_idx_c2_min = np.argmin(abs(paulin_stats_rca_testgal_listGals[:,4,0]))

lbda_gal_idx_c2_max = np.argmax(abs(paulin_stats_lbda_testgal_listGals[:,4,0]))
lbda_gal_idx_c2_min = np.argmin(abs(paulin_stats_lbda_testgal_listGals[:,4,0]))


cindx = 1.0/(3-1)*np.array(range(3)) # for galaxies

fig, ax = plt.subplots()
plt.errorbar(sizes, paulin_stats_rca_testgal_listGals[rca_gal_idx_c2_max,4,:],yerr=paulin_stats_rca_testgal_listGals[rca_gal_idx_c2_max,5,:], c='firebrick', lw=2, capsize=5,elinewidth=1.2,label=r'RCA')
plt.errorbar(sizes, paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_max,4,:],c='aqua', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_rca_testgal_listGals[rca_gal_idx_c2_min,4,:],paulin_stats_rca_testgal_listGals[rca_gal_idx_c2_min,5,:], c='red', lw=2, capsize=5,elinewidth=1.2)
plt.errorbar(sizes, paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_min,4,:], c='mediumblue', lw=2, capsize=5,elinewidth=1.2)


plt.fill_between(sizes,paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_max,4,:]-paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_max,5,:],paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_max,4,:]+paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_max,5,:],color='dodgerblue',alpha=0.3,label=r'$\lambda$RCA')
plt.fill_between(sizes,paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_min,4,:]-paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_min,5,:],paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_min,4,:]+paulin_stats_lbda_testgal_listGals[lbda_gal_idx_c2_min,5,:],color='mediumblue',alpha=0.3)
plt.legend()

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_2$')

plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (2nd component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbPos_downup.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbPos_downup.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1

plt.show()
plt.close()


#%%
fig, ax = plt.subplots()
plt.plot(sizes, np.mean(paulin_stats_rca_testgal_listGals[:,4,:],axis=0), c='red', lw=2)
plt.plot(sizes, np.mean(paulin_stats_rca_testgal_listGals[:,5,:],axis=0),'--k' ,c='red',lw=2)
plt.plot(sizes, np.mean(paulin_stats_lbda_testgal_listGals[:,4,:],axis=0), c='darkorchid', lw=2)
plt.plot(sizes, np.mean(paulin_stats_lbda_testgal_listGals[:,5,:],axis=0),'--k', c='darkorchid',lw=2)

plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_2$')
plt.legend([r'RCA mean',r'RCA std',r'$\lambda$RCA mean',r'$\lambda$RCA std'])
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title(r'Additive bias (2nd component)')
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbPos_meanstd.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c2_cbPos_meanstd.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1
plt.show()
plt.close()


#%% Stack it all together

# Multiplicative bias
fig, ax = plt.subplots()
plt.errorbar(sizes,paulin_stats_rca_testgal_All[0], yerr=paulin_stats_rca_testgal_All[1], c='darkorchid', lw=2, capsize=5)
plt.errorbar(sizes, paulin_stats_lbda_testgal_All[0], yerr=paulin_stats_lbda_testgal_All[1], c='red', alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $m$')
plt.legend(leg)
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title('Multiplicative bias')# Every position has one single SED. Errorbar position.
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_m_stack_cbPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_m_stack_cbPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1
plt.show()
plt.close()
#%%
# Additive bias c1
fig, ax = plt.subplots()
plt.errorbar(sizes,paulin_stats_rca_testgal_All[2], yerr=paulin_stats_rca_testgal_All[3], c='darkorchid', lw=2, capsize=5)
plt.errorbar(sizes, paulin_stats_lbda_testgal_All[2], yerr=paulin_stats_lbda_testgal_All[3], c='red', alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_1$')
plt.legend(leg)
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title('Additive bias')# Every position has one single SED. Errorbar position.
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c1_stack_cbPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c1_stack_cbPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1
plt.show()
plt.close()
#%%
# Additive bias c2
fig, ax = plt.subplots()
plt.errorbar(sizes,paulin_stats_rca_testgal_All[4], yerr=paulin_stats_rca_testgal_All[5], c='darkorchid', lw=2, capsize=5)
plt.errorbar(sizes, paulin_stats_lbda_testgal_All[4], yerr=paulin_stats_lbda_testgal_All[5], c='red', alpha=.7, capsize=5)
plt.xlabel(r'Galaxy size $R$ (arcsec)')
plt.ylabel(r'Paulin predicted $c_2$')
plt.legend(leg)
plt.xlim(.075,0.525)
plt.plot([0,0.6], [0,0], 'k--')
plt.title('Additive bias')# Every position has one single SED. Errorbar position.
if save_plot:
    plt.savefig(plot_path+'rca_lambda_paulin_c2_stack_cbPos.eps'.format(ID),format='eps',dpi=1200)
    plt.savefig(plot_path+'rca_lambda_paulin_c2_stack_cbPos.jpg'.format(ID),format='jpeg',dpi=1200)
    ID +=1
plt.show()
plt.close()

#%%




 