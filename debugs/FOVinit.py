#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:31:57 2018

@author: rararipe
"""
#%%
import numpy as np
import sys
sys.path.append('../')
from psf_learning_utils import SnA
import psf_toolkit as tk
import matplotlib.pyplot as plt


#%%
load_path = '/Users/rararipe/Documents/Data/QuickestGenerator/trueSEDs/42x42pixels_8lbdas80pos/'
stars = np.load(load_path+'stars.npy')
fov = np.load(load_path+'fov.npy')
shifts = np.load(load_path+'shifts.npy')
nb_obj = stars.shape[-1]
#%%
nb_comp = 3
min_el_per_grp = 15
W = 0.3 # inches?
H = 0.25
center = [0.0,0.775]
r_last = min(W,H)/2 

#%%
segment = r_last/(nb_comp-1)
r_list = []
for i in range(nb_comp-1):
    r_list.append(segment*(i+1))
    
#%
  
groups = []
for i in range(nb_comp):
    groups.append([])
for obj in range(nb_obj):
    sector = 0
    r_obj = np.sqrt((fov[obj,0] - center[0])**2 + (fov[obj,1] - center[1])**2)
#    if r_obj > 0.125:
#        print "LAST SECTOR"
    
    # Find sector
    not_found = True
    j = int((len(r_list)-1)/2) # middle of the list
    while not_found:
        if r_obj < r_list[j]:
            j -= 1
            if j < 0:
                not_found = False
        elif r_obj >= r_list[j]:
            if r_obj < r_list[j+1]:
                not_found = False
            else:
                j +=1
                if j == len(r_list)-1:
                    not_found = False
    sector = j+1
    
    # Put at the sector
    groups[sector].append(obj)
#%%
# Check if all groups have minimum stars requirement (NOT OPTIMAL)
for i in range(nb_comp):
    if len(groups[i]) < min_el_per_grp:
        deficit = min_el_per_grp - len(groups[i])
        # go through neighbor groups to find a donator
        neighbors = [i-1,i+1]
        for j in neighbors:
            if j < 0:
                continue
            sobra = len(groups[j]) - min_el_per_grp
            if sobra > 0:
                gain = min(sobra,deficit)
                groups[i].extend([groups[j].pop(el) for el in range(gain)])
                deficit -= gain
                if deficit == 0:
                    break
            
    
    
#%%    
sr_stars = []
for i in range(nb_comp):
    print len(groups[i])," stars"
    sr_stars.append(SnA(stars[:,:,groups[i]],shifts[groups[i]]))

sr_stars = np.array(sr_stars)

for sr in sr_stars:
    print np.argwhere(np.isnan(sr))
    tk.plot_func(sr)

#%%
# RETURN SR STARS AND GROUPS
# CHOOSING A (I think uniform would be the best go)    
    
#for i in range(nb_comp):
#    print "-----------------"
#    for obj in groups[i]:
#        tk.plot_func(stars[:,:,obj])
        
        
#%% 
plt.figure()
plt.plot(fov[groups[0],0], fov[groups[0],1], '.',color='red')
plt.plot(fov[groups[1],0], fov[groups[1],1], '.',color='blue') 
plt.plot(fov[groups[2],0], fov[groups[2],1], '.',color='green')   
      
#%% A init

A = np.zeros((nb_comp,nb_obj))
for i in range(nb_comp):
    A[i,groups[i]] = 1.0

tk.plot_func(A)      
        
        
        
        
    