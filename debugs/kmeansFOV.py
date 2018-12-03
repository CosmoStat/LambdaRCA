#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:59:23 2018

@author: rararipe
"""

#%%
import numpy as np
import sys
sys.path.append('../')
from psf_learning_utils import SnA
import psf_toolkit as tk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
kmeans = KMeans(n_clusters=nb_comp, random_state=0).fit(fov)
labels = kmeans.predict(fov)
plt.scatter(fov[:, 0], fov[:, 1], c=labels)
#%%
groups = []
for i in range(nb_comp):
    groups.append([])
for obj in range(nb_obj):
    groups[labels[obj]].append(obj)
    
#%%
sr_stars = []
for i in range(nb_comp):
    print len(groups[i])," stars"
    sr_stars.append(SnA(stars[:,:,groups[i]],shifts[groups[i]]))

sr_stars = np.array(sr_stars)

for sr in sr_stars:
    print np.argwhere(np.isnan(sr))
    tk.plot_func(sr)