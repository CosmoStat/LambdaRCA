#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:25:54 2018

@author: rararipe
"""

        import pdb; pdb.set_trace()  # breakpoint 3d87ba10 //


CORES

#    color_indxs_1 = np.array([int(round((len(cp.my_color_list)/2)/len(paulin_rca_testgal_m_mean_list))*i) for i in range(len(paulin_rca_testgal_m_mean_list))])
#    color_indxs_2 = color_indxs_1 + int(round((len(cp.my_color_list)/2)))
  #    cNorm  = colors.Normalize(vmin=0, vmax=len(paulin_rca_testgal_m_mean_list)-1)
#    scalarMap_1 = cm.ScalarMappable(norm=cNorm, cmap=cm.Reds)
#    scalarMap_2 = cm.ScalarMappable(norm=cNorm, cmap=cm.winter)
  
  import matplotlib.colors as colors
  
  
  
  
 paulin_stats_rca = np.empty(3,dtype=np.ndarray) # <m,c1,c2>
for i in range(3):
    paulin_stats_rca[i] = np.empty(2,dtype=np.ndarray) # <mean,std>