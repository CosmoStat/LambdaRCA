#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:25:54 2018

@author: rararipe
"""

        import pdb; pdb.set_trace()  # breakpoint 3d87ba10 //
raise ValueError('Ops')

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
    
    
    
    ## COLORBAR PERFEITO
    
    
fig, ax = plt.subplots()
this_cmap = cm.get_cmap('Spectral',full_shapes_testgal.shape[1])
c = np.arange(1,full_shapes_testgal.shape[1]+1)
dummie_cax = ax.scatter(c, c, c=c, cmap=this_cmap)
# Clear axis
ax.cla()
for gal_i in range(full_shapes_testgal.shape[1]):
    plt.scatter(truInt_shapes_testgal[:,gal_i,0],lbdaInt_shapes_testgal[:,gal_i,0],marker='o',c=this_cmap(color_indx[gal_i]))
#    plt.scatter(truInt_shapes_testgal[:,gal_i,1],lbdaInt_shapes_testgal[:,gal_i,1],marker='*',c=this_cmap(color_indx[gal_i]))
#    plt.plot(truInt_shapes_testgal[:,gal_i,0],rca_shapes_test[:,0],'o',label=r'RCA $e_1$',c=cmap_2(color_indx[gal_i]))
#    plt.plot(truInt_shapes_testgal[:,gal_i,1],rca_shapes_test[:,1],'*',label=r'RCA $e_2$',c=cmap_2(color_indx[gal_i]))
plt.xlabel('`Known\' PSF ellipticity $e_1$')
plt.ylabel('Recovered PSF ellipticity $e_1$')
plt.title(r'PSF ellipticity with galaxy SED at galaxy positions')
cb=fig.colorbar(dummie_cax)
cb.set_ticks([])

plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]
plt.plot(lims, lims, 'k--')
plt.show()