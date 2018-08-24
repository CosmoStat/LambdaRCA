# Lambda RCA V0 test code
import sys
sys.path.append('../utilities')
from optim_utils import polychromatic_psf_field_est_2
import numpy as np

# load small data
stars = np.load('/Users/rararipe/Documents/Data/QuickestGenerator/22x22pixels_5lbdas10pos_25_07_2018/stars.npy')
SEDs = np.load('/Users/rararipe/Documents/Data/QuickestGenerator/22x22pixels_5lbdas10pos_25_07_2018/SEDs.npy')
lbdas = np.load('/Users/rararipe/Documents/Data/QuickestGenerator/22x22pixels_5lbdas10pos_25_07_2018/lbdas.npy')
fov = np.load('/Users/rararipe/Documents/Data/QuickestGenerator/22x22pixels_5lbdas10pos_25_07_2018/fov.npy')

# stars (21,21,10)
#SEDs (5,10)
#lbdas (5,)
#fov (10,2)


nb_comp = 3
D = 2
opt_shift = ['-t2','-n2']
wvl_opt = ['-t2','-n2']

psf_est,D,A,res =  polychromatic_psf_field_est_2(stars,SEDs,lbdas,D,opt_shift,nb_comp,nb_iter=1,nb_subiter=3,mu=0.3,\
                        tol = 0.1,sig_supp = 6,sig=None,shifts=None,flux=None,nsig_shift_est=4,simplex_en=True,wvl_en=True,wvl_opt =            wvl_opt,field_pos=fov,nsig=5,graph_cons_en=True)
                        
np.save('/Users/rararipe/Documents/Data/Results_quickestGenerator/22x22pixels_5lbdas10pos_25_07_2018/wdl_psf_est_quicktest.npy', psf_est)
np.save('/Users/rararipe/Documents/Data/Results_quickestGenerator/22x22pixels_5lbdas10pos_25_07_2018/wdl_D_quicktest.npy', D)
np.save('/Users/rararipe/Documents/Data/Results_quickestGenerator/22x22pixels_5lbdas10pos_25_07_2018/wdl_A_quicktest.npy', A)
np.save('/Users/rararipe/Documents/Data/Results_quickestGenerator/22x22pixels_5lbdas10pos_25_07_2018/wdl_res_quicktest.npy', res)
