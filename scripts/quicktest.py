# Lambda RCA V0 test code
import sys
sys.path.append('../utilities')
from optim_utils import polychromatic_psf_field_est_2
import numpy as np
import time as time

# load small data
load_path = '/Users/rararipe/Documents/Data/QuickestGenerator/42x42pixels_8lbdas10pos/'
# load_path ='/Users/rararipe/Documents/Data/lbdaRCA_wdl/Fake_SEDs/morgan_stars/'
stars = np.load(load_path+'stars.npy')
SEDs = np.load(load_path+'SEDs.npy')
lbdas = np.load(load_path+'lbdas.npy')
fov = np.load(load_path+'fov.npy')
gt_stars_2euclidres = np.load(load_path+'stars_2euclidrec.npy')
first_guesses = np.load('/Users/rararipe/Documents/Data/lbdaRCA_wdl/Fake_SEDs/morgan_stars/first_guesses.npy')
# D_stack_ini = np.load('/Users/rararipe/Documents/Data/lbdaRCA_wdl/Fake_SEDs/super_res_first_guess/D_stack_ini_guess.npy')
# gt_PSFs = np.load(load_path+'PSFs_2euclidrec.npy')

# stars (21,21,10)
#SEDs (5,10)
#lbdas (5,)
#fov (10,2)


nb_comp = 3
D = 2
opt_shift = ['-t2','-n2']
wvl_opt = ['-t2','-n2']



tic = time.time()
psf_est,D,A,res,obs_est,barycenters,D_stack_1 =  polychromatic_psf_field_est_2(stars,SEDs,lbdas,D,opt_shift,nb_comp,sr_first_guesses=first_guesses,nb_iter=1,nb_subiter=3,mu=0.3,\
                        tol = 0.1,sig_supp = 6,sig=None,shifts=None,flux=None,nsig_shift_est=4,simplex_en=True,wvl_en=True,wvl_opt =wvl_opt,field_pos=fov,nsig=5,graph_cons_en=True,feat_init="super_res_2")
toc = time.time()

print "Total time: "+ str((toc-tic)/60.0) + " min"

save_path = '/Users/rararipe/Documents/Data/lbdaRCA_wdl/Fake_SEDs/42x42pixels_8lbdas10pos_sr_gamma01_all_power_methods/'                       
np.save(save_path+'wdl_psf_est.npy', psf_est)
np.save(save_path+'wdl_D.npy', D)
np.save(save_path+'wdl_A.npy', A)
np.save(save_path+'wdl_res.npy', res)
np.save(save_path+'wdl_obs_est.npy', obs_est)
np.save(save_path+'wdl_barycenters.npy', barycenters)
np.save(save_path+'D_stack_1.npy', D_stack_1)