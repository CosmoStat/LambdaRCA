# Lambda RCA V0 test code
import sys
sys.path.append('../utilities')
from optim_utils import polychromatic_psf_field_est_2
import numpy as np

# load small data
load_path = '/Users/rararipe/Documents/Data/QuickestGenerator/22x22pixels_5lbdas10pos/'
stars = np.load(load_path+'stars.npy')
SEDs = np.load(load_path+'SEDs.npy')
lbdas = np.load(load_path+'lbdas.npy')
fov = np.load(load_path+'fov.npy')
stars_first_guess = np.load(load_path+'stars_2euclidrec.npy')

# stars (21,21,10)
#SEDs (5,10)
#lbdas (5,)
#fov (10,2)


nb_comp = 3
D = 2
opt_shift = ['-t2','-n2']
wvl_opt = ['-t2','-n2']

psf_est,D,A,res,obs_est =  polychromatic_psf_field_est_2(stars,SEDs,lbdas,D,opt_shift,nb_comp,stars_first_guess,nb_iter=1,nb_subiter=3,mu=0.3,\
                        tol = 0.1,sig_supp = 6,sig=None,shifts=None,flux=None,nsig_shift_est=4,simplex_en=True,wvl_en=True,wvl_opt =wvl_opt,field_pos=fov,nsig=5,graph_cons_en=True)
 

save_path = '/Users/rararipe/Documents/Data/lbdaRCA_wdl/Fake_SEDs/22x22pixels_5lbdas10pos/'                       
np.save(save_path+'wdl_psf_est.npy', psf_est)
np.save(save_path+'wdl_D.npy', D)
np.save(save_path+'wdl_A.npy', A)
np.save(save_path+'wdl_res.npy', res)
np.save(save_path+'wdl_obs_est.npy', obs_est)