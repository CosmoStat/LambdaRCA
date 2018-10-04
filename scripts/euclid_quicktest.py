import sys
sys.path.append('../utilities')
from optim_utils import polychromatic_psf_field_est_2
import pickle
from numpy import arange
import numpy as np
import time

[output_stack,dec_stack,sig,flux,spectrums,field_pos] = pickle.load( open( "/Users/rararipe/Documents/Data/LambdaRCA/sim_starsx100_10dB.sav", "rb" ) )
wvl = arange(0,dec_stack[0].shape[2])

# print output_stack.shape # (21, 21, 100)        LR pixel x LR pixel x nobj number of objects -   inputs?
# print dec_stack.shape    # (42, 42, 100, 100)   HR pixel x HR pixel x wavelength for each object? x nobj      -   ground truth?
# print sig.shape          # (100, 1)             Noise estimation
# print flux.shape         # (100,)               ... flux. 
# print spectrums.shape    # (100, 100)           SED 100 objects, 100 lambdas each
# print field_pos.shape    # (100, 2)             FOV positions of the 100 objects

W_lr = output_stack.shape[0]
W = dec_stack.shape[0]
nb_comp = 3
D = 2
opt_shift = ['-t2','-n2']
wvl_opt = ['-t2','-n2']

#choose 10 positions and 5 labdas (deterministic)
nb_obj = 10

np.random.seed(11)


gt = np.zeros((W,W,dec_stack.shape[2],nb_obj))
stars = np.zeros((W_lr,W_lr,nb_obj))
SEDs = []
fov = np.zeros((nb_obj,2))
for obj,pos in enumerate(np.random.choice(output_stack.shape[2],nb_obj,replace=False)):
	stars[:,:,obj] = output_stack[:,:,pos]
	SEDs.append(spectrums[:,pos])
	fov[obj,:] = field_pos[pos,:]
	gt[:,:,:,obj] = dec_stack[:,:,:,pos]

SEDs = np.array(SEDs).swapaxes(0,1)

SEDs = SEDs[15:86,:] #15:86 is useful
gt = gt[:,:,15:86,:]


lbdas = np.arange(550.0,935.0,35.0)#min step = 5, remember to put 1 step forward in right limit of arange
nb_wvl = lbdas.size
SEDs = SEDs[::7,:]
gt = gt[:,:,::7,:] # 11 wvls

save_path = '/Users/rararipe/Documents/Data/lbdaRCA_wdl/euclid/42x42pixels_11lbdas10pos_gt/'                       


np.save(save_path+'gt.npy', gt)
np.save(save_path+'lbdas.npy', lbdas)
np.save(save_path+'fov.npy', fov)


psf_est,D,A,res,obs_est,barycenters =  polychromatic_psf_field_est_2(stars,SEDs,lbdas,D,opt_shift,nb_comp,gt,nb_iter=1,nb_subiter=3,mu=0.3,\
                        tol = 0.1,sig_supp = 6,sig=None,shifts=None,flux=None,nsig_shift_est=4,simplex_en=True,wvl_en=True,wvl_opt =wvl_opt,
                        field_pos=fov,nsig=5,graph_cons_en=True,feat_init = "ground_truth")

np.save(save_path+'wdl_psf_est.npy', psf_est)
np.save(save_path+'wdl_D.npy', D)
np.save(save_path+'wdl_A.npy', A)
np.save(save_path+'wdl_res.npy', res)
np.save(save_path+'wdl_obs_est.npy', obs_est)
np.save(save_path+'wdl_barycenters.npy', barycenters)