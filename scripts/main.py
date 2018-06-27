# Lambda RCA V0 test code
import sys
sys.path.append('../utilities')
from optim_utils import polychromatic_psf_field_est_2
import pickle
from numpy import arange
import numpy as np

[output_stack,dec_stack,sig,flux,spectrums,field_pos] = pickle.load( open( "../Data/sim_starsx100_10dB.sav", "rb" ) )
wvl = arange(0,dec_stack[0].shape[2])


nb_comp = 5
D = 2
opt_shift = ['-t2','-n2']
wvl_opt = ['-t2','-n2']
wvl = wvl.astype(float)

psf_est,P,A,res =  polychromatic_psf_field_est_2(output_stack,spectrums,wvl,D,opt_shift,nb_comp,nb_iter=2,nb_subiter=10,mu=0.3,\
                        tol = 0.1,sig_supp = 6,sig=None,shifts=None,flux=None,nsig_shift_est=4,simplex_en=True,wvl_en=True,wvl_opt = wvl_opt,field_pos=field_pos,nsig=5,graph_cons_en=True)
                        
np.save('psf_est.npy', psf_est)
np.save('P.npy', P)
np.save('A.npy', A)
np.save('res.npy', res)
