mport sys
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



nb_comp = 3
D = 2
opt_shift = ['-t2','-n2']
wvl_opt = ['-t2','-n2']

#choose 10 positions and 5 labdas (deterministic)

nb_obj = 10
np.random.seed(11)

for pos in np.randint(output_stack.shape[2],size=nb_obj):
	


stars,SEDs,lbdas,D,opt_shift,nb_comp,stars_first_guess