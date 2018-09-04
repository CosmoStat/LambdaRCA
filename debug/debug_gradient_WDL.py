
import numpy as np
import sys
sys.path.append('../baryOT')
import logOT_bary as ot
import time
from scipy.ndimage import gaussian_filter
import scipy.signal as scisig



#from utils
def transpose_decim(im,decim_fact,av_en=0):
    """ Applies the transpose of the decimation matrix."""
    shap = im.shape
    im_out = np.zeros((shap[0]*decim_fact,shap[1]*decim_fact))

    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            im_out[decim_fact*i,decim_fact*j]=im[i,j]

    if av_en==1:
        siz = decim_fact+1-(decim_fact%2)
        mask = np.ones((siz,siz))/siz**2
        im_out = scisig.fftconvolve(im, mask, mode='same')

    return im_out


def nab_L2(data,MX):
    """
        PARAMETERS
        ----------
        data: <N,N> observed image
        MX: <N,N> estimated image
    """
    return transpose_decim((data-MX),2) 


def my_conv(x,gamma,trunc=500):
	N = np.size(x)
	sqrtN = int(np.sqrt(N))	
	return gaussian_filter(x.reshape(sqrtN,sqrtN), gamma, truncate=trunc).flatten()


def feat_fullgrad(MX,b,phi,datapoint, lbda, Ys, gamma, nabL=nab_L2, trunc=500):
    """Computes the gradient wrt the atoms of a given component
         PARAMETERS
         ----------
         MX: <N,N,nb_objs> estimated data
         b: <N*N,nb_atoms,n_iter> beta
         phi: <N*N,nb_atoms,n_iter> alpha
         datapoint: <N,N,nb_objs> observed data
         lbda: <nb_atoms,> current weights
         Ys: <N*N,nb_atoms> dictionary, current features
         gamma:
    """
    gamma = np.sqrt(gamma)
    n_iter = b.shape[2]
    N, S = Ys.shape
    sqrtN = int(np.sqrt(N))


    conv = lambda x: gaussian_filter(x.reshape(sqrtN,sqrtN), gamma, truncate=trunc).flatten()
    
    
    
    n = nabL(datapoint,MX).reshape(-1)
    v = np.zeros((N,S))
    c = np.zeros((N,S))
    grad = np.zeros((N,S))
    
    for j in range(n_iter-1,1,-1):

        for s in range(S):
	        c[:,s] = conv((lbda[s]*n - v[:,s]) * b[:,s,j])
	        grad[:,s] += c[:,s] / conv(b[:,s,j-1])
	        v[:,s] = -1./phi[:,s,j-1] * conv((Ys[:,s] * c[:,s])/(conv(b[:,s,j-1])**2))
        n = np.sum(v, axis=1)
    return grad
    
   
def WDL_full_grad(im_stack,D_stack,w_stack,A,spectrums,flux,sig,ker,ker_rot,C,gamma,n_iter_sink):
	start = time.time()
	[MX,barys,alphas,betas] = ot.Theano_wdl_MX(A,spectrums,flux,sig,ker,D_stack,w_stack,C,gamma,n_iter_sink)
	fwd = time.time()
	print 'Forward computation: {}s'.format(fwd-start)
	nb_wvl = w_stack.shape[0]
	nb_comp = D_stack.shape[2]
	nb_obj = im_stack.shape[2]
	pixels = D_stack.shape[0]
	nb_atoms = D_stack.shape[1]
	grad = np.zeros(D_stack.shape)
	for k in range(nb_obj):
	    grad_k = np.zeros(D_stack.shape)
	    for i in range(nb_comp):
	        for v in range(nb_wvl):#this could be done in parallel
	        	temp = spectrums[v,k]*feat_fullgrad(MX[:,:,k],betas[:,:,:,i,v],alphas[:,:,:,i,v],im_stack[:,:,k], w_stack[v,:], D_stack[:,:,i], gamma)
	         	grad_k[:,:,i] += temp
	        grad_k[:,:,i] *=  A[i,k]
	    grad +=  grad_k   
	return grad,MX,barys




A = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/A.npy')
spectrums = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/spectrums.npy')
flux = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/flux.npy')
sig = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/sig.npy')
ker = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/ker.npy')
ker_rot = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/ker_rot.npy')
D_stack = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/D_stack.npy')
w_stack = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/w_stack.npy')
C = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/C.npy')
gamma = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/gamma.npy')
datapoint = np.load('/Users/rararipe/Documents/Data/debug_WDL_gradi/datapoint.npy')

[grad,MX,barys] = WDL_full_grad(datapoint,D_stack,w_stack,A,spectrums,flux,sig,ker,ker_rot,C,gamma,100)

np.save('/Users/rararipe/Documents/Data/debug_WDL_gradi/grad.npy',grad)
np.save('/Users/rararipe/Documents/Data/debug_WDL_gradi/MX.npy',MX)
np.save('/Users/rararipe/Documents/Data/debug_WDL_gradi/barys.npy',barys)






