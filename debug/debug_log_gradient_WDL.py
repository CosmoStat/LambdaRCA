
import numpy as np
import sys
sys.path.append('../baryOT')
import logOT_bary as ot
import time
from scipy.ndimage import gaussian_filter
import scipy.signal as scisig
from tqdm import tqdm
import psf_toolkit as tk
import pickle
import operator as operator


def EuclidCost_split(Nr, Nc, divmed=False, timeit=False, trunc=False, maxtol=745.13, truncval=745.13):
    if timeit:
        start = time.time()

    Cr = np.zeros((Nr,Nr))
    Cc = np.zeros((Nc,Nc))

    for k1 in range(Nr):
        for k2 in range(k1):
            r1, r2 = int(float(k1) / Nr)+1, int(float(k2) / Nr)+1
            Cr[k1, k2]= (r1-r2)**2
            Cr[k2, k1] = Cr[k1, k2]

    for k1 in range(Nc):
        for k2 in range(k1):
            c1, c2 = k1%Nc + 1, k2%Nc + 1 # why this assymetry between row and col costs?
            Cc[k1, k2] = (c1-c2)**2
            Cc[k2, k1] = Cc[k1, k2]

    if timeit:
        print 'cost matrix computed in '+str(time.time()-start)+'s.'
    if divmed:
        Cr /= np.median(Cr)
        Cc /= np.median(Cc)
    if trunc:
        Cr[Cr>maxtol] = truncval
        Cc[Cc>maxtol] = truncval
    return Cr,Cc



def signed(x):
    return {'mod':np.abs(x), 'sign': np.sign(x)}


def safelog(signed_x):
    return {'val': np.log(signed_x['mod']),
            'sign': signed_x['sign']}

def safeexp(*args):
    sign = 1
    vals = []
    for arg in args:
        sign *= arg['sign']
        vals.append(arg['val'])
    return signed(sign*np.exp(np.sum(vals)))

def safeadd(*args): # addition for signed variables
    res = 0.0
    for arg in args:
        res += arg['sign']*arg['mod']
    return signed(res)

def safemax(l_signed_x): # max for outputs of safelog
    """
        PARAMETERS
        ----------
        l_signed_x: dictionary of kind {val: , sign:}

        OUTPUT
        ------
        max of log values , keeping track of the signal of the previous log argument
    """
    import pdb; pdb.set_trace()  # breakpoint ffbd453d //

    key = operator.max(l_signed_x.iteritems(),key=operator.itemgetter(1))[0]






def LogSeparableKer(v,Cr,Cc,gamma):
    """
        PARAMETERS
        ----------
        Cc: cost of transporting mass in the x axis
        Cr: cost of transporting mass in the y axis
        y: image in log domain LxL
    """

    I = K = Cr.shape[0]
    J = L = Cc.shape[0]

    X = []
    for l in range(L):
        x = np.zeros((K,J))
        for k in range(K):
            for j in range(J):
                x[k,j] = Cr[j,l]/gamma + v[k,l]
        X.append(x)
    X = np.array(X)


    # A = np.zeros((K,J))
    A = {}
    for k in range(K):
        for j in range(J):
            max_x = np.max(X[:,k,j])
            temp = safelog(signed(np.sum(np.exp(X[:,k,j] - max_x))))
            temp['val'] += max_x 
            A[(k,j)] = temp



    Y = {}
    for k in range(K):
        for i in range(I):
            for j in range(J):
                temp = A[(k,j)]
                temp['val'] += Cc[i,k]/gamma
                Y[(k,i,j)] = temp

        

    R = np.zeros((I,J))
    # R = {}
    for i in range(I):
        for j in range(J):
            values = {}
            for key,v in Y.items():
                if key[1] == i and key[2] == j:
                    print "FOUND"
                    import pdb; pdb.set_trace()  # breakpoint 50181719 //
                    values.update(v)

            safemax(values)
            # max_y = np.max(values)
            # soma = signed(0.0)
            # for k in range(K):
            #     res = safeexp(Y[(i,j,k)],   )
            #     soma = safeadd(soma,res)
            # R[i,j] = np.log(soma['mod']) + max_y

    return R


#one object, one component. Do big loop later to have the barycenters for every
def logSinkornLoop(Cr,Cc,D,lbda,gamma,n_iter):
    """
        PARAMETERS
        ----------
        D: <N,nb_atoms>
        lbda: <nb_atoms,>

    """

    N,S = D.shape
    sqrtN = int(np.sqrt(N))

    v = np.zeros((n_iter+1,N,S))
    phi = np.zeros((n_iter+1,N,S))
    for it in range(1,n_iter+1):
        logp = np.zeros(N)
        for s in range(S):
            phi[it,:,s] = LogSeparableKer(np.log(np.abs(D[:,s])).reshape(sqrtN,sqrtN) - LogSeparableKer(v[it-1,:,s].reshape(sqrtN,sqrtN),Cr,Cc,gamma),Cr,Cc,gamma).flatten()
            logp += lbda[s]* phi[it,:,s]
            v[it,:,s] = logp - phi[it,:,s]




    p = np.exp(logp)

    return p,phi,v


def computeBarycenters(Cc,Cr,D_stack,w_stack,gamma,n_iter): # DEBUGGED BY test_barycenter_WDL.ipynb

    N = D_stack.shape[0]
    nb_atoms = D_stack.shape[1]
    nb_comp = D_stack.shape[2]
    nb_wvl = w_stack.shape[0]



    barycenters = np.zeros((N,nb_comp,nb_wvl))
    phi = np.zeros((n_iter+1,N,nb_atoms,nb_comp,nb_wvl))
    v = np.zeros((n_iter+1,N,nb_atoms,nb_comp,nb_wvl))
    for i in tqdm(range(nb_comp)):
        for w in tqdm(range(nb_wvl)):
            temp = logSinkornLoop(Cr,Cc,D_stack[:,:,i],w_stack[w,:],gamma,n_iter)
            x,y,z = temp
            barycenters[:,i,w],phi[:,:,:,i,w],v[:,:,:,i,w]  =  temp



    return barycenters


#========================== TEST KLS ========================================
shap = (22,22)
Cr,Cc = EuclidCost_split(shap[0],shap[1])
gamma = 2
im = np.random.rand(shap[0],shap[1]) 
R = LogSeparableKer(np.log(im),Cr,Cc,gamma)





# #======================== TEST BARYCENTER =================================

# Lambda_RCA_path = '/Users/rararipe/Documents/Data/LambdaRCA/sim_starsx100_10dB.sav'
# [out_stack,dec_stack,sig,flux,spectrums,field_pos] = pickle.load(open( Lambda_RCA_path, "rb" ) )
# nb_comp = 1

# print out_stack.shape # (21, 21, 100)        LR pixel x LR pixel x nobj number of objects -   inputs?
# print dec_stack.shape    # (42, 42, 100, 100)   HR pixel x HR pixel x wavelength for each object? x nobj      -   ground truth?
# print sig.shape          # (100, 1)             Noise estimation
# print flux.shape         # (100,)               ... flux. 
# print spectrums.shape    # (100, 100)           100 lambdas each, SED 100 objects
# print field_pos.shape    # (100, 2)             FOV positions of the 100 objects

# #Use estimated flux and sigma
# [sig2,flux2] = pickle.load(open( "/Users/rararipe/Documents/Data/LambdaRCA_OTbaryv2/sigma_flux_estimated_full_original_data.pickle", "rb" ) )
# sig = sig2
# flux = flux2

# pos = 57
# PSFs = np.expand_dims(dec_stack[:,:,15:86,pos],axis=3)
# SEDs = np.expand_dims(spectrums[15:86,pos],axis=1)
# flux = np.expand_dims(flux2[pos],axis=1)
# sigma = np.expand_dims(sig2[pos],axis=1)
# stars = np.expand_dims(out_stack[:,:,pos],axis=2)

# # Dictionary atoms and weights
# shap = (42,42)
# range_ab = np.arange(550.0,950.0,50.0)#min step = 5, remember to put 1 step forward in right limit of arange
# PSFs_chosen = PSFs[:,:,::10,:] #step/5


# SEDs_chosen = SEDs[::10,:]
# range_01 = (range_ab-range_ab[0])/(range_ab[-1] - range_ab[0])
# w_stack_ = np.array([range_01 + 1e-10, 1-range_01 - 1e-10]).T 


# PSF_550 = PSFs_chosen[:,:,0,0] 
# PSF_900 = PSFs_chosen[:,:,-1,0]

# atoms = np.array([PSF_550.reshape(-1),PSF_900.reshape(-1)])
# atoms = atoms.T
# D_stack_ = atoms.reshape(shap[0]*shap[1], 2,1)

# #%% Wasserstein params and ground metric
# gamma = 0.3
# n_iter_sink = 13
# Cr,Cc = EuclidCost_split(shap[0],shap[1])


# barycenters = computeBarycenters(Cc,Cr,D_stack_,w_stack_,gamma,n_iter_sink)

# np.save('/Users/rararipe/Documents/Data/debug_log_wdl_bary/barycenters.npy',barycenters)
# np.save('/Users/rararipe/Documents/Data/debug_log_wdl_bary/PSFs_chosen.npy',PSFs_chosen)



















