import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import conv
from theano.gradient import jacobian, hessian
import scipy.signal as scisig
from scipy.stats import norm
import time
from scipy.ndimage import gaussian_filter

#TO DO import this constants from optim_utils 
Shap = (42,42) # Image full dimensions
Fshap = (9,9) # Filter dimensions
nb_atoms = 2
nb_wvl =8



#############
# UTILITIES #
#############
def EuclidCost(Nr, Nc, divmed=False, timeit=False, trunc=False, maxtol=745.13, truncval=745.13):
    if timeit:
        start = time.time()
    N = Nr * Nc
    C = np.zeros((N,N))
    for k1 in range(N):
        for k2 in range(k1):
            r1, r2 = int(float(k1) / Nc)+1, int(float(k2) / Nc)+1
            c1, c2 = k1%Nc + 1, k2%Nc + 1
            C[k1, k2] = (r1-r2)**2 + (c1-c2)**2
            C[k2, k1] = C[k1, k2]
    if timeit:
        print 'cost matrix computed in '+str(time.time()-start)+'s.'
    if divmed:
        C /= np.median(C)
    if trunc:
        C[C>maxtol] = truncval
    return C
    
def alphatolbda(alpha):
    return (np.exp(alpha).T / np.sum(np.exp(alpha), axis=1)).T


def LogCost(xgrid): # not used
    N = len(xgrid)
    C = np.zeros((N,N))
    for i in range(N):
        for j in range(i):
            val = (np.log10(i) - np.log10(j))**2
            C[i,j] = val
            C[j,i] = val
    return C


##############
### THEANO ###
##############
# define Theano variables

Cost = T.matrix('Cost')
Gamma = T.scalar('Gamma')
Ker = T.exp(-Cost/Gamma)
GaussKer = T.matrix('GaussKer')
n_iter = T.iscalar('n_iter')
Tau = T.scalar('Tau')
Rho = T.scalar('Rho')
spectrum = T.vector('spectrum')
A = T.vector('A')
K = T.matrix('K')#(1,42x42)
# define weights and dictionary
D_stack = T.tensor3('D_stack')
lbda_stack = T.matrix('lbda_stack')
Sigma = T.scalar('Sigma')
Flux = T.scalar('Flux')
newa = T.ones_like(D_stack[:,:,0])
kera = T.ones_like(D_stack[:,:,0])


#new A
Datapoint = T.tensor3('Datapoint')
K_all = T.tensor3() #<9,9,100>
Sigma_all = T.vector('Sigma_all')
Flux_all = T.vector('Flux_all')
barycenters = T.tensor3('barycenters') #<42x42,5,100wvl>
A_mat = T.matrix('A_mat') # <5,100objs>
beta_mat = T.matrix('beta_mat')  #<100wvls,100objs>


## Logdomain version ##
logD_stack = T.log(abs(D_stack))
Epsilon = T.scalar('Epsilon')

# Stabilized kernel computation
def StabKer(Cost, alpha, beta, Gamma):
    M = -Cost.dimshuffle(0,1,'x') + alpha.dimshuffle(0,'x',1) + beta.dimshuffle('x',0,1)
    # alpha.dimshuffle(0,'x',1) is u1t and beta.dimshuffle('x',0,1) is 1vt
    M = T.exp(M / Gamma)
    return M


# Log Sinkhorn iteration
def log_sinkhorn_step(alpha, beta, logp, logD_stack, lbda_stack, Gamma, Cost, Tau, Epsilon,i,v):
    M = StabKer(Cost,alpha,beta,Gamma)
    newalpha = Gamma * (logD_stack[:,:,i] - T.log(T.sum(M,axis=1) + Epsilon)) + alpha # new u
    # alpha = newalpha
    alpha = Tau*alpha + (1.-Tau)*newalpha # sinkhorn heavyball
    M = StabKer(Cost,alpha,beta,Gamma)
    lKta = T.log(T.sum(M, axis=0) + Epsilon) - beta/Gamma
    logp = T.sum(lbda_stack[v,:]*lKta, axis=1)
    newbeta = Gamma * (logp.dimshuffle(0,'x') - lKta) # new v
    # beta = newbeta
    beta = Tau*beta + (1.-Tau)*newbeta
    return alpha, beta, logp,lKta



def log_sinkhorn_algoritm(v,i,logD_stack,lbda_stack,Cost,Tau,Epsilon,n_iter):
    
    res, updates = theano.scan(log_sinkhorn_step, outputs_info=[T.zeros_like(logD_stack[:,:,0]),
                              T.zeros_like(logD_stack[:,:,0]), T.zeros_like(logD_stack[:,0,0]),None], 
                              non_sequences=[logD_stack,lbda_stack,Gamma,Cost,Tau,Epsilon,i,v], n_steps=n_iter)
    
    log_bary = T.exp(res[2][-1])
    lKta = res[3]
    beta = res[1]
    
    return log_bary,lKta,beta




def log_compute_bary(i,logD_stack,lbda_stack,Cost,Tau,Epsilon,n_iter):
    res,updates = theano.scan(log_sinkhorn_algoritm,
                             outputs_info = None,
                             sequences = T.arange(lbda_stack.shape[0]),
                             non_sequences = [i,logD_stack,lbda_stack,Cost,Tau,Epsilon,n_iter])
  
    barys = res[0]
    lKtas = res[1]
    betas = res[2]

    return barys,lKtas,betas
        

log_result_bary,log_updates_bary = theano.scan(log_compute_bary,
                                      outputs_info=None,
                                      sequences=T.arange(logD_stack.shape[2]),
                                      non_sequences = [logD_stack,lbda_stack,Cost,Tau,Epsilon,n_iter]) #<nb_comp, nb_wvl, pixels>

log_result_bary_rshp = T.swapaxes(T.swapaxes(log_result_bary[0],0,1),0,2) #<pixels,nb_comp, nb_wvl>
log_lKtas_rshp = T.swapaxes(T.swapaxes(log_result_bary[1],0,3),1,4) #<pixels,nb_atoms, nb_comp, nb_wvl>
log_betas_rshp = T.swapaxes(T.swapaxes(log_result_bary[2],0,3),1,4) #<pixels,nb_atoms, nb_comp,nb_wvl>



Theano_bary = theano.function([D_stack,lbda_stack,Gamma,Cost,n_iter,theano.In(Tau,value=-0.3),theano.In(Epsilon,value=1e-200)],log_result_bary_rshp)

barycenters = log_result_bary_rshp

def lin_comb_A(v,barycente,A_mat):
    return theano.dot(barycente[:,:,v],A_mat)


def lin_comb_B(k,barycenters,A_mat,res_A_rshp,K_all,Flux_all,Sigma_all):
    temp = T.reshape(theano.dot(res_A_rshp[:,k,:],T.reshape(beta_mat[:,k],(nb_wvl,1))),Shap)
    
    #===Decimate theano conv===
    im_full = conv.conv2d(temp,K_all[:,:,k],border_mode='full',subsample=(1, 1)) #30x30
    coord_ini = (Fshap[0]-1)/2
    coord_end = coord_ini + Shap[0]
    im_center = im_full[coord_ini:coord_end,coord_ini:coord_end]
    MX = (Flux_all[k]/Sigma_all[k])*im_center[::2,::2]
    
    return MX


res_A,updates_A = theano.scan(lin_comb_A,
                             outputs_info = None,
                             sequences = T.arange(beta_mat.shape[0]),
                             non_sequences = [barycenters,A_mat]) # <5comp,484,2>



res_A_rshp = T.swapaxes(T.swapaxes(res_A,0,1),1,2) #<484,2obj,5wvl>

res_B,updates_B = theano.scan(lin_comb_B,#for each object
                             outputs_info = None,
                             sequences = T.arange(beta_mat.shape[1]),
                             non_sequences = [barycenters,A_mat,res_A_rshp,K_all,Flux_all,Sigma_all]) #<2objs,22,22>

res_B_rshp = T.swapaxes(T.swapaxes(res_B,0,1),1,2)

Theano_coeff_MX = theano.function([A_mat,beta_mat,barycenters,Flux_all,Sigma_all,K_all],res_B_rshp)
Loss = 1./2*(Datapoint-res_B_rshp).norm(2)**2 

MtX_wdl = T.grad(Loss, D_stack)

Theano_wdl_MtX = theano.function([A_mat,beta_mat,Flux_all,Sigma_all,K_all,D_stack,lbda_stack,Cost,Gamma,n_iter,Datapoint,\
	theano.In(Tau,value=-0.0),theano.In(Epsilon,value=1e-200)],[MtX_wdl,res_B_rshp,barycenters],on_unused_input='ignore')


Theano_wdl_MX = theano.function([A_mat,beta_mat,Flux_all,Sigma_all,K_all,D_stack,lbda_stack,Cost,Gamma,n_iter,\
	theano.In(Tau,value=-0.0),theano.In(Epsilon,value=1e-200)],[res_B_rshp,barycenters,log_lKtas_rshp,log_betas_rshp])

MtX_coeff = T.grad(Loss,A_mat)

Theano_coeff_MtX = theano.function([A_mat,beta_mat,barycenters,Flux_all,Sigma_all,K_all,Datapoint],[MtX_coeff,res_B_rshp])
















