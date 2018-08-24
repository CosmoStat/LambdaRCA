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
Shap = (22,22) # Image full dimensions
Fshap = (9,9) # Filter dimensions
nb_atoms = 2
nb_wvl =5


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
    

    
##############
### THEANO ###
##############
# define Theano variables

Cost = T.matrix('Cost')
Gamma = T.scalar('Gamma')
Ker = T.exp(-Cost/Gamma)
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



Datapoint = T.tensor3('Datapoint')
nb_wvl =5
K_all = T.tensor3() #<9,9,100>
Sigma_all = T.vector('Sigma_all')
Flux_all = T.vector('Flux_all')
barycenters = T.tensor3('barycenters') #<42x42,5,100wvl>
A_mat = T.matrix('A_mat') # <5,100objs>
beta_mat = T.matrix('beta_mat')  #<100wvls,100objs>


# Sinkhorn barycenter iteration
def sinkhorn_step(a,b,p,D_stack,lbda_stack,Ker,Tau,i,v):
    newa = D_stack[:,:,i]/T.dot(Ker,b)
    a = a**Tau * abs(newa)**(1.-Tau)
    p = T.prod(abs(T.dot(Ker.T,a))**lbda_stack[v,:], axis=1)
    newb = p.dimshuffle(0,'x')/T.dot(Ker.T,a)
    b = b**Tau * newb**(1.-Tau)
    return a,b,p



def sinkhorn_algorithm(v,i,D_stack,lbda_stack,Ker,Tau,n_iter):
    res, updates = theano.scan(sinkhorn_step, outputs_info=[T.ones_like(D_stack[:,:,0]),
                              T.ones_like(D_stack[:,:,0]), T.ones_like(D_stack[:,0,0])], 
                              non_sequences=[D_stack,lbda_stack,Ker,Tau,i,v], n_steps=n_iter)  
    bary = res[2][-1]
    
    return bary
    

def compute_bary(i,D_stack,lbda_stack,Ker,Tau,n_iter):
    res,updates = theano.scan(sinkhorn_algorithm,
                             outputs_info = None,
                             sequences = T.arange(lbda_stack.shape[0]),
                             non_sequences = [i,D_stack,lbda_stack,Ker,Tau,n_iter])
    
    res_flip = res[::-1]

    
    return res_flip
    

result_bary,updates_bary = theano.scan(compute_bary,
                                      outputs_info=None,
                                      sequences=T.arange(D_stack.shape[2]),
                                      non_sequences = [D_stack,lbda_stack,Ker,Tau,n_iter]) #<nb_comp, nb_wvl, pixels>

result_bary_rshp = T.swapaxes(T.swapaxes(result_bary,0,1),0,2) #<pixels,nb_comp, nb_wvl>

Theano_bary = theano.function([D_stack,lbda_stack,Gamma,Cost,n_iter,theano.In(Tau,value=-0.3)],result_bary_rshp,on_unused_input='ignore')

barycenters = result_bary_rshp

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

res_B,updates_B = theano.scan(lin_comb_B,
                             outputs_info = None,
                             sequences = T.arange(beta_mat.shape[1]),
                             non_sequences = [barycenters,A_mat,res_A_rshp,K_all,Flux_all,Sigma_all]) #<2objs,22,22>

res_B_rshp = T.swapaxes(T.swapaxes(res_B,0,1),1,2)

Theano_coeff_MX = theano.function([A_mat,beta_mat,barycenters,Flux_all,Sigma_all,K_all],res_B_rshp)


Loss = 1./2*(Datapoint-res_B_rshp).norm(2)**2 

MtX_wdl = T.grad(Loss, D_stack)

Theano_wdl_MX = theano.function([A_mat,beta_mat,Flux_all,Sigma_all,K_all,D_stack,lbda_stack,Cost,Gamma,n_iter,theano.In(Tau,value=-0.3)],res_B_rshp)

Theano_wdl_MtX = theano.function([A_mat,beta_mat,Flux_all,Sigma_all,K_all,D_stack,lbda_stack,Cost,Gamma,n_iter,Datapoint,theano.In(Tau,value=-0.3)],[MtX_wdl,res_B_rshp,barycenters])

MtX_coeff = T.grad(Loss,A_mat)

Theano_coeff_MtX = theano.function([A_mat,beta_mat,barycenters,Flux_all,Sigma_all,K_all,Datapoint],[MtX_coeff,res_B_rshp])







# Theano_bary = theano.function([D_stack,lbda_stack,Gamma,Cost,n_iter,theano.In(Tau,value=-0.3)],only_bary_stack,on_unused_input='ignore')
# Theano_coeff_MX = theano.function([A_mat,beta_mat,barycenters,Flux_all,Sigma_all,K_all],[res_B_rshp],on_unused_input='ignore')










