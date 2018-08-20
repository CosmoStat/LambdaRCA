import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import conv
from theano.gradient import jacobian, hessian
import scipy.signal as scisig
from scipy.stats import norm
import time
from scipy.ndimage import gaussian_filter
Shap = (22,22) # Image full dimensions
Fshap = (9,9) # Filter dimensions
nb_atoms = 2


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
Datapoint = T.matrix('Datapoint')
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


# Sinkhorn barycenter iteration
def sinkhorn_step(a,b,p,D_stack,lbda_stack,Ker,Tau,i,v):
    newa = D_stack[:,:,i]/T.dot(Ker,b)
    a = a**Tau * abs(newa)**(1.-Tau)
    p = T.prod(T.dot(Ker.T,a)**lbda_stack[v,:], axis=1)
    newb = p.dimshuffle(0,'x')/T.dot(Ker.T,a)
    b = b**Tau * newb**(1.-Tau)
    return a,b,p



def only_bary(v,lbda_stack,D_stack,n_iter,Tau,Ker):
    # Sinkhorn algorithm
    i=0 #theano is called for each component separately in the case of barycenters
    result, updates = theano.scan(sinkhorn_step, outputs_info=[T.ones_like(D_stack[:,:,0]),
                              T.ones_like(D_stack[:,:,0]), T.ones_like(D_stack[:,0,0])], 
                              non_sequences=[D_stack,lbda_stack,Ker,Tau,i,v], n_steps=n_iter)
    # keep only the final barycenter
    bary = result[2][-1]
    
    return bary
    



    


def inner_loop(v,i,lbda_stack,D_stack,spectrum,n_iter,Tau,Ker):
    # Sinkhorn algorithm
    result, updates = theano.scan(sinkhorn_step, outputs_info=[T.ones_like(D_stack[:,:,0]),
                              T.ones_like(D_stack[:,:,0]), T.ones_like(D_stack[:,0,0])], 
                              non_sequences=[D_stack,lbda_stack,Ker,Tau,i,v], n_steps=n_iter)
    # keep only the final barycenter
    bary = result[2][-1]
    res_i = bary*spectrum[v]
    
    return res_i,bary



def outer_loop(i,D_stack,lbda_stack,spectrum,A,n_iter,Tau,Ker):
    
    res,updatess = theano.scan(inner_loop,
                                     sequences = T.arange(spectrum.shape[0]),
                                     non_sequences=[i,lbda_stack,D_stack,spectrum,n_iter,Tau,Ker])
    bary_stack_i_allv = res[0]
    bary_stack_i_allv = bary_stack_i_allv[::-1]
    
    bary = res[1]
    bary=bary[::-1]
    
    sum1 = T.sum(bary_stack_i_allv,axis=0)
    res_o = sum1*A[i]
 
    
    return res_o,bary





res_ob,updates_ob = theano.scan(only_bary,
                                     sequences = T.arange(lbda_stack.shape[0]),
                                     non_sequences=[lbda_stack,D_stack,n_iter,Tau,Ker])

only_bary_stack = res_ob[::-1]





res,updates = theano.scan(outer_loop,
                                sequences=T.arange(D_stack.shape[2]),
                                non_sequences=[D_stack,lbda_stack,spectrum,A,n_iter,Tau,Ker])

bary_stack = res[0]
bary = res[1] #(nb_comp, nb_wvl, 42x42)
sum2 = T.sum(bary_stack,axis=0)
sum_rshp = T.reshape(sum2,Shap,ndim=2)


#===Decimate theano conv===
MXfull = conv.conv2d(sum_rshp,K,border_mode='full',subsample=(1, 1)) #30x30
coord_ini = (Fshap[0]-1)/2
coord_end = coord_ini + Shap[0]
MXcenter = MXfull[coord_ini:coord_end,coord_ini:coord_end]
MX = (Flux/Sigma)*MXcenter[::2,::2]


Loss = 1./2*(Datapoint-MX).norm(2)**2 

MtX = T.grad(Loss, D_stack)




Theano_wass_MXMtX = theano.function([D_stack,lbda_stack,K,spectrum,A,Gamma,\
    Cost,n_iter,Sigma,Flux,Datapoint,theano.In(Tau,value=-0.3)],\
    [MX,MtX,bary],on_unused_input='ignore')



Theano_bary = theano.function([D_stack,lbda_stack,Gamma,Cost,n_iter,theano.In(Tau,value=-0.3)],only_bary_stack,on_unused_input='ignore')











