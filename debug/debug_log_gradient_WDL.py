
import numpy as np
import sys
sys.path.append('../baryOT')
import logOT_bary as ot
import time
from scipy.ndimage import gaussian_filter
import scipy.signal as scisig



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


    A = np.zeros((K,J))
    for k in range(K):
        for j in range(J):
            max_x = np.max(X[:,k,j])
            A[k,j] = np.log(np.sum(np.exp(X[:,k,j] - max_x))) + max_x

    Y = []
    for k in range(K):
        y = np.zeros((I,J))
        for i in range(I):
            for j in range(J):
                y[i,j] = Cc[i,k]/gamma + A[k,j]
        Y.append(y)
    Y = np.array(Y)

    R = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            max_y = np.max(Y[:,i,j])
            R[i,j] = np.log(np.sum(np.exp(Y[:,i,j] - max_y))) + max_y

    return R


#one object, one component. Do big loop later to have the barycenters for every
def logSinkornLoop(Cr,Cc,D,lbda,n_iter):
    """
        PARAMETERS
        ----------
        D: <N,nb_atoms>
        lbda: <nb_atoms,>

    """

    N,S = D.shape
    sqrtN = int(np.sqrt(N))

    v = np.zeros(n_iter,N,S)
    phi = np.zeros(n_iter+1,N,S)
    for it in range(1,n_iter+1):
        logp = np.zeros(N)
        for s in range(S):
            phi[it,:,s] = LogSeparableKer(np.log(np.abs(D[:,s])).reshape(sqrtN,sqrtN) - LogSeparableKer(v[it-1,:,s].reshape(sqrtN,sqrtN),Cr,Cc,gamma),Cr,Cc,gamma).flatten()
            logp += lbda[s]* phi[it,:,s]
            v[it,:,s] = logp - phi[it,:,s]

    p = np.exp(logp)

    return p,phi,v


def computeBarycenters(Cc,Cr,D_stack,w_stack,n_iter):

    nb_comp = D_stack.shape[2]
    nb_wvl = w_stack.shape[0]

    for i in range(nb_comp):
        for v in range(nb_wvl):






























