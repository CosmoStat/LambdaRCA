import numpy as np
import sys
sys.path.append('../baryOT')
import logOT_bary as ot





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

    Y = np.zeros((K,I,J))
    for k in range(K):
        for i in range(I):
            for j in range(J): 
                Y[k,i,j] = Cc[i,k]/gamma + A[k,j]


    R = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            max_y = np.max(Y[:,i,j])
            R[i,j] = np.log(np.sum(np.exp(Y[:,i,j] - max_y))) + max_y

    return R


def BackwardLoopDictionary():

    theano_MX = ot.Theano_wdl_MX    




def feat_fullgrad(Xw, Ys, gamma, n_iter=20, nabL=nab_L2, trunc=500):
    X, lbda = Xw
    S, N = Ys.shape
    sqrtN = int(sqrt(N))
    b = np.zeros((S,n_iter+1,N))
    b[:,0,:] = 1.
    phi = np.zeros((S,n_iter+1,N))
    gamma = sqrt(gamma)
    conv = lambda x: gaussian_filter(x.reshape(sqrtN,sqrtN), gamma, truncate=trunc).flatten()
    
    # Sinkhorn
    for j in range(1,n_iter+1):
        p = np.ones((N)).astype(float)
        for s in range(S):
            phi[s,j,:] = conv(Ys[s,:] / conv(b[s,j-1,:])) 
            p *= phi[s,j,:]**(lbda[s])
        b[:,j,:] = p / phi[:,j,:]
    
    n = nabL(p,X)
    v = np.zeros((S,N))
    c = np.zeros((S,N))
    grad = np.zeros((S,N))
    
    for j in range(n_iter,1,-1):
        for s in range(S):
            c[s,:] = conv((lbda[s]*n - v[s,:]) * b[s,j,:])
            grad[s,:] += c[s,:] / conv(b[s,j-1,:])
            v[s,:] = -1./phi[s,j-1,:] * conv((Ys[s,:] * c[s,:])/(conv(b[s,j-1,:])**2))
        n = np.sum(v, axis=0)
    return grad, p

















