

import numpy as np
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



#========================== TEST KLS ========================================================
shap = (22,22)
Cr,Cc = EuclidCost_split(shap[0],shap[1])
gamma = 2
im = np.random.rand(shap[0],shap[1]) 
R = LogSeparableKer(np.log(im),Cr,Cc,gamma)







