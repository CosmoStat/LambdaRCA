import gc
import numpy.linalg as LA
from numpy import *
import numpy as np
import scipy.sparse.linalg as SLA
from multiprocessing import Process, Queue, Pool
import numpy.random as random
import scipy.signal as scisig
import isap
import sys
sys.path.append('../utilities')
import utils
import os
import scipy.stats as scistats
import copy as cp
from pyflann import *
import psf_learning_utils
import scipy
from scipy.optimize import minimize#,linear_sum_assignment
sys.path.append('../sams')
from modopt.opt.cost import costObj
import grads as grad
import linear as sams_linear
import proximity as sams_prox
import modopt.opt.algorithms as optimalg

try:
    import pyct
except ImportError, e:
    pass # module doesn't exist, deal with it.



def ineq_proj(u,v): # Projects u onto the constraint u>=v
    u_out = copy(u)
    ind = where(u<v)
    u_out[ind] = v[ind]
    return u_out

def prox_coeff_sum(u,a): # Projects the vector u onto the contraint sum_i vi=a; we assume that u is a column vector
    dev = u.sum()-a
    siz = size(u)
    ones_vect = ones((siz,))
    b = dev/siz
    z = u - b*ones_vect
    return z

def prox_coeff_sum_mat(U,a): # Projects each column of u onto the contraint sum_i vi=a; we assume that u is a column vector
    Z = copy(U)
    shap = U.shape
    for i in range(0,shap[1]):
        Z[:,i] = prox_coeff_sum(U[:,i],a)
    return Z

def pos_proj(z,tol=0): # Puts negative entries of z to zero
    u = copy(z)
    u = u.reshape(size(z))
    i = where(u<0)
    u[i[0]] = 0
    shap = z.shape
    u = u.reshape(shap)
    return u

def pos_proj_mat(m,tol=0):
    u = copy(m)
    shap = m.shape
    j=0
    for j in range(0,shap[1]):
        u[:,j]=pos_proj(squeeze(m[:,j]),tol=tol)
    return u

def pos_proj_mat_2(m1,m2):
    m = m1+m2
    I1 = (m>=0)
    I2 = (m<0)
    m1_proj = m1*I1 + I2*(m1-m2)/2
    m2_proj = m2*I1 + I2*(m2-m1)/2
    return m1_proj,m2_proj

def pos_proj_cube(m,tol=0):
    u = copy(m)
    i,j,k = where(u<0)
    u[i,j,k] = 0
    return u

def prox_dot_prod(M,map): # Proximity operator of m -> <m,map> at M
    Mp = M-map
    return Mp

def positive_centering(stack):
    inf = copy(stack[:,:,0])
    shap = stack.shape
    stack_cent = copy(stack)
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            inf[i,j] = stack[i,j,:].min()

    for i in range(0,stack.shape[2]):
        stack_cent[:,:,i] -=inf

    return stack_cent,inf

def level_images(stack):
    stack_cp = copy(stack)
    stack_out = stack*0
    ind_stack = stack*0
    for i in range(0,stack.shape[2]):
        for k in range(0,stack.shape[0]):
            for l in range(0,stack.shape[1]):
                z = where(stack_cp[k,l,:]==stack_cp[k,l,:].max())
                ind_stack[k,l,z[0]] = i
                stack_out[k,l,i] = stack_cp[k,l,:].max()
                stack_cp[k,l,z[0]] = 0

    return stack_out,ind_stack


def prox_mat_marg(M,b): # Projects a matrix onto the constraints transpose(M)*ones_vect = b (the sum of each row is constraint

    shap = M.shape
    ones_vect = ones((shap[0],1))
    dev = (transpose(M).dot(ones_vect)-b)/shap[0]
    Mp = M - ones_vect.dot(transpose(dev))
    return Mp

def prox_mat_marg_l(M,b): # Projects a matrix onto the constraints M*ones_vect = b (the sum of each row is constraint

    Mt  =transpose(M)
    Mpt = prox_mat_marg(Mt,b)
    Mp = transpose(Mpt)
    return Mp

def prox_cube_marg(M,B):
    shap = M.shape
    Mp  =copy(M)
    k=0
    for k in range(0,shap[2]):
        bk = B[:,k]
        bk = bk.reshape((shap[0],1))
        Mp[:,:,k] = prox_mat_marg(squeeze(M[:,:,k]),bk)
    return Mp

def prox_cube_marg_homog(M):
    shap = M.shape
    k = 0
    ones_vect = ones((shap[1],1))
    mean_marg = zeros((shap[0],1))
    Mp = copy(M)
    for k in range(0,shap[2]):
        mk = M[:,:,k]
        mean_marg = mean_marg + mk.dot(ones_vect)/shap[2]
    print mean_marg.sum()
    for k in range(0,shap[2]):
        mk = transpose(M[:,:,k])
        mkp = prox_mat_marg(mk,mean_marg)

        Mp[:,:,k] = transpose(mkp)

    return Mp

def prox_hub(x,delta,lambd):
    shap = x.shape
    ones_mat = ones((shap[0],shap[1]))
    i,j = where(abs(x)<delta)

    mask = zeros((shap[0],shap[1]))
    mask[i,j] = 1

    projx = ((1+lambd)**(-1))*mask*x + (ones_mat-mask)*utils.thresholding(x,ones_mat*lambd*delta,1)

    return projx

def prox_hub_stack(x,delta,lambd):
    shap = x.shape
    prox_out = copy(x)
    for i in range(0,shap[2]):
        prox_out[:,:,i] = prox_hub(x[:,:,i],delta[i],lambd)

    return prox_out

def bar_coord_proj(pos,target_pos,est_coeff_bar_coeff): # Locations arte stored in pos lines
    vect_pos = ones((3,1))
    vect_pos[:-1,:] = target_pos.reshape((2,1))
    shap = pos.shape
    A = ones((3,shap[0]))
    A[:-1,:] = transpose(pos)
    AAT_inv = LA.inv(A.dot(transpose(A)))
    B = transpose(A).dot(AAT_inv)

    out = est_coeff_bar_coeff - B.dot(A.dot(est_coeff_bar_coeff)-vect_pos)

    return out

def bar_coord_pb_dijkstra(pos,target_pos,nb_iter=10):
    shap = pos.shape
    init_pos = zeros((shap[0],1))
    id_flag = False
    ind_id = -1
    for i in range(0,shap[0]):
        init_pos[i,0] = sqrt((target_pos[0]-pos[i,0])**2+(target_pos[1]-pos[i,1])**2)
        if init_pos[i,0]==0:
            id_flag = True
            ind_id = i
    if id_flag:
        a = init_pos[ind_id,0]
        init_pos*=0
        init_pos[ind_id,0] = 1
    else:
        for i in range(0,shap[0]):
            init_pos[i,0] = init_pos[i,0]**(-1)
        t = sum(init_pos)
        init_pos/=t

    x = copy(init_pos)
    p = x*0
    q = x*0
    if id_flag is not True:
        for i in range(0,nb_iter):
            y = bar_coord_proj(pos,target_pos,x+p)
            p = x+p-y
            x = pos_proj(y+q)
            q = y+q-x
            temp = transpose(pos).dot(x)
            print "Check: min val,",x.min()," sum: ",x.sum()," coordinates error: ",(temp[0,0]-target_pos[0])**2+(temp[1,0]-target_pos[1])**2

    return x

def bar_coord_pb(A,B,nb_iter): # Solve the problem min ||At-B||^2, s.t. sum_i ti=1, ti>=0; calculates the best "barycentric" coordinates of B with respect to A columns
    from numpy.linalg import norm
    w1 = 0.5
    w2 = 1-w1
    M = transpose(A).dot(A)
    w,v = LA.eigh(M)
    tol = 0.1
    lip = abs(w).max()
    print "lip coeff: ",lip
    mu = 1/((1+tol)*lip)
    shap = A.shape

    # Initialization
    i=0
    minim = 1e16
    ind=0
    for i in range(0,shap[1]):
        ai = A[:,i]
        dist = ((ai-B)**2).sum
        if dist < minim:
            ind = i
            minim = dist
    t = zeros((shap[1],1))
    t[ind] = 1
    z1 = zeros((shap[1],1))
    z2 = zeros((shap[1],1))
    mse = zeros((nb_iter,))
    lambd = 1.5
    for i in range(0,nb_iter):
        res = A.dot(t)-B
        grad = transpose(A).dot(res)
        mse[i] = (res**2).sum()
        #print 100*sqrt(mse[i])/norm(B)
        temp1 = 2*t - z1 - mu*grad
        z1 = z1 + lambd*((prox_coeff_sum(temp1.reshape((shap[1],)),1)).reshape((shap[1],1))-t)
        temp2 = 2*t - z2 - mu*grad
        z2 = z2 + lambd*(pos_proj(temp2.reshape((shap[1],))).reshape((shap[1],1))-t)
        t = w1*z1+w2*z2
    t = pos_proj(t.reshape((shap[1],)))/t.sum()
    return t,100*sqrt(mse[-1])/norm(t)

def bar_coord_pb_field(A,B,tinit=None,nb_iter=100,tol=1e-15): # Solve the problem min ||At-B||^2, s.t. sum_i ti=1, ti>=0; calculates the best "barycentric" coordinates of B with respect to A columns; B and t are now matrices
    w1 = 0.5
    w2 = 1-w1
    M = transpose(A).dot(A)
    w,v = LA.eigh(M)
    tol0 = 0.1
    lip = abs(w).max()
    mu = 1/((1+tol0)*lip)
    shap = A.shape
    shap1 = B.shape

    # Initialization
    t = zeros((shap[1],shap1[1]))
    if tinit is None:
        for j in range(0,shap1[1]):
            minim = 1e16
            ind=0
            for i in range(0,shap[1]):
                dist = ((A[:,i]-B[:,j])**2).sum
                if dist < minim:
                    ind = i
                    minim = dist
            t[ind,j] = 1
    else:
        t = copy(tinit)
    z1 = copy(t)
    z2 = copy(t)
    mse = zeros((nb_iter,))
    lambd = 1.5
    var = 100
    i=0
    while i<nb_iter and var>tol:

        res = A.dot(t)-B
        grad = transpose(A).dot(res)
        mse[i] = (res**2).sum()
        print mse[i]
        temp1 = 2*t - z1 - mu*grad
        z1 = z1 + lambd*(prox_coeff_sum_mat(temp1,1)-t)
        temp2 = 2*t - z2 - mu*grad
        z2 = z2 + lambd*(pos_proj(temp2)-t)
        told = copy(t)
        t = w1*z1+w2*z2
        var  = 100*((t-told)**2).sum()/(told**2).sum()
        i+=1
    print "coeff res: ",mse[i-1]
    """t = pos_proj(t)
    one_vect = ones((shap[1],1))
    t = t/(one_vect.dot((t.sum(axis=0)).reshape(1,shap1[1])))"""
    return t

def lsq_mat_inv(A,B,neg_tol=-1e-5):
    col_basis, s, Vt = linalg.svd(A,full_matrices=False)
    M = A.dot(transpose(Vt)).dot(diag(s**(-1)))
    coeff = transpose(Vt).dot(diag(s**(-1))).dot(transpose(M)).dot(B)
    B_approx = A.dot(coeff)
    #err = ((B-B_approx)**2).sum()
    #print "lsq err: ",err
    nb_samp = 1
    siz = B.shape
    if len(siz)>1:
        nb_samp = siz[1]
    flags = zeros((nb_samp,))
    nb_flags = 0
    for i in range(0,nb_samp):
        ind = where(coeff[:,i]<0)
        if sum(coeff[ind,i])<neg_tol:
            print sum(coeff[ind,i])
            flags[i]=1
            nb_flags+=1
    print nb_flags
    return coeff,flags


def lsq_mat_inv_lin(A,B):
    dim_data = A.shape[0]
    nb_data = B.shape[1]
    nb_src = A.shape[1]
    output = zeros((nb_src,nb_data))
    ones_vect = ones((nb_src,))
    for i in range(0,nb_data):
        optim_res = scipy.optimize.linprog(ones_vect, A_ub=-eye(nb_src), b_ub=ones_vect*0,A_eq=transpose(A).dot(A),b_eq = transpose(A).dot(B[:,i]))
        output[:,i] = optim_res.x
    return output

def lsq_mat(S,Y): # Solves min_A ||Y-SA||_2^2
    Y2 = transpose(S).dot(Y)
    S2 = transpose(S).dot(S)
    A = LA.inv(S2).dot(Y2)
    return A


def coeff_adjust(coeff,A,nb_iter=100,tol_neg=-1e-5): # computes a zero mean vector u with the same length as coeff, in A null space so that coeff+u>=0, with dijkstra algorithm
    x = copy(coeff)
    p = copy(x)*0
    q = copy(x)*0
    i = 0
    cons_mat = zeros((A.shape[0]+1,A.shape[1]))
    cons_mat[:-1,:] = A
    cons_mat[-1,:] = 1
    cons_mat = transpose(cons_mat)
    cons_mat, s, Vt = linalg.svd(cons_mat,full_matrices=False)
    ind = where(x<0)
    neg_w = sum(x[ind])
    while i<nb_iter and neg_w<tol_neg:
        y = ineq_proj(x+p,-coeff)
        p = x+p-y
        x = y+q-utils.proj_vect(y+q,cons_mat,ortho_en=False)
        q = y+q-x
        ind = where(x+coeff<0)
        neg_w = sum(x[ind]+coeff[ind])
        print neg_w
        i+=1
    return x,cons_mat


def cv_feasibility_pb(coeff,A,w1=0.5,nb_iter=100,tol_neg=-1e-5):
    x = copy(coeff)
    w2 = 1-w1
    i = 0
    cons_mat = zeros((A.shape[0]+1,A.shape[1]))
    cons_mat[:-1,:] = A
    cons_mat[-1,:] = 1
    cons_mat = transpose(cons_mat)
    cons_mat, s, Vt = linalg.svd(cons_mat,full_matrices=False)
    ind = where(x<0)
    neg_w = sum(x[ind])

    while i<nb_iter and neg_w<tol_neg:
        y = ineq_proj(x,-coeff)
        x = y-utils.proj_vect(y,cons_mat,ortho_en=False)
        #x = w1*ineq_proj(x,-coeff)+w2*(x-utils.proj_vect(x,cons_mat,ortho_en=False))
        ind = where(x+coeff<0)
        neg_w = sum(x[ind]+coeff[ind])
        print neg_w
        i+=1
    return x,cons_mat

def coeff_adjust_dr(coeff,A,nb_iter=100,tol_neg=-1e-5,gamma=1):
    x = copy(coeff)
    y = copy(x)*0
    i = 0
    cons_mat = zeros((A.shape[0]+1,A.shape[1]))
    cons_mat[:-1,:] = A
    cons_mat[-1,:] = 1
    cons_mat = transpose(cons_mat)
    cons_mat, s, Vt = linalg.svd(cons_mat,full_matrices=False)
    ind = where(x<0)
    neg_w = sum(x[ind])
    lambd = 1
    thresh = gamma*ones((len(coeff),))
    while i<nb_iter: #and neg_w<tol_neg:
        x = y-((double(i)/(nb_iter-1))**(10))*utils.proj_vect(y,cons_mat,ortho_en=False)
        y = y + lambd*(utils.thresholding(2*x-y+coeff,thresh,1)-x-coeff)
        ind = where(x+coeff<0)
        neg_w = sum(x[ind]+coeff[ind])
        i+=1
    print 'neg weight: ',neg_w,'l1 norm',abs(x).sum()

    return x,cons_mat



def pos_lsq_mat_inv(A,B):
    coeff = lsq_mat_inv(A,B)
    nb_vect = A.shape[1]
    nb_samp = 1
    siz = B.shape
    if len(siz)>1:
        nb_samp = siz[1]
    coeff_out = copy(coeff)*0
    for i in range(0,nb_samp):
        ind = where(coeff[:,i]>min(coeff[:,i].min(),0))
        coeff_out[ind[0],i] = lsq_mat_inv(A[:,ind[0]],B[:,i])

    B_approx_1 = A.dot(coeff)
    B_approx_2 = A.dot(coeff_out)
    err_1 = ((B-B_approx_1)**2).sum()
    err_2 = ((B-B_approx_2)**2).sum()
    print "lsq err: ",err_1," ",err_2
    print "min val:",coeff.min()," ",coeff_out.min()
    return coeff,coeff_out


def iter_lsq(A,B,nb_iter=100,tinit=None,pos_en=False,tol=1e-15): # Solves ||AX-B||^2
    M = transpose(A).dot(A)
    w,v = LA.eigh(M)
    tol0 = 0.1
    lip = abs(w).max()
    mu = 1/((1+tol0)*lip)
    shap = A.shape
    shap1 = B.shape

    # Initialization
    t = zeros((shap[1],shap1[1]))
    if tinit is None:
        for j in range(0,shap1[1]):
            minim = 1e16
            ind=0
            for i in range(0,shap[1]):
                dist = ((A[:,i]-B[:,j])**2).sum
                if dist < minim:
                    ind = i
                    minim = dist
            t[ind,j] = 1
    else:
        t = copy(tinit)
    mse = zeros((nb_iter,))
    lambd = 1.5
    var = 100
    i=0
    print " ----- ",nb_iter," ------ "
    while i<nb_iter:# and var>tol:

        res = A.dot(t)-B
        grad = transpose(A).dot(res)
        fw_grad = A.dot(grad)
        mse[i] = (res**2).sum()
        # Optimal steps computation
        fw_grad_norm = diag(transpose(fw_grad).dot(fw_grad))
        #print fw_grad_norm.min()

        steps = (res*fw_grad).sum(axis=0)
        ind = where(fw_grad_norm>0)
        steps[ind] /=fw_grad_norm[ind]
        #print mse[i]
        told = copy(t)
        t = t - grad.dot(diag(steps))
        #t = t - mu*grad
        if pos_en:
            t = pos_proj(t)
        var  = 100*((t-told)**2).sum()/(told**2).sum()
        i+=1
    print "coeff res: ",mse[i-1]
    """t = pos_proj(t)
        one_vect = ones((shap[1],1))
        t = t/(one_vect.dot((t.sum(axis=0)).reshape(1,shap1[1])))"""
    return t

def iter_lsq_accel(A,B,nb_iter=100,zinit=None,pos_en=False,tol=1e-15): # Solves ||AX-B||^2


    M = transpose(A).dot(A)
    w,v = LA.eigh(M)
    tol0 = 0.1
    lip = abs(w).max()
    mu = 1/((1+tol0)*lip)
    shap = A.shape
    shap1 = B.shape

    X2 = B.dot(transpose(B))
    U, s, Vt = linalg.svd(X2,full_matrices=False)
    ref_res = s[shap[1]:].sum()
    print 'ref res: ',ref_res
    # Initialization
    z = zeros((shap[1],shap1[1]))
    if zinit is None:
        for j in range(0,shap1[1]):
            minim = 1e16
            ind=0
            for i in range(0,shap[1]):
                dist = ((A[:,i]-B[:,j])**2).sum
                if dist < minim:
                    ind = i
                    minim = dist
            z[ind,j] = 1
    else:
        z = copy(zinit)
    x = copy(z)
    mse = zeros((nb_iter,))
    var = 100
    i=0
    t=1
    print " ----- ",nb_iter," ------ "
    while i<nb_iter:# and var>tol:

        res = A.dot(z)-B
        grad = transpose(A).dot(res)
        mse[i] = (res**2).sum()
        print mse[i]
        y = z - mu*grad
        xold = copy(x)
        if pos_en:
            x = pos_proj(y)
        else:
            x = copy(y)
        told = t
        t = (1+sqrt(4*t**2 +1))/2
        lambd = 1+(told-1)/t
        zold = copy(z)
        z = xold+lambd*(x-xold)
        var  = 100*((z-zold)**2).sum()/(zold**2).sum()
        i+=1
    print "coeff res: ",mse[i-1]
    """t = pos_proj(t)
        one_vect = ones((shap[1],1))
        t = t/(one_vect.dot((t.sum(axis=0)).reshape(1,shap1[1])))"""
    return z


#def iter_hard_thresh_lsq():

def sparse_bar_coord_pb_field(A,B,tinit=None,thresh_map=None,thresh_type=1,nb_iter=100,tol=1): # Solve the problem min ||At-B||^2, s.t. sum_i ti=1, ti>=0; calculates the best "barycentric" coordinates of B with respect to A columns; B and t are now matrices
    w1 = 0.5
    w2 = 1-w1
    M = transpose(A).dot(A)
    w,v = LA.eigh(M)
    tol0 = 0.1
    lip = abs(w).max()
    mu = 1/((1+tol0)*lip)
    shap = A.shape
    shap1 = B.shape

    # Initialization
    t = zeros((shap[1],shap1[1]))
    if tinit is None:
        for j in range(0,shap1[1]):
            minim = 1e16
            ind=0
            for i in range(0,shap[1]):
                dist = ((A[:,i]-B[:,j])**2).sum
                if dist < minim:
                    ind = i
                    minim = dist
            t[ind,j] = 1
    else:
        t = copy(tinit)
    z1 = copy(t)
    z2 = copy(t)
    mse = zeros((nb_iter,))
    lambd = 1.5
    var = 100
    i=0
    while i<nb_iter and var>tol:

        res = A.dot(t)-B
        grad = transpose(A).dot(res)
        mse[i] = (res**2).sum()
        print mse[i]
        temp1 = 2*t - z1 - mu*grad
        z1 = z1 + lambd*(prox_coeff_sum_mat(temp1,1)-t)
        temp2 = 2*t - z2 - mu*grad
        if thresh_map is not None:
            z2 = z2 + lambd*(pos_proj(utils.thresholding(temp2,thresh_map,thresh_type))-t)
        else:
            z2 = z2 + lambd*(pos_proj(temp2)-t)
        told = copy(t)
        t = w1*z1+w2*z2
        var  = 100*((t-told)**2).sum()/(told**2).sum()
        i+=1

    print "coeff res: ",mse[i-1]
    t = pos_proj(t)
    one_vect = ones((shap[1],1))
    t = t/(one_vect.dot((t.sum(axis=0)).reshape(1,shap1[1])))
    return t

def displacement_interp(distrib_mat,cross_maps,bar_coordinates,thresh,nb_iter): # Calculate optimal mass displacement at a specified location between given some given distributions, in the sense of wassertein L2 distance. The masses of the given distribuiton are in the columns of distrib_mat. The total mass of each distribution must be the same.
    w = 1.0/3
    lambd = 1 #1.5
    shap = distrib_mat.shape
    P = zeros((shap[0],shap[0],shap[1]))
    k=0
    mean_distrib = zeros((shap[0],1))
    for k in range(0,shap[1]):
        mean_distrib = mean_distrib + distrib_mat[:,k]/shap[1]
    i=0
    for i in range(0,shap[0]):
        for k in range(0,shap[1]):
            P[i,i,k] = mean_distrib[i,0]
    z1 = copy(P)
    z2 = copy(P)
    z3 = copy(P)
    z4 = copy(P)
    cost = zeros((nb_iter,))
    l = 0
    grad = copy(cross_maps)
    thresh_map = 0*P
    ones_vect = ones((shap[0],1))
    for k in range(0,shap[1]):
        distrib_k = distrib_mat[:,k]
        distrib_k = distrib_k.reshape((1,shap[0]))
        thresh_map[:,:,k] = ones_vect.dot(distrib_k)*thresh
    thresh_type = 1 # Soft thresh
    mu = 1.0/((grad**2).sum())
    for k in range(0,shap[1]):
        grad[:,:,k] = bar_coordinates[k]*grad[:,:,k]
    a1 = 0.2
    a2 = 9*(1-a1)/10
    a3 = 2*(1 - a1-a2)/3
    a4 = 1 -a1-a2-a3
    for l in range(0,nb_iter):
        for k in range(0,shap[1]):
            cost[l] = cost[l]+bar_coordinates[k]*(P[:,:,k]*cross_maps[:,:,k]).sum()
        print cost[l]
        temp1 = 2*P - z1 - mu*grad
        z1 = z1 + lambd*(prox_cube_marg(temp1,distrib_mat)-P)
        temp2 = 2*P - z2 - mu*grad
        z2 = z2 + lambd*(prox_cube_marg_homog(temp2)-P)
        temp3 = 2*P - z3 - mu*grad
        z3 = z3 + lambd*(pos_proj_cube(temp3)-P)
        temp4 = 2*P - z4 - mu*grad
        z4 = z4 + lambd*(utils.thresholding_3D(temp4,thresh_map,thresh_type)-P)
        #P = (z1+z2+z3+z4)/4
        P = a1*z1+a2*z2+a3*z3+a4*z4
    return P,cost


def opt_coupling(distrib1,distrib2,cross_map,thresh,nb_iter):

    w = 1.0/3
    lambd = 1
    shap = cross_map.shape
    P = zeros((shap[0],shap[0])) # We assume cross_map is a square matrix
    k=0
    mean_distrib = (distrib1+distrib2)/2
    i=0
    for i in range(0,shap[0]):
        P[i,i] = mean_distrib[i,0]
    z1 = copy(P)
    z2 = copy(P)
    z3 = copy(P)
    z4 = copy(P)
    cost = zeros((nb_iter,))
    l = 0
    grad = copy(cross_map)
    mu_max = 1.0/(grad.sum())
    mu_min=0

    ones_vect = ones((1,shap[0]))
    thresh_map_1 = distrib1.dot(ones_vect)*thresh
    thresh_map_2 = transpose(distrib2.dot(ones_vect))*thresh
    thresh_map = thresh_map_1
    thresh_type=1 # Soft thresholding
    for l in range(0,nb_iter):
        #mu = (mu_max-mu_min)*(nb_iter-1-l)/(nb_iter-1) + mu_min
        mu = mu_max
        cost[l] = (P[:,:]*cross_map[:,:]).sum()
        print cost[l]
        temp1 = 2*P - z1 - mu*grad

        z1 = z1 + lambd*(prox_mat_marg_l(temp1,distrib1)-P)
        temp2 = 2*P - z2 - mu*grad
        z2 = z2 + lambd*(prox_mat_marg(temp2,distrib2)-P)
        temp3 = 2*P - z3 - mu*grad
        z3 = z3 + lambd*(pos_proj_mat(temp3)-P)
        temp4 = 2*P - z4 - mu*grad
        z4 = z4 + lambd*(utils.thresholding(temp4,thresh_map,thresh_type)-P)
        P = (z1+z2+z3+z4)/4
    #P = (z1+z2+z3)/3

    return P,cost,grad

def opt_rotation(map_1,map_2,coupling_weights,nb_iter=100):
    M = zeros((2,2))
    B = zeros((2,1))
    Y = zeros((2,1))
    i = 0
    j = 0
    shap = map_1.shape
    O = zeros((2,1))
    O[0,0]=map_1[(shap[0]-1)/2,(shap[1]-1)/2,0]
    O[1,0]=map_1[(shap[0]-1)/2,(shap[1]-1)/2,1]
    for i in range(0,shap[0]*shap[1]):
        tempi=0
        i1 = floor(i/shap[1]) # Line index in map_1
        j1 = mod(i,shap[1])
        Mi  = array([[map_1[i1,j1,0]-O[0,0],-(map_1[i1,j1,1]-O[1,0])],[(map_1[i1,j1,1]-O[1,0]),map_1[i1,j1,0]-O[0,0]]])
        for j in range(0,shap[0]*shap[1]):
            tempi = tempi + coupling_weights[i,j]
            i2 = floor(j/shap[1]) # Line index in map_2
            j2 = mod(j,shap[1])
            Y[0,0] = map_2[i2,j2,0]
            Y[1,0] = map_2[i2,j2,1]
            B = B + coupling_weights[i,j]*transpose(Mi).dot(Y-O)
        M = M + tempi*transpose(Mi).dot(Mi)

    w,v = LA.eigh(M)
    tol = 0.1
    lip = abs(w).max()
    mu = 1/((1+tol)*lip)
    print mu
    l = 0
    mse = zeros((nb_iter,))

    U = zeros((2,1))
    U[0,0]=1
    V = copy(U)
    V_old = copy(V)

    t=1
    told=t

    for l in range(0,nb_iter):
        grad = M.dot(U)-B
        mse[l] = (grad**2).sum()
        print mse[l]
        V = U - mu*grad
        V = V/sqrt((V**2).sum())
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        U = V_old + lamb*(V-V_old)
        V_old = copy(V)
        theta = arccos(U[0,0])
        if (U[1,0]<0):
            theta=-theta
        print 'theta = ',theta*180/pi
        print 'B = ',B

    theta = arccos(U[0,0])
    if (U[1,0]<0):
        theta=-theta

    rot_mat = array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    rot_map = 0*map_1
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            Y[0,0] = map_1[i,j,0]
            Y[1,0] = map_1[i,j,1]
            Yrot = rot_mat.dot(Y-O)+O
            rot_map[i,j,0] = Yrot[0,0]
            rot_map[i,j,1] = Yrot[1,0]
    return theta,rot_map



#def rotation_registration(im1,im2) # The images are supposed to be centered

def nuc_norm_thresh(M,thresh,thresh_type):

    U, s, Vt = linalg.svd(M,full_matrices=False)
    sthresh = utils.thresholding(s,thresh,thresh_type)
    S = diag(sthresh)
    Mthresh = U.dot(S.dot(Vt))

    return Mthresh,s,U,Vt

def trunc_svd(M,nb_comp):
    U, s, Vt = linalg.svd(M,full_matrices=False)
    s[nb_comp:]=0
    S = diag(s)
    Mtrunc = U.dot(S.dot(Vt))
    return Mtrunc

def trunc_svd_cube(cube,nb_comp):
    shap = cube.shape
    mat = zeros((shap[2],shap[0]*shap[1]))
    for k in range(0,shap[2]):
        mat[k,:] = cube[:,:,k].reshape((shap[0]*shap[1],))
    mat_trunc = trunc_svd(mat,nb_comp)
    cube_out = copy(cube)
    for k in range(0,shap[2]):
        cube_out[:,:,k] = mat_trunc[k,:].reshape((shap[0],shap[1]))
    return cube_out

def svd_cube_compression(data_cube,var_percent=0.9):
    shap = data_cube.shape
    mat = utils.cube_to_mat(data_cube)
    U, s, Vt = linalg.svd(mat,full_matrices=False)
    e = s.sum()
    ec = 0
    rank=0
    while ec<e*0.99:
        ec +=s[rank]
        rank+=1
    basis = utils.mat_to_cube(Vt[0:rank,:],shap[0],shap[1])
    return basis,rank

def spca(M,shap=None,k=3,nb_iter=10,nb_rw=1,nb_process=5):

    shap0 = M.shape
    #U,s,Vt = SLA.svds(M,2*nb_comp_max)
    U, s, Vt = linalg.svd(M,full_matrices=False)
    sigma = s[int(min(shap0[0],shap0[1])/2)]
    n=max(shap0[0],shap0[1])
    nu = k*sigma*ones((min(shap0[0],shap0[1]),)) #sqrt(2*n)*sigma*ones((nb_comp_max,))
    to = sqrt(2)*sigma*ones((shap0[0],shap0[1]))
    Md = 0*M
    O = 0*M
    gradk = 0*O
    lambd = 1.5
    mu=1
    i=0
    thresh_type=1
    mse = zeros((nb_iter,))
    l=0

    for l in range(0,nb_rw+1):
        for i in range(0,nb_iter):
            gradk = Md+O - M
            mse[i] = (gradk**2).sum()
            print 'mse[i]: ',mse[i]
            temp = Md-mu*gradk
            temp1m = Md-mu*gradk
            temp1o = O-mu*gradk
            Md,si,U,Vt = nuc_norm_thresh(temp1m,nu,thresh_type)
            O = utils.thresholding(temp1o,to,thresh_type)
        U, s, Vt = linalg.svd(Md,full_matrices=False)
        nu = nu/((s/(sqrt(2)*sigma))+1)
    return Md,O,gradk,sigma

def positive_spca(M,shap=None,k=3,nb_iter=10,nb_rw=1,nb_process=5):
    shap0 = M.shape
    #U,s,Vt = SLA.svds(M,2*nb_comp_max)
    U, s, Vt = linalg.svd(M,full_matrices=False)
    sigma = s[int(min(shap0[0],shap0[1])/2)]
    n=max(shap0[0],shap0[1])
    nu = k*sigma*ones((min(shap0[0],shap0[1]),)) #sqrt(2*n)*sigma*ones((nb_comp_max,))
    to = sqrt(2)*sigma*ones((shap0[0],shap0[1]))
    Md = 0*M
    z1m = 0*M
    z2m = 0*M
    O = 0*M
    z1o = 0*M
    z2o = 0*M
    gradk = 0*O
    lambd = 1
    mu=0.9
    i=0
    w1=0.7
    w2 = 1-w1
    thresh_type=1
    mse = zeros((nb_iter,))
    l=0
    pool = Pool(processes=3)

    for l in range(0,nb_rw+1):
        for i in range(0,nb_iter):
            gradk = Md+O - M
            mse[i] = (gradk**2).sum()
            print mse[i]
            args1 = [Md,z1m,gradk,mu,nu,thresh_type,lambd,1]
            args2 = [O,z1o,gradk,mu,to,thresh_type,lambd,2]
            args3 = [Md,z2m,gradk,mu,lambd,3]
            args4 = [O,z2o,gradk,mu,lambd,4]
            res = pool.map(mp_positive_spca_aux, [args2,args3,args4])

            #z1m  = res[0]
            z1m = mp_positive_spca_aux(args1)
            z1o = res[0]
            z2m = res[1]
            z2o = res[2]
            #print (z1m**2).sum()
            #    q = Queue()
            #    jobs = []
            #    for i in range(0,nb_proc):
            #        ib = i*slice
            #        ie = ib+slice-1
            #        p = Process(target=utils.mp_rot_registration_cube, args=(gal_test,ib,ie,q))
            #        p.start()
            #        jobs.append(p)
            #    for i in range(0,nb_proc):
            #        a = q.get()


            """temp1m = 2*Md-z1m-mu*gradk
                temp1m,si,U,Vt = nuc_norm_thresh(temp1m,nu,thresh_type)
                z1m = z1m + lambd*(temp1m-Md)

                temp1o = 2*O-z1o-mu*gradk
                temp1o = utils.thresholding(temp1o,to,thresh_type)
                z1o = z1o + lambd*(temp1o-O)

                temp2m = 2*Md-z2m-mu*gradk
                temp2m = pos_proj_mat(temp2m)
                z2m = z2m + lambd*(temp2m-Md)

                z2o = z2o + lambd*(O-z2o-mu*gradk)"""
            Md = w1*z1m+w2*z2m
            O = w1*z1o+w2*z2o
        U, s, Vt = linalg.svd(Md,full_matrices=False)
        nu = nu/((s/(sqrt(2)*sigma))+1)
    return Md,O,gradk,sigma

def mp_positive_spca_aux(args):
    if args[-1]==1:
        return mp_nuc_norm_aux(args)
    if args[-1]==2:
        return mp_thresh_aux(args)
    if args[-1]==3:
        return mp_pos_aux(args)
    if args[-1]==4:
        return mp_id_aux(args)

def mp_nuc_norm_aux(args):

    x = args[0]
    z = args[1]
    grad = args[2]
    mu = args[3]
    nu = args[4]
    thresh_type = args[5]
    lambd = args[6]
    tempx = 2*x-z-mu*grad
    tempx,s,U,Vt = nuc_norm_thresh(tempx,nu,thresh_type)
    z = z + lambd*(tempx-x)

    return z



def mp_thresh_aux(args):
    x = args[0]
    z = args[1]
    grad = args[2]
    mu = args[3]
    to = args[4]
    thresh_type = args[5]
    lambd = args[6]
    tempx = 2*x-z-mu*grad
    tempx = utils.thresholding(tempx,to,thresh_type)
    z = z + lambd*(tempx-x)
    return z

def mp_pos_aux(args):
    x = args[0]
    z = args[1]
    grad = args[2]
    mu = args[3]
    lambd = args[4]
    tempx = 2*x-z-mu*grad
    tempx = pos_proj_mat(tempx)
    z = z + lambd*(tempx-x)
    return z

def mp_id_aux(args):
    x = args[0]
    z = args[1]
    grad = args[2]
    mu = args[3]
    lambd = args[4]
    z = z + lambd*(x-z-mu*grad)
    return z

def positive_spca_2(M,nb_comp_max,shap=None,k=3,nb_iter=20,nb_iter_2=100,nb_rw=5,nb_process=5):

    shap0 = M.shape
    #U,s,Vt = SLA.svds(M,2*nb_comp_max)
    U, s, Vt = linalg.svd(M,full_matrices=False)
    sigma = s[int(min(shap0[0],shap0[1])/2)]
    n=max(shap0[0],shap0[1])
    nu = k*sigma*ones((min(shap0[0],shap0[1]),)) #sqrt(2*n)*sigma*ones((nb_comp_max,))
    to = sqrt(2)*sigma*ones((shap0[0],shap0[1]))
    w1=0.66666666666666666
    w2 = 1-w1

    Md = 0*M
    O = 0*M
    z1m = 0*Md
    z1O = 0*Md
    z1fs = 0*Md
    z2m = 0*Md
    z2O = 0*Md
    z2fs = 0*Md
    xm = 0*Md
    xO = 0*Md
    xfs = 0*Md
    gradk = 0*O
    lambd = 1.5
    mu=0.5
    i=0
    thresh_type=1
    mse = zeros((nb_iter_2,))
    #mse = zeros((100,))
    l=0
    faint_struct_supp = None
    faint_struct = None
    if shap is not None:
        faint_struct_supp = zeros((shap[0],shap[1],shap[2]))
        faint_struct = zeros((shap[0],shap[1],shap[2]))

    for l in range(0,nb_rw-1):
        for i in range(0,nb_iter):
            gradk = Md+O - M
            mse[i] = (gradk**2).sum()
            print mse[i]
            temp1m = Md-mu*gradk
            temp1o = O-mu*gradk
            Md,si,U,Vt = nuc_norm_thresh(temp1m,nu,thresh_type)
            O = utils.thresholding(temp1o,to,thresh_type)
        U, s, Vt = linalg.svd(Md,full_matrices=False)
        nu = nu/((s/(sqrt(2)*sigma))+1)
    xm,xO = Md,O

    res = utils.mat_to_cube(gradk,shap[0],shap[1])
    faint_struct_supp = utils.mp_struct_res_support(res,nb_process,thresh=20,wind=10)
    supp = utils.cube_to_mat(faint_struct_supp)

    for i in range(0,nb_iter_2):
        #for i in range(0,100):
        temp1m = 2*xm - z1m - mu*(xm+xO+supp*xfs-M)
        proj_temp1m,si,U,Vt = nuc_norm_thresh(temp1m,nu,thresh_type)
        z1m = z1m + lambd*(proj_temp1m-xm)

        temp1O  = 2*xO - z1O - mu*(xm+xO+supp*xfs-M)
        proj_temp1O = utils.thresholding(temp1O,to,thresh_type)
        z1O = z1O + lambd*(proj_temp1O-xO)

        temp1fs = 2*xfs - z1fs - mu*supp*(xm+xO+supp*xfs-M)
        z1fs = z1fs + lambd*(temp1fs-xfs)
        temp2m = 2*xm - z2m - mu*(xm+xO+supp*xfs-M)

        temp2fs = 2*xfs - z2fs - mu*supp*(xm+xO+supp*xfs-M)
        proj_temp2m,proj_temp2fs = pos_proj_mat_2(temp2m,temp2fs)
        temp2O = 2*xO - z2O - mu*(xm+xO+supp*xfs-M)

        z2O = z2O + lambd*(temp2O-xO)
        xm = w1*z1m+w2*z2m
        xO = w1*z1O+w2*z2O
        xfs = w1*z1fs+w2*z2fs
        gradk = xm+xO+supp*xfs-M
        mse[i] = (gradk**2).sum()
        print mse[i]
    Md = xm
    O =xO
    faint_struct = utils.mat_to_cube(xfs,shap[0],shap[1])

    return Md,O,gradk,sigma,faint_struct,faint_struct_supp


def spca_interf(cube,k=3,nb_iter=5,nb_iter_2=100,nb_rw=2,nb_process=5):
    siz = cube.shape
    M = utils.cube_to_mat(cube)
    Md,O,gradk,sigma = positive_spca(M,shap=siz,k=k,nb_iter=nb_iter,nb_rw=nb_rw,nb_process=nb_process) #positive_spca_2(M,nb_comp_max,shap=siz,k=k,nb_iter=nb_iter,nb_iter_2=100,nb_rw=nb_rw,nb_process=nb_process)
    cube_den = utils.mat_to_cube(Md,siz[0],siz[1])
    Om = utils.mat_to_cube(O,siz[0],siz[1])
    res = utils.mat_to_cube(gradk,siz[0],siz[1])
    return cube_den,Om,res

def lanczos_gridding(samples_coord,samples_val,target_siz,lanczos_rad=4,nb_iter=50,step_size=1,im_init=None):

    mat,supp_cols,supp_lines,lip = utils.space_var_conv_mat_feat(target_siz,samples_coord,lanczos_rad=lanczos_rad)
    U = zeros(target_siz[0]*target_siz[1])
    if im_init is not None:
        U = im_init.reshape((target_siz[0]*target_siz[1],))
    V = copy(U)
    V_old = copy(V)

    t=1
    told=t
    l=0
    mse = zeros((nb_iter,))
    #print 'lip cons = ',lip**2
    U0, s, Vt = linalg.svd(transpose(mat).dot(mat),full_matrices=False)
    lip = max(s)
    print 'Conditioning number: ',abs(s).max()/abs(s).min()
    for l in range(0,nb_iter):
        res = samples_val-utils.space_var_conv_mat(mat,U,supp_lines)
        mse[l] = (res**2).sum()
        print 'iter ',l,'/',nb_iter-1,' mse = ',mse[l]
        grad = -utils.space_var_conv_transpose_mat(mat,res,supp_cols)
        V = U - (step_size/lip**2)*grad
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        U = V_old + lamb*(V-V_old)
        V_old = copy(V)
    output = U.reshape((target_siz[0],target_siz[1]))
    return output,mse,supp_cols,supp_lines,mat


def certify_adjoint(func,func_adj,siz_input,siz_output):
    rand1 = random.randn(siz_input)
    rand2 = random.randn(siz_output)
    out1 = map(funct,[rand1])
    out2 = map(funct_adj,[rand2])
    scl1 = (rand1*out2).sum()
    scl2 = (rand2*out1).sum()
    scl3 = sqrt(((rand1*out2)**2).sum())
    err = (scl1-scl2)/scl3
    print 'Adjoint accuracy: ',err


def deconvol(y,h,mu=1,nb_iter=50,tol=0.001):
    i=0
    lip = (abs(h).sum())**2
    x = copy(y)
    v = copy(y)
    mse = zeros((nb_iter,))
    h_adj = rot90(h,2)
    t=1
    told=t
    i=0
    err = 1.0
    ell = zeros((2,))
    shap = y.shape
    Wc = ones(shap)
    while i < nb_iter:# and err>tol:
        res = y-scisig.fftconvolve(x,h,mode='same')
        mse[i] = (res**2).sum()
        print mse[i]
        grad = -scisig.fftconvolve(res,h_adj,mode='same')

        vold = copy(v)
        v = pos_proj(x - (mu/lip)*grad)
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        x = vold +lamb*(v-vold)
        ell_old = copy(ell)
        ell,centroid,fwhm = utils.mk_ellipticity_2(x,Wc)
        err = sqrt(sum((ell-ell_old)**2))
        i+=1
    return x#,res,mse


def deconvol_sparse(y,h,k,mu=1,nb_iter=50,tol=0.001,thresh_type=1):
    i=0
    lip = (abs(h).sum())**2
    x = copy(y)
    v = copy(y)
    mse = zeros((nb_iter,))
    h_adj = rot90(h,2)
    t=1
    told=t
    i=0
    err = 1.0
    ell = zeros((2,))
    shap = y.shape
    Wc = ones(shap)
    suppi=0
    suppj=0
    while i < nb_iter:# and err>tol:
        res = y-scisig.fftconvolve(x,h,mode='same')
        mse[i] = (res**2).sum()
        print mse[i]
        grad = -scisig.fftconvolve(res,h_adj,mode='same')

        vold = copy(v)
        if i<nb_iter:
            v = pos_proj(utils.kthresholding_im(x - (mu/lip)*grad,sqrt(nb_iter+1),thresh_type))
            suppi,suppj = where(v>0)
        else:
            t1 = x - (mu/lip)*grad
            t2 = t1*0
            t2[suppi,suppj] = t1[suppi,suppj]
            v = pos_proj(t2)
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        x = vold +lamb*(v-vold)
        ell_old = copy(ell)
        ell,centroid,fwhm = utils.mk_ellipticity_2(x,Wc)
        err = sqrt(sum((ell-ell_old)**2))
        i+=1
    est = scisig.fftconvolve(x,h,mode='same')
    return x,est#,res,mse




def wvl_analysis_op(im,weights,mu=1,opt=None,coeff_init=None,mr_file=None,nb_iter=100):
    x = None
    if coeff_init is None:
        coeff_init,mr_file = isap.mr_trans(im,opt=opt)
        x = 0*coeff_init
    else:
        x = copy(coeff_init)
    xold = copy(x)
    z = copy(x)
    y = copy(x)
    i=0
    mse = zeros((nb_iter,))
    thresh_type=1
    t=1
    told = t
    for i in range(0,nb_iter):
        rec = isap.mr_recons_coeff(z,mr_file)
        res = im - rec
        mse[i] = (res**2).sum()

        grad,mr_file_temp = isap.mr_trans(-res,opt=opt)
        os.remove(mr_file_temp)
        y = z - mu*grad
        xold = copy(x)
        xtemp = utils.l_inf_ball_proj_3D(y,weights,thresh_type) # The coarse scale is not thresholded
        x = copy(xtemp)
        told = t
        t = (1+sqrt(4*(t**2)+1))/2
        lambd = 1 + (told-1)/t
        z = xold+lambd*(x-xold)

    coeff_init = z
    n = isap.mr_recons_coeff(z,mr_file)
    result = im - n
    return result,mr_file,n,coeff_init

def wvl_analysis_op_src(im,u,weights,a,mu=1,opt=None,coeff_init=None,mr_file=None,nb_iter=100):
    x = None
    nb_im = len(a)
    if coeff_init is None:
        temp,mr_file = isap.mr_trans(im,opt=opt)
        shap = temp.shape
        coeff_init = zeros((shap[0],shap[1],shap[2],nb_im))
    x = copy(coeff_init)
    grad = copy(coeff_init)
    xold = copy(x)
    z = copy(x)
    y = copy(x)
    i=0
    mse = zeros((nb_iter,))
    thresh_type=1
    t=1
    told = t

    w = copy(coeff_init[:,:,:,0])*0
    for k in range(0,nb_im):
        w += (a[k]**2)*weights[:,:,:,k]**2
    spec_rad = w.max()
    shap = coeff_init.shape
    ones_w = ones((shap[0],shap[1],shap[2]))
    for i in range(0,nb_iter):
        rec = im*0
        for k in range(0,nb_im):
            rec += isap.mr_recons_coeff(z[:,:,:,k]*weights[:,:,:,k]*a[k],mr_file)
        res = rec - im
        mse[i] = (res**2).sum()-(z*u).sum()
        if i == nb_iter-1:
            print "mse prox analysis: ",mse[i]
        gradtemp,mr_file_temp = isap.mr_trans(res,opt=opt)
        for k in range(0,nb_im):
            grad[:,:,:,k] = gradtemp*weights[:,:,:,k]*a[k]
        os.remove(mr_file_temp)
        grad -= u
        y = z - mu*grad/spec_rad
        xold = copy(x)
        for k in range(0,nb_im):
            x[:,:,:,k] = utils.l_inf_ball_proj_3D(y[:,:,:,k],ones_w,thresh_type) # The coarse scale is not thresholded
        told = t
        t = (1+sqrt(4*(t**2)+1))/2
        lambd = 1 + (told-1)/t
        z = xold+lambd*(x-xold)

    coeff_init = z
    rec = im*0
    for k in range(0,nb_im):
        rec += isap.mr_recons_coeff(z[:,:,:,k]*weights[:,:,:,k]*a[k],mr_file)
    result = im - rec
    return result,mr_file,rec,coeff_init


def wvl_analysis_op_ell_cons(im,cur_point,weights,mu=1,opt=None,coeff_init=None,mr_file=None,nb_iter=100,nb_comp=2000):
    x = None
    if coeff_init is None:
        coeff_init,mr_file = isap.mr_trans(im,opt=opt)
        x = 0*coeff_init
    else:
        x = copy(coeff_init)
    xold = copy(x)
    z = copy(x)
    y = copy(x)
    i=0
    mse = zeros((nb_iter,))
    thresh_type=1
    t=1
    told = t
    for i in range(0,nb_iter):
        rec = isap.mr_recons_coeff(z,mr_file)
        res = im - rec
        mse[i] = (res**2).sum()
        grad,mr_file_temp = isap.mr_trans(-res,opt=opt)
        os.remove(mr_file_temp)
        y = z - mu*grad
        xold = copy(x)
        xtemp = utils.l_inf_ball_proj_3D(y[:,:,:-1],weights,thresh_type) # The coarse scale is not thresholded
        x[:,:,:-1] = xtemp
        told = t
        t = (1+sqrt(4*(t**2)+1))/2
        lambd = 1 + (told-1)/t
        z = xold+lambd*(x-xold)

    #coeff_init = z
    n = isap.mr_recons_coeff(z,mr_file)
    result = im - n
    result,ref_ell,test_ell,fit_ell = ell_controled_fit(result,cur_point,nb_iter=100,nb_comp=nb_comp)
    coeff_init,mr_file_temp = isap.mr_trans(im-result,opt=opt)
    os.remove(mr_file_temp)

    return result,mr_file,n,coeff_init

def pos_spars_deconvol(y,h,opt,nsig=4,nb_rw=1,nb_iter=20):
    w1 = 0.8
    w2 = 0.2
    z1 = y*0
    z2 = y*0
    x = y*0
    coeff_init=None
    mr_file = None
    l = 0
    lip = (abs(h).sum())**2
    h_adj = rot90(h,2)
    lambd = 1.5
    weights = None
    for l in range(0,nb_rw+1):
        i=0
        if l>0:
            nb_iter = 10
        nb_subiter = 50
        for i in range(0,nb_iter):
            print i
            if i>0:
                nb_subiter = 5
            res = y-scisig.fftconvolve(x,h,mode='same')
            gradi = -scisig.convolve(res,h_adj,mode='same')

            # ---- Analysis constraint ---- #
            # Wavelet noise estimation
            sig_map = utils.res_sig_map(gradi/lip,opt=opt)
            temp1 = 2*x - z1 - gradi/lip
            thresh_map = nsig*sig_map
            if weights is not None:
                thresh_map = thresh_map*weights
            result,mr_file,n,coeff_init = wvl_analysis_op(temp1,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter)
            z1 = z1+lambd*(result-x)

            # ---- Positivity constraint ---- #
            z2 = z2+lambd*(pos_proj_mat(2*x - z2 - gradi/lip)-x)

            # ---- Main variable update ---- #
            x = w1*z1 + w2*z2

        # ---- Weights update ---- #
        coeffx,mr_file = isap.mr_trans(x,opt=opt)
        weights  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)

    res = y-scisig.fftconvolve(x,h,mode='same')
    os.remove(mr_file)

    return x,res


def pos_spars_deconvol_shap_cons(y,h,opt,nell1=1,nell2=1,nsig=4,nb_rw=1,nb_iter=20,r=1):
    w1 = 0.8
    w2 = 0.2
    z1 = y*0
    z2 = y*0
    x = y*0
    coeff_init=None
    mr_file = None
    l = 0
    lip = (abs(h).sum())**2
    h_adj = rot90(h,2)
    lambd = 1.5
    weights = None
    nb_subiter = 50
    for l in range(0,nb_rw+1):
        i=0
        if l>0:
            nb_iter = 10

        for i in range(0,nb_iter):
            print i
            if i>0:
                nb_subiter = 5
            res = y-scisig.fftconvolve(x,h,mode='same')
            gradi = -scisig.convolve(res,h_adj,mode='same')
            print 'res: ', (res**2).sum()
            centroid,U,ell = utils.mk_ellipticity_atoms(x)
            print 'ell: ', ell
            # ---- Analysis constraint ---- #
            # Wavelet noise estimation
            sig_map = utils.res_sig_map(gradi/lip,opt=opt)
            temp1 = 2*x - z1 - gradi/lip
            thresh_map = nsig*sig_map
            if weights is not None:
                thresh_map = thresh_map*weights
            result,mr_file,n,coeff_init = wvl_analysis_op(temp1,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter)
            z1 = z1+lambd*(result-x)

            # ---- Positivity constraint ---- #
            z2 = z2+lambd*(pos_proj_mat(2*x - z2 - gradi/lip)-x)

            # ---- Main variable update ---- #
            x = w1*z1 + w2*z2

        # ---- Weights update ---- #
        coeffx,mr_file = isap.mr_trans(x,opt=opt)
        weights  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)
    res1 = y-scisig.fftconvolve(x,h,mode='same')
    sol1 = copy(x)
    w1 = 0.0
    w2 = 0.5
    w3 = 0.5
    centroid,U,ell_ref= utils.mk_ellipticity_atoms(x)
    #proj1,proj2,proj3,ell1,ell2,ell3 = utils.nul_ell_proj(x)
    proj3,ell3=nul_ell_proj_pos(x)
    grad1,grad2 = utils.ellipticity_grad(proj3)
    shap_weights_1 = (abs(grad1)**(-1))*abs(ell_ref[0,0])/(size(x)*nell1)
    shap_weights_2 = (abs(grad2)**(-1))*abs(ell_ref[0,1])/(size(x)*nell2)
    shap_weights = utils.min_pt(shap_weights_1,shap_weights_2)
    z3 = copy(x)
    print 'ell_ref: ',ell_ref
    nb_subiter = 5
    nb_iter = 100
    rcons = r*sqrt((proj3**2).sum())
    for i in range(0,nb_iter):
        res = y-scisig.fftconvolve(x,h,mode='same')
        gradi = -scisig.convolve(res,h_adj,mode='same')
        print 'res: ', (res**2).sum()
        centroid,U,ell = utils.mk_ellipticity_atoms(x)
        print 'ell: ',ell
        # ---- Analysis constraint ---- #
        # Wavelet noise estimation
        sig_map = utils.res_sig_map(gradi/lip,opt=opt)
        temp1 = 2*x - z1 - gradi/lip
        thresh_map = nsig*sig_map
        #if weights is not None:
        #    thresh_map = thresh_map*weights
        #result,mr_file,n,coeff_init = wvl_analysis_op(temp1,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter)
        #z1 = z1+lambd*(result-x)

        # ---- Positivity constraint ---- #
        #z2 = z2+lambd*(pos_proj_mat(2*x - z2 - gradi/lip)-x)


        # ---- Shape constraint ---- #
        #z2 = z2+lambd*(pos_proj_mat(proj_ellip_cons(2*x - z3 - gradi/lip,proj3,abs(ell_ref[0,0])/(size(x)*nell1),abs(ell_ref[0,1])/(size(x)*nell2)))-x)

        # ---- Main variable update ---- #
        #x = w1*z1 + w2*z2 + w3*z3
        #x = pos_proj_mat(proj_ellip_cons(x - gradi/(lip),proj3,abs(ell_ref[0,0])/(size(x)*nell1),abs(ell_ref[0,1])/(size(x)*nell2)))
        #x = utils.l_inf_ball_proj(x - gradi/lip,shap_weights,1,cent=proj3)
        #x = proj_ellip_cons_2(x - gradi/(lip),proj3,abs(ell_ref[0,0]),abs(ell_ref[0,1]),grad1,grad2)

        x = dyks_alg_shap_cons(x - gradi/(lip),proj3,abs(ell_ref[0,0]),abs(ell_ref[0,1]),grad1,grad2,rcons,nb_iter=50)
        print 'proj ell1 ',abs((grad1*(x-proj3)).sum())
        print 'proj ell2 ',abs((grad2*(x-proj3)).sum())

    res2 = y-scisig.fftconvolve(x,h,mode='same')
    os.remove(mr_file)

    return sol1,x,res1,res2,proj3,grad1,grad2

def pos_spars_deconvol_shap_cons_2(y,h,opt,nell1=1,nell2=1,nsig=4,nb_rw=1,nb_iter=20,r=1):
    w1 = 0.8
    w2 = 0.2
    z1 = y*0
    z2 = y*0
    x = y*0
    coeff_init=None
    mr_file = None
    l = 0
    lip = (abs(h).sum())**2
    h_adj = rot90(h,2)
    lambd = 1.5
    weights = None
    nb_subiter = 50
    for l in range(0,nb_rw+1):
        i=0
        if l>0:
            nb_iter = 10

        for i in range(0,nb_iter):
            print i
            if i>0:
                nb_subiter = 5
            res = y-scisig.fftconvolve(x,h,mode='same')
            gradi = -scisig.convolve(res,h_adj,mode='same')
            print 'res: ', (res**2).sum()
            centroid,U,ell = utils.mk_ellipticity_atoms(x)
            print 'ell: ', ell
            # ---- Analysis constraint ---- #
            # Wavelet noise estimation
            sig_map = utils.res_sig_map(gradi/lip,opt=opt)
            temp1 = 2*x - z1 - gradi/lip
            thresh_map = nsig*sig_map
            if weights is not None:
                thresh_map = thresh_map*weights
            result,mr_file,n,coeff_init = wvl_analysis_op(temp1,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter)
            z1 = z1+lambd*(result-x)

            # ---- Positivity constraint ---- #
            z2 = z2+lambd*(pos_proj_mat(2*x - z2 - gradi/lip)-x)

            # ---- Main variable update ---- #
            x = w1*z1 + w2*z2

            # ---- Weights update ---- #
            coeffx,mr_file = isap.mr_trans(x,opt=opt)
            weights  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)
    res1 = y-scisig.fftconvolve(x,h,mode='same')
    sol1 = copy(x)
    centroid,U,ell_ref= utils.mk_ellipticity_atoms(x)
    #proj1,proj2,proj3,ell1,ell2,ell3 = utils.nul_ell_proj(x)
    proj3,ell3=nul_ell_proj_pos(x)
    grad1,grad2 = utils.ellipticity_grad(proj3)
    shap_weights_1 = (abs(grad1)**(-1))*abs(ell_ref[0,0])/(size(x)*nell1)
    shap_weights_2 = (abs(grad2)**(-1))*abs(ell_ref[0,1])/(size(x)*nell2)
    shap_weights = utils.min_pt(shap_weights_1,shap_weights_2)
    z3 = copy(x)
    print 'ell_ref: ',ell_ref
    nb_subiter = 5
    nb_iter = 20
    rcons = r*sqrt((proj3**2).sum())
    t=1
    told=t
    for i in range(0,nb_iter):
        res = y-scisig.fftconvolve(x,h,mode='same')
        gradi = -scisig.convolve(res,h_adj,mode='same')
        print 'res: ', (res**2).sum()
        centroid,U,ell = utils.mk_ellipticity_atoms(x)
        print 'ell: ',ell
        # ---- Analysis constraint ---- #
        # Wavelet noise estimation
        sig_map = utils.res_sig_map(gradi/lip,opt=opt)
        thresh_map = nsig*sig_map
        x,mr_file,n,coeff_init = wvl_analysis_op_ell_cons(x-gradi/lip,x,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter)

    res2 = y-scisig.fftconvolve(x,h,mode='same')
    os.remove(mr_file)

    return sol1,x,res1,res2,proj3,grad1,grad2


def proj_ellip_cons(im,ref,mu1,mu2):
    grad1,grad2 = utils.ellipticity_grad(ref)
    #weights = utils.min_pt(mu1*(abs(grad1)**(-1)),mu2*(abs(grad2)**(-1)))
    weights = mu1*(abs(grad1)**(-1))
    thresh_type = 1
    im_proj = utils.l_inf_ball_proj(im,weights,thresh_type,cent=ref)

    return im_proj

"""def proj_ellip_param(ref,tol1,tol2,ell_ref):
    eps = 1e-13
    siz = ref.shape
    cent = zeros((siz[0],siz[1]))
    weights = zeros((siz[0],siz[1]))
    i,j = 0,0
    grad1,grad2 = utils.ellipticity_grad(ref)
    ind1,ind2 = where(grad1==0)
    grad1[ind1,ind2]=eps
    ind1,ind2 = where(grad2==0)
    grad2[ind1,ind2]=eps
    for i in range(0,siz[0]):
        for j in range(0,siz[1]):
            r1 = abs(ell_ref[0,0])*tol1/(abs(grad1[i,j])*siz[0]*siz[1])
            r2 = abs(ell_ref[0,1])*tol2/(abs(grad2[i,j])*siz[0]*siz[1])
            cent1 = ref[i,j]/abs(grad1[i,j])
            cent2 ="""

#def wvl_analysis_shap_cons(im,weights,mu=1,opt=None,coeff_init=None,mr_file=None,nb_iter=10):



def sparse_deconvol_err(y,h,opt,xref,nell1=1,nell2=1,nsig=4,nb_rw=1,nb_iter=20):
    sol1,sol2,res1,res2,proj3 = pos_spars_deconvol_shap_cons(y,h,opt,nell1=1,nell2=1,nsig=4,nb_rw=1,nb_iter=20)
    centroid,U,ell_ref = utils.mk_ellipticity_atoms(xref)
    centroid,U,ell_conv = utils.mk_ellipticity_atoms(y)
    centroid,U,ell_deconv1 = utils.mk_ellipticity_atoms(sol1)
    centroid,U,ell_deconv2 = utils.mk_ellipticity_atoms(sol2)

    rell_err_obs = abs((ell_ref-ell_conv)/ell_conv)*100
    rell_err_bias1 = abs((ell_ref-ell_deconv1)/ell_conv)*100
    rell_err_bias2 = abs((ell_ref-ell_deconv2)/ell_conv)*100
    mse1 = ((sol1-xref)**2).sum()
    mse2 = ((sol2-xref)**2).sum()
    return rell_err_obs,rell_err_bias1,rell_err_bias2,mse1,mse2,sol1,sol2,res1,res2

def sparse_convol_err_arr(y,h,opt,xref,nell1=1,nell2=1,nsig=4,nb_rw=1,nb_iter=20):
    shap = y.shape
    nb_im = shap[2]




def nul_ell_proj_pos(im,mu=0.8,nb_iter=10):
    u = im*0
    k = 0
    mat_proj,normv = utils.proj_mat(im.shape,0)
    shap = im.shape

    """mat_proj1 = zeros((shap[0],shap[1],2))
    mat_proj1[:,:,0] = mat_proj[:,:,4]
    mat_proj1[:,:,1] = mat_proj[:,:,0]-mat_proj[:,:,1]
    mat_proj2 = zeros((shap[0],shap[1],3))
    mat_proj2[:,:,0] = mat_proj[:,:,5]
    mat_proj2[:,:,1] = mat_proj[:,:,0]
    mat_proj2[:,:,2] = mat_proj[:,:,1]"""

    mat_proj3 = zeros((shap[0],shap[1],4))
    mat_proj3[:,:,0] = mat_proj[:,:,5]
    mat_proj3[:,:,1] = mat_proj[:,:,0]
    mat_proj3[:,:,2] = mat_proj[:,:,1]
    mat_proj3[:,:,3] = mat_proj[:,:,4]
    ell=None
    proj = None
    for k in range(0,nb_iter):
        res = im-u
        grad = -res+utils.proj_cube(res,mat_proj3,ortho=1)
        u = -pos_proj_mat(-u+mu*grad)

    proj1 = im-u-utils.proj_cube(im-u,mat_proj3,ortho=1)
#proj1 = pos_proj_mat(proj1)
    centroid,v,ell1 = utils.mk_ellipticity_atoms(proj1)

    return proj1,ell1


def abs_val_proj(x,u,b,eps): # calculate the projection onto |<u,x>-b|<eps
    proj = copy(x)
    if abs((x*u).sum()-b)>eps:
        z1 = utils.proj_hyp(u,eps+b,x)
        z2 = utils.proj_hyp(-u,eps-b,x)
        if (((x-z1)**2).sum()<((x-z2)**2).sum()):
            proj = z1
        else:
            proj = z2
    return proj

def abs_val_proj_2(x,u1,u2,c1,c2,eps1,eps2): # calculate the projection onto |<u1,x>-c1|<eps1 and |<u2,x>-c2|<eps2
    proj = copy(x)
    if (abs((x*u1).sum()-c1)>eps1 or abs((x*u2).sum()-c2)>eps2):
        shap = x.shape
        proj_cand = zeros((shap[0],shap[1],4))
        proj_cand[:,:,0] = utils.proj_2_hyp(u1,eps1+c1,u2,eps2+c2,x)
        proj_cand[:,:,1] = utils.proj_2_hyp(u1,eps1+c1,-u2,eps2-c2,x)
        proj_cand[:,:,2] = utils.proj_2_hyp(-u1,eps1-c1,u2,eps2+c2,x)
        proj_cand[:,:,3] = utils.proj_2_hyp(-u1,eps1-c1,-u2,eps2-c2,x)

        i=0
        d = 1e16

        for i in range(0,3):
            di = ((x-proj_cand[:,:,i])**2).sum()
            if di < d :
                proj = proj_cand[:,:,i]
                d =di
    return proj

def proj_ellip_cons_2(im,ref,mu1,mu2,grad1,grad2):
    c1 = (grad1*ref).sum()
    c2 = (grad2*ref).sum()
    proj = abs_val_proj_2(im,grad1,grad2,c1,c2,mu1,mu2)
    return proj

def proj_dir(V,u): # Vectors are stored in columns, u is assumed to be a column vector
    u1 = copy(u)
    u1 = u1/utils.norm2(u1)
    proju_mat = u1.dot(transpose(u1))
    projV = V - proju_mat.dot(V)
    return V


def proj_sphere(x,o,r):
    proj = copy(x)
    l = sqrt(((x-o)**2).sum())
    if l>r:
        proj = r*(x-o)/l + o
    return proj

def proj_sphere_mat(x,o,r):
    shap = x.shape
    i=0
    proj = copy(x)
    for i in range(0,shap[0]):
        proj[i,:] = proj_sphere(x[i,:],o[i,:],r[i])
    return proj

def proj_sphere_cube(x,o,r):
    shap = x.shape
    i=0
    proj = copy(x)
    for i in range(0,shap[2]):
        proj[:,:,i] = proj_sphere(x[:,:,i],o[:,:,i],r[i])
    return proj

def proj_sphere_decim(z,y,r,D): # z is the HR image to be projected, y is the center of the LR projection l2 sphere of radius r, D is the downsampling factor
    x = copy(z)
    x_lr_proj = proj_sphere(utils.decim(z,D,av_en=0,fft=0),y,r)
    shap = x_lr_proj.shape
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            x[i*D,j*D] = x_lr_proj[i,j]

    return x

def proj_sphere_decim_conv(z,y,r,D,ker,ker_adj):
    x = proj_sphere_decim(z,y,r,D)
    out = conv_pseudo_inv(x,ker,ker_adj)
    res = y-utils.decim(scisig.fftconvolve(out,ker,mode='same'),D,av_en=0)
    print "res after inversion : ",(res**2).sum()," reference: ",r**2
    return out

def proj_sphere_decim_conv_pd(cur_point,o,r,D,ker,ker_adj,xinit=None,nb_iter=100,tol=0.1):
    t=1
    x = None
    if xinit is None:
        x = copy(cur_point)*0
    else:
        x = copy(xinit)
    z = copy(x)
    lip = (abs(ker).sum())**2
    i=0
    cost = 1
    cost_old = 0
    out = copy(z)
    while(i<nb_iter and abs(cost-cost_old)*100/abs(cost)>tol):
        i+=1
        out = cur_point - scisig.fftconvolve(z,ker_adj,mode='same')
        grad = - scisig.fftconvolve(out,ker,mode='same')
        y = z - grad/lip
        xold = copy(x)
        x = y - proj_sphere_decim(y*lip,o,r,D)/lip # Moreau
        told = t
        t = (1+sqrt(1+4*t**2))/2
        lamb = 1+(told-1)/t
        z = xold +lamb*(x-xold)
        cost_old = cost
        cost = ((cur_point-out)**2).sum()
        print "proj sphere cost: ",(out**2).sum()
    out = cur_point - scisig.fftconvolve(z,ker_adj,mode='same')
    print "res after inversion : ",cost," reference: ",r**2
    return out

def proj_sphere_decim_conv_stack(z,y,r,D,ker,ker_adj):
    zout = copy(z)
    shap = z.shape
    for i in range(0,shap[2]):
        zout[:,:,i] = proj_sphere_decim_conv_pd(z[:,:,i],y[:,:,i],r[i],D,ker[:,:,i],ker_adj[:,:,i])
    return zout

def dyks_alg_shap_cons(im,ref,mu1,mu2,grad1,grad2,r,nb_iter=20):
    x = copy(im)
    p = x*0
    q = x*0
    i=0
    for i in range(0,nb_iter):
        y = proj_sphere(x+p,ref,r)
        p = x+p-y
        x = proj_ellip_cons_2(y+q,ref,mu1,mu2,grad1,grad2)
        q = y+q-x

    rest = sqrt(((x-ref)**2).sum())
    print 'target radius: ',r,' radius achieved: ',rest
    return x

def poc_shap_cons(im,ref,mu1,mu2,grad1,grad2,r,nb_iter=20):
    proj = copy(im)
    i=0
    for i in range(0,nb_iter):
        proj = proj_sphere(proj_ellip_cons_2(proj,ref,mu1,mu2,grad1,grad2),ref,r)
    rest = sqrt(((proj-ref)**2).sum())
    print 'target radius: ',r,' radius achieved: ',rest

    return proj


def least_square_desc_steep(y,x,u,t):
    topt = ((y-x)*u).sum()/((u**2).sum())
    steepness = t*((u**2).sum())-((y-x)*u).sum()
    return topt,steepness


def lsq_ell_desc_check(dir_stack,ref_ell,cur_point,y):
    siz = dir_stack.shape
    opt_step_lsq=0
    opt_step=0
    lsq_steepness = zeros((siz[2],))
    dir_cand,ell_steepness = utils.dir_check(dir_stack,ref_ell,cur_point)
    i = 0
    rmax = 0
    lsq_max_steep=0
    dir_id = -1
    pinf=False
    for i in range(0,siz[2]):
        topti,steepnessi = least_square_desc_steep(y,cur_point,dir_stack[:,:,i],0)
        if (steepnessi<0) and (dir_cand[i]==1):
            if (ell_steepness[0,0]==0) or (ell_steepness[0,1]==0):
                pinf=True
                if (-steepnessi>lsq_max_steep):
                    dir_id = i
                    lsq_max_steep = -steepnessi
                    opt_step_lsq = topti
            else:
                if pinf==False:
                    ri = -steepnessi*(abs(ell_steepness[i,0])**(-1)+abs(ell_steepness[i,1])**(-1))
                    if ri>rmax:
                        dir_id = i
                        rmax = ri
                        opt_step_lsq = topti

    if opt_step_lsq>0:
        root_ell1,root_ell2,root_ell3,root_ell4 = utils.ellt_zeros(cur_point,dir_stack[:,:,dir_id])
        roots = zeros((8,))
        roots[0] = root_ell1[0]
        roots[1] = root_ell1[1]
        roots[2] = root_ell2[0]
        roots[3] = root_ell2[1]
        roots[4] = root_ell3[0]
        roots[5] = root_ell3[1]
        roots[6] = root_ell4[0]
        roots[7] = root_ell4[1]
        ind = where((imag(roots)==0))
        if size(ind)==0:
            opt_step = opt_step_lsq
        else:
            roots_real = roots[ind]
            ind2 = where(roots_real>0)
            if size(ind2) ==0:
                opt_step = opt_step_lsq
            else:
                roots_real_pos = roots_real[ind2]
                opt_step = min(opt_step_lsq,roots_real_pos.min())

    else:
        print "Failed to find a descend direction"
        ell_steepness_l1 = abs(ell_steepness[:,0])+abs(ell_steepness[:,1])
        dir_id_temp = where(ell_steepness_l1==ell_steepness_l1.min())
        if sum(shape(dir_id_temp))>1:
            dir_id = dir_id_temp[0][0]
        else:
            dir_id = dir_id_temp
        toptid,steepnessid = least_square_desc_steep(y,cur_point,dir_stack[:,:,dir_id],0)
        root_ell1,root_ell2,root_ell3,root_ell4 = utils.ellt_zeros(cur_point,dir_stack[:,:,dir_id])
        roots = zeros((8,))
        roots[0] = root_ell1[0]
        roots[1] = root_ell1[1]
        roots[2] = root_ell2[0]
        roots[3] = root_ell2[1]
        roots[4] = root_ell3[0]
        roots[5] = root_ell3[1]
        roots[6] = root_ell4[0]
        roots[7] = root_ell4[1]
        ind = where((imag(roots)==0))
        if size(ind)==0:
            opt_step = toptid
        else:
            if toptid>0:
                roots_real = roots[ind]
                ind2 = where(roots_real>0)
                if size(ind2) ==0:
                    opt_step = toptid
                else:
                    roots_real_pos = roots_real[ind2]
                    opt_step = min(toptid,roots_real_pos.min())
            else:
                roots_real = roots[ind]
                ind2 = where(roots_real<=0)
                if size(ind2) ==0:
                    opt_step = toptid
                else:
                    roots_real_neg = roots_real[ind2]
                    opt_step = max(toptid,roots_real_neg.max())

    return opt_step,dir_id

def ell_controled_fit(y,init_pt,nb_iter=100,nb_comp=2000):
    output = copy(init_pt)
    centroid,U,ref_ell=utils.mk_ellipticity_atoms(output)
    centroid,U,fit_ell=utils.mk_ellipticity_atoms(y)
    i=0
    test_ell=None

    print 'Ref Ell: ',ref_ell
    for i in range(0,nb_iter):
        print 'mse: ',((y-output)**2).sum()
        grad_lsq = -y+output
        grad1_ell,grad2_ell = utils.ellipticity_grad(output)

        com_stack,comb_coeff = utils.rand_lin_comb(grad_lsq,grad1_ell,grad2_ell,nb_comp)
        centroid,U,ref_ell=utils.mk_ellipticity_atoms(output)
        opt_step,dir_id = lsq_ell_desc_check(com_stack,ref_ell,output,y)

        output = output+opt_step*com_stack[:,:,dir_id]

        centroid,U,test_ell=utils.mk_ellipticity_atoms(output)

        print 'Test Ell: ',test_ell

        print 'Dir: ',comb_coeff[dir_id,:],'; step: ',opt_step

    return output,ref_ell,test_ell,fit_ell




def prox_ellipsoid(alpha,y,M,mu,r,nb_iter=100,u_init=None): # projection of alpha on {x/||y-Mx||^2<r^2}


    shap = M.shape
    i = 0
    if u_init is None:
        u_init = zeros((shap[0],1))
    o = zeros((shap[0],1))
    u = copy(u_init)
    p = alpha - transpose(M).dot(u_init)
    for i in range(0,nb_iter):
        z = u/mu + M.dot(p)-y
        u = mu*(z-proj_sphere(z,o,r))
        p = alpha - transpose(M).dot(u)
    print 'radius :',r,' Current dist: ',sqrt(((y-M.dot(p))**2).sum())
    u_init = u
    return p,u_init

def shap_cons_denoising(obs,opt,nsig,nb_iter=50,nb_rw=1):
    siz = obs.shape
    obs_vect = obs.reshape((obs.size,1))
    Mshap = utils.proj_mat_2(siz)
    inv_cor = LA.inv(Mshap.dot(transpose(Mshap)))
    U, s, Vt = linalg.svd(inv_cor,full_matrices=False)
    S = sqrt(diag(s))
    inv_cor_sqrt = U.dot(S.dot(Vt))
    Mshap_white = inv_cor_sqrt.dot(Mshap)
    U, s0, Vt = linalg.svd(Mshap_white,full_matrices=False)
    obs_proj = Mshap_white.dot(obs_vect)

    i=0
    z1 = copy(obs)*0
    z2 = copy(obs)*0
    x = copy(obs)*0
    #z3 = copy(obs)*0
    w1 = 0.5
    w2 = 0.5
    mu = 1
    lambd = 1.5

    coeff_init=None
    mr_file = None
    weights = None
    u_init = None
    nb_subiter1 = 50
    nb_subiter2 = 50
    l = 0
    for l in range(0,nb_rw+1):
        for i in range(0,nb_iter):
            if i >0:
                nb_subiter1 = 5
                nb_subiter2 = 50
            grad = x-obs
            # ---- Analysis constraint ---- #
            # Wavelet noise estimation
            sig_map = utils.res_sig_map(grad*mu,opt=opt)
            thresh_map = nsig*sig_map
            if weights is not None:
                thresh_map = thresh_map*weights
            # Prox step
            temp1,mr_file,n,coeff_init = wvl_analysis_op(2*x-z1-mu*grad,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter1)
            z1 = z1+lambd*(temp1-x)

            # ---- Shape constraint ---- #
            # Image noise est
            sig = utils.im_gauss_nois_est(grad)
            print 'sig :',sig
            temp2 = 2*x-z2-mu*grad
            temp2_vect = temp2.reshape((temp2.size,1))
            mat_in = Mshap_white
            temp3,u_init = prox_ellipsoid(temp2_vect,obs_proj,mat_in,0.5/(abs(s0).max()),sqrt(6)*sig*0.5,nb_iter=nb_subiter2)
            temp3 = temp3.reshape(siz)
            z2 = z2+lambd*(temp3-x)
            x = w1*z1+w2*z2
            centroid,U,test_ell=utils.mk_ellipticity_atoms(x)
            print 'Current ellipticity: ',test_ell

        # ---- Weights update ---- #
        coeffx,mr_file = isap.mr_trans(x,opt=opt)
        weights  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)

    return x

def analysis_rw_denoising(obs,opt,nsig,nb_iter=2,nb_rw=1):
    siz = obs.shape

    i=0
    z1 = copy(obs)*0
    z2 = copy(obs)*0
    x = copy(obs)*0
    mu = 1

    coeff_init=None
    mr_file = None
    weights = None
    nb_subiter = 50
    l = 0
    for l in range(0,nb_rw+1):
        if l>0:
            nb_iter=1
        for i in range(0,nb_iter):
            if i >0:
                nb_subiter = 5

            grad = x-obs
            # ---- Analysis constraint ---- #
            # Wavelet noise estimation
            sig_map = utils.res_sig_map(grad*mu,opt=opt)
            thresh_map = nsig*sig_map
            if weights is not None:
                thresh_map = thresh_map*weights
            # Prox step
            x,mr_file,n,coeff_init = wvl_analysis_op(x-mu*grad,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter)
            centroid,U,test_ell=utils.mk_ellipticity_atoms(x)
            print 'Current ellipticity: ',test_ell

        # ------- Weights update ------- #
        coeffx,mr_file = isap.mr_trans(x,opt=opt)
        weights  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)

    return x


def lin_prog_transp(distrib1,distrib2,dist_map):

    siz = dist_map.shape
    dist_vect = dist_map.reshape((siz[0]*siz[1],))
    eq_vect = zeros((2*siz[0],))
    eq_vect[0:siz[0]] = distrib1
    eq_vect[siz[0]:] = distrib2
    mat_eq = zeros((2*siz[0],siz[0]*siz[1]))
    for i in range(0,siz[0]):
        mat_eq[i,i*siz[0]:(i+1)*siz[0]] = ones((siz[0],))
        mat_eq[siz[0]:,i*siz[0]:(i+1)*siz[0]] = eye(siz[0])
    mat_uq = -eye(siz[0]*siz[1])
    uq_vect = zeros((siz[0]*siz[1],))
    optim_res = scipy.optimize.linprog(dist_vect,A_ub=mat_uq,b_ub=uq_vect,A_eq=mat_eq,b_eq = eq_vect)
    x = optim_res.x
    mapping = x.reshape((siz[0],siz[1]))


    return mapping,optim_res


"""def lin_prog_transp_bar(distrib_mat,dist_map,weights): # The distributions are in distrib_mat columns
    siz = distrib_mat.shape
    dist_vect = dist_map.reshape((siz[0]**2,))
    dist_vect_n = zeros((siz[1]*siz[0]**2,))
    for i in range(0,siz[1]):
        dist_vect_n[i*siz[0]**2:(i+1)*siz[0]**2] = weights[i]*dist_vect
    distrib_vect = reshape(distrib_mat,(siz[1]*siz[0],),'F') # reshaping in columns order
    mat_eq = zeros(((2*siz[1]-1)*siz[0],siz[1]*siz[0]**2))
    for i in range(0,siz[0]):
        for j in range(0,siz[1]):
            mat_eq[j*siz[0]+i,j*siz[0]**2+i*siz[0]:j*siz[0]**2+(i+1)*siz[0]] = ones((siz[0],))
            if j<siz[1]-1:
                mat_eq[siz[0]*siz[1]+:,i*siz[0]:(i+1)*siz[0]]"""

def sinkh_transp(M,r,c,beta=200,nb_iter=1000):
    lambd = beta/M.max()
    K = exp(-lambd*M)
    x = ones((size(r),1))/size(r)
    i = 0
    for i in range(0,nb_iter):
        temp = K.dot(c*(transpose(K).dot(x**(-1)))**(-1))

        x=diag((r.reshape(size(r),))**(-1)).dot(temp)
        u = x**(-1)
        v = c*((transpose(K).dot(u))**(-1))
        a = diag(u.reshape((size(u),)))
        b = diag(v.reshape((size(v),)))
        P = a.dot(K.dot(b))
        ri = sum(P,axis=0)
        ri = ri.reshape((size(ri),1))
        ci = sum(P,axis=1)
        ci = ci.reshape((size(ci),1))
        print 'Bias :',((c-ri)**2).sum()/(c**2).sum(),' ',((r-ci)**2).sum()/(r**2).sum()
        print 'Check sum :',ri.sum(),' ',ci.sum()
    dist=sum(u*((K*M).dot(v)))

    return P,dist


def kl_marg_proj(P,u): # P is a matrix and u u is a column; this function perfoms the projection of P on the set of matrices of columns marginal equal to u, with respect to the KL divergence

    v = u.reshape((size(u),))
    marg = sum(P,axis=1)
    id = where(marg>0)
    temp = v*0
    temp[id] = double(v[id])/marg[id]
    a = diag(temp)

    """if (sum(P,axis=1)).min()==0:
        print "warning zero div in kl_marg_proj"""
    out = a.dot(P)

    return out



def kl_clust_proj(P,cart_prods,w):
    nb_clust = len(cart_prods)
    P_out = copy(P)
    for i in range(0,nb_clust):
        P_out[cart_prods[i][:,0],cart_prods[i][:,1]] *= w[i]/sum(P_out[cart_prods[i][:,0],cart_prods[i][:,1]])

    return P_out


def beg_proj_transp(M,r,c,supp=None,supp_comp_en=False,beta=200,nb_iter=1000,step=50):
    lambd = beta/M.max()
    K = None
    nb_min = 1

    i = where(r>0)
    j = where(c>0)
    supp_size_min = size(i)+size(j)+1
    #supp_size_min = max(size(i),size(j))
    print "Support minimal size: ",supp_size_min
    supp_size_init = None

    for l in range(0,nb_min):
        K = exp(-lambd*M)

        if supp is not None:
            K *= supp
        P = copy(K)
        #suppi = P*0
        for i in range(0,nb_iter):
            P = kl_marg_proj(P,c)
            P = transpose(kl_marg_proj(transpose(P),r))
            if i%step==0:
                ci = sum(P,axis=1)
                ci = ci.reshape((size(ci),1))
                ri = sum(P,axis=0)
                ri = ri.reshape((size(ri),1))
                print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
                print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
                print 'Check sum :',ri.sum(),' ',ci.sum()

                #supp_old = copy(suppi)
                #suppi = P*0
                """if l==0:
                    if supp_size_init is None:
                        n,m = where(P>0)
                        supp_size_init = size(n)
                        print "initial support size: ",supp_size_init
                    next_size = int(-(supp_size_init-supp_size_min)*(double(i)/(nb_iter-1-(nb_iter-1)%step))**0.5 +supp_size_init)
                    print "Next size: ",next_size
                    supp = utils.khighest(P,next_size)
                    #supp = support_checking(P,supp,r,c)
                    print "Actual size: ",supp.sum()
                    K = exp(-lambd*M)
                    if supp is not None:
                        K *= supp
                    P = copy(K)"""
                #suppi[n,m] = 1
                #print "Support error: ",sum(abs(suppi-supp_old))," Support size: ",suppi.sum()
        ci = sum(P,axis=1)
        ci = ci.reshape((size(ci),1))
        ri = sum(P,axis=0)
        ri = ri.reshape((size(ri),1))
        print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
        print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
        print 'Check sum :',ri.sum(),' ',r.sum()

    P = transpose(map_sparse(transpose(P),c,r))
    dist=sum(M*P)
    return P,dist

def sliced_transport_time_assessing(f,g,nb_real,nb_iter=1000,tol=0.001,alph = 0.0,basis=None,rand=False,disc_err=0.1,indf=None,indg=None,smart_init_en = True,monte_carlo=100):
    from numpy import zeros
    import time
    time_val = zeros((nb_real.shape[0],))
    nb_disc = zeros((nb_real.shape[0],))
    for i in range(0,nb_real.shape[0]):
        print i+1,"th real/",nb_real.shape[0]
        for k in range(0,monte_carlo):
            t = time.time()
            projf,dist,sig,disc_pts = sliced_transport(f,g,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,nb_real=nb_real[i],rand=rand,disc_err=disc_err,indf=indf,indg=indg,smart_init_en = smart_init_en)
            time_val[i] += time.time()-t
            nb_disc[i] += float(disc_pts.shape[1])

    return time_val/monte_carlo,nb_disc/monte_carlo

def sliced_transport_time_assessing_2(f,g,nb_iter=1000,tol=0.001,alph = 0.0,basis=None,nb_real=30,rand=False,disc_err=0.1,indf=None,indg=None,smart_init_en = True,monte_carlo=100,gap=None,shap=None,rad=15):
    from numpy import zeros
    import time
    time_val = zeros((monte_carlo,))
    nb_disc = zeros((monte_carlo,))


    for k in range(0,monte_carlo):
        print "Realization ",k+1,"/",monte_carlo
        t = time.time()
        projf,dist,sig,disc_pts = sliced_transport(f,g,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,nb_real=nb_real,rand=rand,disc_err=disc_err,smart_init_en = smart_init_en,gap=gap,shap=shap,rad=rad)
        time_val[k] = time.time()-t
        nb_disc[k] = float(disc_pts.shape[1])

    return time_val,nb_disc


def sliced_transport_time_assessing_m(f,g,nb_real,nb_iter=1000,tol=0.001,alph = 0.0,basis=None,rand=False,disc_err=0.1,indf=None,indg=None,smart_init_en = True):
    from numpy import zeros
    time_val = zeros((f.shape[2],nb_real.shape[0]))
    nb_disc = zeros((f.shape[2],nb_real.shape[0]))
    indfin = None
    indgin = None

    for i in range(0,f.shape[2]):
        print i+1,"th test/",f.shape[2]
        if indf is not None and indg is not None:
            indfin = indf[i,:]
            indgin = indg[i,:]
        time_val[i,:],nb_disc[i,:] = sliced_transport_time_assessing(f[:,:,i],g[:,:,i],nb_real,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand,disc_err=disc_err,indf=indfin,indg=indgin,smart_init_en = smart_init_en)

    return time_val,nb_disc



def sliced_transport(f,g,nb_iter=1000,tol=0.001,alph = 0.0,basis=None,nb_real=200,rand=True,disc_err=0.5,indf=None,indg=None,smart_init_en = True,output_control=True,iter_control=False,rad=15,shap=None,gap=None):
    import time
    from numpy.linalg import norm
    from numpy import copy,zeros,ones,argsort
    from scipy.optimize import linear_sum_assignment
    from utils import window_ind
    iter = None
    t = time.time()
    if iter_control:
        iter = zeros((f.shape[0],f.shape[1],nb_iter+2))
        iter[:,:,0] = copy(g)

    pf = copy(f)
    if smart_init_en:


        if indg is None or indf is None:

            i = where(f[2,:]==f[2,:].max())
            print i,gap,f.shape
            cent1 = f[0:2,i[0][0]]/gap
            j = where(g[2,:]==g[2,:].max())
            cent2 = g[0:2,j[0][0]]/gap
            indf = window_ind(cent1.astype(int),shap,rad)
            indg = window_ind(cent2.astype(int),shap,rad)


        print "On going smart initialization..."
        #indg = arange(0,f.shape[1])
        #indf = indg
        dist_init,pf[:,indf],sig_init = opt_assignment(f[:,indf],g[:,indg])
        print "Done."

    if iter_control:
        iter[:,:,1] = copy(pf)

    i = 0
    err = 100
    dim  = f.shape[0]

    sig_global = zeros((dim,f.shape[1],nb_real))

    bias = None
    max_disc_cv = None
    if basis is None and rand == False:
        rand = True
    while i<nb_iter and err>tol:
        i+=1
        pf_old = copy(pf)
        step = 1.0/i**alph
        tempf = pf_old*0
        if rand:
            #tempf = None
            bias = 0
            max_disc_cv = 0
            discr_max = f.shape[1]
            for k in range(0,nb_real):
                basis = utils.random_orth_basis(dim)
                tempfi,err_proj,sig_global[:,:,k] = coord_wise_proj(pf,g,basis)
                #max_disc,disc_id = discrepancies_analysis(sig_global[:,:,,k])
                #bias += max_disc-disc_id
                #max_disc_cv += max_disc
                tempf+=tempfi
                """if max_disc<discr_max:
                    discr_max = max_disc
                    tempf=copy(tempfi)"""
                #print "Inter-discrepancy: ",max_disc," Id discrepancy: ",disc_id

            tempf/=nb_real
        else:
            tempf,err_proj,sig = coord_wise_proj(pf,g,basis)
        # sprint "Projection error: ",err_proj
        pf = (1-step)*pf_old+step*tempf
        if iter_control:
            iter[:,:,i] = copy(pf)
        err = 100*sum((pf-pf_old)**2)/sum(pf**2)
        print "Relative change of the iterate: ",err,"%"

    sig_global = sig_global.astype(int)

    print "Elapsed time: ", time.time()-t



    # Assignements diagnostic
    """for i in range(0,shap[0]):
        print "Departure: ",size(where(abs(sig[i_opt,:]-sig[i,:])>0))"""

    bias = 0
    max_disc_cv = 0
    discr_max = f.shape[1]
    for k in range(0,nb_real):
        max_disc,disc_id = discrepancies_analysis(sig_global[:,:,k])
        bias += max_disc-disc_id
        max_disc_cv += max_disc

    print "Mean departure: ",float(max_disc_cv)/nb_real

    disc_supp = None
    cost = norm(f)+norm(g)
    iopt = 0
    for k in range(0,nb_real):
        disc_supp_k,agreem_supp_k = discrepancies_global_analysis(sig_global[:,:,k])
        cost_k = norm(f[:,agreem_supp_k]-g[:,sig_global[0,agreem_supp_k,k]])
        if cost_k<cost:
            cost = cost_k
            iopt = k
            disc_supp = disc_supp_k

    sig_approx = copy(sig_global[0,:,iopt])

    print "Global disagrement: ",len(disc_supp)

    if output_control==False:
        dist = norm(f-pf)
        if iter_control:
            return pf,dist,sig_approx,f[0:2,disc_supp],iter[:,:,0:i+1]
        else:
            return pf,dist,sig_approx,f[0:2,disc_supp]
    else:
        if len(disc_supp)<disc_err*f.shape[1]:

            if len(disc_supp)==0:

                dist = norm(f-pf)
                print "Transport correction: ",dist/norm(f-g)
                print "Success"
                if iter_control:
                    return g[:,sig_approx],dist,sig_approx,f[0:2,disc_supp],iter
                else:
                    return g[:,sig_approx],dist,sig_approx,f[0:2,disc_supp]
            else:
                dist_disc = zeros((len(disc_supp),len(disc_supp)))
                for i in range(0,len(disc_supp)):
                    for j in range(0,len(disc_supp)):
                        dist_disc[i,j] = norm(f[:,disc_supp[i]]-g[:,sig_global[0,disc_supp[j],iopt]])**2
                # Linear programming
                print "Linear programming solving..."
                row_ind,col_ind = linear_sum_assignment(dist_disc)

                print "Done."
                sig_approx[disc_supp] = sig_approx[disc_supp[col_ind[argsort(row_ind)]]]
                dist = norm(f-g[:,sig_approx])
                rho = dist/norm(f-g)
                print "Transport corrections: ",dist/norm(f-g), norm(f[:,disc_supp]-g[:,sig_approx[disc_supp]])/norm(f[:,disc_supp]-g[:,sig_global[0,disc_supp,iopt]])
                if rho <1:
                    print "Success"
                    if iter_control:
                        return g[:,sig_approx],dist,sig_approx,f[0:2,disc_supp],iter_control
                    else:
                        return g[:,sig_approx],dist,sig_approx,f[0:2,disc_supp]
                else:
                    print "Fail"
                    if iter_control:
                        return g,norm(f-g),sig_approx,f[0:2,disc_supp],iter_control
                    else:
                        return g,norm(f-g),sig_approx,f[0:2,disc_supp]
        else:
            print "Fail"
            if iter_control:
                return g,norm(f-g),sig_approx,f[0:2,disc_supp],iter
            else:
                return g,norm(f-g),sig_approx,f[0:2,disc_supp]

def sliced_transport_proj(ref,pt_clouds,gap,nb_iter=1000,tol=0.001,alph = 0.0,basis=None,nb_real=200,rand=True,disc_err=0.5,indf=None,indg=None,smart_init_en = True,output_control=True,iter_control=False,rad=15,shap=None):
    from numpy import copy
    proj_clouds = copy(pt_clouds)

    for i in range(0,pt_clouds.shape[2]):
        pt_i,dist_i,sig_i,ref_disc = sliced_transport(ref,pt_clouds[:,:,i],nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,nb_real=nb_real,rand=rand,disc_err=disc_err,indf=indf,indg=indg,smart_init_en = smart_init_en,output_control=output_control,iter_control=iter_control,rad=rad,shap=shap,gap=gap)
        proj_clouds[:,:,i] = pt_i-ref

    return proj_clouds

def cloud_size_constraint_simp(clouds_in,cent,siz):
    from numpy import ones,sqrt,sign
    from numpy.linalg import norm
    shap = clouds_in.shape
    norm_vect = ones((2,1)).dot(clouds_in[2,:].reshape((1,shap[1])))/sum(clouds_in[2,:])
    cent_mat = cent.reshape((2,1)).dot(ones((1,shap[1])))
    cent_dist = (cent_mat-clouds_in[0:2,:])**2
    ajust_pos = sqrt(pos_proj_mat(cent_dist-(sum(norm_vect*cent_dist)-siz)*norm_vect/norm(norm_vect)**2))*sign(clouds_in[0:2,:]-cent_mat)+cent_mat
    return ajust_pos

def cloud_size_constraint(clouds_in,cent,siz,inertia,nb_iter=100,tol=1e-5,acc=True):
    from numpy import ones,zeros,transpose,sqrt
    from numpy.linalg import norm,inv

    cloud_out = copy(clouds_in)
    ind = where(cloud_out[2,:]<0)
    if size(ind)>0:
        cloud_out[2,ind]=0
    nb_points = cloud_out.shape[1]
    cent_mat = cent.reshape((2,1)).dot(ones((1,nb_points)))
    """# Elliptical projection
    nb_points = clouds.shape[1]

    weights = sqrt(ones((2,1)).dot(clouds[2,:].reshape((1,nb_points)))/siz)
    #weights = ones((2,nb_points))

    cloud_out[0:2,:] = proj_ellip(cent_mat,weights,cloud_out[0:2,:],nb_iter=nb_iter,tol=tol)
    print norm(cloud_out[0:2,:]-clouds[0:2,:])
    # Intensities adjustement
    cloud_out[0:2,:] = inertia*(cloud_out[0:2,:]-cent_mat)/norm(cloud_out[0:2,:]-cent_mat)+cent_mat"""
    norms = norm(cloud_out[0:2,:]-cent_mat,axis=0)**2
    print "Max siz: ",siz,"; effective size: ",sum(norms*cloud_out[2,:])
    const_mat = ones((2,nb_points))
    const_mat[0,:] = norms
    const_vect = ones((2,1))
    const_vect[0,0] = siz
    if acc:
        cloud_out[2,:] = cloud_size_constraint_acc(cloud_out[2,:],const_mat,const_vect,nb_iter=500,tol=1.e-12)
    else:
        temp_m = transpose(const_mat).dot(inv(const_mat.dot(transpose(const_mat)))).dot(const_mat.dot(cloud_out[2,:].reshape((nb_points,1)))-const_vect)
        cloud_out[2,:]  = cloud_out[2,:] - temp_m.reshape((nb_points,))
    print "Size after correction: ",sum(norms*cloud_out[2,:])
    pos_ind = where(cloud_out[2,:]<0)
    cloud_out[2,pos_ind[0]] = 0

    return cloud_out

def cloud_size_constraint_acc(gray_lev,const_mat,const_vect,nb_iter=2000,tol=1.e-15):
    from numpy import copy,zeros
    from numpy.linalg import norm,inv

    nb_points = size(gray_lev)

    w1 = 0.5
    w2 = 1-w1
    z1 = copy(gray_lev)
    z2 = copy(gray_lev)
    x = copy(gray_lev)
    gamma = 1.0
    lambd = 1

    i=0
    var = 100

    while i<nb_iter and var > tol:

        # Moment constraint
        temp1 = 2*x-z1-gamma*(x-gray_lev)
        temp_m = transpose(const_mat).dot(inv(const_mat.dot(transpose(const_mat)))).dot(const_mat.dot(temp1.reshape((nb_points,1)))-const_vect)
        prox_step = temp1 - temp_m.reshape((nb_points,))
        z1 += lambd*(prox_step-x)

        # Positivity constraint
        temp1 = 2*x-z2-gamma*(x-gray_lev)
        z2 += lambd*(pos_proj(temp1)-x)

        # Average
        xold = copy(x)
        x = w1*z1+w2*z2
        var = 100*norm(x-xold)/norm(x)
        i+=1

    return x

def cloud_segmentation(arr,frac_size,rand_en=False): # The number of points is supposed to be a multiple of frac_size; the sample are in the rows.
    from pyflann import FLANN
    from numpy import arange,zeros
    from numpy.random import permutation

    knn = FLANN()
    params = knn.build_index(array(arr, dtype=float64))
    nb_samp  = arr.shape[0]
    result, dists = knn.nn_index(arr,nb_samp)

    ind = arange(nb_samp).astype(int)
    if rand_en:
        ind = permutation(ind)

    nb_parts = nb_samp/frac_size
    parts = zeros((nb_parts,frac_size)).astype(int)
    ref = zeros((arr.shape[1],nb_parts))
    diff_vect = zeros((arr.shape[1],frac_size,nb_parts))

    check = zeros((nb_samp,)).astype(bool)
    i=0
    uncheck = nb_samp

    while uncheck>0:
        if not check[ind[i]]:
            ref[:,i] = arr[ind[i],:]
            n_check = where(check[result[ind[i],:]] ==False)
            check[n_check[0][0:frac_size]] = True
            for j in range(0,frac_size):
                parts[ind[i],j] = result[ind[i],n_check[0][j]]
                diff_vect[:,count,i] = -arr[result[ind[i],n_check[0][j]],:]+arr[result[ind[i],n_check[0][(j+1)%frac_size]],:]
        i+=1
        uncheck-=frac_size

    return parts,diff_vect,ref

def cloud_reconstruction(parts,diff_vect,ref,eps=1.e-15): # The output samples are in the rows
    from numpy import zeros,size
    from numpy.linalg import norm
    nb_parts = ref.shape[1]
    nb_samp = size(parts)
    frac_size = parts.shape[1]
    samp_dim = ref.shape[0]
    arr = zeros((nb_samp,samp_dim))
    disc_perc = zeros((nb_parts,))

    for i in range(0,nb_parts):
        arr[parts[i,0],:] = ref[:,i]
        for j in range(0,parts.shape[1]-1):
            arr[parts[i,j+1],:]=arr[parts[i,j],:]+diff_vect[:,j,i]
        disc_perc[i] = 100*norm(ref[:,i]-arr[parts[i,-1],:]-diff_vect[:,-1,i])/(norm(ref[:,i])+eps)

    return arr,disc_perc


"""def cloud_segments_constraint(diff_vect,rad,nb_iter=10):
    from numpy import copy
    from numpy.linalg import norm

    diff_vect_out = copy(diff_vect)
    norms = norm(diff_vect_out,axis=0)
    i=0"""


def lasso_cumul(arr_in,lines=True):
    from numpy import zeros,cumsum,transpose
    shap = arr_in.shape
    arr = copy(arr_in)
    if not lines:
        arr = transpose(arr)
    out = copy(arr)
    total = 0
    for i in range(0,shap[0]):

        l = copy(arr[i,:])
        if i%2==0:
            l[0]+=total
            out[i,:] = cumsum(l)
            total = out[i,-1]
        else:
            l[-1]+=total
            out[i,:] = cumsum(l[::-1])[::-1]
            total = out[i,0]

    if not lines:
        out = transpose(out)
    return out

def lasso_cumul_transp(arr,lines=True):
    from numpy import flipud,rot90

    if arr.shape[0]%2==0:
        return flipud(lasso_cumul(flipud(arr),lines=lines))
    else:
        return rot90(lasso_cumul(rot90(arr,2),lines=lines),2)

def angle_proj(a,b,angle_min,nb_samp=10):
    from numpy.linalg import norm
    from numpy import arccos,pi
    a_out = copy(a)
    b_out = copy(b)
    na = norm(a)
    nb = norm(b)
    theta = angle_min
    if na >0 and nb >0:
        cos_theta = sum(a*b)/(na*nb)
        theta = arccos(cos_theta)

    if theta>=angle_min or theta==0.0:
        return a_out,b_out
    else:
        v1 = copy(a)
        v2 = copy(b)
        det = a[0]*b[1]+a[1]*b[0]
        if det<0:
            v1 = copy(b)
            v2 = copy(a)
        v1,v2 = direction_optim_brute_force(v1,v2,angle_min-theta,nb_samp=nb_samp)
        if det>0:
            return v1,v2
        else:
            return v2,v1

def angle_proj_array(vect_x_l,vect_y_l,vect_x_c,vect_y_c,angle_min,nb_samp=10):
    from numpy.linalg import norm

    for i in range(0,vect_x_l.shape[0]):
        for j in range(0,vect_x_l.shape[1]):
            a = array([vect_x_l[i,j],vect_y_l[i,j]])
            b = array([vect_x_c[i,j],vect_y_c[i,j]])
            a,b = angle_proj(a,b,angle_min,nb_samp=nb_samp)
            vect_x_l[i,j],vect_y_l[i,j] = a
            vect_x_c[i,j],vect_y_c[i,j] = b

    return vect_x_l,vect_y_l,vect_x_c,vect_y_c

def rot_2D(u,theta):
    return array([cos(theta)*u[0]-sin(theta)*u[1],sin(theta)*u[0]+cos(theta)*u[1]])


def direction_optim_brute_force(v1,v2,theta_max,nb_samp=10):
    from numpy.linalg import norm
    from numpy import zeros,copy

    cos_fact_max = 0
    v1_out = copy(v1)
    v2_out = copy(v2)
    nv1 = norm(v1)
    nv2 = norm(v2)
    count = 0.0

    while(count<nb_samp+1):
        cos_fact = nv1*cos(count*theta_max/nb_samp)+nv2*cos((nb_samp-count)*theta_max/nb_samp)
        if cos_fact>cos_fact_max:
            v1_out = rot_2D(v1,-count*theta_max/nb_samp)
            v2_out = rot_2D(v2,(nb_samp-count)*theta_max/nb_samp)
        count+=1.0

    return v1_out,v2_out

def diff_coord_mat(arr_in,lines=True):
    from numpy import diff,transpose

    arr = copy(arr_in)
    if not lines:
        arr = transpose(arr)
    out = copy(arr)
    shap = arr.shape
    for i in range(0,shap[0]):
        if i%2==0:
            out[i,1:] = diff(arr[i,:])
            if i>0:
                out[i,0] = arr[i,0]-arr[i-1,0]
        else:
            out[i,-2::-1] = diff(arr[i,:][::-1])
            out[i,-1] = arr[i,-1]-arr[i-1,-1]
    if not lines:
        out = transpose(out)

    return out

def lasso_constraint(pos_mat,shap,r_min,angle_min,nb_iter=50,mu=1,tol=1,var_tol=1.e-25):
    from numpy import sqrt,copy
    from numpy.linalg import norm

    xcoord = pos_mat[0,:].reshape(shap)
    ycoord = pos_mat[1,:].reshape(shap)

    vect_x_l = diff_coord_mat(xcoord,lines=True)
    vect_x_c = diff_coord_mat(xcoord,lines=False)

    vect_y_l = diff_coord_mat(ycoord,lines=True)
    vect_y_c = diff_coord_mat(ycoord,lines=False)


    vect_x_l,vect_x_c,vect_y_l,vect_y_c = retraction_lasso(vect_x_l,vect_x_c,vect_y_l,vect_y_c,r_min,angle_min)

    i = 0
    r_data = 0
    max_disc = 100
    rel_var = 100

    while i<nb_iter and max_disc>tol and rel_var>var_tol:

        # Riemanian gradient
        tg_grad_x_l,tg_grad_x_c,tg_grad_y_l,tg_grad_y_c = tg_gradient(vect_x_l,vect_x_c,vect_y_l,vect_y_c,r_min)

        # Update using Armijo rule
        vect_x_l_old = copy(vect_x_l)
        vect_y_l_old = copy(vect_y_l)
        vect_x_c_old = copy(vect_x_c)
        vect_y_c_old = copy(vect_y_c)
        vect_x_l,vect_x_c,vect_y_l,vect_y_c = armijo_lasso(vect_x_l,vect_x_c,vect_y_l,vect_y_c,tg_grad_x_l,tg_grad_x_c,tg_grad_y_l,tg_grad_y_c,r_min,angle_min,alpha=0.5,beta=0.5,sigma=1.e-5)
        i+=1


        # Lassos discrepancies check
        disc1 = 100*norm(lasso_cumul(vect_x_l,lines=True)-lasso_cumul(vect_x_c,lines=False))/norm(lasso_cumul(vect_x_l,lines=True))
        disc2 = 100*norm(lasso_cumul(vect_y_l,lines=True)-lasso_cumul(vect_y_c,lines=False))/norm(lasso_cumul(vect_y_l,lines=True))
        max_disc = max(disc1,disc2)/xcoord.shape[1]
        print "Position average discrepancy: ",max_disc,"%"," Tolerance ",tol," relative iterates variation: ",rel_var,"%"," Variation tolerance: ",var_tol," Nb iter max: ",nb_iter

        # Relative variation
        rel_var = 100*(norm(vect_x_l_old-vect_x_l)/norm(vect_x_l_old) + norm(vect_y_l_old-vect_y_l)/norm(vect_y_l_old) + norm(vect_x_c_old-vect_x_c)/norm(vect_x_c_old) + norm(vect_y_c_old-vect_y_c)/norm(vect_y_c_old))/4

    print "Position average discrepancy: ",max_disc,"%"," Tolerance ",tol," relative iterates variation: ",rel_var,"%"," Variation tolerance: ",var_tol," Nb iter max: ",nb_iter

    vect_x_l[0,0] = xcoord[0,0]
    vect_y_l[0,0] = ycoord[0,0]

    pos_out = copy(pos_mat)
    pos_out[0,:] = lasso_cumul(vect_x_l,lines=True).reshape((shap[0]*shap[1],))
    pos_out[1,:] = lasso_cumul(vect_y_l,lines=True).reshape((shap[0]*shap[1],))

    d = dist_map_2(pos_out)
    i,j = where(d>0)
    print "Mean dist after correction: ",d[i,j].mean()


    return pos_out

def tg_gradient(vect_x_l,vect_x_c,vect_y_l,vect_y_c,r_min):

    from numpy import sqrt,copy,array
    from numpy.linalg import norm

    tg_grad_x_l = lasso_cumul_transp(lasso_cumul(vect_x_l,lines=True)-lasso_cumul(vect_x_c,lines=False),lines=True)
    tg_grad_x_c = lasso_cumul_transp(-lasso_cumul(vect_x_l,lines=True)+lasso_cumul(vect_x_c,lines=False),lines=False)

    tg_grad_y_l = lasso_cumul_transp(lasso_cumul(vect_y_l,lines=True)-lasso_cumul(vect_y_c,lines=False),lines=True)
    tg_grad_y_c = lasso_cumul_transp(-lasso_cumul(vect_y_l,lines=True)+lasso_cumul(vect_y_c,lines=False),lines=False)

    tg_grad_x_l[0,0] = 0
    tg_grad_x_c[0,0] = 0
    tg_grad_y_l[0,0] = 0
    tg_grad_y_c[0,0] = 0

    d_l = sqrt(vect_x_l**2+vect_y_l**2)
    indx1,indy1 = where(d_l>0)
    id_bound = where(d_l[indx1,indy1]==r_min)
    id_bound = id_bound[0]
    for i in range(0,size(id_bound)):
        # Projection of the euclidian gradients at points at the boundary on the corresponding tangent planes
        tg_grad_x_l[indx1[id_bound[i]],indy1[id_bound[i]]],tg_grad_y_l[indx1[id_bound[i]],indy1[id_bound[i]]] = array([tg_grad_x_l[indx1[id_bound[i]],indy1[id_bound[i]]],tg_grad_y_l[indx1[id_bound[i]],indy1[id_bound[i]]]]) - (tg_grad_x_l[indx1[id_bound[i]],indy1[id_bound[i]]]*vect_x_l[indx1[id_bound[i]],indy1[id_bound[i]]] + tg_grad_y_l[indx1[id_bound[i]],indy1[id_bound[i]]]*vect_y_l[indx1[id_bound[i]],indy1[id_bound[i]]])*array([vect_x_l[indx1[id_bound[i]],indy1[id_bound[i]]],vect_y_l[indx1[id_bound[i]],indy1[id_bound[i]]]])/d_l[indx1[id_bound[i]],indy1[id_bound[i]]]**2

    d_c = sqrt(vect_x_c**2+vect_y_c**2)
    indx1,indy1 = where(d_c>0)
    id_bound = where(d_c[indx1,indy1]==r_min)
    id_bound = id_bound[0]
    for i in range(0,size(id_bound)):
        # Projection of the euclidian gradients at points at the boundary on the corresponding tangent planes
        tg_grad_x_c[indx1[id_bound[i]],indy1[id_bound[i]]],tg_grad_y_c[indx1[id_bound[i]],indy1[id_bound[i]]] = array([tg_grad_x_c[indx1[id_bound[i]],indy1[id_bound[i]]],tg_grad_y_c[indx1[id_bound[i]],indy1[id_bound[i]]]]) - (tg_grad_x_c[indx1[id_bound[i]],indy1[id_bound[i]]]*vect_x_c[indx1[id_bound[i]],indy1[id_bound[i]]] + tg_grad_y_c[indx1[id_bound[i]],indy1[id_bound[i]]]*vect_y_c[indx1[id_bound[i]],indy1[id_bound[i]]])*array([vect_x_c[indx1[id_bound[i]],indy1[id_bound[i]]],vect_y_c[indx1[id_bound[i]],indy1[id_bound[i]]]])/d_c[indx1[id_bound[i]],indy1[id_bound[i]]]**2

    return tg_grad_x_l,tg_grad_x_c,tg_grad_y_l,tg_grad_y_c


def cost_funct_lasso(vect_x_l,vect_x_c,vect_y_l,vect_y_c):
    from numpy.linalg import norm
    return norm(lasso_cumul(vect_x_l,lines=True)-lasso_cumul(vect_x_c,lines=False))+norm(lasso_cumul(vect_y_l,lines=True)-lasso_cumul(vect_y_c,lines=False))

def retraction_lasso(vect_x_l,vect_x_c,vect_y_l,vect_y_c,r_min,angle_min,nb_samp=10,angle_cons=False):
    from numpy import sqrt
    from numpy.linalg import norm
    # Fixing the origin
    """vect_x_l[0,0] = 0
    vect_x_c[0,0] = 0
    vect_y_l[0,0] = 0
    vect_y_c[0,0] = 0"""

    d_l = sqrt(vect_x_l**2+vect_y_l**2)
    indx1,indy1 = where(d_l>0)
    ind2 = where(d_l[indx1,indy1]<r_min)
    vect_x_l[indx1[ind2[0]],indy1[ind2[0]]] *= r_min/(d_l[indx1[ind2[0]],indy1[ind2[0]]])
    vect_y_l[indx1[ind2[0]],indy1[ind2[0]]] *= r_min/(d_l[indx1[ind2[0]],indy1[ind2[0]]])

    d_c = sqrt(vect_x_c**2+vect_y_c**2)
    indx1,indy1 = where(d_c>0)
    ind2 = where(d_c[indx1,indy1]<r_min)
    vect_x_c[indx1[ind2[0]],indy1[ind2[0]]] *= r_min/(d_c[indx1[ind2[0]],indy1[ind2[0]]])
    vect_y_c[indx1[ind2[0]],indy1[ind2[0]]] *= r_min/(d_c[indx1[ind2[0]],indy1[ind2[0]]])

    if angle_cons:
        vect_x_l,vect_y_l,vect_x_c,vect_y_c = angle_proj_array(vect_x_l,vect_y_l,vect_x_c,vect_y_c,angle_min,nb_samp=nb_samp)

    return vect_x_l,vect_x_c,vect_y_l,vect_y_c

def armijo_lasso(vect_x_l,vect_x_c,vect_y_l,vect_y_c,tg_grad_x_l,tg_grad_x_c,tg_grad_y_l,tg_grad_y_c,r_min,angle_min,alpha=0.5,beta=0.5,sigma=1.e-10,nb_iter=1000,eps = 1.e-13):
    from numpy import sqrt
    from numpy.linalg import norm
    m=0
    cur_cost = cost_funct_lasso(vect_x_l,vect_x_c,vect_y_l,vect_y_c)
    grad_norm = norm(tg_grad_x_l)**2+norm(tg_grad_y_l)**2+norm(tg_grad_x_c)**2+norm(tg_grad_y_c)**2
    descent = 0

    mu = alpha*beta**m
    vect_x_l_0,vect_x_c_0,vect_y_l_0,vect_y_c_0 = retraction_lasso(vect_x_l-mu*tg_grad_x_l,vect_x_c-mu*tg_grad_x_c,vect_y_l-mu*tg_grad_y_l,vect_y_c-mu*tg_grad_y_c,r_min,angle_min)
    descent = cur_cost - cost_funct_lasso(vect_x_l_0,vect_x_c_0,vect_y_l_0,vect_y_c_0)
    count = 0
    while descent<=grad_norm*sigma*mu and grad_norm>eps and count<nb_iter:
        m+=1
        mu = alpha*beta**m
        vect_x_l_0,vect_x_c_0,vect_y_l_0,vect_y_c_0 = retraction_lasso(vect_x_l-mu*tg_grad_x_l,vect_x_c-mu*tg_grad_x_c,vect_y_l-mu*tg_grad_y_l,vect_y_c-mu*tg_grad_y_c,r_min,angle_min)
        descent = cur_cost - cost_funct_lasso(vect_x_l_0,vect_x_c_0,vect_y_l_0,vect_y_c_0)
        count+=1
    print "Armijo descent: ",descent," Step: ",mu," upper bound: ",grad_norm*sigma*mu
    return vect_x_l_0,vect_x_c_0,vect_y_l_0,vect_y_c_0



def proj_ellip(cent,weights,y,nb_iter=20,tol=0.01): # Calculates the nearest point on the ellipsoid defined by \|weights*(X-cent)\|_2^2 = 1 to the point y; this routine assumes that the entriers are matrices
    from numpy.linalg import norm

    i,j = where(weights>0)
    z = cent*0
    x = cent*0
    t=1
    mu = weights[i,j].min()**2

    var = 1
    i=0
    while i<nb_iter and var>tol:
        v = z[i,j] - mu*((weights[i,j]**(-2))*z[i,j] + (weights[i,j]**(-1))*(cent[i,j]-y[i,j]))
        x_old = copy(x[i,j])
        x[i,j] = v
        if norm(x[i,j])>1:
            x[i,j] /= norm(x[i,j])
        told = t
        t = (1+sqrt(4*t**2+1))/2
        lambd = 1+(told-1)/t
        z[i,j] = x_old + lambd*(x[i,j]-x_old)
        var = 100*norm(x[i,j]-x_old)/norm(x_old)
    print "Relative change of iterates: ",var

    proj = copy(y)
    proj[i,j] = cent[i,j]+z[i,j]/weights[i,j]

    return proj



def opt_assignment(f,g):
    from numpy import zeros
    from numpy.linalg import norm
    from scipy.optimize import linear_sum_assignment

    shap = f.shape

    dist = zeros((shap[1],shap[1]))
    sig = zeros((shap[1],))
    for i in range(0,shap[1]):
        for j in range(0,shap[1]):
            dist[i,j] = norm(f[:,i]-g[:,j])

    row_ind,col_ind = linear_sum_assignment(dist)
    for i in range(0,shap[1]):
        sig[row_ind[i]] = col_ind[i]

    dist = norm(f-g[:,sig.astype(int)])


    return dist,g[:,sig.astype(int)],sig.astype(int)


def discrepancies_analysis(sig):
    max_disc = 0
    nb_perm = sig.shape[0]
    for i in range(0,nb_perm-1):
        for j in range(i+1,nb_perm):
            supp = where(abs(sig[i,:]-sig[j,:])>0)
            max_discij = size(supp[0])
            if max_discij>max_disc:
                max_disc = max_discij
    disc_id = 0
    id = arange(0,sig.shape[1])
    for i in range(0,nb_perm):
        supp = where(abs(sig[i,:]-id)>0)
        disc_idij = size(supp[0])
        if disc_idij > disc_id:
            disc_id = disc_idij

    return max_disc,disc_id

def discrepancies_global_analysis(sig):
    from numpy import ones,array
    from numpy.linalg import norm
    shap = sig.shape
    disc_support = list()
    agreem_support = list()
    ones_mat = ones((shap[0],))

    for i in range(0,shap[1]):
        if norm(sig[:,i]-ones_mat*sig[0,i])>0:
            disc_support.append(i)
        else:
            agreem_support.append(i)

    return array(disc_support).astype(int),array(agreem_support).astype(int)

def bar_discrepancies_analysis(sig):
    from numpy import zeros
    shap = sig.shape
    disc = zeros((shap[2],))
    for i in range(0,shap[2]):
        maxi = 0
        for j in range(1,shap[0]):
            a = abs(sig[0,:,i]-sig[j,:,i])
            ind = where(a>0)
            if maxi < ind[0].shape[0]:
                maxi = ind[0].shape[0]
        disc[i] = maxi

    return disc
def sliced_transport_ptbar(F,weights,nb_iter=1000,tol=0.000000000000000001,alph=0.01,bar_init=None,dim=1,samp=3,rand_en=True,basis=None,adapted_basis=True,pos_en=False,min_basis=False,support=False,assign_out=False,single_dir=False,nb_real=20,gap=None,shap=None,rad=20,smart_init_en=True):

    from numpy import argsort,copy
    indw = argsort(weights)

    bar = copy(F[:,:,indw[-1]])
    w = weights[indw[-1]]
    for i in range(2,size(weights)+1):
        proj,dist,sig,i_opt = sliced_transport(F[:,:,indw[-i]],bar,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand_en,nb_real=nb_real,gap=gap,shap=shap,rad=rad,iter_control=False,smart_init_en=smart_init_en)
        t = w/(w+weights[indw[-i]])
        print "current weight:", t
        bar = t*proj+(1-t)*F[:,:,indw[-i]]
        w+= weights[indw[-i]]

    return bar

def sym_sliced_transport(f,g,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False,approx_en=True):
    from numpy import argsort,copy
    fin = copy(f)
    gin = copy(g)
    if approx_en:
        indf = where(f[2,:]>0)
        indg = where(g[2,:]>0)

        if len(indf[0])< len(indg[0]):
            fin = f[:,indf[0]]
            ind = argsort(g[2,indg[0]])
            gin = g[:,indg[0][ind[0:len(indf[0])]]]
        else:
            gin = g[:,indg[0]]
            ind = argsort(f[2,indf[0]])
            fin = f[:,indf[0][ind[0:len(indg[0])]]]


    pf,dist1,sig1,i_opt = sliced_transport(fin,gin,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand)
    pg,dist2,sig2,i_opt = sliced_transport(gin,fin,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand)

    return (dist1+dist2)/2,sig1,sig2



def sym_sliced_transport_supp(f,g,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False): # Compute the approximated sliced distance between the non-overlapping parts (with respect to the 2 first dimensions) of the input 3D points clouds

    from numpy import copy

    # computing the intersection of the supports
    supp1 = where(abs(f[2,:])>0)
    supp12 = where(abs(g[2,supp1[0]])>0)

    fc = copy(f)
    gc = copy(g)

    fc[2,supp1[0][supp12[0]]] = 0
    gc[2,supp1[0][supp12[0]]] = 0

    if sum(fc[2,:]**2)==0 or sum(gc[2,:]**2)==0:
        return 0
    else:
        return sym_sliced_transport(fc,gc,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand,approx_en=True)


def stack_im_coherence(cube):
    from numpy import zeros
    from numpy.linalg import norm

    shap = cube.shape
    corr_coeff = zeros((shap[2],shap[2]))

    for i in range(0,shap[2]-1):
        for j in range(i+1,shap[2]):
            corr_coeff[i,j] = sum(abs(cube[:,:,i]*cube[:,:,j]))/(norm(cube[:,:,i])*norm(cube[:,:,j]))
            corr_coeff[j,i] = corr_coeff[i,j]

    return corr_coeff


def local_sliced_dist_mat(stack,pos_field,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False,nb_neigh=30,max_val =2**63):
    from numpy import zeros,transpose,ones
    from pyflann import FLANN
    knn = FLANN()
    params = knn.build_index(array(pos_field, dtype=float64))


    nb_points = stack.shape[2]
    dist_mat = zeros((nb_points,nb_points))

    for i in range(0,nb_points):
        result, dists = knn.nn_index(pos_field, nb_neigh+1)
        print i+1,"th PSF"
        for j in range(1,nb_neigh+1):
            if dist_mat[i,result[0,j]]==0:
                dist_mat[i,result[0,j]],sig_1,sig_2 = sym_sliced_transport(stack[:,:,i],stack[:,:,result[0,j]],nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand)
                dist_mat[result[0,j],i] = dist_mat[i,result[0,j]]


    i,j = where(dist_mat==0)
    dist_mat[i,j] = max_val

    i = range(0,nb_points)
    dist_mat[i,i] = 0

    i,j = where(dist_mat<max_val)

    if dist_mat[i,j].max()>0:
        dist_mat[i,j]/=dist_mat[i,j].max()

    return dist_mat

def local_sliced_dist_mat_target(stack,pos_field,target_pos,nb_neigh,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False,max_val =2**63):

    from numpy import zeros,transpose,ones
    from pyflann import FLANN
    knn = FLANN()
    params = knn.build_index(array(pos_field, dtype=float64))
    result, dists = knn.nn_index(target_pos, nb_neigh)

    dist_mat = zeros((nb_neigh,nb_neigh))
    dict_assignments = {}

    for i in range(0,nb_neigh-1):
        print i+1,"th PSF/",nb_neigh
        for j in range(i+1,nb_neigh):
            if dist_mat[i,j]==0:
                dist_mat[i,j],sig_1,sig_2 = sym_sliced_transport(stack[:,:,i],stack[:,:,result[0,j]],nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand)
                dict_assignments[j+i*nb_neigh] = [sig_1,sig_2]
                dist_mat[j,i] = dist_mat[i,j]

    return dist_mat,result,dict_assignments







def local_sliced_dist_mat_supp(stack,pos_field,knn=None,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False,nb_neigh=30,max_val =2**63):
    from numpy import zeros,transpose,ones
    from pyflann import FLANN
    if knn is None:
        knn = FLANN()
        params = knn.build_index(array(pos_field, dtype=float64))




    nb_points = stack.shape[2]
    dist_mat = ones((nb_points,nb_points))*max_val
    i = range(0,nb_points)
    dist_mat[i,i] = 0
    result, dists = knn.nn_index(pos_field, nb_neigh+1)

    for i in range(0,nb_points):

        print i+1,"th PSF/",nb_points
        for j in range(1,nb_neigh+1):
            if dist_mat[i,result[i,j]]== max_val:
                dist_mat[i,result[i,j]] = sym_sliced_transport_supp(stack[:,:,i],stack[:,:,result[i,j]],nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand)
                dist_mat[result[i,j],i] = dist_mat[i,result[i,j]]

    return dist_mat



def local_euc_dist_mat(stack,pos_field,tol=0.001,nb_neigh=30,max_val =2**63):
    from numpy import zeros,transpose,ones,sqrt
    from pyflann import FLANN
    knn = FLANN()
    params = knn.build_index(array(pos_field, dtype=float64))


    nb_points = stack.shape[2]
    dist_mat = zeros((nb_points,nb_points))

    for i in range(0,nb_points):
        result, dists = knn.nn_index(pos_field, nb_neigh+1)
        print i+1,"th PSF"
        for j in range(i+1,nb_neigh+1):
            if dist_mat[i,result[0,j]]==0:
                dist_mat[i,result[0,j]] = sqrt(sum((stack[:,:,i]-stack[:,:,result[0,j]])**2))
                dist_mat[result[0,j],i] = dist_mat[i,result[0,j]]


    """i,j = where(dist_mat==0)
    dist_mat[i,j] = max_val

    i = range(0,nb_points)
    dist_mat[i,i] = 0"""

    i,j = where(dist_mat<max_val)
    dist_mat/=dist_mat.max()

    return dist_mat

def dist_map(stack):
    from numpy import zeros
    from numpy.linalg import norm
    shap = stack.shape
    dist_mat = zeros((shap[2],shap[2]))

    for i in range(0,shap[2]-1):
        for j in range(i+1,shap[2]):
            dist_mat[i,j] = norm(stack[:,:,i]-stack[:,:,j])
            dist_mat[j,i] = dist_mat[i,j]

    return dist_mat

def dist_map_2(arr): # The samples are in the columns
    """Pairwise distances...?"""
    from numpy.linalg import norm
    from numpy import ones,transpose,fill_diagonal

    nb_samp = arr.shape[1]
    norm_mat = ((norm(arr,axis=0).reshape((nb_samp,1)))**2).dot(ones((1,nb_samp)))
    scalar_prod = transpose(arr).dot(arr)
    res = norm_mat+transpose(norm_mat)-2*scalar_prod
    i,j = where(res<0)
    res[i,j] = 0
    return res


def dist_map2(mat,rel=True): # The data are stored in the rows
    from numpy import ones,transpose
    from numpy.linalg import norm
    ones_vect = ones((1,mat.shape[1]))

    norm_vect = (norm(mat,axis=0).reshape((mat.shape[1],1)))**2

    co_dist_mat = norm_vect.dot(ones_vect)

    if rel:
        return co_dist_mat+transpose(co_dist_mat)-2*transpose(mat).dot(mat)/(0.01*(co_dist_mat+transpose(co_dist_mat)))
    else:
        return co_dist_mat+transpose(co_dist_mat)-2*transpose(mat).dot(mat)

def max_gap(stack,pos_field,tol=0.001,nb_neigh=30,max_val =2**63,dist_mat=None,pol_en=False,cent=None):
    from utils import compute_centroid
    from numpy import arange,ones,median,pi
    if dist_mat is None:
        dist_mat = dist_map(stack)

    i,j = where(dist_mat>0)
    k = where(dist_mat[i,j]==dist_mat[i,j].min())

    inear = i[k[0][0]]
    jnear = j[k[0][0]]

    gap = abs(stack[:,:,inear]-stack[:,:,jnear]).max()
    """if pol_en:
        if cent is None:
            im_mean = stack.mean(axis=2)
            cent,w = compute_centroid(im_mean,sigw=100000000)
            cent = squeeze(cent)
        shap = stack.shape

        point_cloud = zeros((2,shap[0]*shap[1]))
        point_cloud[0,:] = (arange(0,shap[0]).reshape((shap[0],1)).dot(ones((1,shap[1])))).reshape((shap[0]*shap[1],))
        point_cloud[1,:] = ones((shap[0],1)).dot(arange(0,shap[1]).reshape((1,shap[1]))).reshape((shap[0]*shap[1],))
        point_cloud = setting_polar_2(point_cloud,1,cent)
        space_dist = dist_map_2(point_cloud)
        i,j = where(space_dist>0)
        gap/=median(space_dist[i,j])"""


    print "Unit mass single pixel move cost: ",gap

    return gap,inear,jnear,dist_mat

def setting_polar(coord_cart,gap,cent,max_dist):

    from numpy import zeros,pi,sqrt
    from utils import polar_coord_cloud
    """from pyflann import FLANN
    knn = FLANN()
    params = knn.build_index(array(coord_cart, dtype=float64))
    result, dists = knn.nn_index(coord_cart, 2)"""

    pol_coord = polar_coord_cloud(coord_cart,cent)
    pol_coord[1,:]*=(gap/pi)#(max_dist/(pi*sqrt(coord_cart.shape[1])))
    pol_coord[0,:]*=gap

    return pol_coord

def setting_polar_2(coord_cart,gap,cent):

    from numpy import zeros,pi,sqrt
    from utils import polar_coord_cloud
    """from pyflann import FLANN
        knn = FLANN()
        params = knn.build_index(array(coord_cart, dtype=float64))
        result, dists = knn.nn_index(coord_cart, 2)"""

    pol_coord = polar_coord_cloud(coord_cart,cent)
    pol_coord[1,:]*=(gap/(2*pi))*pol_coord[0,:]#(max_dist/(pi*sqrt(coord_cart.shape[1])))
    pol_coord[0,:]*=gap

    return pol_coord


def setting_cart_from_polar(coord_pol,gap,cent,max_dist):

    from numpy import zeros,pi,sqrt,copy
    from utils import polar_to_cart_cloud
    """from pyflann import FLANN
        knn = FLANN()
        params = knn.build_index(array(coord_cart, dtype=float64))
        result, dists = knn.nn_index(coord_cart, 2)"""

    cart_coord = copy(coord_pol)
    cart_coord[1,:]/=(gap/pi)#(max_dist/(pi*sqrt(coord_pol.shape[1])))
    cart_coord[0,:]/=gap

    cart_coord = polar_to_cart_cloud(cart_coord,cent)

    return cart_coord

def setting_cart_from_polar_2(coord_pol,gap,cent,max_dist):

    from numpy import zeros,pi,sqrt,copy
    from utils import polar_to_cart_cloud
    """from pyflann import FLANN
        knn = FLANN()
        params = knn.build_index(array(coord_cart, dtype=float64))
        result, dists = knn.nn_index(coord_cart, 2)"""

    cart_coord = copy(coord_pol)
    cart_coord[0,:]/=gap

    ind = where(cart_coord[0,:]>0)
    cart_coord[1,ind[0]]/=(cart_coord[0,ind[0]]*gap/(2*pi))#(max_dist/(pi*sqrt(coord_pol.shape[1])))


    cart_coord = polar_to_cart_cloud(cart_coord,cent)

    return cart_coord


def max_gap_abs(stack):
    from numpy import gradient,median,zeros,mean

    diff_val = zeros((stack.shape[2]*2,))

    for i in range(0,stack.shape[2]):
        dx,dy = gradient(stack[:,:,i])
        diff_val[2*i] = median(dx)
        diff_val[2*i+1] = median(dy)

    return mean(diff_val)

def corr_map(stack):
    from numpy import ones
    from numpy.linalg import norm
    shap = stack.shape
    corr_mat = ones((shap[2],shap[2]))

    for i in range(0,shap[2]-1):
        for j in range(i+1,shap[2]):
            corr_mat[i,j] = abs(sum(stack[:,:,i]*stack[:,:,j]))/(norm(stack[:,:,i])*norm(stack[:,:,j]))

    return corr_mat

def distance_mat(embed):
    from numpy import zeros
    shap = embed.shape
    dist = zeros((shap[1],shap[1]))
    for i in range(0,shap[1]-1):
        for j in range(i+1,shap[1]):
            dist[i,j] = sqrt(sum((embed[:,i]-embed[:,j])**2))
            dist[j,i] = dist[i,j]
    return dist

def local_euc_dist_mat_support(stack,pos_field,knn=None,nb_neigh=30,max_val =2**63): # Compute the euclidian distance over the intersection of the support of each considered pair of images
    from numpy import zeros,transpose,ones,sqrt
    from pyflann import FLANN
    if knn is None:
        knn = FLANN()
        params = knn.build_index(array(pos_field, dtype=float64))

    nb_points = stack.shape[2]
    dist_mat = ones((nb_points,nb_points))*max_val
    i = range(0,nb_points)
    dist_mat[i,i] = 0
    result, dists = knn.nn_index(pos_field, nb_neigh+1)

    for i in range(0,nb_points):

        print i+1,"th PSF"
        for j in range(1,nb_neigh+1):
            if dist_mat[i,result[i,j]]==max_val:
                supp1i,supp1j =  where(abs(stack[:,:,i])>0)
                supp12 = where(abs(stack[supp1i,supp1j,result[0,j]])>0)
                dist_mat[i,result[i,j]] = sqrt(sum((stack[supp1i[supp12[0]],supp1j[supp12[0]],i]-stack[supp1i[supp12[0]],supp1j[supp12[0]],result[i,j]])**2))
                dist_mat[result[i,j],i] = dist_mat[i,result[i,j]]


    return dist_mat



def local_dist_euc_interface_m(stack_list,pos_field,nb_neigh=30):

    dist_list = list()
    for i in range(0,len(stack_list)):
        print "Feature ",i+1
        dist_i = local_euc_dist_mat(stack_list[i],pos_field,nb_neigh=nb_neigh)
        dist_list.append(dist_i)

    return dist_list

def local_dist_euc_interface_supp_m(stack_list,pos_field,nb_neigh=30):

    dist_list = list()
    for i in range(0,len(stack_list)):
        print "Feature ",i+1
        dist_i = local_euc_dist_mat_support(stack_list[i],pos_field,nb_neigh=nb_neigh)
        dist_list.append(dist_i)

    return dist_list


def euc_support_barycenter(stack,weights):
    from numpy import zeros
    shap = stack.shape

    bar = zeros(shap[0:2])
    i,j = where(abs(stack.prod(axis=2))>0)
    for k in range(0,shap[2]):
        bar[i,j]+=stack[i,j,k]*weights[k]

    return bar


def local_tang_coord(dist_mat,nb_neigh,nb_components=None,test=False,tol=0.999):
    from numpy import zeros,argsort,transpose
    from utils import mds
    nb_components_0 = nb_components
    if nb_components is None:
        nb_components_0 = nb_components
    nb_points = dist_mat.shape[0]
    coord_out = zeros((nb_components_0,nb_neigh+1,nb_points))
    indices = zeros((nb_points,nb_neigh+1))
    weights = zeros((nb_points,nb_components_0))
    for i in range(0,nb_points):
        ind  = argsort(dist_mat[i,:])

        indices[i,:] = ind[0:nb_neigh+1]
        loc_dist = zeros((nb_neigh+1,nb_neigh+1))
        for k in range(0,nb_neigh):
            for l in range(k+1,nb_neigh+1):
                loc_dist[k,l] = dist_mat[ind[k],ind[l]]
        loc_dist+=transpose(loc_dist)

        loc_coord,w = mds(loc_dist,nb_components = nb_components_0)
        weights[i,:] = w
        coord_out[:,:,i] = transpose(loc_coord)

    # Intrinsic dimension esttimation
    dim = 1
    if nb_components is None:
        E = (weights**2).sum()
        Etrunc = (weights[:,0:dim]**2).sum()
        while Etrunc<tol*E:
            dim+=1
            Etrunc = (weights[:,0:dim]**2).sum()

        print "Estimated intrinsic dimension: ",dim
    else:
        dim = nb_components

    if test:
        return coord_out[0:dim,:,:],indices,weights
    else:
        return coord_out[0:dim,:,:],indices

def local_reduction(data_mat,nb_neigh,nb_comp,dist_mat):

    from numpy.linalg import svd
    from numpy import argsort,copy,diag

    data_approx = copy(data_mat)

    shap = data_mat.shape

    for i in range(0,shap[1]):
        ind = argsort(dist_mat[i,:])
        U,s,V = svd(data_mat[:,ind[0:nb_neigh]])
        data_approx[:,i] = (U[:,ind[0:nb_neigh]].dot(diag(s[0:nb_neigh]))).dot(V[0:nb_neigh,0].reshape((nb_neigh,1))).reshape((shap[0],))

    return data_approx


def hessian_estimator(coord_out,indices):
    from numpy import zeros,ones,transpose
    from utils import gram_schmidt

    nb_points = coord_out.shape[2]
    d = coord_out.shape[0]
    nb_neigh = coord_out.shape[1]
    q = (d*(d+1))/2
    X = zeros((nb_neigh,1+d+q,nb_points))
    Q = zeros((nb_neigh,q,nb_points))
    H = zeros((nb_points,nb_points))

    for i in range(0,nb_points):
        print i+1," th point/",nb_points
        X[:,0,i] = ones((nb_neigh,))
        X[:,1:1+d,i] = transpose(coord_out[:,:,i])
        Ei = zeros((nb_points,nb_neigh))
        for j in range(0,nb_neigh):
            Ei[indices[i,j],j] = 1
            count=0
            for l in range(0,d):
                for k in range(l,d):
                    X[j,d+1+count,i] = coord_out[l,j,i]*coord_out[k,j,i]
                    count+=1
        Q[:,:,i] = gram_schmidt(X[:,:,i])[:,-q:]
        temp_m = Ei.dot(Q[:,:,i])
        H+= temp_m.dot(transpose(temp_m))

    return H,Q,X

def HLLE(dist_mat,nb_neigh,nb_comp,test=False,tol=0.999):
    from numpy.linalg import svd
    from numpy import transpose
    from pyflann import FLANN

    coord_out,indices,s = local_tang_coord(dist_mat,nb_neigh,nb_components=nb_comp,tol=tol,test=True)
    H,Q,X = hessian_estimator(coord_out,indices)

    U,s0,V =svd(H)

    embedding = transpose(U[:,-(coord_out.shape[0]+1):-1])

    knn = FLANN()
    params = knn.build_index(array(U[:,-(coord_out.shape[0]+1):-1], dtype=float64))

    if test:
        return embedding,s,knn
    else:
        return embedding,knn


def approx_HLLE(data_mat,dist_mat,nb_neigh,nb_comp): # The observations are in the columns
    from numpy import zeros,argsort,transpose
    from numpy.linalg import pinv
    shap = data_mat.shape
    loc_comp = zeros((shap[0],nb_comp,shap[1]))
    data_approx = copy(data_mat)

    embedding,knn = HLLE(dist_mat,nb_neigh,nb_comp,test=False,tol=0.999)

    for i in range(0,shap[1]):
        ind = argsort(dist_mat[i,:])
        loc_comp[:,:,i] = data_mat[:,ind[0:nb_neigh]].dot(transpose(pinv(transpose(embedding[:,ind[0:nb_neigh]]))))
        data_approx[:,i] = (loc_comp[:,:,i].dot(embedding[:,ind[0]].reshape((nb_comp,1)))).reshape((shap[0],))

    return embedding,loc_comp,data_approx

def pair_wise_distances_constraint(Y,dist_mat,details_mat=None,nb_iter=100,tol=0.0001,spec_rad=None,nb_eig=200): # The observations are in the rows
    from scipy.sparse.linalg import spsolve
    from numpy.linalg import solve,norm
    from numpy import zeros,transpose,array,copy,diag
    from scipy.sparse.linalg import svds,eigsh
    shap = Y.shape
    diff_mat = zeros((shap[0],(shap[1]*(shap[1]-1))/2))
    ordered_dist = zeros(((shap[1]*(shap[1]-1))/2,))
    spec_rad = 0
    if details_mat is None:
        details_mat = list()
        full_details_mat = zeros(((shap[1]*(shap[1]-1))/2,(shap[1]*(shap[1]-1))/2))
        print "Building implicit details matrix..."
        for i in range(0,shap[1]-1):
            for j in range(i+1,shap[1]):

                ind = i*shap[1] - (i*(i+1))/2 + j-i-1
                diff_mat[:,ind] = (Y[:,i]-Y[:,j])
                ordered_dist[ind] = dist_mat[i,j]
                full_details_mat[ind,ind] = 2
                for k in range(i+1,shap[1]):
                    if k!=j:
                        full_details_mat[i*shap[1] - (i*(i+1))/2 + k-i-1,ind] = 1
                for l in range(0,j):
                    if l!=i:
                        full_details_mat[l*shap[1] - (l*(l+1))/2 + j-l-1,ind] = 1
                for k in range(j+1,shap[1]):
                    full_details_mat[j*shap[1] - (j*(j+1))/2 + k-j-1,ind] = -1.0
                for l in range(0,i):
                    full_details_mat[l*shap[1] - (l*(l+1))/2 + i-l-1,ind] = -1.0

        print "Eigen analysis..."
        #U,s,V = svds(full_details_mat,k=nb_eig)
        s,U = eigsh(full_details_mat,k=nb_eig)
        V = transpose(U)
        ind = where(s>0)
        U = U[:,ind[0]]
        s = s[ind[0]]
        V = V[ind[0],:]

        print "Details matrix approximation error: ",100*abs(norm(s)-norm(full_details_mat))/norm(full_details_mat),norm(full_details_mat-U.dot(diag(s)).dot(V))

        details_mat.append(U)
        details_mat.append(s)
        details_mat.append(V)
    spec_rad = details_mat[1].max()
    Z = zeros((shap[0],(shap[1]*(shap[1]-1))/2))
    X = zeros((shap[0],(shap[1]*(shap[1]-1))/2))

    i=0
    err = 1
    t=1
    told=t
    nb_samp_prox = (shap[1]*(shap[1]-1))/2
    while(i<nb_iter and err>tol):
        grad = Z.dot(details_mat[0]).dot(diag(details_mat[1])).dot(details_mat[2]) -diff_mat
        Ytemp = Z-grad/spec_rad
        Xold = copy(X)
        X = pair_wise_dist_dual_proximity_op(Ytemp,spec_rad**(-1),ordered_dist,nb_samp=nb_samp_prox)
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        Z = Xold + lamb*(X-Xold)
        err = 100*norm(X-Xold)/norm(X)

        print "Gradient's norm: ",norm(grad)," Ieration variation rate: ",err,"%"
        i+=1


    print "Computing details manifold..."
    details = copy(Y)*0
    for k in range(0,shap[1]-1):
        for l in range(k+1,shap[1]):
            ind = k*shap[1] - (k*(k+1))/2 + l-k-1
            details[:,k]+=Z[:,ind]
            details[:,l]-=Z[:,ind]
    print "Done."
    return details,Z,details_mat

def pair_wis_dist_grad_mat(input,nb_samp=100):
    from utils import rand_diff_integ
    from numpy import diag
    cur_iter = input[0]
    coeff = input[1]
    supports = input[1]
    nb_vect = cur_iter.shape[1]
    if nb_vect>nb_samp:
        ind = rand_diff_integ(0,nb_vect,nb_samp)
    else:
        ind = range(0,nb_vect)
    dim = cur_iter.shape[0]
    output = cur_iter*0

    for i in range(0,len(ind)):

        output[:,ind[i]] = (cur_iter[:,supports[ind[i]]].dot(coeff[ind[i]].reshape((len(supports[ind[i]]),1)))).reshape((dim,))

    return output

def pair_wise_dist_dual_proximity_op(U,mu,ordered_dist,nb_samp=100):
    from utils import rand_diff_integ
    from numpy import diag
    from numpy.linalg import norm

    nb_vect = U.shape[1]
    if nb_vect>nb_samp:
        ind = rand_diff_integ(0,nb_vect,nb_samp)
    else:
        ind = range(0,nb_vect)

    U[:,ind] = U[:,ind] - mu*U[:,ind].dot(diag(ordered_dist[ind]*(norm(U[:,ind],axis=0))**(-1)))

    return U

def pair_wise_distances_constraint_lagrangian(Y,dist_ref,dist_target,beta=3):
    from numpy import zeros,eye,diag,transpose
    from numpy.linalg import norm,pinv,eigh


    shap = Y.shape
    M = eye(shap[1])
    for i in range(0,shap[1]-1):
        for j in range(i+1,shap[1]):
            lambd = beta*max(0,(dist_ref[i,j]-dist_target[i,j])/(2*dist_target[i,j]))
            M[i,i] += lambd/(shap[1]-1)
            M[j,i] -= lambd/(shap[1]-1)
            M[i,j] -= lambd/(shap[1]-1)
            M[j,j] += lambd/(shap[1]-1)

    s,U = eigh(M)
    ind = where(s>0)
    Yout = Y.dot(U[:,ind[0]].dot(diag(s[ind[0]])).dot(transpose(U[:,ind[0]])))
    return Yout,[s,U]



def HLLEm(dist_mat_list,nb_neigh,nb_comp,tol=0.999):
    knn_list = list()
    embedding_list = list()
    for i in range(0,len(dist_mat_list)):
        print "feature ",i+1,"/",len(dist_mat_list)
        embed_i,knn_i = HLLE(dist_mat_list[i],nb_neigh,nb_comp,tol = tol)
        embedding_list.append(embed_i)
        knn_list.append(knn_i)

    return embedding_list,knn_list

def sliced_bar_gradient(basis,obs,iterate,weights,assign_out=False,single_dir=False):
    from numpy import zeros,copy
    shap = obs.shape
    shapb = basis.shape
    H = zeros((shapb[0],shapb[0]))
    for i in range(0,shapb[1]):
        ui = basis[:,i].reshape((shapb[0],1))
        H+= ui.dot(transpose(ui))
    U, s, Vt = linalg.svd(H,full_matrices=False)

    coord_proj = zeros(shap)
    nb_obs = None
    grad = None
    sig_out = None
    if len(shap)==2:
        nb_obs = shap[0]

        for i in range(0,shap[1]):
            temp_proj,err_proj,sig = coord_wise_proj(transpose(iterate),obs[:,i].reshape((1,shap[0])),basis)
            coord_proj[:,i] = temp_proj.reshape((size(temp_proj),))
        grad = utils.vect(iterate) - coord_proj.dot(utils.vect(weights))
    elif len(shap)==3:
        nb_obs = shap[2]
        sig_out = zeros((shapb[1],shap[1],shap[2]))
        for i in range(0,shap[2]):
            temp_proj,err_proj,sig = coord_wise_proj(iterate,obs[:,:,i],basis,single_dir=single_dir)
            sig_out[:,:,i] = copy(sig)
            coord_proj[:,:,i] = temp_proj
        #grad = LA.inv(H).dot(iterate)
        grad = H.dot(copy(iterate))
        for i in range(0,nb_obs):
            grad -= weights[i]*coord_proj[:,:,i]
    else:
        raise ValueError("Bad input data shape")
    grad*= s[0]**(-1)
    if assign_out:
        return grad,sig_out
    else:
        return grad

def sliced_bar_gradient_compos(basis,obs,iterate,weights):
    from numpy import zeros

    list_supp_trans,list_supp_euc = supports_setting(iterate,obs)
    shap = obs.shape
    shapb = basis.shape
    H = zeros((shapb[0],shapb[0]))
    for i in range(0,shapb[1]):
        ui = basis[:,i].reshape((shapb[0],1))
        H+= ui.dot(transpose(ui))
    U, s, Vt = linalg.svd(H,full_matrices=False)

    coord_proj = zeros(shap)
    nb_obs = None
    grad = zeros(shap[0:2])


    nb_obs = shap[2]
    for i in range(0,shap[2]):
        temp_proj,err_proj,sig = coord_wise_proj(iterate[:,list_supp_trans[i]],obs[:,list_supp_trans[i],i],basis)
        grad[:,list_supp_trans[i]] += weights[i]*(H.dot(iterate[:,list_supp_trans[i]])-temp_proj)
        grad[2,list_supp_euc[i]] += weights[i]*(iterate[2,list_supp_euc[i]]-obs[:,list_supp_euc[i],i])

    grad*= (s[0]+1)**(-1)
    return grad




def init_sliced(basis,obs,weights):
    shap = obs.shape
    shapb = basis.shape

    coord_sort = zeros(shap)
    nb_obs = None
    init = None
    if len(shap)==2:
        nb_obs = shap[0]
        for i in range(0,shap[1]):
            temp_sort = coord_wise_sort(obs[:,i].reshape((1,shap[0])),basis)
            coord_sort[:,i] = temp_sort.reshape((size(temp_proj),))
        init = coord_sort.dot(utils.vect(weights))
    elif len(shap)==3:
        nb_obs = shap[2]
        for i in range(0,shap[2]):
            temp_sort = coord_wise_sort(obs[:,:,i],basis)
            coord_sort[:,:,i] = temp_sort
        init = weights[0]*coord_sort[:,:,0]
        for i in range(1,nb_obs):
            init += weights[i]*coord_sort[:,:,i]
    else:
        raise ValueError("Bad input data shape")
    return init



def sliced_transport_bar(F,weights,nb_iter=1000,tol=0.000000000000000001,alph=0.0,bar_init=None,dim=1,samp=3,rand_en=False,basis=None,adapted_basis=True,pos_en=False,min_basis=False,support=False,assign_out=False,single_dir=False,nb_real=1):
    shap = F.shape

    if len(shap)==2:
        if bar_init is None:
            bar_init = F.dot(utils.vect(weights))
        dim=1

    elif len(shap)==3:
        if bar_init is None:
            bar_init = weights[0]*F[:,:,0]
            for i in range(1,F.shape[2]):
                bar_init+=weights[i]*F[:,:,i]
        dim = shap[0]
    else:
        raise ValueError("Bad input data shape")

    bar = copy(bar_init)
    err = 100
    i = 1.0
    #basis = None
    grad = None
    sig_out = None
    while i < nb_iter and err>tol:

        step = 1.0/i**alph
        """
        if dim==3:
            basis = zeros((3,3))
            basis[0,0] = sign(random.randn(1))
            basis[1:,1:] = utils.random_orth_basis(dim-1)"""

        if rand_en:

            basis = utils.random_orth_basis_cat(dim,nb_real=nb_real)



        """elif adapted_basis and len(shap)==3 and i==0:
            if basis is None:
                basis = utils.adapted_positive_basis(F)
            if min_basis:
                basis = basis[:,0:2]
            #mat = utils.gram_schmidt(eye(3)+ np.random.randn(3,3)/(i+1)**2)
            #basis = mat.dot(basis)

        elif len(shap)==3 and i==0:
            basis = utils.sphere_vect(samp)
        elif i==0:
            basis = ones((1,1))"""

        bar_old = copy(bar)

        if support:
            grad = sliced_bar_gradient_compos(basis,F,bar,weights)
        else:
            grad,sig_out = sliced_bar_gradient(basis,F,bar,weights,assign_out=True,single_dir=single_dir)

        bar = bar - step*grad
        if pos_en:
            bar = pos_proj_mat(bar)
        i+=1
        err = 100*sum((bar-bar_old)**2)/sum(bar**2)
        print "Relative change of the iterate: ",err,"%."," Gradient's norm: ",sqrt(sum((step*grad)**2))

    #print "Adapted basis: ",basis
    disc_vect = bar_discrepancies_analysis(sig_out)
    print "Discrepancy vector: ",disc_vect
    if assign_out:
        return bar,basis,sig_out
    else:
        return bar,basis



def sliced_transport_bar_support_2d(F,weights,approx_en=True,nb_iter=1000,tol=0.000000000000000001,alph=0.01,bar_init=None,dim=1,samp=3,rand_en=False,basis=None,adapted_basis=True,pos_en=False,min_basis=False):
    from numpy import copy,zeros,argsort


    shap = F.shape

    if len(shap)==2:
        if bar_init is None:
            bar_init = F.dot(utils.vect(weights))
        dim=1

    elif len(shap)==3:
        if bar_init is None:
            bar_init = weights[0]*F[:,:,0]
            for i in range(1,F.shape[2]):
                bar_init+=weights[i]*F[:,:,i]
        dim = shap[0]
    else:
        raise ValueError("Bad input data shape")

    # ------ Support size ------ #
    Fc = copy(F)
    I = (F[2,:,:]).prod(axis=1)
    ind_supp = where(abs(I)>0)
    Fc[2,ind_supp[0],:] = 0
    bar_init[2,ind_supp[0]] = 0
    Fapprox = None
    if approx_en:
        min_supp = shap[1]
        for i in range(0,shap[2]):
            ind = where(abs(Fc[2,:,i])>0)

            supp_size_i = size(ind[0])
            if supp_size_i <min_supp:
                min_supp = supp_size_i
        Fapprox = zeros((shap[0],min_supp,shap[2]))
        for i in range(0,shap[2]):
            ind_sort = argsort(Fc[2,:,i])
            Fapprox[:,:,i] = Fc[:,ind_sort[-min_supp:],i]
        ind_sort = argsort(bar_init[2,:])
        bar_init = bar_init[:,ind_sort[-min_supp:]]
    else:
        Fapprox = Fc

    bar,basis = sliced_transport_bar(Fapprox,weights,nb_iter=nb_iter,tol=tol,alph=alph,bar_init=bar_init,dim=dim,samp=samp,rand_en=rand_en,basis=basis,adapted_basis=adapted_basis,pos_en=pos_en,min_basis=min_basis)

    return bar,basis,Fapprox


def supports_setting(iterate,F,acc=False):

    shap = F.shape
    I = (F[2,:,:]).prod(axis=1)
    ind_supp = where(abs(I)==0)
    list_trans_supp = list()
    list_eucl_supp = list()

    for i in range(0,shap[2]):
        I = F[2,:,i]*iterate[2,:]
        if acc:
            list_trans_supp.append((where(abs(I)==0))[0])
        else:
            ind1 = (where(abs(I)==0))[0]
            ind2 = (where(F[2,ind1,i]))[0]
            list_trans_supp.append(ind1[ind2])
        list_eucl_supp.append((where(abs(I)>0))[0])


    return list_supp_trans,list_supp_euc
"""def sliced_transport_bar_support_2d_acc(F,weights,approx_en=True,nb_iter=1000,tol=0.000000000000000001,alph=0.01,bar_init=None,dim=1,samp=3,rand_en=False,basis=None,adapted_basis=True,pos_en=False,min_basis=False):


    """




def robust_sliced_transport_bar(F,weights,template,ref,nb_iter=1000,tol=0.0000001,alph=0.01,bar_init=None,dim=1,samp=3,rand_en=False,adapted_basis=True):
    shap = F.shape
    if bar_init is None:
        if len(shap)==2:
            bar_init = F.dot(utils.vect(weights))
            dim=1

        elif len(shap)==3:
            bar_init = weights[0]*F[:,:,0]
            dim = shap[0]
            for i in range(1,F.shape[2]):
                bar_init+=weights[i]*F[:,:,i]
        else:
            raise ValueError("Bad input data shape")

    bar = copy(bar_init)
    err = 100
    i = 0
    basis = None
    while i < nb_iter and err>tol:

        step = 1.0#/i**alph
        """
            if dim==3:
            basis = zeros((3,3))
            basis[0,0] = sign(random.randn(1))
            basis[1:,1:] = utils.random_orth_basis(dim-1)"""

        if rand_en:
            basis = utils.random_orth_basis(dim)
        elif adapted_basis and len(shap)==3 and i==0:
            basis,s = utils.adapted_positive_basis(F)
        #mat = utils.gram_schmidt(eye(3)+ np.random.randn(3,3)/(i+1)**2)
        #basis = mat.dot(basis)

        elif len(shap)==3 and i==0:
            basis = utils.sphere_vect(samp)
        elif i==0:
            basis = ones((1,1))

        bar_old = copy(bar)
        grad = sliced_bar_gradient(basis,F,bar,weights)
        bar = bar - step*grad
        opt_ind,bar[0,:] = optimal_assignement_1d(bar[0,:]+ref,template)-ref
        err = 100*sum((bar-bar_old)**2)/sum(bar**2)
        print "Relative change of the iterate: ",err,"%."," Gradient's norm: ",sqrt(sum(grad**2))
        i+=1
    print "Max val: ",bar.max()
    return bar,basis


def l1_mean(distribs,weights,nb_iter=100,tol=0.001):
    optim_var1 = distribs*0
    optim_var2 = optim_var1.mean(axis=1)
    thresh = ones(distribs.shape)
    thresh = thresh.dot(diag(weights))
    ones_vect = zeros((1,distribs.shape[1]))
    lambd = 1
    cost_old=0.001
    cost = sum(abs(distribs - utils.vect(optim_var2).dot(transpose(utils.vect(weights)))))
    gamma = mean(distribs)
    i=0
    while i<nb_iter:# and 100*abs(cost_old-cost)/cost_old>tol:
        optim_var2 = optim_var1.mean(axis=1)
        temp1 = utils.vect(optim_var2).dot(ones_vect)
        temp2 = 2*temp1 - optim_var1
        optim_var1 += lambd*(distribs - temp1 + utils.thresholding(temp2 - distribs,gamma*thresh,1))
        cost_old = cost
        cost = sum(abs(distribs - utils.vect(optim_var2).dot(transpose(utils.vect(weights)))))
        print cost
        i+=1
    optim_var2 = optim_var1.mean(axis=1)
    return optim_var2

def optimal_assignement_1d(f,g,indg=None):
    from numpy import argsort
    indf = argsort(f)
    if indg is None:
        indg = argsort(g)
    indf_inv = argsort(indf)
    opt_ind = indg[indf_inv]
    projf = g[opt_ind]
    return opt_ind,projf

def optimal_assignement_im(im1,im2):
    f = im1.reshape((size(im1),))
    g = im2.reshape((size(im1),))
    opt_ind,projf = optimal_assignement_1d(f,g)
    return projf.reshape(im1.shape)

def coord_wise_proj(f,g,basis,single_dir=False):
    from numpy import zeros,argsort,copy
    from numpy.linalg import norm
    pf = transpose(basis).dot(f)
    pg = transpose(basis).dot(g)
    indg = None
    ind = None
    if single_dir:
        norm_proj = norm(pg,axis=1)
        ind = where(norm_proj==norm_proj.max())
        ind = ind[0][0]
        indg = argsort(pg[ind,:])
    eqf = copy(pf)
    sig_out = zeros((basis.shape[1],f.shape[1]))

    if single_dir:
        sig_out[ind,:],eqf[ind,:] = optimal_assignement_1d(pf[ind,:],pg[ind,:],indg=indg)
    else:
        for i in range(0,basis.shape[1]):
            sig_out[i,:],eqf[i,:] = optimal_assignement_1d(pf[i,:],pg[i,:],indg=indg)
            #print "assignement ",i," ",sig

    projf = basis.dot(eqf)
    err = sum((projf-f)**2)
    return projf,err,sig_out.astype(int)

def coord_wise_proj_bas(f,g,basis):
    from numpy import zeros,copy,transpose
    from numpy.linalg import norm

    pf = transpose(basis).dot(f)
    pg = transpose(basis).dot(g)

    eqf = copy(pf)
    sig_out = zeros((basis.shape[1],f.shape[1]))

    for i in range(0,basis.shape[1]):
        sig_out[i,:],eqf[i,:] = optimal_assignement_1d(pf[i,:],pg[i,:],indg=None)

    return pf,eqf,sig_out



def sliced_transport_gradient(f,g,nb_real=1,basis=None):
    from numpy import copy
    dim = f.shape[0]
    if basis is None:
        basis = utils.random_orth_basis_cat(dim,nb_real=nb_real)
    projf,err,sig_out = coord_wise_proj(f,g,basis,single_dir=False)

    grad_out = copy(f) - projf/nb_real

    return grad_out,basis


def size_tangent_proj(f,grad,cent,gap=1,cloud_siz=None):
    from numpy.linalg import norm
    from numpy import copy

    nb_points = f.shape[1]
    cent_mat = cent.reshape((2,1)).dot(ones((1,nb_points)))
    cent_sq_dist = norm(f[0:2]/gap - cent_mat,axis=0)**2

    if cloud_siz is None:
        cloud_siz = 0
        for i in range(0,nb_points):
            cloud_siz+=f[2,i]*cent_sq_dist[i]

    grad_proj = copy(grad)
    grad_proj[2,:] -= ((sum(cent_sq_dist*grad[2,:])-cloud_siz)/norm(cent_sq_dist)**2)*cent_sq_dist

    pos_normal_vect = f[0:2]/gap - cent_mat
    for i in range(0,nb_points):
        pos_normal_vect[:,i]*=f[2,i]

    grad_proj[0:2,:] -= (sum(grad[0:2,:]*pos_normal_vect)/norm(pos_normal_vect)**2)*pos_normal_vect

    return grad_proj

def exact_transport_line_search(f,g,basis,ascent_vect):

    from numpy import transpose
    from numpy.linalg import norm
    nb_points = f.shape[1]
    pf,eqf,sig_out = coord_wise_proj_bas(f,g,basis)
    proj_ascent = transpose(basis).dot(ascent_vect)
    step = sum((pf-eqf)*proj_ascent)/norm(proj_ascent)**2

    return step

def sliced_transport_siz(f,g,cent,cloud_siz=None,nb_iter=1000,tol=1.e-10,nb_real=30,gap=1,rand_en=False):

    from numpy import copy
    from numpy.linalg import norm
    shapf = f.shape
    i = 0
    var = 100
    pf = copy(f)
    basis = None
    sliced_grad = None
    displacement = zeros((shapf[0],shapf[1],nb_iter+2))
    displacement[:,:,0] = copy(f)

    while i<nb_iter and var>tol :
        if rand_en:
            sliced_grad,basis = sliced_transport_gradient(pf,g,nb_real=nb_real)
        else:
            sliced_grad,basis = sliced_transport_gradient(pf,g,nb_real=nb_real,basis=basis)
        tg_grad = size_tangent_proj(pf,sliced_grad,cent,gap=gap,cloud_siz=cloud_siz)
        #tg_grad = copy(sliced_grad)
        step = exact_transport_line_search(pf,g,basis,tg_grad)
        pf -= step*tg_grad
        var = 100*norm(pf-displacement[:,:,i])/norm(displacement[:,:,i])
        print "Relative variation: ",var," Total mass: ",pf[2,:].sum()," Cost: ",sliced_transport_cost(pf,g,basis)
        displacement[:,:,i+1] = copy(pf)
        i+=1
    displacement[:,:,i+1] = copy(g)

    return displacement[:,:,0:i+2]

def sliced_transport_single_dir(f,g,nb_iter=1000,tol=1.e-10,nb_real=30,rand_en=False):

    from numpy import copy
    from numpy.linalg import norm,svd
    shapf = f.shape
    i = 0
    var = 100
    pf = copy(f)
    basis = None
    sliced_grad = None
    displacement = list()
    displacement.append(copy(f))

    while i<nb_iter and var>tol :
        if rand_en:
            sliced_grad,basis = sliced_transport_gradient(pf,g,nb_real=nb_real)
        else:
            sliced_grad,basis = sliced_transport_gradient(pf,g,nb_real=nb_real,basis=basis)
        U,s,V = svd(sliced_grad.dot(transpose(sliced_grad)))
        #print "Advection dispersion: ",s
        tg_grad = U[:,0].reshape((shapf[0],1)).dot(U[:,0].reshape((1,shapf[0]))).dot(sliced_grad)
        #tg_grad = copy(sliced_grad)
        step = exact_transport_line_search(pf,g,basis,tg_grad)
        pf -= step*tg_grad
        var = 100*norm(pf-displacement[i])/norm(displacement[i])
        print "Relative variation: ",var," Total mass: ",pf[2,:].sum()," Cost: ",sliced_transport_cost(pf,g,basis)
        displacement.append(copy(pf))
        i+=1
    displacement.append(copy(g))

    return displacement



def lasso_transport_gradient(vect_x_l,vect_y_l,vect_x_c,vect_y_c,pix_val,g,shap,nb_real=1):
    from numpy import copy
    f = copy(g)
    f[2,:] = copy(pix_val)
    f[0,:] = lasso_cumul(vect_x_l,lines=True).reshape((shap[0]*shap[1],))+lasso_cumul(vect_x_c,lines=False).reshape((shap[0]*shap[1],))
    f[1,:] = lasso_cumul(vect_y_l,lines=True).reshape((shap[0]*shap[1],))+lasso_cumul(vect_y_c,lines=False).reshape((shap[0]*shap[1],))


    sliced_grad,basis = sliced_transport_gradient(f,g,nb_real=nb_real)
    grad_x_l = lasso_cumul_transp(sliced_grad[0,:].reshape(shap),lines=True)
    grad_x_c = lasso_cumul_transp(sliced_grad[0,:].reshape(shap),lines=False)
    grad_y_l = lasso_cumul_transp(sliced_grad[1,:].reshape(shap),lines=True)
    grad_y_c = lasso_cumul_transp(sliced_grad[1,:].reshape(shap),lines=False)

    return grad_x_l,grad_x_c,grad_y_l,grad_y_c,sliced_grad[2,:],basis

"""def tg_lasso_proj(vect_x_l,vect_y_l,vect_x_c,vect_y_c,pix_val,g,rmin,shap,nb_real=10):


    tg_grad_x_l,tg_grad_x_c,tg_grad_y_l,tg_grad_y_c,pix_grad,basis = lasso_transport_gradient(vect_x_l,vect_y_l,vect_x_c,vect_y_c,pix_val,g,shap,nb_real=nb_real)

    d_l = sqrt(vect_x_l**2+vect_y_l**2)
    indx1,indy1 = where(d_l==r_min)
    for i in range(0,size(indx1)):
        # Projection of the euclidian gradients at points at the boundary on the corresponding tangent planes
        tg_grad_x_l[indx1[id_bound[i]],indy1[id_bound[i]]],tg_grad_y_l[indx1[id_bound[i]],indy1[id_bound[i]]] = array([tg_grad_x_l[indx1[id_bound[i]],indy1[id_bound[i]]],tg_grad_y_l[indx1[id_bound[i]],indy1[id_bound[i]]]]) - (tg_grad_x_l[indx1[id_bound[i]],indy1[id_bound[i]]]*vect_x_l[indx1[id_bound[i]],indy1[id_bound[i]]] + tg_grad_y_l[indx1[id_bound[i]],indy1[id_bound[i]]]*vect_y_l[indx1[id_bound[i]],indy1[id_bound[i]]])*array([vect_x_l[indx1[id_bound[i]],indy1[id_bound[i]]],vect_y_l[indx1[id_bound[i]],indy1[id_bound[i]]]])/d_l[indx1[id_bound[i]],indy1[id_bound[i]]]**2

    d_c = sqrt(vect_x_c**2+vect_y_c**2)
    indx1,indy1 = where(d_c>0)
    id_bound = where(d_c[indx1,indy1]==r_min)
    id_bound = id_bound[0]
    for i in range(0,size(id_bound)):
    # Projection of the euclidian gradients at points at the boundary on the corresponding tangent planes
    tg_grad_x_c[indx1[id_bound[i]],indy1[id_bound[i]]],tg_grad_y_c[indx1[id_bound[i]],indy1[id_bound[i]]] = array([tg_grad_x_c[indx1[id_bound[i]],indy1[id_bound[i]]],tg_grad_y_c[indx1[id_bound[i]],indy1[id_bound[i]]]]) - (tg_grad_x_c[indx1[id_bound[i]],indy1[id_bound[i]]]*vect_x_c[indx1[id_bound[i]],indy1[id_bound[i]]] + tg_grad_y_c[indx1[id_bound[i]],indy1[id_bound[i]]]*vect_y_c[indx1[id_bound[i]],indy1[id_bound[i]]])*array([vect_x_c[indx1[id_bound[i]],indy1[id_bound[i]]],vect_y_c[indx1[id_bound[i]],indy1[id_bound[i]]]])/d_c[indx1[id_bound[i]],indy1[id_bound[i]]]**2

"""

def sliced_transport_cost(f,g,basis):
    from numpy import transpose,sort
    from numpy.linalg import norm
    slices_f = transpose(basis).dot(f)
    slices_g = transpose(basis).dot(g)
    cost = 0
    for i in range(0,basis.shape[1]):
        cost += norm(sort(slices_f[i,:])-sort(slices_g[i,:]))**2

    return cost

def lasso_transport_cost(vect_x_l,vect_y_l,vect_x_c,vect_y_c,pix_val,g,basis):
    from numpy import copy
    f = copy(g)
    f[2,:] = copy(pix_val)
    f[0,:] = lasso_cumul(vect_x_l,lines=True).reshape((shap[0]*shap[1],))+lasso_cumul(vect_x_c,lines=False).reshape((shap[0]*shap[1],))
    f[1,:] = lasso_cumul(vect_y_l,lines=True).reshape((shap[0]*shap[1],))+lasso_cumul(vect_y_c,lines=False).reshape((shap[0]*shap[1],))
    return sliced_transport_cost(f,g,basis)

def armijo_lasso_transport(vect_x_l,vect_x_c,vect_y_l,vect_y_c,pix_val,tg_grad_x_l,tg_grad_x_c,tg_grad_y_l,tg_grad_y_c,pix_grad,g,basis,r_min,angle_min,alpha=0.5,beta=0.5,sigma=1.e-10,nb_iter=1000,eps = 1.e-13):
    from numpy import sqrt
    from numpy.linalg import norm
    m=0
    cur_cost = lasso_transport_cost(vect_x_l,vect_y_l,vect_x_c,vect_y_c,pix_val,g,basis)
    grad_norm = norm(tg_grad_x_l)**2+norm(tg_grad_y_l)**2+norm(tg_grad_x_c)**2+norm(tg_grad_y_c)**2+norm(pix_grad)**2
    descent = 0

    mu = alpha*beta**m
    vect_x_l_0,vect_x_c_0,vect_y_l_0,vect_y_c_0 = retraction_lasso(vect_x_l-mu*tg_grad_x_l,vect_x_c-mu*tg_grad_x_c,vect_y_l-mu*tg_grad_y_l,vect_y_c-mu*tg_grad_y_c,r_min,angle_min)
    pix_val_0 = pix_val-mu*pix_grad
    new_cost = lasso_transport_cost(vect_x_l_0,vect_y_l_0,vect_x_c_0,vect_y_c_0,pix_val_0,g,basis)
    descent = cur_cost - new_cost

    count = 0
    while descent<=grad_norm*sigma*mu and grad_norm>eps and count<nb_iter:
        m+=1
        mu = alpha*beta**m
        vect_x_l_0,vect_x_c_0,vect_y_l_0,vect_y_c_0 = retraction_lasso(vect_x_l-mu*tg_grad_x_l,vect_x_c-mu*tg_grad_x_c,vect_y_l-mu*tg_grad_y_l,vect_y_c-mu*tg_grad_y_c,r_min,angle_min)
        pix_val_0 = pix_val-mu*pix_grad
        new_cost = lasso_transport_cost(vect_x_l_0,vect_y_l_0,vect_x_c_0,vect_y_c_0,pix_val_0,g,basis)
        descent = cur_cost - new_cost
        count+=1
    print "Armijo descent: ",descent," Step: ",mu," new cost: ",new_cost
    return vect_x_l_0,vect_x_c_0,vect_y_l_0,vect_y_c_0,pix_val_0,new_cost

"""def sliced_transport_lasso(f,g,rmin,nb_iter=1000,tol=1.e-15,nb_real=30,shap=None,gap=None):
    from numpy import copy

    fout = copy(f)
    var = 100
    i = 0

    vect_x_l = diff_coord_mat(f[0,:].reshape(shap),lines=True)
    vect_x_c = diff_coord_mat(f[0,:].reshape(shap),lines=False)

    vect_y_l = diff_coord_mat(f[1,:].reshape(shap),lines=True)
    vect_y_c = diff_coord_mat(f[1,:].reshape(shap),lines=False)

    pix_val = copy(f[2,:])

    while i<nb_iter or var>tol:
        grad_x_l,grad_x_c,grad_y_l,grad_y_c,pix_grad,basis = lasso_transport_gradient(vect_x_l,vect_y_l,vect_x_c,vect_y_c,pix_val,g,nb_real=nb_real,shap)
"""
def shape_tg_proj(f,ascent_dir,cent,gap=1,cloud_siz=None):
    from numpy import ones,copy
    from numpy.linalg import norm
    nb_points = f.shape[1]
    cent_mat = cent.reshape((2,1)).dot(ones((1,nb_points)))
    cent_dir = f[0:2,:]/gap - cent_mat
    dist_w = norm(cent_dir,axis=0)**2
    if cloud_siz is None:
        cloud_siz = sum(dist_w*f[2,:])

    ascent_dir_proj = copy(ascent_dir)
    ascent_dir_proj[2,:] -= (sum(dist_w*ascent_dir[2,:])-cloud_siz)*dist_w/norm(dist_w)**2
    normal_pos = ones((2,1)).dot(f[2,:].reshape((1,nb_points)))
    ascent_dir_proj[0:2,:] -= sum(normal_pos*ascent_dir_proj[0:2,:])*normal_pos/norm(normal_pos)**2

    return ascent_dir_proj


def energy_warping_metric(f,nb_neigh=4,eps=1.e-15):

    from pyflann import FLANN
    from numpy import transpose,zeros,float64
    nb_samp = f.shape[1]
    knn = FLANN()
    params = knn.build_index(array(transpose(f[0:2,:]), dtype=float64))
    result, dists = knn.nn_index(transpose(f[0:2,:]),nb_neigh+1)
    dists+=eps
    intensities = zeros((nb_samp,))
    inv_dist_weights = zeros((nb_samp,))

    for i in range(0,nb_samp):
        intensities[i] = sum(f[2,result[i,1:nb_neigh+1]]**2)
        inv_dist_weights[i] = sum(dists[i,1:nb_neigh+1]**(-2))

    return intensities,inv_dist_weights,sum(intensities*inv_dist_weights),knn

def energy_warping_tg_proj(f,ascent_dir,nb_neigh=4,eps=1.e-15):
    from pyflann import FLANN
    from numpy import transpose,zeros,float64
    from numpy.linalg import norm

    shap = f.shape
    normal_pos = zeros((2,shap[1]))
    nb_samp = f.shape[1]
    inv_dist_weights = zeros((nb_samp,))
    intensities = zeros((nb_samp,))
    knn = FLANN()
    params = knn.build_index(array(transpose(f[0:2,:]), dtype=float64))
    result, dists = knn.nn_index(transpose(f[0:2,:]),nb_neigh+1)
    dists += eps
    dist_w = zeros((shap[1],))
    for i in range(0,nb_samp):
        inv_dist_weights[i] = sum(dists[i,1:nb_neigh+1]**(-2))
        intensities[i] = sum(f[2,result[i,0:nb_neigh+1]])
        for j in range(0,nb_neigh+1):
            if j<nb_neigh:
                normal_pos[:,i] += (intensities[i]**2)*(f[0:2,i]-f[0:2,result[i,j+1]])/dists[i,j+1]**4
                normal_pos[:,result[i,j+1]] -= (intensities[i]**2)*(f[0:2,i]-f[0:2,result[i,j+1]])/dists[i,j+1]**4
            dist_w[result[i,j]] = inv_dist_weights[i]*intensities[i]

    ascent_dir_proj = copy(ascent_dir)
    ascent_dir_proj[2,:] -= sum(dist_w*ascent_dir[2,:])*dist_w/norm(dist_w)**2
    normal_pos = ones((2,1)).dot(f[2,:].reshape((1,nb_samp)))
    ascent_dir_proj[0:2,:] -= sum(normal_pos*ascent_dir_proj[0:2,:])*normal_pos/norm(normal_pos)**2

    return ascent_dir_proj

def coord_wise_sort(g,basis,first = True):
    pg = transpose(basis).dot(g)
    eqg = pg*0
    ind = None
    sortg = None
    if first:
        ind = argsort(pg[0,:])
        sortg = g[:,ind]
    else:
        for i in range(0,basis.shape[1]):
            eqg[i,:] = sort(pg[i,:])
        sortg = basis.dot(eqg)
    return sortg


def support_computing(P,r,c):
    i = where(r>0)
    j = where(c>0)
    supp_size_init = size(i)+size(j)+1
    P_thresh = utils.kthresholding_im(P,supp_size_init,0)
    thresh_ind = where(P_thresh>0)
    supp = zeros(P.shape)
    supp[thresh_ind[0],thresh_ind[1]] = 1

    margc = P_thresh.sum(axis=1)
    margl = P_thresh.sum(axis=0)
    zc = where(margc==0)
    zl = where(margl==0)
    Ptemp = copy(P)
    supp[thresh_ind[0],thresh_ind[1]] = 0
    for i in range(0,size(zc)):
        if c[zc[0][i]] >0:
            j = where(Ptemp[i,:]==Ptemp[i,:].max())
            supp[i,j[0][0]] = 1 # Extending the support

    for j in range(0,size(zl)):
        if c[zl[0][j]] >0:
            i = where(Ptemp[:,j]==Ptemp[:,j].max())
            supp[i[0][0],j] = 1 # Extending the support


    return supp

def map_sparse(P,r,c):
    ri = sum(P,axis=0)
    ri = ri.reshape((size(ri),1))
    print 'Bias :',((r-ri)**2).sum()/(r**2).sum()

    Psp = copy(P)
    shap = P.shape
    for i in range(0,shap[0]):
        Psp[i,:]*=0
        j = where(P[i,:]==P[i,:].max())
        l = 100000000
        opt = 0
        for k in range(0,size(j[0])):
            err = abs(c[j[0][k]]-r[i])
            if err<l:
                l = err
                opt = k
        Psp[i,j[0][opt]] = r[i]

    return Psp





def support_checking(P,supp,r,c):
    margc = supp.sum(axis=1)
    margl = supp.sum(axis=0)
    zc = where(margc==0)
    zl = where(margl==0)
    for i in range(0,size(zc)):
        if c[zc[0][i]] >0:
            indj = argsort(P[i,:])
            search = True
            count = -1
            while search and count >-size(r):
                j = indj[count]
                count-=1
                if r[j]>0:
                    search = False
                #print "columns count: ",count
            supp[i,j] = 1 # Extending the support

    for j in range(0,size(zl)):
        if c[zl[0][j]] >0:
            indi = argsort(P[:,j])
            search = True
            count = -1
            while search and count > -size(c):
                i = indi[count]
                count-=1
                if c[i]>0:
                    search = False
                #print "lines count: ",count
            supp[i,j] = 1 # Extending the support

    return supp

def beg_proj_transp_clust(M,r,c,cart_prods,w,inf_val=None,beta=200,nb_iter=1000):
    lambd = beta/M.max()
    K = None
    i,j=None,None
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0

    P = copy(K)
    P1 = copy
    for i in range(0,nb_iter):


        P = kl_marg_proj(P,c)
        P = transpose(kl_marg_proj(transpose(P),r))
        ci = sum(P,axis=1)
        ci = ci.reshape((size(ci),1))
        ri = sum(P,axis=0)
        ri = ri.reshape((size(ri),1))
    print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
    print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
    P = kl_clust_proj(P,cart_prods,w)
    nb_clust = len(w)
    bias_3 = 0
    for i in range(0,nb_clust):
        bias_3+= abs(sum(P[cart_prods[i][:,0],cart_prods[i][:,1]]) - w[i])/w[i]
    print 'Bias 3:',bias_3
    print 'Check sum :',ri.sum(),' ',ci.sum()

    dist=sum(M*P)
    return P,dist


def beg_normalize(M,c,inf_val=None,beta=200,nb_iter=1000,w1=0.5):
    w2 = 1-w1
    cin = c*(c>0)
    lambd = beta/M.max()
    K = None
    i,j=None,None
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0

    P = copy(K)

    for i in range(0,nb_iter):
        P1 = P/P.sum()
        P2 = kl_marg_proj(P,cin)
        P = (P1**w1)*(P2**w2)
        ci = sum(P,axis=1)
        ci = ci.reshape((size(ci),1))
        print 'Bias normalization:',((cin-ci)**2).sum()/(c**2).sum()

    dist=sum(M*P)
    proj = sum(P1,axis=0)
    return proj,P,dist


def beg_proj_transp_rw(M,r,c,inf_val=None,beta=200,alpha=0.8,nb_iter=20,nb_rw=1,nw=3):
    max_num_exp = 2000
    lambd = beta/M.max()
    K = None
    i,j=None,None
    Mtemp=None
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0

    P = copy(K)
    for i in range(0,nb_iter):
        P = kl_marg_proj(P,c)
        ci = sum(P,axis=1)
        ci = ci.reshape((size(ci),1))
        P = transpose(kl_marg_proj(transpose(P),r))
        ri = sum(P,axis=0)
        ri = ri.reshape((size(ri),1))
        print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
        print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
        print 'Check sum :',ri.sum(),' ',ci.sum()

    dist=sum(M*P)
    Pw = copy(P)
    shap = Pw.shape
    maxw = max_num_exp/(Mtemp.max()*lambd)
    print "maxw = ",maxw," M.max() = ",Mtemp.max()
    w = None
    for l in range(0,nb_rw):
        w = ones(shap)
        (i,j) = where(Pw>0)
        n = size(i)
        for m in range(0,n):
            w1 = Pw[i[m],:].max()
            w2 = Pw[:,j[m]].max()
            w[i[m],j[m]] = utils.scaling((w1-Pw[i[m],j[m]])/w1,nw,ainit=0,afinal=maxw)
            #print "w1-Pw[i[m],j[m]])/w1,w[i[m],j[m]]",(w1-Pw[i[m],j[m]])/w1,w[i[m],j[m]]
        Mw = M*w
        num_pb = True
        Kw = exp(-lambd*Mw)
        Kwsym = (Kw+transpose(Kw))/2
        if inf_val is not None:
            i,j = where(M==inf_val)
            Kw[i,j]=0
        count_max = floor(-log(10)/log(alpha))+1
        count = 0
        """while (num_pb==True and count<count_max):
            k1 = sum(Kw,axis=0)
            k2 = sum(Kw,axis=1)
            if prod(k1)==0 or prod(k2)==0:
                print "Warning: static or isolated point(s); increasing entropy weight"
                lambd=lambd*alpha
                Kw = exp(-lambd*Mw)
                if inf_val is not None:
                    Kw[i,j]=0
            else:
                num_pb = False
            count+=1
        print "num_pb:", num_pb
        if count==count_max and num_pb==True:
            i,j = where(Kw==0)
            Kw[i,j] = K[i,j]"""

        Pw = copy(Kw)
        for i in range(0,4*nb_iter):
            Pw = kl_marg_proj(Pw,c)
            ci = sum(Pw,axis=1)
            ci = ci.reshape((size(ci),1))
            Pw = transpose(kl_marg_proj(transpose(Pw),r))
            ri = sum(Pw,axis=0)
            ri = ri.reshape((size(ri),1))
            print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
            print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
            print 'Check sum :',ri.sum(),' ',ci.sum()


    distw=sum(Mw*Pw)
    return P,Pw,dist,distw,w


def wkl_cons_lag_comp(list_arg):
    a = list_arg[3]
    p = list_arg[1]
    lambd = list_arg[2]
    evl_pt = list_arg[0]
    i = where(lambd>0)
    output = ((p[i]*exp(-(lambd[i]**(-1))*evl_pt)).sum()-a)*exp(-evl_pt/(p[i].sum()))
    return output

def der_wkl_cons_lag_comp(list_arg):
    #a = list_arg[3]
    p = list_arg[1]
    lambd = list_arg[2]
    evl_pt = list_arg[0]
    i = where(lambd>0)
    output = (-(lambd[i]**(-1))*p[i]*exp(-(lambd[i]**(-1))*evl_pt)).sum()*exp(-evl_pt/(p[i].sum())) - ((p[i].sum())**(-1))*wkl_cons_lag_comp(list_arg)

    return output

def newton_1d(fun,der_fun,input,init=0,nb_iter=80,tol=10**(-20)):
    x = init
    if input[1].sum()<tol:
        return 0
    else:
        for i in range(0,nb_iter):
            input[0] = x
            x = x - fun(input)/der_fun(input)
            """print "weights sum ",input[1].sum()
            print "target val ",input[3]
            print "eval pt ",input[0]"""

    print "fun(input) ",fun(input)/exp(-x/(input[1].sum()))
#print "der_fun(input) ",der_fun(input)
    return x


def wkl_cons_lag_lines(P,Lambd,a):
    shap = P.shape
    mu = zeros((shap[0],1))
    for i in range(0,shap[0]):
        list_arg = list()
        list_arg.append(0)
        list_arg.append(copy(P[i,:]))
        list_arg.append(copy(Lambd[i,:]))
        list_arg.append(a[i])
        mu[i] = newton_1d(wkl_cons_lag_comp,der_wkl_cons_lag_comp,list_arg)

    return mu

def wkl_proj_lines(P,Lambd,a):
    mu = wkl_cons_lag_lines(P,Lambd,a)
    #print "sum(mu) ",sum(mu)
    shap = P.shape
    one_vect = ones((1,shap[1]))
    Mu = mu.dot(one_vect)
    P_proj = 0*P
    i,j = where(Lambd>0)
    P_proj[i,j] = P[i,j]*exp(-Mu[i,j]/Lambd[i,j])
    return P_proj


def beg_proj_transp_rw_2(M,r,c,inf_val=None,beta=200,alpha=0.8,nb_iter=20,nb_rw=1,nw=9):
    max_num_exp = 2000
    lambd = beta/M.max()
    K = None
    i,j=None,None
    Mtemp=None
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0

    P = copy(K)
    for i in range(0,nb_iter):
        P = kl_marg_proj(P,c)
        ci = sum(P,axis=1)
        ci = ci.reshape((size(ci),1))
        P = transpose(kl_marg_proj(transpose(P),r))
        ri = sum(P,axis=0)
        ri = ri.reshape((size(ri),1))
        print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
        print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
        print 'Check sum :',ri.sum(),' ',ci.sum()

    dist=sum(M*P)
    Pw = copy(P)
    shap = Pw.shape
    minw = (Mtemp.max()*lambd)/max_num_exp
    print "minw = ",minw," M.max() = ",Mtemp.max()
    w = None
    for l in range(0,nb_rw):
        w = ones(shap)
        (i,j) = where(Pw>0)
        n = size(i)
        for m in range(0,n):
            w1 = Pw[i[m],:].min()
            w2 = Pw[i[m],:].max()
            w[i[m],j[m]] = utils.scaling(((Pw[i[m],j[m]])-w1)/(w2-w1),nw,ainit=0.999,afinal=1)
        #print "w1-Pw[i[m],j[m]])/w1,w[i[m],j[m]]",(w1-Pw[i[m],j[m]])/w1,w[i[m],j[m]]

        num_pb = True
        Kw = exp(-lambd*M/w)
        print "min(w) ",w.min()," max(w) ",w.max()
        #Kwsym = (Kw+transpose(Kw))/2
        if inf_val is not None:
            i,j = where(M==inf_val)
            Kw[i,j]=0
        count_max = floor(-log(10)/log(alpha))+1
        count = 0
        """while (num_pb==True and count<count_max):
            k1 = sum(Kw,axis=0)
            k2 = sum(Kw,axis=1)
            if prod(k1)==0 or prod(k2)==0:
            print "Warning: static or isolated point(s); increasing entropy weight"
            lambd=lambd*alpha
            Kw = exp(-lambd*Mw)
            if inf_val is not None:
            Kw[i,j]=0
            else:
            num_pb = False
            count+=1
            print "num_pb:", num_pb
            if count==count_max and num_pb==True:
            i,j = where(Kw==0)
            Kw[i,j] = K[i,j]"""

        Pw = copy(Kw)
        for i in range(0,4*nb_iter):
            Pw = wkl_proj_lines(Pw,w,c)
            ci = sum(Pw,axis=1)
            ci = ci.reshape((size(ci),1))
            Pw = transpose(wkl_proj_lines(transpose(Pw),transpose(w),r))
            ri = sum(Pw,axis=0)
            ri = ri.reshape((size(ri),1))
            print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
            print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
            print 'Check sum :',ri.sum(),' ',ci.sum()

    distw=sum(M*Pw)
    return P,Pw,dist,distw,w

def kl_capacity_proj(M,Mc):
    proj = M
    i,j = where(Mc<M)
    proj[i,j] = Mc[i,j]
    return proj


def beg_proj_transp_rw_3(M,r,c,inf_val=None,beta=200,alpha=0.8,nb_iter=20,nb_rw=1,nw=4):
    max_num_exp = 2000
    lambd = beta/M.max()
    K = None
    i,j=None,None
    Mtemp=None
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0

    P = copy(K)
    for i in range(0,nb_iter):
        P = kl_marg_proj(P,c)
        P = transpose(kl_marg_proj(transpose(P),r))
        ci = sum(P,axis=1)
        ci = ci.reshape((size(ci),1))
        ri = sum(P,axis=0)
        ri = ri.reshape((size(ri),1))
        print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
        print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
        print 'Check sum :',ri.sum(),' ',ci.sum()

    dist=sum(M*P)

    lamdb = lambd*5
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0
    Pn = copy(K)
    shap = Pn.shape
    w = None
    print ' ------------- Reweighting ------------ '
    for l in range(0,nb_rw):
        w = zeros(shap)
        (i,j) = where(K>0)
        n = size(i)
        for m in range(0,n):
            w1 = P[i[m],:].max()
            w[i[m],j[m]] = utils.scaling_2((P[i[m],j[m]])/w1,nw,ainit=P[i[m],:].sum()/20,afinal=P[i[m],:].sum())
        qn = ones(shap)
        qn_1 = ones(shap)
        qn_2 = ones(shap)
        #nb_iter=1
        for n in range(0,nb_iter):
            Pn_1 = copy(Pn)
            Pn = kl_marg_proj(Pn_1*qn_2,c)
            l,m = where(Pn>0)
            qn_2[l,m] = qn_2[l,m]*Pn_1[l,m]/Pn[l,m]

            Pn_1 = copy(Pn)
            Pn = transpose(kl_marg_proj(transpose(Pn_1*qn_1),r))
            l,m = where(Pn>0)
            qn_1[l,m] = qn_1[l,m]*Pn_1[l,m]/Pn[l,m]
            #qn_1 = copy(q1_n)
            #qn = copy(q2_n)


            Pn_1 = copy(Pn)
            Pn = kl_capacity_proj(Pn_1*qn,w)
            l,m = where(Pn>0)
            qn[l,m] = qn[l,m]*Pn_1[l,m]/Pn[l,m]

            ci = sum(Pn,axis=1)
            ci = ci.reshape((size(ci),1))

            ri = sum(Pn,axis=0)
            ri = ri.reshape((size(ri),1))

            print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
            print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
            print 'Check sum :',ri.sum(),' ',ci.sum()

    distw=sum(M*Pn)
    return P,Pn,dist,distw,w


def beg_proj_transp_capacity(M,Pmax,c,inf_val=None,beta=200,nb_iter=100):
    max_num_exp = 2000
    lambd = beta/M.max()
    K = None
    i,j=None,None
    Mtemp=None
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0


    K = exp(-lambd*M)

    Pn = copy(K)
    shap = Pn.shape
    qn = ones(shap)
    qn_1 = ones(shap)
    qn_2 = ones(shap)
    #nb_iter=1
    for n in range(0,nb_iter):
        Pn_1 = copy(Pn)
        Pn = kl_marg_proj(Pn_1*qn_2,c)
        l,m = where(Pn>0)
        qn_2[l,m] = qn_2[l,m]*Pn_1[l,m]/Pn[l,m]

        #Pn_1 = copy(Pn)
        #Pn = transpose(kl_marg_proj(transpose(Pn_1*qn_1),r))
        #l,m = where(Pn>0)
        #qn_1[l,m] = qn_1[l,m]*Pn_1[l,m]/Pn[l,m]
        #qn_1 = copy(q1_n)
        #qn = copy(q2_n)


        Pn_1 = copy(Pn)
        Pn = kl_capacity_proj(Pn_1*qn,Pmax)
        l,m = where(Pn>0)
        qn[l,m] = qn[l,m]*Pn_1[l,m]/Pn[l,m]

    ci = sum(Pn,axis=1)
    ci = ci.reshape((size(ci),1))

    print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
    #print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
    print 'Check sum: ',ci.sum()


    interp = sum(Pn,axis=0)
    return Pn,interp


def beg_proj_transp_rw_4(M,r,c,inf_val=None,beta=200,alpha=0.8,nb_iter=20,nb_rw=1,nw=4):
    max_num_exp = 2000
    lambd = beta/M.max()
    K = None
    i,j=None,None
    Mtemp=None
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0

    P = copy(K)
    for i in range(0,nb_iter):
        P = kl_marg_proj(P,c)
        P = transpose(kl_marg_proj(transpose(P),r))
        ci = sum(P,axis=1)
        ci = ci.reshape((size(ci),1))
        ri = sum(P,axis=0)
        ri = ri.reshape((size(ri),1))
        print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
        print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
        print 'Check sum :',ri.sum(),' ',ci.sum()

    dist=sum(M*P)

    lamdb = lambd*5
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0
    Pn = copy(K)
    shap = Pn.shape
    w = None
    r = None
    print ' ------------- Reweighting ------------ '
    for l in range(0,nb_rw):
        w = zeros(shap)
        r = zeros((shap[0],1))
        for m in range(0,shap[0]):
            i = where(w[m,:]==w[m,:].max())
            w[m,i] = w[m,:].sum()

            w[m,:] = utils.scaling_2((P[i[m],j[m]])/w1,nw,ainit=P[i[m],:].sum()/20,afinal=P[i[m],:].sum())
        qn = ones(shap)
        qn_1 = ones(shap)
        qn_2 = ones(shap)
        #nb_iter=1
        for n in range(0,nb_iter):
            Pn_1 = copy(Pn)
            Pn = kl_marg_proj(Pn_1*qn_2,c)
            l,m = where(Pn>0)
            qn_2[l,m] = qn_2[l,m]*Pn_1[l,m]/Pn[l,m]

            Pn_1 = copy(Pn)
            Pn = transpose(kl_marg_proj(transpose(Pn_1*qn_1),r))
            l,m = where(Pn>0)
            qn_1[l,m] = qn_1[l,m]*Pn_1[l,m]/Pn[l,m]
            #qn_1 = copy(q1_n)
            #qn = copy(q2_n)


            Pn_1 = copy(Pn)
            Pn = kl_capacity_proj(Pn_1*qn,w)
            l,m = where(Pn>0)
            qn[l,m] = qn[l,m]*Pn_1[l,m]/Pn[l,m]

            ci = sum(Pn,axis=1)
            ci = ci.reshape((size(ci),1))

            ri = sum(Pn,axis=0)
            ri = ri.reshape((size(ri),1))

            print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
            print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
            print 'Check sum :',ri.sum(),' ',ci.sum()

    distw=sum(M*Pn)
    return P,Pn,dist,distw,w

def beg_proj_transp_rw_4(M,r,c,corr_coeff,inf_val=None,beta=200,alpha=0.8,nb_iter=100,nb_rw=1,nw=2):
    max_num_exp = 2000
    lambd = beta/M.max()
    K = None
    i,j=None,None
    Mtemp=None
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0

    P = copy(K)
    for i in range(0,nb_iter):
        P = kl_marg_proj(P,c)
        P = transpose(kl_marg_proj(transpose(P),r))
        ci = sum(P,axis=1)
        ci = ci.reshape((size(ci),1))
        ri = sum(P,axis=0)
        ri = ri.reshape((size(ri),1))
        print 'Bias 1:',((c-ci)**2).sum()/(c**2).sum()
        print 'Bias 2:',((r-ri)**2).sum()/(r**2).sum()
        print 'Check sum :',ri.sum(),' ',ci.sum()

    dist=sum(M*P)

#lambd = lambd*5
    if inf_val is not None:
        i,j = where(M==inf_val)
        i0,j0 = where(M<inf_val)
        Mtemp = M[i0,j0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*M)
    if inf_val is not None:
        K[i,j] = 0
    Pn = copy(K)
    shap = Pn.shape
    w = None
    rad = None
    print ' ------------- Reweighting ------------ '
    for l in range(0,nb_rw):
        w = zeros(shap)
        rad = zeros((shap[0],1))
        for m in range(0,shap[0]):
            i = where(P[m,:]==P[m,:].max())
            w[m,i] = P[m,:].sum()
            rmax = sqrt(((w[m,:]-P[m,:])**2).sum())
            rad[m] = rmax#-utils.scaling_2(corr_coeff,nw,ainit=0,afinal=rmax)

        qn = ones(shap)
        qn_1 = ones(shap)
        qn_2 = ones(shap)
        #nb_iter=1
        for n in range(0,nb_iter):
            Pn_1 = copy(Pn)
            Pn = kl_marg_proj(Pn_1*qn_2,c)
            l,m = where(Pn>0)
            qn_2[l,m] = qn_2[l,m]*Pn_1[l,m]/Pn[l,m]

            Pn_1 = copy(Pn)
            Pn = transpose(kl_marg_proj(transpose(Pn_1*qn_1),r))
            l,m = where(Pn>0)
            qn_1[l,m] = qn_1[l,m]*Pn_1[l,m]/Pn[l,m]



            Pn_1 = copy(Pn)
            #Pn = kl_capacity_proj(Pn_1*qn,w)
            Pn,bias3 = kl_proj_sphere_mat(Pn_1*qn,w,rad)
            l,m = where(Pn>0)
            qn[l,m] = qn[l,m]*Pn_1[l,m]/Pn[l,m]

            ci = sum(Pn,axis=1)
            ci = ci.reshape((size(ci),1))

            ri = sum(Pn,axis=0)
            ri = ri.reshape((size(ri),1))

            print 'Bias 1: ',((c-ci)**2).sum()/(c**2).sum()
            print 'Bias 2: ',((r-ri)**2).sum()/(r**2).sum()
            print 'Bias 3: ',bias3.min()
            print 'Check sum :',r.sum(),' ',ci.sum()

    distw=sum(M*Pn)
    return P,Pn,dist,distw,w,rad,Pn_1*qn


def multi_marg_proj(P,distribs): # Marginalizes over columns, marginal distributions correspond to distribs lines
    shap = P.shape
    l = shap[2]
    proj = copy(P)
    bias = zeros((l,))
    for k in range(0,l):
        c = distribs[k,:].reshape((size(distribs[k,:]),1))
        proj[:,:,k] = kl_marg_proj(P[:,:,k],c)
        bias[k] = ((sum(proj[:,:,k],axis=1)-distribs[k,:])**2).sum()/((distribs[k,:])**2).sum()

    return proj,bias

def common_marg_proj(P,weights): # Common Marginal over columns
    shap = P.shape
    geo_bar = ones((shap[0],))
    for i in range(0,shap[2]):
        geo_bar = geo_bar*(sum(P[:,:,i],axis=1))**weights[i]
    bias = zeros((shap[2],))
    geo_bar_c = geo_bar.reshape((size(geo_bar),1))
    ones_vect = ones((1,shap[2]))
    geo_bar_in = transpose(geo_bar_c.dot(ones_vect))

    proj,bias = multi_marg_proj(P,geo_bar_in)
    for i in range(0,shap[2]):
        bias[i] = ((sum(proj[:,:,i],axis=1)-geo_bar)**2).sum()/(geo_bar**2).sum()
    return proj,bias

def beg_proj_transp_bar(dist_mats,distribs,weights,inf_val=None,beta=200,nb_iter=20):
    max_num_exp = 2000
    lambd = beta/dist_mats.max()
    K = None
    i,j=None,None
    Mtemp=None
    if inf_val is not None:
        i,j,k = where(dist_mats==inf_val)
        i0,j0,k0 = where(dist_mats<inf_val)
        Mtemp = dist_mats[i0,j0,k0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*dist_mats)
    if inf_val is not None:
        K[i,j,k] = 0

    P = copy(K)
    l = dist_mats.shape[2]
    bias1 = zeros((l,))
    for i in range(0,nb_iter):
        P,bias1 = multi_marg_proj(P,distribs)
        tP,bias2 = common_marg_proj(utils.stack_transpose(P),weights)
        P = utils.stack_transpose(tP)


    for k in range(0,l):
        bias1[k] = ((sum(P[:,:,k],axis=1)-distribs[k,:])**2).sum()/((distribs[k,:])**2).sum()
    print 'Multimarg bias: ',bias1
    print 'Common marg bias: ',bias2
    bar=sum(P[:,:,0],axis=0)
    return P,bar

def beg_proj_transp_bar_located(dist_mats,distribs,weights,target_pos,src_pos1,src_pos2,inf_val=None,beta=200,nb_iter=20):
    max_num_exp = 2000
    lambd = beta/dist_mats.max()
    #print "dist_mats.max(): ",dist_mats.max()
    shap = dist_mats.shape
    K = None
    i,j=None,None
    Mtemp=None
    if inf_val is not None:
        i,j,k = where(dist_mats==inf_val)
        i0,j0,k0 = where(dist_mats<inf_val)
        Mtemp = dist_mats[i0,j0,k0]
        lambd = beta/Mtemp.max()

    K = exp(-lambd*dist_mats)
    if inf_val is not None:
        K[i,j,k] = 0

    P = copy(K)
    l = dist_mats.shape[2]
    bias1 = zeros((l,))

    # Location contraint parameters
    b = target_pos.reshape((2,1))
    A = zeros((2,size(src_pos1)))
    A[0,:] = src_pos1.reshape((size(src_pos1),))
    A[1,:] = src_pos2.reshape((size(src_pos2),))
    xinit = None
    for i in range(0,nb_iter):



        tP,bias2 = common_marg_proj(utils.stack_transpose(P),weights)
        P = utils.stack_transpose(tP)

        # Location constraint
        lines_marg = sum(P[:,:,0],axis=0).reshape((shap[0],1))
        pivot_loc = pos_lin_cons(A,b,lines_marg,xinit,nb_iter=100,tol=1e-32,mu=1)
        xinit = copy(pivot_loc)
        pivn = pivot_loc.dot(ones((1,shap[2])))
        tP,bias2 = multi_marg_proj(utils.stack_transpose(P),transpose(pivn))
        P = utils.stack_transpose(tP)
        P,bias1 = multi_marg_proj(P,distribs)
        for k in range(0,l):
            bias1[k] = ((sum(P[:,:,k],axis=1)-distribs[k,:])**2).sum()/((distribs[k,:])**2).sum()
            bias2[k] = ((sum(P[:,:,k],axis=0)-sum(P[:,:,0],axis=0))**2).sum()/((sum(P[:,:,k],axis=0))**2).sum()
        pos_err = abs(A.dot(pivot_loc)-b)
    print 'Multimarg bias: ',bias1
    print 'Common marg bias: ',bias2
    print 'Location error: ',pos_err

    bar=sum(P[:,:,0],axis=0)
    return P,bar


def beg_proj_transp_ptbar(dist_mat,next_mat,distribs,weights,inf_val=None,beta=500,nb_iter=100):
    shap = distribs.shape
    bar = distribs[0,:].reshape(shap[1],1)
    wbar = weights[0]
    wold = 0
    for i in range(1,shap[0]):
        distrib2 = distribs[i,:].reshape(shap[1],1)
        P,dist = beg_proj_transp(double(dist_mat),bar,distrib2,beta=beta,nb_iter=nb_iter)
        t = weights[i]/(wbar+weights[i])
        bar = displacement_interp_graph(P,sqrt(dist_mat),next_mat,t)
        wbar = wbar+weights[i]
    return bar

def graph_transport_ptbar(dist_mat,next_mat,graph_ind,S,target_feat,anchor_ind,nb_loops=1,tol=0.001,lin_prog_en=False,beta=60): # The distribution are in distribs columns;  they are assumed to be sorted in increasing order of distance to the target location of interpolation
    nb_src = len(graph_ind)
    starting_ind = graph_ind[0]
    target_ind = graph_ind[1]
    path = utils.get_path(next_mat,starting_ind,target_ind)
    ind_activ = zeros((len(path)+1,))

    ind_activ[0] = starting_ind
    ind_activ[1:] = array(path)

    distrib_final = zeros((dist_mat.shape[0],1))
    distrib_final[starting_ind] = 1
    i=0
    err = 100
    ind_activ = ind_activ.astype(int)
    l_ind_activ = list(ind_activ)
    for l in range(0,nb_loops):
        while i < nb_src-1 and err >tol:
            distrib_1 = distrib_final[ind_activ]
            distrib_2 = zeros((len(distrib_1),1))
            distrib_2[-1,0] = 1
            dist_mat_in = dist_mat[ix_(ind_activ,ind_activ)]
            if lin_prog_en:
                mapping,optim_res = lin_prog_transp(distrib_1.reshape((len(distrib_1),)),distrib_2.reshape((len(distrib_2),)),double(dist_mat_in))
            else:
                mapping,sinkh_dist = beg_proj_transp(double(dist_mat_in),distrib_1,distrib_2,inf_val=None,beta=beta,nb_iter=50)
            interp_opt,err = acc_displacement_interp_graph(mapping,dist_mat,next_mat,S,anchor_ind,target_feat,samp=10,tol=tol,nb_iter_max = 10,ind_cv=ind_activ)

            distrib_final = interp_opt.reshape((dist_mat.shape[0],1))
            if l==0:
                if i < nb_src-2:
                    for k in range(0,len(ind_activ)):

                        path = utils.get_path(next_mat,ind_activ[k],graph_ind[i+2])
                        for z in range(0,len(path)):
                            if path[z] not in l_ind_activ:
                                l_ind_activ.append(path[z])
                ind_activ = array(l_ind_activ)
                ind_activ = ind_activ.astype(int)
            i+=1

    Sout = S.dot(distrib_final)
    return Sout,distrib_final


def displacement_interp_graph(P,geod_dist,next_mat,t,ind_cv=None):
    shap = P.shape
    if ind_cv is None:
        ind_cv = range(0,shap[0])
    interpt = copy(diag(geod_dist))*0
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            dist = geod_dist[ind_cv[i],ind_cv[j]]*t
            if P[i,j]>0:
                if i!=j:
                    path = utils.get_path(next_mat,ind_cv[i],ind_cv[j])
                    dist_cur= geod_dist[ind_cv[i],path[0]]
                    ind=0
                    while(dist_cur<dist):
                        ind+=1
                        dist_cur= geod_dist[ind_cv[i],path[ind]]
                    ind_1 = ind_cv[i]
                    if ind>0:
                        ind_1 = path[ind-1]
                    delta = geod_dist[ind_1,path[ind]]
                    deltat = dist - geod_dist[ind_cv[i],ind_1]
                    w = deltat/delta
                    #print w,ind_1,path[ind]
                    if w>1 or w<0:
                        "warning: bug in displacement interpolation on the graph"
                    interpt[ind_1]+=(1-w)*P[i,j]
                    interpt[path[ind]]+=w*P[i,j]
                else:
                    interpt[ind_cv[i]] = P[i,i]
    #interpt = interpt.reshape((shap[0],1))
    #k = where(interpt>0)
    #print k
    return interpt

def displacement_interp_graph_samp(P,geod_dist,next_mat,t,ind_cv=None):
    nb_points = len(t)
    shap = geod_dist.shape
    interpt = zeros((shap[0],nb_points))
    for i in range(0,nb_points):
        interpt[:,i] = displacement_interp_graph(P,geod_dist,next_mat,t[i],ind_cv=ind_cv)
    return interpt

def acc_displacement_interp_graph(P,geod_dist,next_mat,S,anchor_ind,target_feat,samp=4,tol=0.01,nb_iter_max = 1,ind_cv=None):
    range_ref = array(range(0,samp+1))
    ti = 0.
    te = 1.
    range_i = ((te-ti)/samp)*range_ref +ti
    err = 100
    nb_iter = 0
    ones_vect = ones((1,samp+1))
    target_feat_mat = target_feat.dot(ones_vect)
    interp_opt = None
    while err>tol and nb_iter<nb_iter_max:
        interpt = displacement_interp_graph_samp(P,geod_dist,next_mat,range_i,ind_cv=ind_cv)
        target_interp = S[anchor_ind,:].dot(interpt)
        err_sq = ((target_interp - target_feat_mat)**2).sum(axis=0)
        ind =   argsort(err_sq)
        err = err_sq[ind[0]]/sum(target_feat**2)
        interp_opt = copy(interpt[:,ind[0]])
        ti = min(range_i[ind[0]],range_i[ind[1]])
        te = max(range_i[ind[0]],range_i[ind[1]])
        #print range_i
        #print "ti, te, err",range_i[ind[0]]," ",range_i[ind[1]]," ",err," tol: ",tol
        #print err_sq
        range_i = ((te-ti)/samp)*range_ref +ti
        nb_iter+=1
        #print where(interp_opt>0)
    return interp_opt,err

def low_rank_deconv(im_stack,psf_stack,mu,ksig,nb_iter=100,nb_rw=2):
    shap = im_stack.shape
    w = ones((shap[2],))
    mat_stack = zeros((shap[2],shap[0]*shap[1])) # Images correspond to lines
    for i in range(0,shap[2]):
        mat_stack[i,:] = im_stack[:,:,i].reshape((shap[0]*shap[1],))
    U, s, Vt = linalg.svd(mat_stack,full_matrices=True)
    sigma = s[int(min(shap[0]*shap[1])/2)]
    #n=max(shap0[0],shap0[1])
    nu = ksig*sigma*ones((shap[2],))
    lip_coeff = zeros((shap[2],))
    psf_stack_adj = copy(psf_stack)
    psf_norm_2 = zeros((shap[2],))
    thresh_type=1

    for i in range(0,shap[2]):
        lip_coeff[i] = (abs(psf_stack[:,:,i]).sum())**2
        psf_stack_adj[:,:,i] = rot90(psf_stack[:,:,i],2)
        psf_norm_2[i] = sqrt((psf_stack[:,:,i]**2).sum())

    thresh_coeff = max(mu*psf_norm_2/lip_coeff)
    nu = thresh_coeff*nu
    grad = copy(im_stack)
    res = copy(im_stack)
    im_deconv = im_stack*0
    mat_deconv = mat_stack*0
    mse = zeros((nb_iter,))

    for l in range(0,nb_rw+1):
        for i in range(0,nb_iter):
            for k in range(0,shap[2]):
                res[:,:,k] = im_stack[:,:,k]-scisig.fftconvolve(im_deconv[:,:,k],psf_stack[:,:,k],mode='same')
                mse[i] += (res[:,:,k]**2).sum()
                grad[:,:,k] = -scisig.convolve(res[:,:,k],psf_stack_adj[:,:,i],mode='same')
            print mse[i]
            # -------- Gradient step -------- #
            for k in range(0,shap[2]):
                im_deconv[:,:,k] = im_deconv[:,:,k] - (mu/lip_coeff[k])*grad[:,:,k]
                mat_deconv[k,:] = im_deconv[:,:,k].reshape((shap[0]*shap[1],))

            # ------ Low rank constraint ---- #
            mat_deconv,si,U,Vt = nuc_norm_thresh(mat_deconv,nu*w,thresh_type)
            for k in range(0,shap[2]):
                im_deconv[:,:,k] = mat_deconv[k,:].reshape((shap[0],shap[1]))
        U, s, Vt = linalg.svd(mat_deconv,full_matrices=False)
        w = 1/((s/(thresh_coeff*ksig*sigma))+1)

    return im_deconv,mat_deconv,res


def low_rank_sr(im_stack,mu,upfact,klambd,shifts=None,nb_iter=100,nb_rw=1,lanc_rad=10):
    shap = im_stack.shape
    w = ones((shap[2],))
    mat_stack = zeros((shap[2],shap[0]*shap[1])) # Images correspond to lines
    if shifts is None:
        shifts = utils.shift_est(im_stack)*upfact

    for i in range(0,shap[2]):
        mat_stack[i,:] = im_stack[:,:,i].reshape((shap[0]*shap[1],))
    U, s, Vt = linalg.svd(mat_stack,full_matrices=True)

       #n=max(shap0[0],shap0[1])
    nu = klambd*s[0]*ones((shap[2],))
    nu[0]=0
    nu[1]=0
    lip_coeff = zeros((shap[2],))

    shift_ker_stack = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))
    shift_ker_stack_adj = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))
    ker_norm_2 = zeros((shap[2],))
    thresh_type=1

    for i in range(0,shap[2]):
        uin = shifts[i,:].reshape((1,2))*upfact
        shift_ker_stack[:,:,i] = utils.lanczos(uin,n=lanc_rad)
        lip_coeff[i] = (abs(shift_ker_stack[:,:,i]).sum())**2
        shift_ker_stack_adj[:,:,i] = rot90(shift_ker_stack[:,:,i],2)
        ker_norm_2[i] = sqrt((shift_ker_stack[:,:,i]**2).sum())

    thresh_coeff = max(mu*ker_norm_2/lip_coeff)
    nu = thresh_coeff*nu
    grad = zeros((shap[0]*upfact,shap[1]*upfact,shap[2]))
    res = zeros((shap[0],shap[1],shap[2]))
    im_hr = zeros((shap[0]*upfact,shap[1]*upfact,shap[2]))
    mat_hr = zeros((shap[2],shap[0]*shap[1]*upfact**2))
    mse = zeros((nb_iter,))

    for l in range(0,nb_rw+1):
        for i in range(0,nb_iter):
            if l>0:
                mse[i]=0
            for k in range(0,shap[2]):
                res[:,:,k] = im_stack[:,:,k]-utils.decim(scisig.fftconvolve(im_hr[:,:,k],shift_ker_stack[:,:,k],mode='same'),upfact,av_en=0)
                mse[i] += (res[:,:,k]**2).sum()
                grad[:,:,k] = -scisig.convolve(utils.transpose_decim(res[:,:,k],upfact),shift_ker_stack_adj[:,:,k],mode='same')
            print mse[i]
            # -------- Gradient step -------- #
            for k in range(0,shap[2]):
                im_hr[:,:,k] = im_hr[:,:,k] - (mu/lip_coeff[k])*grad[:,:,k]
                mat_hr[k,:] = im_hr[:,:,k].reshape((shap[0]*shap[1]*upfact**2,))

            # ------ Low rank constraint ---- #
            mat_hr,si,U,Vt = nuc_norm_thresh(mat_hr,nu*w,thresh_type)
            for k in range(0,shap[2]):
                im_hr[:,:,k] = mat_hr[k,:].reshape((shap[0]*upfact,shap[1]*upfact))
        U, s, Vt = linalg.svd(mat_hr,full_matrices=False)
        w[2:] = 1/((s[2:]/nu[2:])+1)

    return im_hr,mat_hr,res


def robust_stacking(im_stack,mu,upfact,opt,shifts=None,nsig=4,nb_iter=100,nb_rw=2,lanc_rad=10):

    shap = im_stack.shape
    w = ones((shap[2],))
    if shifts is None:
        shifts = utils.shift_est(im_stack)*upfact

    lip_coeff = zeros((shap[2],))

    shift_ker_stack = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))
    shift_ker_stack_adj = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))
    ker_norm_2 = zeros((shap[2],))
    thresh_type=1

    for i in range(0,shap[2]):
        uin = shifts[i,:].reshape((1,2))*upfact
        shift_ker_stack[:,:,i] = utils.lanczos(uin,n=lanc_rad)
        lip_coeff[i] = (abs(shift_ker_stack[:,:,i]).sum())**2
        shift_ker_stack_adj[:,:,i] = rot90(shift_ker_stack[:,:,i],2)
        ker_norm_2[i] = sqrt((shift_ker_stack[:,:,i]**2).sum())

    lip_coeff_t = lip_coeff.sum()
    thresh_coeff = max(mu*ker_norm_2/lip_coeff)
    grad = zeros((shap[0]*upfact,shap[1]*upfact))
    res = zeros((shap[0],shap[1],shap[2]))
    im_hr = zeros((shap[0]*upfact,shap[1]*upfact))
    mse = zeros((nb_iter,))

    weights = None
    coeff_init = None
    mr_file = None

    for l in range(0,nb_rw+1):
        nb_subiter = 50
        for i in range(0,nb_iter):
            if i >0:
                nb_subiter=5
            if l>0:
                mse[i]=0
            for k in range(0,shap[2]):
                res[:,:,k] = im_stack[:,:,k]-utils.decim(scisig.fftconvolve(im_hr,shift_ker_stack[:,:,k],mode='same'),upfact,av_en=0)
                mse[i] += (res[:,:,k]**2).sum()
                grad = grad -scisig.convolve(utils.transpose_decim(res[:,:,k],upfact),shift_ker_stack_adj[:,:,k],mode='same')
            print mse[i]
            # -------- Gradient step -------- #
            im_hr = im_hr - (mu/lip_coeff_t)*grad

            # ---- Analysis constraint ---- #
            # Wavelet noise estimation
            sig_map = utils.res_sig_map((mu/lip_coeff_t)*grad,opt=opt)
            grad=grad*0
            thresh_map = nsig*sig_map
            if weights is not None:
                thresh_map = thresh_map*weights
            im_hr,mr_file,n,coeff_init = wvl_analysis_op(im_hr,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter)
        # ---- Weights update ---- #
        coeffx,mr_file = isap.mr_trans(im_hr,opt=opt)
        weights  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)

    return im_hr,res


def low_rank_sr_mod(im_stack,model,mu,upfact,ksig,nb_iter=100,nb_rw=2,lanc_rad=10):
    shap = im_stack.shape
    w = ones((shap[2],))
    mat_stack = zeros((shap[2],shap[0]*shap[1])) # Images correspond to lines
    shifts = utils.int_grid_shift(im_stack)*upfact

    for i in range(0,shap[2]):
        mat_stack[i,:] = im_stack[:,:,i].reshape((shap[0]*shap[1],))
    U, s, Vt = linalg.svd(mat_stack,full_matrices=True)

    sigma = s[int(min(shap[0],shap[1])/2)]
    #n=max(shap0[0],shap0[1])
    nu = ksig*sigma*ones((shap[2],))
    lip_coeff = zeros((shap[2],))

    shift_ker_stack = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))
    shift_ker_stack_adj = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))
    ker_norm_2 = zeros((shap[2],))
    thresh_type=1

    for i in range(0,shap[2]):
        uin = shifts[i,:].reshape((1,2))*upfact
        shift_ker_stack[:,:,i] = utils.lanczos(uin,n=lanc_rad)
        lip_coeff[i] = (abs(shift_ker_stack[:,:,i]).sum())**2
        shift_ker_stack_adj[:,:,i] = rot90(shift_ker_stack[:,:,i],2)
        ker_norm_2[i] = sqrt((shift_ker_stack[:,:,i]**2).sum())

    thresh_coeff = max(mu*ker_norm_2/lip_coeff)
    nu = thresh_coeff*nu
    grad = zeros((shap[0]*upfact,shap[1]*upfact,shap[2]))
    res = zeros((shap[0],shap[1],shap[2]))
    im_hr = zeros((shap[0]*upfact,shap[1]*upfact,shap[2]))
    delta_hr = zeros((shap[0]*upfact,shap[1]*upfact,shap[2]))
    mat_hr = zeros((shap[2],shap[0]*shap[1]*upfact**2))
    mse = zeros((nb_iter,))

    for l in range(0,nb_rw+1):
        for i in range(0,nb_iter):
            if l>0:
                mse[i]=0
            for k in range(0,shap[2]):
                res[:,:,k] = im_stack[:,:,k]-utils.decim(scisig.fftconvolve(delta_hr[:,:,k]+model,shift_ker_stack[:,:,k],mode='same'),upfact,av_en=0)
                mse[i] += (res[:,:,k]**2).sum()
                grad[:,:,k] = -scisig.convolve(utils.transpose_decim(res[:,:,k],upfact),shift_ker_stack_adj[:,:,k],mode='same')
            print mse[i]
            # -------- Gradient step -------- #
            for k in range(0,shap[2]):
                delta_hr[:,:,k] = delta_hr[:,:,k] - (mu/lip_coeff[k])*grad[:,:,k]
                mat_hr[k,:] = delta_hr[:,:,k].reshape((shap[0]*shap[1]*upfact**2,))

            # ------ Low rank constraint ---- #
            mat_hr,si,U,Vt = nuc_norm_thresh(mat_hr,nu*w,thresh_type)
            for k in range(0,shap[2]):
                delta_hr[:,:,k] = mat_hr[k,:].reshape((shap[0]*upfact,shap[1]*upfact))
        U, s, Vt = linalg.svd(mat_hr,full_matrices=False)
        w = 1/((s/(thresh_coeff*ksig*sigma))+1)

    for k in range(0,shap[2]):
        im_hr[:,:,k] = delta_hr[:,:,k]+model
    return im_hr,mat_hr,res,delta_hr

def proj_tg_sph(y,x,o,r): # Projects y on a the tangent plane at x on a sphere centered on o, with a radius r
    a = ((y-x)*(x-o)).sum()/r**2
    projy = y - a*(x-o)
    return projy

def KL(x,y,inf_val=double(2)**63):
    id = where(y>0)
    kl_d = (x[id]*(log(x[id]/y[id])-1)+y[id]).sum()
    id_check = where(x>0)
    y_check = x[id_check]
    l = size(where(y_check==0))
    if l>0:
        kl_d = inf_val
    return kl_d

def kl_proj_sphere(y,o,r,mu=1,nb_iter=40,tol=10**(-10)): # Finds argmin_x KL(x|y) s.t. ||y-x||_2 = r; O is assumed to be a dirac mass
    xout = proj_sphere(y,o,r)
    io = where(o>0)
    if o[io]>=r and y[io]==0:
        return xout
    else:
        xout = y
        id = where(y>0)
        #step = mu*r/sqrt((y**2).sum())
        grad = zeros((size(y),))
        des_dir = zeros((size(y),))
        i=0
        kl_old = KL(xout,y)
        kl = kl_old+1
        while (i<nb_iter and abs(kl-kl_old)/kl >tol):
            #print xout[id].min()
            grad[id] = proj_tg_sph(log(xout[id]/y[id]),xout[id],o[id],r)
            des_dir[id] = -xout[id]*grad[id]
            step,m,count = armijo_step_kl(grad,des_dir,xout,o,r,y)
            xtemp = xout+step
            #print xtemp.min()
            xout = proj_sphere(xtemp,o,r)
            kl_old = kl
            kl = KL(xout,y)
            #print "divergence: ",kl," armijo power: ",m," count: ",count
            i+=1
        return xout

def kl_proj_sphere_mat(y,o,r,mu=0.01,nb_iter=40):
    shap = y.shape
    output = y*0
    bias = zeros((size(r),))
    for i in range(0,shap[0]):

        disti = sqrt(((y[i,:]-o[i,:])**2).sum())
        #print i,"/",shap[0]," r = ",r[i]," disti = ",disti
        if disti <= r[i]:
            output[i,:] = y[i,:]
            bias[i] = r[i]-disti
        else:
            yi = kl_proj_sphere(y[i,:],o[i,:],r[i],mu=mu,nb_iter=nb_iter)
            disti = sqrt(((yi-o[i,:])**2).sum())
            bias[i] = r[i]-disti
            output[i,:]=yi
    return output,bias


def armijo_step_kl(grad,des_dir,xk,o,r,y,sigma=0.5,beta=0.5,alpha=1,tol=10**(-30),count_max=50):
    m=0
    a = True
    i = where(y>0)
    n = (grad*des_dir).sum()
    ind = where(y>0)

    cur_cost = (xk[ind]*(log(xk[ind]/y[ind])-1)).sum()
    xtemp = zeros((size(y),))
    nb_iter=0
    while a==True and nb_iter<count_max:
        rhs = -sigma*alpha*(beta**m)*n
        xtemp[ind] = proj_sphere(xk[ind]+alpha*(beta**m)*des_dir[ind],o[ind],r)
        ind_neg = where(xtemp[ind]<=0)
        if size(ind_neg)==0:
            cost_updt = (xtemp[ind]*(log(xtemp[ind]/y[ind])-1)).sum()
            if cur_cost-cost_updt>=rhs or abs(cur_cost-cost_updt)/cur_cost <tol:
                a=False
            else:
                m+=2
                #print "rhs: ",rhs
                #print "cur_cost-cost_updt = ",cur_cost-cost_updt
        else:
            m+=2
        nb_iter+=1
        #else:
            #print "size(ind_neg) = ",size(ind_neg)
    step = alpha*(beta**m)*des_dir
    return step,m,nb_iter

def pow_meth(opname,op_param,siz,tol=0.5,ainit=None,nb_iter_max=30,opt_vect=None):

    a = None
    if ainit is not None:
        a = ainit
    else:
        if size(siz)==1:
            a = random.randn(siz)
        elif size(siz)==2:
            a = random.randn(siz[0],siz[1])
        elif size(siz)==3:
            a = random.randn(siz[0],siz[1],siz[2])
        a = a/sqrt(((a)**2).sum())

    L_old = 10
    L=1
    i=0
    print "----------- spec_rad est ---------- : "

    while i<nb_iter_max and (100*abs(L-L_old))/L>tol:
        print L
        op_param[0] = a
        b = opname(op_param)
        if opt_vect is not None:
            b-=sum(b*opt_vect)*opt_vect
        L_old = L
        L = sqrt(((b)**2).sum())
        a = b/L
        i+=1

    if i==nb_iter_max:
        print "Warning max number if iterations reached in pow_meth"
    return a,L

def op_eig_space(input): # Let A be a matrix representing op_name, this function applies I-A/eig_val, eig_val being an non zero eigenvalue of A
    op_name = input[-2]
    eig_val = input[-1]
    out = input[0] - op_name(input)/eig_val
    return out

def min_eig_val(opname,op_param,siz,tol=0.5,eig_max=None,eig_min_init=None,nb_iter_max=30,L_max=None): # The operation is supposed to be positive and non singular


    eig_max,L_max = pow_meth(opname,op_param,siz,tol=tol,ainit=eig_max,nb_iter_max=nb_iter_max)
    op_param2 = cp.deepcopy(op_param)
    op_param2.append(opname)
    op_param2.append(L_max)
    eig_min,coeff_min = pow_meth(op_eig_space,op_param2,siz,tol=tol,ainit=eig_min_init,nb_iter_max=nb_iter_max)
    # Smaller eigenvalue estimation
    print "Coeff_min: ",coeff_min
    L_min = L_max*(1-coeff_min)
    return eig_min,L_min

def conjugate_grad(op,op_param,nb_iter=100,tol=10**(-10)):
    target_val = copy(op_param[0])
    op_param_in = cp.deepcopy(op_param)
    x = target_val*0
    r = copy(target_val)
    p = copy(r)
    temp1 = (r**2).sum()
    temp1_old = temp1
    i=0
    err_rel = 1
    op_param_test = cp.deepcopy(op_param)
    while i<nb_iter and err_rel>tol:
        op_param_in[0] = p
        temp2 = op(op_param_in)
        alpha = temp1/((temp2*p).sum())
        x = x + alpha*p
        r = r - alpha*temp2
        op_param_test[0] = x
        temp3 = op(op_param_test)
        err_rel = ((temp3-target_val)**2).sum()/(target_val**2).sum()

        temp1_old = temp1
        temp1 = (r**2).sum()
        beta = temp1/temp1_old
        p = r + beta*p
        i+=1

    print "res: ",err_rel
    x = x + alpha*p
    return x


def trans_sr_op(input):
    x = input[0]
    upfact = input[1]
    ker_adj = input[2]
    output = scisig.convolve(utils.transpose_decim(x,upfact),ker_adj,mode='same')
    return output

def sr_op(input):
    x = input[0]
    upfact = input[1]
    ker = input[2]
    output = utils.decim(scisig.fftconvolve(x,ker,mode='same'),upfact,av_en=0)
    return output

def sr_op_trans(input): # AA^T
    x = input[0]
    upfact = input[1]
    ker = input[2]
    ker_adj = input[3]

    input_trans = list()
    input_trans.append(copy(x))
    input_trans.append(upfact)
    input_trans.append(ker_adj)
    output_temp = trans_sr_op(input_trans)

    input_op = list()
    input_op.append(copy(output_temp))
    input_op.append(upfact)
    input_op.append(ker)
    output = sr_opt(input_op)

    return output

def conv_op_trans(input): # M^TM M being a convolutive operator
    x = input[0]
    ker = input[1]
    ker_adj = input[2]
    #output = scisig.fftconvolve(scisig.fftconvolve(x,ker,mode='same'),ker_adj,mode='same')
    output = scisig.convolve2d(scisig.convolve2d(x,ker,mode='same'),ker_adj,mode='same')
    return output

def conv_pseudo_inv(y,ker,ker_adj,nb_iter=20,tol=10**(-15)):
    x = scisig.fftconvolve(y,ker_adj,mode='same')
    input = list()
    input.append(x)
    input.append(ker)
    input.append(ker_adj)
    out = conjugate_grad(conv_op_trans,input,nb_iter=nb_iter,tol=tol)
    return out


def sr_op_trans_stack(input): # sum_i w_i*A_i^TA_i
    x = input[0]
    upfact = input[1]
    ker = input[2]
    ker_adj = input[3]
    shap = ker_adj.shape
    w = input[4]
    sig = input[5]
    flux = input[6]
    flux_ref = np.median(flux)
    output = x*0

    for i in range(0,shap[2]):
        input_op = list()
        input_op.append(x)
        input_op.append(upfact)
        input_op.append(ker[:,:,i])
        output_temp = sr_op(input_op)

        input_trans = list()
        input_trans.append(output_temp)
        input_trans.append(upfact)
        input_trans.append(ker_adj[:,:,i])
        output = output + ((w[i]*flux_ref/(flux[i]*sig[i]))**2)*trans_sr_op(input_trans)

    return output

def sr_stack_op(input):
    S = input[0]
    upfact = input[1]
    ker = input[2]
    ker_adj = input[3]
    shap = ker_adj.shape
    A = input[4]
    sig = input[5]
    flux = input[6]
    flux_ref = np.median(flux)
    nb_im = shap[2]
    shapS = S.shape
    output = zeros((shapS[0]/upfact,shapS[1]/upfact,nb_im))
    for i in range(0,nb_im):
        im_i = zeros((shapS[0],shapS[1]))
        for j in range(0,shapS[2]):
            im_i+=S[:,:,j]*A[j,i]
        output[:,:,i] = (flux[i]/(flux_ref*sig[i]))*utils.decim(scisig.fftconvolve(im_i,ker[:,:,i],mode='same'),upfact,av_en=0)
    return output

def sr_stack_trans(input):
    y = input[0]
    upfact = input[1]
    ker = input[2]
    ker_adj = input[3]
    A = input[4]
    shapA = A.shape
    sig = input[5]
    flux = input[6]
    flux_ref = np.median(flux)
    shapy = y.shape
    output = zeros((shapy[0]*upfact,shapy[1]*upfact,shapA[0]))
    nb_im = shapy[2]

    for i in range(0,nb_im):
        im_i = (flux[i]/(flux_ref*sig[i]))*scisig.convolve(utils.transpose_decim(y[:,:,i],upfact),ker_adj[:,:,i],mode='same')
        for j in range(0,shapA[0]):
            output[:,:,j] += im_i*A[j,i]
            #print "========> param optim :",(flux[i]/(flux_ref*sig[i]))
    return output

def sr_stack_trans_op_src(input): # M^TM
    output1 = sr_stack_op(input)
    input2 = cp.deepcopy(input)
    input2[0] = output1
    output = sr_stack_trans(input2)
    return output


def sr_stack_trans_op_cons_src(input): # M^TM+ AA^T
    output1 = sr_stack_trans_op_src(input)
    S = input[0]
    A = input[4]
    A2 = A.dot(transpose(A))
    output2 = output1*0
    shap = A2.shape

    for i in range(0,shap[1]):
        for j in range(0,shap[0]):
            output2[:,:,i] += S[:,:,j]*A2[j,i]
    return output1+output2


def sr_op_stack_pseudo_inv(input):
    stack = input[0]
    upfact = input[1]
    ker = input[2]
    ker_adj = input[3]
    shap = stack.shape
    w = input[4]
    sig = input[5]
    flux = input[6]
    flux_ref = np.median(flux)
    output_temp = zeros((shap[0]*upfact,shap[1]*upfact))

    for i in range(0,shap[2]):
        input_trans = list()
        input_trans.append(copy(stack[:,:,i]))
        input_trans.append(upfact)
        input_trans.append(copy(ker_adj[:,:,i]))
        output_temp = output_temp + (w[i]*flux_ref/(flux[i]*sig[i]))*trans_sr_op(input_trans)

    # Inversion
    input_2 = list()
    input_2.append(output_temp)
    input_2.append(upfact)
    input_2.append(copy(ker))
    input_2.append(copy(ker_adj))
    input_2.append(copy(w))
    input_2.append(copy(sig))
    input_2.append(copy(flux))

    output = conjugate_grad(sr_op_trans_stack,input_2,nb_iter=25,tol=10**(-5))

    return output


def shift_assign(shifts,input_list,lanc_rad=10):
    shap = len(shifts)
    shifts_in = shifts.reshape((shap/2,2))
    ker,ker_stack = utils.shift_ker_stack(shifts_in,1,lanc_rad=lanc_rad)
    input = cp.deepcopy(input_list)
    input[2] = copy(ker)
    input[3] = copy(ker_stack)
    return input

def sr_shift_penalty(input):
    x_opt = sr_op_stack_pseudo_inv(input)
    stack = input[0]
    upfact = input[1]
    ker = input[2]
    ker_adj = input[3]
    shap = stack.shape
    w = input[4]
    sig = input[5]
    flux = input[6]
    flux_ref = np.median(flux)

    cost = 0
    for i in range(0,shap[2]):
        input_op = list()
        input_op.append(copy(x_opt))
        input_op.append(upfact)
        input_op.append(ker[:,:,i])
        output_temp = sr_op(input_op)
        res = stack[:,:,i] - (w[i]*flux_ref/(flux[i]*sig[i]))*output_temp
        cost+=(res**2).sum()

    return cost

def shifts_optim(im_stack,upfact,shifts_0=None,sig=None,nsig_shift_est=5,flux=None,w=None,lanc_rad=10,shifts_range=None,nb_iter=1):


    shap = im_stack.shape

    if shifts_0 is None:
        sig_est=None
        if sig is None:
            sig_est = utils.im_gauss_nois_est_cube(im_stack)
        else:
            sig_est = sig
        #shifts = utils.shift_est(psf_stack)*upfact
        anchor = double([shap[0],shap[0]])/(2*upfact)
        map = ones((shap[0],shap[1],shap[2]))
        for i in range(0,shap[2]):
            map[:,:,i] *= nsig_shift_est*sig_est[i]
        print 'First guess estimation...'
        im_stack_shift = utils.thresholding_3D(im_stack,map,0)
        shifts_0 = -utils.shift_est_2_anch(im_stack_shift,anchor)*upfact
        print 'Done...'

    ker,ker_adj = utils.shift_ker_stack(shifts_0,1,lanc_rad=lanc_rad)

    if sig is None:
        sig = ones((shap[2],))
    if flux is None:
        flux = ones((shap[2],))
    if w is None:
        w = ones((shap[2],))



    input = list()
    input.append(im_stack)
    input.append(upfact)
    input.append(copy(ker))
    input.append(copy(ker_adj))
    input.append(copy(w))
    input.append(copy(sig))
    input.append(copy(flux))
    shifts_0_in = shifts_0.reshape((shap[2]*2,))

    if shifts_range is None:
        list_range = list()
        for i in range(0,2*shap[2]):
            list_range.append((shifts_0_in[i]-upfact/2,shifts_0_in[i]+upfact/2))
        shifts_range = tuple(list_range)
    print shifts_range

    cost_eval = lambda shifts : sr_shift_penalty(shift_assign(shifts,input,lanc_rad=lanc_rad))
    print "Initial cost: ",cost_eval(shifts_0_in)
    print 'Shifts estimation...'
    opt = {'maxiter': nb_iter,'disp':True}
    res = minimize(cost_eval,shifts_0_in,bounds = shifts_range,options={'maxiter': 2, 'disp': True})
    print 'Done...'
    shifts_est = res.x
    print "Final cost: ",cost_eval(shifts_est)

    sr_image = sr_op_stack_pseudo_inv(shift_assign(shifts_est,input,lanc_rad=lanc_rad))

    shifts_est = shifts_est.reshape((shap[0],shap[1]))

    shift_assign(shifts,input_list,lanc_rad=10)

    return res,shifts_est,sr_image





def sr_op_trans_stack_translate(input): # Evaluates (id + lambd*M^TM)x
    lambd = input[-1]
    out = input[0]+lambd*sr_op_trans_stack(input)
    return out

def sr_op_trans_stack_translate_inv(input,nb_iter,tol=10**(-10)): # Evaluates (id + lambd*M^TM)^-1 X
    out = conjugate_grad(sr_op_trans_stack_translate,input,nb_iter=nb_iter,tol=tol)
    return out

def sr_op_trans_stack_prox(input,nb_iter,tol=10**(-15)): # Evaluates the proximity operator of lambd*||y-Mx||^2 at z
    lambd = input[-3]
    z = input[-2]
    lr_stack = input[-1] # LR data
    shap = lr_stack.shape
    temp = z*0
    upfact = input[1]  # Refer to the above sr_op functions for the convention on the structure function
    ker_adj = input[3]
    w = input[4]
    sig = input[5]
    flux = input[6]
    flux_ref = np.median(flux)
    input_trans = list()
    input_trans.append(copy(lr_stack[:,:,0]))
    input_trans.append(upfact)
    input_trans.append(copy(ker_adj[:,:,0]))

    for i in range(0,shap[2]):
        input_trans[0] = copy(lr_stack[:,:,i])
        input_trans[2] = copy(ker_adj[:,:,i])
        temp = temp + (w[i]*flux_ref/(flux[i]*sig[i]))*trans_sr_op(input_trans)

    input[0] = z+lambd*temp
    input_loc = input[:-2]
    out = conjugate_grad(sr_op_trans_stack_translate,input_loc,nb_iter=nb_iter,tol=tol)
    return out

def sr_op_trans_stack_pseudo_inv(x,ker,ker_adj,up_fact,w,sig,flux,nb_iter,tol=10**(-4)):
    input = list()
    shap = x.shape
    input.append(x)
    input.append(up_fact)
    input.append(ker)
    input.append(ker_adj)
    input.append(w)
    input.append(sig)
    input.append(flux)
    flux_ref = np.median(flux)
    y = zeros((shap[0]*up_fact,shap[1]*up_fact))
    input_trans = list()
    input_trans.append(copy(x[:,:,0]))
    input_trans.append(up_fact)
    input_trans.append(copy(ker_adj[:,:,0]))

    for i in range(0,shap[2]):
        input_trans[0] = copy(x[:,:,i])
        input_trans[2] = copy(ker_adj[:,:,i])
        y = y+((w[i]*flux_ref/(flux[i]*sig[i])))*trans_sr_op(input_trans)
    input[0]=y
    out = conjugate_grad(sr_op_trans_stack,input,nb_iter=nb_iter,tol=tol)

    return out

"""def sr_op_trans_stack_pseudo_inv_ortho(x,ker,ker_adj,up_fact,w,sig,flux,nb_iter,src,tol=10**(-4)):
    input = list()
    shap = x.shape
    input.append(x)
    input.append(up_fact)
    input.append(ker)
    input.append(ker_adj)
    input.append(w)
    input.append(sig)
    input.append(flux)
    flux_ref = np.median(flux)
    y = zeros((shap[0]*up_fact,shap[1]*up_fact))
    input_trans = list()
    input_trans.append(copy(x[:,:,0]))
    input_trans.append(up_fact)
    input_trans.append(copy(ker_adj[:,:,0]))

    for i in range(0,shap[2]):
        input_trans[0] = copy(x[:,:,i])
        input_trans[2] = copy(ker_adj[:,:,i])
        y = y+((w[i]*flux_ref/(flux[i]*sig[i])))*trans_sr_op(input_trans)
    input[0]=copy(y)
    lsq = conjugate_grad(sr_op_trans_stack,input,nb_iter=nb_iter,tol=tol)
    shap = src.shape
    temp = copy(src)
    if size(shap)>2:
        for i in range(0,shap[2]):

            temp[:,:,i] =
return out"""

def sr_op_equal_cons(y,target_im,upfact,ker,ker_adj,nb_iter=100): # Projection of y onto the contraint target_im = D*W*x where D and W are respectively shift and downsampling operators

    u = utils.decim(scisig.fftconvolve(y,ker,mode='same'),upfact,av_en=0)-target_im

    op_param = list()
    op_param.append(copy(u))
    op_param.append(upfact)
    op_param.append(ker)
    op_param.append(ker_adj)
    v = conjugate_grad(conjugate_grad,op_param,nb_iter=nb_iter)

    proj = y - scisig.convolve(utils.transpose_decim(v,upfact),ker_adj,mode='same')
    err = ((utils.decim(scisig.fftconvolve(proj,ker,mode='same'),upfact,av_en=0)-target_im)**2).sum()/(target_im**2).sum()
    print "Projection error: ",err
    return proj

def sr_op_inequal_cons(y,target_im,upfact,ker,ker_adj,sigma,u_init=None,nb_iter=100):

    shap = target_im.shape
    i = 0
    if u_init is None:
        u_init = zeros(shap)
    o = zeros(shap)
    u = copy(u_init)
    p = y - scisig.convolve(utils.transpose_decim(u_init,upfact),ker_adj,mode='same')
    mu = 1/(abs(ker).sum())
    for i in range(0,nb_iter):
        z = u/mu + utils.decim(scisig.fftconvolve(p,ker,mode='same'),upfact,av_en=0)-target_im
        u = mu*(z-proj_sphere(z,o,sigma))
        p = alpha - scisig.convolve(utils.transpose_decim(u,upfact),ker_adj,mode='same')
    print 'radius :',sigma,' Current dist: ',sqrt(((target_im-utils.decim(scisig.fftconvolve(p,ker,mode='same'),upfact,av_en=0))**2).sum())
    u_init = u
    return p,u_init

def lowrank_sr_eq_cons_dr(psf_stack,shift,upfact,nb_iter=100,mu=1.0,n_eig=3): # n_eig: order of the reference eigenvalue

    shap = psf_stack.shape
    mat_stack = zeros((shap[2],shap[0]*shap[1])) # Images correspond to lines

    for i in range(0,shap[2]):
        mat_stack[i,:] = psf_stack[:,:,i].reshape((shap[0]*shap[1],))
    U, s, Vt = linalg.svd(mat_stack,full_matrices=True)
    gamma = s[n_eig]*upfact**2
    nu = gamma*ones((shap[2],))
    im_hr = zeros((shap[0]*upfact,shap[1]*upfact,shap[2]))
    mat_hr = zeros((shap[2],shap[0]*shap[1]*upfact**2))
    thresh_type = 1

    shift_ker_stack = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))
    shift_ker_stack_adj = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))

    for i in range(0,shap[2]):
        uin = shifts[i,:].reshape((1,2))*upfact
        shift_ker_stack[:,:,i] = utils.lanczos(uin,n=lanc_rad)
        shift_ker_stack_adj[:,:,i] = rot90(shift_ker_stack[:,:,i],2)


    for i in range(0,nb_iter):

        # ------- Low rank constraint ----- #
        mat_temp,si,U,Vt = nuc_norm_thresh(mat_hr,nu*w,thresh_type)
        mat_hr = 2*mat_temp - mat_hr

        # ------- Equality constraint ----- #
        for k in range(0,shap[2]):
            temp_k = sr_op_equal_cons(mat_hr[k,:].reshape((shap[0],shap[1])),psf_stack[:,:,k],upfact,shift_ker_stack[:,:,k],shift_ker_stack_adj[:,:,k],nb_iter=50)
            im_hr[:,:,k] = (1-mu/2)*im_hr[:,:,k] + (mu/2)*(2*temp_k-mat_hr[k,:].reshape((shap[0],shap[1])))
            mat_hr[k,:] = im_hr[:,:,k].reshape((1,shap[0]*shap[1]))


        U, si, Vt = linalg.svd(mat_hr,full_matrices=False)
        print 'Nuclear norm: ',abs(si).sum()

    return im_hr,mat_hr


def lowrank_sr_ineq_cons_dr(psf_stack,shift,upfact,nb_iter=100,mu=1.0,n_eig=3,p_val_chi2=0.95): # n_eig: order of the reference eigenvalue

    shap = psf_stack.shape
    mat_stack = zeros((shap[2],shap[0]*shap[1])) # Images correspond to lines

    for i in range(0,shap[2]):
        mat_stack[i,:] = psf_stack[:,:,i].reshape((shap[0]*shap[1],))
    U, s, Vt = linalg.svd(mat_stack,full_matrices=True)
    gamma = s[n_eig]*upfact**2
    nu = gamma*ones((shap[2],))
    im_hr = zeros((shap[0]*upfact,shap[1]*upfact,shap[2]))
    mat_hr = zeros((shap[2],shap[0]*shap[1]*upfact**2))
    thresh_type = 1
    res_stack = copy(psf_stack)
    shift_ker_stack = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))
    shift_ker_stack_adj = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[2]))

    for i in range(0,shap[2]):
        uin = shifts[i,:].reshape((1,2))*upfact
        shift_ker_stack[:,:,i] = utils.lanczos(uin,n=lanc_rad)
        shift_ker_stack_adj[:,:,i] = rot90(shift_ker_stack[:,:,i],2)


    for i in range(0,nb_iter):

        # ------- Low rank constraint ----- #
        mat_temp,si,U,Vt = nuc_norm_thresh(mat_hr,nu*w,thresh_type)
        mat_hr = 2*mat_temp - mat_hr

        # ------- Inequality constraint ----- #

        for k in range(0,shap[2]):
            res_stack[:,:,k] = utils.decim(scisig.fftconvolve(mat_hr[k,:].reshape((shap[0],shap[1])),shift_ker_stack[:,:,k],mode='same'),upfact,av_en=0)-psf_stack[:,:,k]
            sig = im_gauss_nois_est(res_stack[:,:,k])
            r = sig*scistats.chi2.interval(p_val_chi2,shap[0]*shap[1])
            temp_k = sr_op_inequal_cons(mat_hr[k,:].reshape((shap[0],shap[1])),psf_stack[:,:,k],upfact,shift_ker_stack[:,:,k],shift_ker_stack_adj[:,:,k],r,nb_iter=50)
            im_hr[:,:,k] = (1-mu/2)*im_hr[:,:,k] + (mu/2)*(2*temp_k-mat_hr[k,:].reshape((shap[0],shap[1])))
            mat_hr[k,:] = im_hr[:,:,k].reshape((1,shap[0]*shap[1]))

        U, si, Vt = linalg.svd(mat_hr,full_matrices=False)
        print 'Nuclear norm: ',abs(si).sum()

    return im_hr,mat_hr

def non_unif_smoothing(tree,weights,first_guess,lr_data,lr_data_est,mu=1,spec_rad=None,nb_iter=20,tol=0.01):
    shap = lr_data.shape
    bound = zeros((shap[2],))
    for i in range(0,shap[2]):
        bound[i] = 2*(lr_data[:,:,i]*lr_data_est[:,:,i]).sum()/(lr_data_est[:,:,i]**2).sum() - first_guess[i]

    a = sum(weights,axis=1)
    if spec_rad is None:
        spec_rad = 2*a.max()
    else:
        print "real spec rad: ",spec_rad," estimate: ",2*a.max()

    x = copy(first_guess)
    z = copy(x)
    t = 1
    cost_ref = 0
    res_ref =0
    for k in range(0,shap[2]):
        res_ref = res_ref+((lr_data[:,:,k]-first_guess[k]*lr_data_est[:,:,k])**2).sum()
    print "-------res_ref: ",res_ref,"-------"
    cost_old = 0
    cost = 1
    i=0
    while i <nb_iter and 100*abs(cost-cost_old)/cost>tol:
        y = z - mu*grad_nn_unif_smooth(z,tree,weights,a)/spec_rad
        x_old = copy(x)
        x = pt_wise_bound(y,first_guess,bound)
        t_old = t
        t = (1+sqrt(4*t**2+1))/2
        lambd = 1+(t_old-1)/t
        z = x_old + lambd*(x-x_old)
        res = 0
        if i>0:
            cost_old = cost
        cost=0
        for k in range(0,shap[2]):
            res = res+((lr_data[:,:,k]-z[k]*lr_data_est[:,:,k])**2).sum()
            cost = cost+ (((z[k]*ones((size(tree[k,:]),))-z[tree[k,:]])**2)*weights[k,:]).sum()
        print "cost: ",cost," res: ",res
        i+=1
    return z

def lsq_coeff2D(u,v): # Find the scalar a which minimizes ||a*u-v||^2
    a = (u*v).sum()/(u**2).sum()
    return a

def lsq_coeff_stack(u,v):
    shap = u.shape
    a = zeros((shap[2],))
    for i in range(0,shap[2]):
        a[i] = lsq_coeff2D(u[:,:,i],v[:,:,i])
    return a

def lsq_mult_coeff(im,src,man=True):
    shap = src.shape
    mat = zeros((shap[2],shap[2]))
    for i in range(0,shap[2]):
        for j in range(0,shap[2]):
            mat[i,j] = (src[:,:,i]*src[:,:,j]).sum()
    v = zeros((shap[2],1))
    for k in range(0,shap[2]):
        v[k,0] = (im*src[:,:,k]).sum()

    output = None
    if man:
        output = man_inv(mat).dot(v)
    else:
        output = LA.inv(mat).dot(v)
    return output,mat,v

def man_inv(mat,cond=None):
    U, s, Vt = linalg.svd(mat,full_matrices=False)

    eig = zeros((s.shape[0],s.shape[0]))
    for i in range(0,s.shape[0]):
        if cond is not None:
            if s[i]>s[0]/cond:
                eig[i,i] = (s[i])**(-1)
            else:
                eig[i,i] = cond/s[0]
        else:
            if s[i]>0:
                eig[i,i] = (s[i])**(-1)
    inv = U.dot(eig.dot(Vt))
    return inv


def lsq_mult_coeff_stack(im,src,man=True):
    shap1 = src.shape
    shap2 = im.shape
    coeff_out = zeros((shap1[2],shap2[2]))
    v = zeros((shap1[2],shap2[2]))
    mat = zeros((shap1[2],shap1[2],shap1[3]))
    for i in range(0,shap2[2]):
        outi,mati,vi = lsq_mult_coeff(im[:,:,i],src[:,:,:,i],man=man)
        coeff_out[:,i] = outi.reshape((shap1[2],))
        mat[:,:,i] = copy(mati)
        v[:,i] = vi.reshape((shap1[2],))
    return coeff_out,mat,v

def lsq_mult_coeff_stack_pos(im,src_hr,src_lr,tol_neg=-10**(-18),nb_iter=100,z_init=None,mu=0.8,tol=0.00001):
    coeff_lsq,mat,v = lsq_mult_coeff_stack(im,src_lr)
    coeff_pos = coeff_lsq*0
    shap = src_lr.shape
    nb_im = shap[3]
    nb_src = shap[2]
    shap2 = src_hr.shape
    z = zeros((shap2[0],shap2[1],nb_im))
    if z_init is not None:
        z = copy(z_init)
    x = copy(z)
    grad = copy(x)
    t=0
    mats = zeros((shap[2],shap[2]))
    for k1 in range(0,shap[2]):
        for k2 in range(k1,shap[2]):
            mats[k1,k2] = (src_hr[:,:,k1]*src_hr[:,:,k2]).sum()
            mats[k2,k1] = mats[k1,k2]
    spec_rad = zeros((shap[3]))
    U, s, Vt = linalg.svd(mats,full_matrices=False)
    spec1 = s.max()
    for i in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,i],full_matrices=False)
        spec_rad[i] = spec1*s.max()
    res=1
    res_old=0
    k = 0
    im_est_hr = zeros((shap2[0],shap2[1],nb_im))
    while k <nb_iter and 100*abs(res-res_old)/res>tol:
        k+=1
        res_old=res
        res=0
        told = t
        t = (1+sqrt(4*t**2+1))/2
        lambd = 1+(told-1)/t
        for i in range(0,nb_im):
            v1 = zeros((nb_src,1))
            for l in range(0,nb_src):
                v1[l,0] = (src_hr[:,:,l]*z[:,:,i]).sum()
            # Primal variable update (not mandatory)
            coeff_pos[:,i] = -(mat[:,:,i].dot(v1)).reshape((nb_src,)) + coeff_lsq[:,i]
            im_est_hr[:,:,i] *= 0
            im_est_lr = zeros((shap[0],shap[1]))
            for l in range(0,nb_src):
                im_est_hr[:,:,i]+=coeff_pos[l,i]*src_hr[:,:,l]
                im_est_lr+=coeff_pos[l,i]*src_lr[:,:,l,i]
            res+=((im[:,:,i]-im_est_lr)**2).sum()
            v2 = mat[:,:,i].dot(v1) - coeff_lsq[:,i]
            grad[:,:,i] = 0*grad[:,:,i]
            for l in range(0,nb_src):
                grad[:,:,i] += v2[l,0]*src_hr[:,:,l]
            y = z[:,:,i] - mu*grad[:,:,i]/spec_rad[i]
            xold = copy(x[:,:,i])
            x[:,:,i] = y-mu*pos_proj_mat(y*spec_rad[i]/mu,tol=-tol_neg)/spec_rad[i]
            z[:,:,i] = xold+lambd*(x[:,:,i]-xold)

        print "Pos coeff est residual: ",res
        print "Min reconstructed val: ",im_est_hr.min()
    v1 = zeros((nb_src,1))
    for i in range(0,nb_im):
        for l in range(0,nb_src):
            v1[l,0] = (src_hr[:,:,l]*z[:,:,i]).sum()
            # Primal variable update (not mandatory)
            coeff_pos[:,i] = -(mat[:,:,i].dot(v1)).reshape((nb_src,)) + coeff_lsq[:,i]
    z_init=z
    return coeff_pos

def lsq_mult_coeff_stack_pos_2(im,src_lr,nb_iter=100,mu=0.8,tol=0.001):
    coeff_lsq,mat,v = lsq_mult_coeff_stack(im,src_lr)
    shap = src_lr.shape
    nb_im = shap[3]
    nb_src = shap[2]

    z = copy(coeff_lsq)
    x = copy(z)
    grad = copy(x)
    t=1
    spec_rad = zeros((shap[3]))
    for i in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,i],full_matrices=False)
        spec_rad[i] = s.max()
    res=1
    res_old=0
    k = 0
    while k <nb_iter and 100*abs(res-res_old)/res>tol:
        k+=1
        res_old=res
        res=0
        told = t
        t = (1+sqrt(4*t**2+1))/2
        lambd = 1+(told-1)/t
        for i in range(0,nb_im):
            im_est_lr = zeros((shap[0],shap[1]))
            for l in range(0,nb_src):
                im_est_lr+=z[l,i]*src_lr[:,:,l,i]
            res+=((im[:,:,i]-im_est_lr)**2).sum()
            grad[:,i] = squeeze(mat[:,:,i].dot(z[:,i].reshape((nb_src,1))))-v[:,i]
            y = z[:,i]-mu*grad[:,i]/spec_rad[i]
            xold = copy(x[:,i])
            x[:,i] = pos_proj(y)
            z[:,i] = xold + lambd*(x[:,i]-xold)
        print "Pos coeff est residual: ",res
    return z

def smoothing_line_search_opt_coeff(tree,weights,u,v,a):
    M = zeros((2,2))
    b = zeros((2,1))
    shap = weights.shape
    nb_pts = shap[0]
    nb_neighs = shap[1]
    for i in range(0,nb_pts):
        for j in range(0,nb_neighs):
            M[0,0] += weights[i,j]*((u[:,i]-u[:,tree[i,j]])**2).sum()
            M[1,1] += weights[i,j]*((v[:,i]-v[:,tree[i,j]])**2).sum()
            M[0,1] += weights[i,j]*((u[:,i]-u[:,tree[i,j]])*(v[:,i]-v[:,tree[i,j]])).sum()
            b[0,0] -= weights[i,j]*((u[:,i]-u[:,tree[i,j]])*(a[:,i]-a[:,tree[i,j]])).sum()
            b[1,0] -= weights[i,j]*((v[:,i]-v[:,tree[i,j]])*(a[:,i]-a[:,tree[i,j]])).sum()
    M[1,0] = M[0,1]
    output = LA.inv(M).dot(b)
    return output

def line_search_lsq_mult_coeff_stack(im,src,u,v,a):
    shap = src.shape
    M = zeros((2,2))
    b = zeros((2,1))
    for i in range(0,shap[3]):
        for j in range(0,shap[2]):
            temp1 = zeros((shap[0],shap[1]))
            temp1 += src[:,:,j,i]*u[j,i]
            M[0,0] += (temp1**2).sum()
            temp2 = zeros((shap[0],shap[1]))
            temp2 += src[:,:,j,i]*v[j,i]
            M[1,1] += (temp2**2).sum()
            M[0,1] += (temp1*temp2).sum()
            temp = copy(im[:,:,i])
            temp -= src[:,:,j,i]*a[j,i]
            b[0,0] += (temp*temp1).sum()
            b[1,0] += (temp*temp2).sum()
    M[1,0] = M[0,1]
    output = LA.inv(M).dot(b)
    return output

def non_unif_smoothing_mult_coeff(im,src,tree,weights,nb_iter=100,spec_rad=None,tol=0.01,Ainit=None):
    a = sum(weights,axis=1)
    shap = src.shape
    shap2 = tree.shape
    nb_neigh = shap2[1]
    if spec_rad is None:
        spec_rad = 2*a.max()*shap[2]
    else:
        print "real spec rad: ",spec_rad," estimate: ",2*a.max()
    ref_smooth = 0
    res_res = 0
    if Ainit is None:
        Ainit = ones((shap[2],shap[3]))
    cost_old = 0
    cost = 1
    A = copy(Ainit)
    i=0
    while i <nb_iter and 100*abs(cost-cost_old)/cost>tol:
        u = 0*Ainit
        v = 0*Ainit
        res_smooth = copy(im)
        for k in range(0,shap[3]):
            i1,i2 = where(tree==k)
            for l in range(0,nb_neigh):
                u[:,k]-=weights[k,l]*A[:,tree[k,l]]
            w=weights[k,:].sum()
            for j in range(0,size(i1)):
                p = where((tree[k,:]==i1[j]))
                if size(p)==0:
                    w+=weights[i1[j],i2[j]]
                    u[:,k]-=weights[i1[j],i2[j]]*A[:,i1[j]]
            u[:,k]+=w*A[:,k]
            for j in range(0,shap[2]):
                res_smooth[:,:,k] -= A[j,k]*src[:,:,j,k]
            for j in range(0,shap[2]):
                v[j,k] = -(res_smooth[:,:,k]*src[:,:,j,k]).sum()
        cost_res = (res_smooth**2).sum()
        print " ------- Cur res: ",cost_res," ----------- "
        cost_smooth = 0
        for k in range(0,shap[3]):
            for l in range(0,nb_neigh):
                cost_smooth+=((A[:,k]-A[:,tree[k,l]])**2).sum()*weights[k,l]
        print " ------- Cur smoothness: ",cost_smooth," ----------- "
        cost_old = cost
        cost = cost_res+cost_smooth
        coeff1 = smoothing_line_search_opt_coeff(tree,weights,-u,-u,A)
        coeff2 = line_search_lsq_mult_coeff_stack(im,src,-u,-u,A)
        mu1 = max(min(coeff1[0,0],coeff2[0,0]),0)
        mu2 = max(min(coeff1[1,0],coeff2[1,0]),0)
        #mu1 = mean([coeff1[0,0],coeff2[0,0]])
        #mu2 = mean([coeff1[1,0],coeff2[1,0]])
        print " Data fidelity direction coeffs: ",coeff2," Smoothness direction: ",coeff1
        A = A-mu1*u-mu2*v
        i+=1
    return A

def non_unif_smoothing_mult_coeff_pos(im,src,tree,weights,mu=0.8,nb_iter=100,spec_rad=None,tol=0.001,Ainit=None):
    a = sum(weights,axis=1)
    shap = src.shape
    shap2 = tree.shape
    nb_neigh = shap2[1]

    smooth_spec_rad = 2*a.max()*shap[2]
    ref_smooth = 0
    res_res = 0
    Atemp,mat,v = lsq_mult_coeff_stack(im,src)
    if Ainit is None:
        if Atemp.shape[0]==1:
            Ainit = pos_proj(Atemp)
        elif Atemp.shape[0]>1:
            Ainit = pos_proj_mat(Atemp)
    lambd_vect = zeros(shap[3]) # Balance paramters

    cost_old = 0
    cost = 1
    A = copy(Ainit)
    i=0
    cost_smooth_ref = 0
    for k in range(0,shap[3]):
        for l in range(0,nb_neigh):
            cost_smooth_ref+=((Ainit[:,k]-Ainit[:,tree[k,l]])**2).sum()*weights[k,l]
    for k in range(0,shap[3]):
        res = im[:,:,k]
        for l in range(0,shap[2]):
            res-=Ainit[l,k]*src[:,:,l,k]
        lambd_vect[k] = cost_smooth_ref/(shap[3]*(res**2).sum())

    spec_rad = zeros((shap[3]))
    for k in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()*lambd_vect[k]
    grad_lip = smooth_spec_rad+spec_rad.max()

    x = copy(A)
    grad = copy(x)
    t=1
    spec_rad = zeros((shap[3]))
    for i in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,i],full_matrices=False)
        spec_rad[i] = s.max()
    res=1
    res_old=0
    iter = 0
    while iter <nb_iter and 100*abs(res-res_old)/res>tol:
        iter+=1
        res_old=res
        told = t
        t = (1+sqrt(4*t**2+1))/2
        lambd = 1+(told-1)/t
        u = 0*Ainit # Smoothness related gradient
        v = 0*Ainit # Gradient related gradient
        res_smooth = copy(im)
        res_smooth_2 = copy(im)
        for k in range(0,shap[3]):
            i1,i2 = where(tree==k)
            for l in range(0,nb_neigh):
                u[:,k]-=weights[k,l]*A[:,tree[k,l]]
            w=weights[k,:].sum()
            for j in range(0,size(i1)):
                p = where((tree[k,:]==i1[j]))
                if size(p)==0:
                    w+=weights[i1[j],i2[j]]
                    u[:,k]-=weights[i1[j],i2[j]]*A[:,i1[j]]
            u[:,k]+=w*A[:,k]
            for j in range(0,shap[2]):
                res_smooth[:,:,k] -= A[j,k]*src[:,:,j,k]
                res_smooth_2[:,:,k] -= lambd_vect[k]*A[j,k]*src[:,:,j,k] # For the objective function value
            for j in range(0,shap[2]):
                v[j,k] = -lambd_vect[k]*(res_smooth[:,:,k]*src[:,:,j,k]).sum()
            cost_smooth = 0
            for k in range(0,shap[3]):
                for l in range(0,nb_neigh):
                    cost_smooth+=((A[:,k]-A[:,tree[k,l]])**2).sum()*weights[k,l]
        res_val = (res_smooth**2).sum()
        res = cost_smooth+(res_smooth_2**2).sum()
        print "smoothness criteria: ",cost_smooth," residual: ",res_val," objective val: ",res
        grad = u+v
        y = A - mu*grad/grad_lip
        xold = copy(x)
        x = pos_proj_mat(y)
        A = xold + lambd*(x-xold)

    return A

def smoothing_matrix(tree,weights):
    shap = tree.shape
    M = zeros((shap[0],shap[0]))
    nb_neigh = shap[1]
    for k in range(shap[0]):
        for l in range(0,nb_neigh):
            M[k,tree[k,l]]=-weights[k,l]
            w=weights[k,:].sum()
            i1,i2 = where(tree==k)
            for j in range(0,size(i1)):
                p = where((tree[k,:]==i1[j]))
                if size(p)==0:
                    w+=weights[i1[j],i2[j]]
                    M[k,i1[j]]=-weights[i1[j],i2[j]]
        M[k,k]=w
    return M

def smoothing_prox(p_smth_mat,M):
    out = M*0
    shap = M.shape
    for k in range(0,shap[0]):
        y1 = M[k,:].reshape((shap[1],1))
        y2 = p_smth_mat[:,:,k].dot(y1)
        out[k,:] = y2.reshape((shap[1],))
    return out

def spreading_constraint_src_update_cp(X,A,centroid,Sinit,res_max,eps=0.8,nb_iter=100,tol=1e-15,accel_en=True):
    shap = Sinit.shape
    # Spectral radius setting
    # -- Coeff matrix
    U, s, Vt = linalg.svd(A.dot(transpose(A)),full_matrices=False)
    spec_rad = sqrt(s.max())
    to = spec_rad**(-1)
    sig = eps/(to*spec_rad**2) # Weight related to the dual variables
    theta = 0.9

    # Dual variable (residual constraint)
    Y1 = Sinit.dot(A)
    O = Y1*0
    # Primal variable (spreading constraint)
    V = copy(Sinit)
    ones_vect = ones((1,shap[1]))
    centroidn = centroid.dot(ones_vect)

    # Main variable
    S = copy(Sinit)

    cost=2
    cost_old=1
    iter=0

    ref_mse = ((X-Sinit.dot(A))**2).sum()
    print "--->>ref res: <<---",ref_mse

    while iter<nb_iter and 100*abs(cost-cost_old)/abs(cost_old)>tol:
        iter+=1

        # Dual variables update
        # -- residual constraint variable update
        temp1 = Y1+sig*S.dot(A)-sig*X
        Y1 = temp1-sig*proj_sphere(temp1/sig,O,sqrt(res_max)) # Moreau id

        # Primal variable update
        Vold = copy(V)
        temp2 = V - to*Y1.dot(transpose(A))
        V = (temp2 + to*centroidn)/(1+to)

        # Main variable update
        if accel_en:
            theta = 1/sqrt(1+2*to)
            to = theta*to
            sig = sig/theta
        S = V + theta*(V-Vold)

        # Sanity check
        min_comp = ((V-centroidn)**2).sum()
        max_comp = (Y1*V.dot(A)).sum()
        res = ((X-S.dot(A))**2).sum()
        cost_old = cost
        cost = min_comp+max_comp
    print "residual: ",res," min comp: ",min_comp," max comp: ",max_comp," cost: ",cost

    return S


def sparse_bar_coord_pb_field_cp(S,X,res_max,Ainit,thresh_map,thresh_type=1,nb_iter=100,tol=1e-15,eps=0.8,accel_en=False): # calculates the best "barycentric" coordinates of X columns with respect to S columns with a sparsity constraint on the coefficients matrix
    shap1 = X.shape
    shap2 = S.shape
    # Spectral radius setting
    # -- Sources matrix
    U, s1, Vt = linalg.svd(transpose(S).dot(S),full_matrices=False)
    # -- Column summation matrix
    s2  = shap2[1]
    spec_rad = sqrt(s1[0]+s2)
    to = spec_rad**(-1)
    sig = eps/(to*spec_rad**2) # Weight related to the dual variables
    theta = 0.9
    # Dual variables
    # -- Residual constraint variable
    Y1 = S.dot(Ainit)
    O = Y1*0
    # -- Columns normalization variable
    Y2 = ones((shap1[1],1))
    ones_vect1 = ones((shap1[1],1))
    ones_vect2 = ones((shap2[1],1))
    # Primal variable
    V = copy(Ainit)

    # Main variable
    A = copy(Ainit)

    ref_mse = ((X-S.dot(Ainit))**2).sum()
    print "--->>ref res: <<---",ref_mse

    cost=2
    cost_old=1
    iter=0
    var = 1
    while iter<nb_iter: #and 100*abs(cost-cost_old)/abs(cost_old)>tol:
        iter+=1
        # Dual variables update

        # -- residual constraint variable update
        temp1 = Y1+sig*S.dot(A)-sig*X
        Y1 = temp1-sig*proj_sphere(temp1/sig,O,sqrt(res_max)) # Moreau id

        # -- columns normalization constraint variable update
        temp2 = Y2+sig*(A.sum(axis=0)).reshape((shap1[1],1))
        Y2 = temp2 - sig*ones_vect1 # Moreau id

        # Primal variable update
        temp3 = V - to*(transpose(S).dot(Y1)+ones_vect2.dot(transpose(Y2)))
        Vold = copy(V)
        V = pos_proj(utils.thresholding(temp3,to*thresh_map,thresh_type))

        # Main variable update
        if accel_en:
            theta = 1/sqrt(1+2*to)
            to = theta*to
            sig = sig/theta
        Aold = copy(A)
        A = V + theta*(V-Vold)

        # Sanity check
        min_comp = abs(V*thresh_map).sum()
        max_comp = (Y1*(S.dot(V))).sum()+(Y2*(V.sum(axis=0)).reshape((shap1[1],1))).sum()
        cost_old = cost
        cost = min_comp+max_comp
        min_val = A.min()
        mean_row_A = A.sum()/shap1[1]
        cur_res = ((X-S.dot(A))**2).sum()
        var = 100*((A-Aold)**2).sum()/(Aold**2).sum()
    print "min_val: ",min_val," min_comp: ",min_comp," max_comp: ",max_comp," residual: ",cur_res," mean rows sum: ",mean_row_A," cost function: ",cost

    #A = pos_proj(A)
    #A = A/(ones_vect2.dot((A.sum(axis=0)).reshape(1,shap1[1])))

    return A

def non_unif_smoothing_mult_coeff_pos_cp(im,src,tree,weights,theta=0.8,p_smth_mat_inv=None,to=None,eps=0.8,nb_iter=100,tol=0.01,Ainit=None,pos_en=1): # to is the weight related to the primal variable

    shap = src.shape
    nb_neigh = tree.shape[1]
    Atemp,mat,v = lsq_mult_coeff_stack(im,src)
    nb_src = Atemp.shape[0]
    if Ainit is None:
        if pos_en==1:
            if nb_src==1:
                Ainit = pos_proj(Atemp)
            elif nb_src>1:
                Ainit = pos_proj_mat(Atemp)
        else:
            Ainit = Atemp
    spec_rad = zeros((shap[3]))
    for k in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()
    spec_norm=None
    if pos_en==1:
        spec_norm = sqrt(spec_rad.sum()+1)
    else:
        spec_norm = sqrt(spec_rad.sum())
    if to is None:
        a = sum(weights,axis=1)
        #to = (2*a.max())**(-1)
        to = 1/spec_norm
    if p_smth_mat_inv is None:
        p_smth_mat_inv = zeros((shap[3],shap[3],nb_src))
        for ind in range(0,nb_src):
            p_smth_mat = zeros((shap[3],shap[3]))
            for k in range(0,shap[3]):
                i1,i2 = where(tree==k)
                for l in range(0,nb_neigh):
                    p_smth_mat[k,tree[k,l]]=-weights[k,l,ind]
                w=weights[k,:,ind].sum()
                for j in range(0,size(i1)):
                    p = where((tree[k,:]==i1[j]))
                    if size(p)==0:
                        w+=weights[i1[j],i2[j],ind]
                        p_smth_mat[k,i1[j]] = -weights[i1[j],i2[j],ind]
                p_smth_mat[k,k] = w
            mat_temp = eye(shap[3])+to*p_smth_mat
            p_smth_mat_inv[:,:,ind] = LA.inv(mat_temp)

    sig = eps/(to*spec_norm**2) # Weight related to the dual variables

    # Dual variables
    y1 = copy(Ainit)
    y2 = im*0
    o = y2*0
    rad = zeros((shap[3],))
    for k in range(0,shap[3]):
        res = copy(im[:,:,k])
        for l in range(0,shap[2]):
            res-=Ainit[l,k]*src[:,:,l,k]
        y2[:,:,k] = -res
        rad[k] = (res**2).sum()
    print "--->>ref res: <<---",rad.sum()
    # Primal variable
    v = copy(Ainit)

    # Main variable
    x = copy(Ainit)

    cost=1
    cost_old=0
    iter=0

    while iter<nb_iter and 100*abs(cost-cost_old)/abs(cost)>tol :

        iter+=1
        # Dual variables update
            # Positivity
        temp1 = y1+sig*x
        if shap[2]==1:
            y1 = -pos_proj(-temp1)
        else:
            y1 = -pos_proj_mat(-temp1)

            # Residual constraint
        temp2 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp2[:,:,k] +=sig*x[l,k]*src[:,:,l,k]
        temp2+=y2-sig*im
        y2 = temp2-sig*proj_sphere_cube(temp2/sig,o,sqrt(rad)) # Moreau id

        # Primal variable update
        temp3 = to*copy(y1)
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp3[l,k]+=to*(src[:,:,l,k]*y2[:,:,k]).sum()
        temp4 = v - temp3
        vold = copy(v)
        v = smoothing_prox(p_smth_mat_inv,temp4)
        x = v+theta*(v-vold)

        # Sanity check
        cost_smooth = 0
        for k in range(0,shap[3]):
            for l in range(0,nb_neigh):
                cost_smooth+=(((x[:,k]-x[:,tree[k,l]])*sqrt(weights[k,l,:]))**2).sum()
        # Objective value
        temp5 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp5[:,:,k] +=x[l,k]*src[:,:,l,k]
        max_comp = (temp5*y2).sum()+(y1*x).sum()
        temp5-=im
        min_comp = cost_smooth
        cost_old = cost
        cost = min_comp+max_comp
        print "residual: ",(temp5**2).sum(), " smoothness: ",cost_smooth," linear part: ",max_comp," objective val: ",cost," min val: ",x.min()


    return x



def non_unif_smoothing_mult_coeff_pos_cp_2(im,src,src_hr,tree,weights,theta=0.8,p_smth_mat_inv=None,to=None,eps=0.8,nb_iter=100,tol=0.01,Ainit=None,pos_en=0): # to is the weight related to the primal variable

    shap = src.shape
    shap1 = src_hr.shape
    src_mat = zeros((shap[2],shap[2]))
    for i in range(0,shap[2]):
        for j in range(i+1,shap[2]):
            src_mat[i,j] = (src_hr[:,:,i]*src_hr[:,:,j]).sum()
    src_mat = src_mat+transpose(src_mat)
    for i in range(0,shap[2]):
        src_mat[i,i] = (src[:,:,i]**2).sum()
    U, s, Vt = linalg.svd(src_mat,full_matrices=False)
    spec_rad_pos = s.max()
    nb_neigh = tree.shape[1]
    Atemp,mat,v = lsq_mult_coeff_stack(im,src)
    nb_src = Atemp.shape[0]
    if Ainit is None:
        if pos_en==1:
            if nb_src==1:
                Ainit = pos_proj(Atemp)
            elif nb_src>1:
                Ainit = pos_proj_mat(Atemp)
        else:
            Ainit = Atemp
    spec_rad = zeros((shap[3]))
    for k in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()
    spec_norm=None
    if pos_en==1:
        spec_norm = sqrt(spec_rad.sum()+spec_rad_pos)
    else:
        spec_norm = sqrt(spec_rad.sum())
    """if to is None:
        a = sum(weights,axis=1)
        #to = (2*a.max())**(-1)
        to = 1/spec_norm"""
    to = 1.0/spec_norm

    if p_smth_mat_inv is None:
        p_smth_mat_inv = zeros((shap[3],shap[3],nb_src))
        p_smth_mat_stack = zeros((shap[3],shap[3],nb_src))
        spec_rad_stack = zeros((nb_src,))
        for ind in range(0,nb_src):
            p_smth_mat = zeros((shap[3],shap[3]))
            for k in range(0,shap[3]):
                i1,i2 = where(tree==k)
                for l in range(0,nb_neigh):
                    p_smth_mat[k,tree[k,l]]=-weights[k,l,ind]
                w=weights[k,:,ind].sum()
                for j in range(0,size(i1)):
                    p = where((tree[k,:]==i1[j]))
                    if size(p)==0:
                        w+=weights[i1[j],i2[j],ind]
                        p_smth_mat[k,i1[j]] = -weights[i1[j],i2[j],ind]
                p_smth_mat[k,k] = w
            p_smth_mat_stack[:,:,ind] = copy(p_smth_mat)#transpose(p_smth_mat).dot(p_smth_mat)
            U, s, Vt = linalg.svd(p_smth_mat_stack[:,:,ind],full_matrices=False)
            spec_rad_stack[ind] = s.max()
        #to = 1.0/spec_rad_stack.max()
        for ind in range(0,nb_src):
            mat_temp = eye(shap[3])+to*p_smth_mat_stack[:,:,ind]
            p_smth_mat_inv[:,:,ind] = LA.inv(mat_temp)

    sig = eps/(to*spec_norm**2) # Weight related to the dual variables

    # Dual variables
    #y1 = copy(Ainit)
    y1 = zeros((shap1[0],shap1[1],shap[3]))
    y2 = im*0
    o = y2*0
    rad = zeros((shap[3],))
    for k in range(0,shap[3]):
        res = copy(im[:,:,k])
        for l in range(0,shap[2]):
            res-=Ainit[l,k]*src[:,:,l,k]
        y2[:,:,k] = -res
        rad[k] = (res**2).sum()
    print "--->> ref res: <<---",rad.sum()
    # Primal variable
    v = copy(Ainit)

    # Main variable
    x = copy(Ainit)

    cost=1
    cost_old=0
    iter=0
    temp6 = None
    while iter<nb_iter and 100*abs(cost-cost_old)/abs(cost)>tol :

        iter+=1
        # Dual variables update
        # Positivity
        #temp1 = y1+sig*x
        temp1 = copy(y1)
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp1[:,:,k] +=sig*x[l,k]*src_hr[:,:,l]
        if shap[2]==1:
            y1 = temp1-sig*pos_proj_mat(temp1/sig)
        else:
            y1 = temp1-sig*pos_proj_cube(temp1/sig)

        # Residual constraint
        temp2 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp2[:,:,k] +=sig*x[l,k]*src[:,:,l,k]
        temp2+=y2-sig*im
        y2 = temp2-sig*proj_sphere_cube(temp2/sig,o,sqrt(rad)) # Moreau id

        # Primal variable update
        temp3 = x*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp3[l,k]+=to*((src[:,:,l,k]*y2[:,:,k]).sum()+(src_hr[:,:,l]*y1[:,:,k]).sum())
        temp4 = v - temp3
        vold = copy(v)
        v = smoothing_prox(p_smth_mat_inv,temp4)
        x = v+theta*(v-vold)

        # Sanity check
        cost_smooth = 0
        for k in range(0,shap[3]):
            for l in range(0,nb_neigh):
                cost_smooth+=(((x[:,k]-x[:,tree[k,l]])*sqrt(weights[k,l,:]))**2).sum()
        # Objective value
        temp5 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp5[:,:,k] +=x[l,k]*src[:,:,l,k]
        temp6 = y1*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp6[:,:,k] +=x[l,k]*src_hr[:,:,l]
        max_comp = (temp5*y2).sum()+(y1*temp6).sum()
        temp5-=im
        min_comp = cost_smooth
        cost_old = cost
        cost = min_comp+max_comp
        print "residual: ",(temp5**2).sum(), " smoothness: ",cost_smooth," linear part: ",max_comp," objective val: ",cost," min val: ",temp6.min()

    min_val_map = temp6[:,:,0]
    for i in range(0,shap1[0]):
        for j in range(0,shap1[0]):
            min_val_map[i,j] = temp6[i,j,:].min()
    print "median min val: ",np.median(min_val_map)

    return x,min_val_map

def non_unif_smoothing_mult_coeff_pos_cp_3(im,src,src_hr,tree,weights,e,theta=0.8,p_smth_mat_inv=None,to=None,eps=0.8,nb_iter=100,tol=0.01,Ainit=None,pos_en=False,reg_param=1000): # to is the weight related to the primal variable;  e is a nothc filter parameter

    shap = src.shape
    shap1 = src_hr.shape
    print shap1
    src_mat = zeros((shap[2],shap[2]))
    for i in range(0,shap[2]):
        for j in range(i+1,shap[2]):
            src_mat[i,j] = (src_hr[:,:,i]*src_hr[:,:,j]).sum()
    src_mat = src_mat+transpose(src_mat)
    for i in range(0,shap[2]):
        src_mat[i,i] = (src[:,:,i]**2).sum()
    U, s, Vt = linalg.svd(src_mat,full_matrices=False)
    spec_rad_pos = s.max()
    nb_neigh = tree.shape[1]
    Atemp,mat,v = lsq_mult_coeff_stack(im,src)
    nb_src = Atemp.shape[0]
    if Ainit is None:
        if pos_en:
            if nb_src==1:
                Ainit = pos_proj(Atemp)
            elif nb_src>1:
                Ainit = pos_proj_mat(Atemp)
        else:
            Ainit = Atemp
    spec_rad = zeros((shap[3]))
    for k in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()
    spec_norm=None
    if pos_en:
        spec_norm = sqrt(spec_rad.sum()+spec_rad_pos)
    else:
        spec_norm = sqrt(spec_rad.sum())
    if to is None:
        a = sum(weights,axis=1)
        #to = (2*a.max())**(-1)
    to = 1/(spec_norm)
    print '----smoothing cp spec_rad-----: ',spec_norm
    lambd_init = None
    p_smth_mat_stack = None
    if p_smth_mat_inv is None:
        p_smth_mat_inv = zeros((shap[3],shap[3],nb_src))
        p_smth_mat_stack = zeros((shap[3],shap[3],nb_src))
        spec_rad_stack = zeros((nb_src,))
        for ind in range(0,nb_src):
            p_smth_mat = non_uniform_smoothing_mat_2(weights[:,:,ind],e[ind])
            #p_smth_mat_stack[:,:,ind] = transpose(p_smth_mat).dot(p_smth_mat)
            p_smth_mat_stack[:,:,ind] = copy(p_smth_mat)
            U, s, Vt = linalg.svd(p_smth_mat_stack[:,:,ind],full_matrices=False)
            spec_rad_stack[ind] = s.max()
        to = 1.0/spec_rad_stack.max()
        lambd_init = reg_param*spec_rad_stack.max()
        for ind in range(0,nb_src):
            mat_temp = eye(shap[3])+to*p_smth_mat_stack[:,:,ind]
            p_smth_mat_inv[:,:,ind] = LA.inv(mat_temp)

    sig = eps/(to*spec_norm**2) # Weight related to the dual variables

    # Dual variables
    #y1 = copy(Ainit)
    y1 = zeros((shap1[0],shap1[1],shap[3]))
    y2 = im*0
    o = y2*0
    rad = zeros((shap[3],))
    for k in range(0,shap[3]):
        res = copy(im[:,:,k])
        for l in range(0,shap[2]):
            res-=Ainit[l,k]*src[:,:,l,k]
        y2[:,:,k] = -res
        rad[k] = (res**2).sum()
    print "--->> ref res: <<---",rad.sum()
    ref_res = rad.sum()
    # Primal variable
    v = copy(Ainit)

    # Main variable
    x = copy(Ainit)
    cost = 1
    cur_res =2*ref_res
    cost_old=0
    iter=0
    temp6 = None
    r = array(range(0,shap[3]))
    while iter<nb_iter and 100*abs((cost-cost_old)/cost)>1  and 100*abs((cur_res-ref_res)/ref_res)>0.0001:

        iter+=1
        # Dual variables update
        # Positivity
        #temp1 = y1+sig*x
        if pos_en:
            temp1 = copy(y1)
            for k in range(0,shap[3]):
                for l in range(0,shap[2]):
                    temp1[:,:,k] +=sig*x[l,k]*src_hr[:,:,l]
            if shap[2]==1:
                y1 = temp1-sig*pos_proj_mat(temp1/sig)
            else:
                y1 = temp1-sig*pos_proj_cube(temp1/sig)

        # Residual constraint
        temp2 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp2[:,:,k] +=sig*x[l,k]*src[:,:,l,k]
        temp2+=y2-sig*im
        y2 = temp2-sig*proj_sphere_cube(temp2/sig,o,sqrt(rad)) # Moreau id

        # Primal variable update
        temp3 = x*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                if pos_en:
                    temp3[l,k]+=to*((src[:,:,l,k]*y2[:,:,k]).sum()+(src_hr[:,:,l]*y1[:,:,k]).sum())
                else:
                    temp3[l,k]+=to*((src[:,:,l,k]*y2[:,:,k]).sum())
        temp4 = v - temp3
        vold = copy(v)
        for ind in range(0,nb_src):
            mat_temp = (1+to*lambd_init*(1-iter/(nb_iter-1)))*eye(shap[3])+to*p_smth_mat_stack[:,:,ind] # /(iter+1)**2
            p_smth_mat_inv[:,:,ind] = LA.inv(mat_temp)
        v = smoothing_prox(p_smth_mat_inv,temp4+(to*lambd_init*(1-iter/(nb_iter-1)))*Ainit) #/(iter+1)**2
        x = v+theta*(v-vold)

        # Sanity check
        cost_smooth = 0
        for k in range(0,shap[3]):
            ind_k = where(r!=k)
            for l in range(0,shap[3]-1):
                cost_smooth+=(((e*x[:,k]-x[:,ind_k[0][l]])*weights[k,l,:]).sum())**2
        # Objective value
        temp5 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp5[:,:,k] +=x[l,k]*src[:,:,l,k]
        temp6 = y1*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp6[:,:,k] +=x[l,k]*src_hr[:,:,l]
        max_comp = (temp5*y2).sum()+(y1*temp6).sum()
        temp5-=im
        min_comp = cost_smooth
        cost_old = cost
        cost = min_comp+max_comp
        cur_res = (temp5**2).sum()
        print "residual: ", cur_res , " smoothness: ",cost_smooth," linear part: ",max_comp," objective val: ",cost," min val: ",temp6.min()

    min_val_map = temp6[:,:,0]
    for i in range(0,shap1[0]):
        for j in range(0,shap1[0]):
            min_val_map[i,j] = temp6[i,j,:].min()
    print "median min val: ",np.median(min_val_map)

    return x,min_val_map

def non_unif_smoothing_mult_coeff_pos_cp_4(im,src,src_hr,tree,basis,alpha_init,theta=0.1,p_smth_mat_inv=None,to=None,eps=0.01,nb_iter=100,tol=0.01,Ainit=None,pos_en=False,reg_param=1000): # to is the weight related to the primal variable;  basis is a concatenation of the optimal notch filter operator eigenvectors

    shap = src.shape
    shap1 = src_hr.shape
    print shap1
    src_mat = zeros((shap[2],shap[2]))
    for i in range(0,shap[2]):
        for j in range(i+1,shap[2]):
            src_mat[i,j] = (src_hr[:,:,i]*src_hr[:,:,j]).sum()
    src_mat = src_mat+transpose(src_mat)
    for i in range(0,shap[2]):
        src_mat[i,i] = (src[:,:,i]**2).sum()
    U, s, Vt = linalg.svd(src_mat,full_matrices=False)
    U, s2, Vt = linalg.svd(basis.dot(transpose(basis)),full_matrices=False)

    spec_rad_pos = s.max()*s2.max()
    nb_neigh = tree.shape[1]
    Atemp,mat,v = lsq_mult_coeff_stack(im,src)
    nb_src = Atemp.shape[0]
    if Ainit is None:
        if pos_en==1:
            if nb_src==1:
                Ainit = pos_proj(Atemp)
            elif nb_src>1:
                Ainit = pos_proj_mat(Atemp)
        else:
            Ainit = Atemp
    spec_rad = zeros((shap[3]))
    for k in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()#*(basis[:,k]**2).sum()
    spec_norm=None
    if pos_en==1:
        spec_norm = sqrt(spec_rad.sum()*s2.max()+spec_rad_pos)
    else:
        spec_norm = sqrt(spec_rad.sum())
    """if to is None:
        a = sum(weights,axis=1)
        #to = (2*a.max())**(-1)
        to = 1/spec_norm"""

    print '----smoothing cp spec_rad-----: ',spec_norm
    lambd_init = None


    to = eps

    shap_alph = alpha_init.shape
    thresh = ones((shap_alph[0],shap_alph[1]))*to

    sig = 0.8/(to*spec_norm**2) # Weight related to the dual variables

    # Dual variables
    #y1 = copy(Ainit)
    y1 = zeros((shap1[0],shap1[1],shap[3]))
    for k in range(0,shap[3]):
        for l in range(0,shap[2]):
            y1[:,:,k] +=Ainit[l,k]*src_hr[:,:,l]
    y2 = im*0
    o = y2*0
    rad = zeros((shap[3],))
    for k in range(0,shap[3]):
        res = copy(im[:,:,k])
        for l in range(0,shap[2]):
            res-=Ainit[l,k]*src[:,:,l,k]
            y2[:,:,k] += Ainit[l,k]*src[:,:,l,k]
        rad[k] = (res**2).sum()
    print "--->> ref res: <<---",rad.sum()," --->> ref l1 norm: <<---",abs(alpha_init).sum()
    ref_res = rad.sum()
    # Primal variable
    v = copy(alpha_init)

    # Main variable
    x = copy(alpha_init)
    cost = 1
    cur_res =2*ref_res
    cost_old=0
    iter=0
    temp6 = None
    r = array(range(0,shap[3]))

    while iter<nb_iter  and 100*abs((cur_res-ref_res)/ref_res)>0.0001:#  and 100*abs((cost-cost_old)/cost)>1:

        iter+=1
        # Dual variables update
        # Positivity
        #temp1 = y1+sig*x
        if pos_en:
            temp1 = copy(y1)
            for k in range(0,shap[3]):
                for l in range(0,shap[2]):
                    temp1[:,:,k] +=sig*x[l,k]*src_hr[:,:,l]
            if shap[2]==1:
                y1 = temp1-sig*pos_proj_mat(temp1/sig)
            else:
                y1 = temp1-sig*pos_proj_cube(temp1/sig)

        # Residual constraint
        temp2 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp2[:,:,k] +=sig*x[l,k]*src[:,:,l,k]
        temp2+=y2-sig*im
        y2 = temp2-sig*proj_sphere_cube(temp2/sig,o,sqrt(rad)) # Moreau id

        # Primal variable update
        temp3 = Ainit*0
        temp30 = Ainit*0
        temp31 = alpha_init*0
        for k in range(0,shap[3]):
            temp30 = zeros((shap[2],1))
            bask = basis[:,k].reshape((1,len(basis[:,k])))
            for l in range(0,shap[2]):
                temp30[l]+=to*(src[:,:,l,k]*y2[:,:,k]).sum()
                if pos_en:
                    temp3[l,k]+=to*(src_hr[:,:,l]*y1[:,:,k]).sum()
            temp31+=temp30.dot(bask)

        temp4 = v - temp31 - temp3.dot(transpose(basis))
        vold = copy(v)
        v = utils.thresholding(temp4,thresh,1)
        x = v+theta*(v-vold)

        # Sanity check
        cost_smooth = abs(x).sum()
        xcheck = x.dot(basis)
        # Objective value
        temp5 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp5[:,:,k] +=xcheck[l,k]*src[:,:,l,k]
        temp6 = y1*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp6[:,:,k] +=xcheck[l,k]*src_hr[:,:,l]
        max_comp = (temp5*y2).sum()+(y1*temp6).sum()
        temp5-=im
        min_comp = cost_smooth
        cost_old = cost
        cost = min_comp+max_comp
        cur_res = (temp5**2).sum()
        print "residual: ", cur_res , " l1 norm: ",cost_smooth," linear part: ",max_comp," objective val: ",cost," min val: ",temp6.min()


    min_val_map = temp6[:,:,0]
    for i in range(0,shap1[0]):
        for j in range(0,shap1[0]):
            min_val_map[i,j] = temp6[i,j,:].min()
    print "median min val: ",np.median(min_val_map)
    mat_out = x.dot(basis)


    return mat_out,x,min_val_map



def non_unif_smoothing_mult_coeff_pos_cp_5(im,src,src_hr,tree,basis,alpha_init,theta=0.1,p_smth_mat_inv=None,to=None,eps=0.01,nb_iter=100,tol=0.01,Ainit=None,pos_en=False,reg_param=1000,spars_en=True): # to is the weight related to the primal variable;  basis is a concatenation of the optimal notch filter operator eigenvectors

    shap = src.shape
    shap1 = src_hr.shape

    src_mat = zeros((shap[2],shap[2]))
    for i in range(0,shap[2]):
        for j in range(i+1,shap[2]):
            src_mat[i,j] = (src_hr[:,:,i]*src_hr[:,:,j]).sum()
    src_mat = src_mat+transpose(src_mat)
    for i in range(0,shap[2]):
        src_mat[i,i] = (src[:,:,i]**2).sum()
    U, s, Vt = linalg.svd(src_mat,full_matrices=False)
    U, s2, Vt = linalg.svd(basis.dot(transpose(basis)),full_matrices=False)

    spec_rad_pos = s.max()*s2.max()
    nb_neigh = tree.shape[1]
    Atemp,mat,v = lsq_mult_coeff_stack(im,src)
    spec_rad = zeros((shap[3]))

    rad = zeros((shap[3],))
    for k in range(0,shap[3]):
        res = copy(im[:,:,k])
        for l in range(0,shap[2]):
            res-=Ainit[l,k]*src[:,:,l,k]
        rad[k] = (res**2).sum()
    print "--->> ref res: <<---",rad.sum()

    ref_res = rad.sum()
    cost = 1
    cost_old=0


    for k in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()#*(basis[:,k]**2).sum()
    spec_norm=spec_rad.sum()*s2.max()
    shapb = basis.shape
    alpha = copy(alpha_init)*0
    i=0

    t = 1
    alphax = copy(alpha)
    shap_alpha = alpha.shape
    supports = zeros((shap_alpha[0],shap_alpha[1],min(nb_iter,shapb[0])))
    while (i < min(nb_iter,shapb[0]) and 100*abs((cost-cost_old)/cost)>0.01) or cost>1.1*ref_res :
        A = alpha.dot(basis)
        res = copy(im)
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                res[:,:,k]-=A[l,k]*src[:,:,l,k]
        print " -------- mse: ",sum(res**2),"-----------"
        cost_old = cost
        cost = sum(res**2)
        temp = Ainit*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp[l,k]+=(src[:,:,l,k]*res[:,:,k]).sum()
        grad = -temp.dot(transpose(basis))
        alphay = alpha - grad/spec_norm
        alphax_old = copy(alphax)
        if spars_en:
            alphax = utils.lineskthresholding(alphay,np.int(floor(sqrt(i)))+1)
            supports[:,:,i] = copy(alphax)
        else:
            alphax = copy(alphay)
        told = t
        t = (1+sqrt(4*t**2 +1))/2
        lambd = 1 + (told-1)/t
        alpha  = alphax_old + lambd*(alphax-alphax_old)
        supp = where(abs(alpha[0,:])>0)
        #print "-------- Support: ",supp," ----------"
        i+=1
    mat_out = alpha.dot(basis)



    return mat_out,alpha,supports

def non_unif_smoothing_mult_coeff_pos_cp_6(im,src,nb_iter=100): # to is the weight related to the primal variable;  basis is a concatenation of the optimal notch filter operator eigenvectors


    Ainit,mat,v = lsq_mult_coeff_stack(im,src)

    shap = src.shape
    spec_rad = zeros((shap[3]))
    for k in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()#*(basis[:,k]**2).sum()
    spec_norm=spec_rad.sum()

    A = copy(Ainit)

    t = 1
    Ax = copy(A)
    rad = zeros((shap[3],))
    for k in range(0,shap[3]):
        res = copy(im[:,:,k])
        for l in range(0,shap[2]):
            res-=Ainit[l,k]*src[:,:,l,k]
        rad[k] = (res**2).sum()
    print "--->> ref res: <<---",rad.sum()

    ref_res = rad.sum()
    cost = 1
    cost_old=0
    i=0
    while (i < nb_iter and 100*abs((cost-cost_old)/cost)>0.01) or cost>1.1*ref_res :
        res = copy(im)
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                res[:,:,k]-=A[l,k]*src[:,:,l,k]
        print " -------- mse: ",sum(res**2),"-----------"
        cost_old = cost
        cost = sum(res**2)
        temp = Ainit*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp[l,k]+=(src[:,:,l,k]*res[:,:,k]).sum()
        grad = -temp
        Ay = A - grad/spec_norm
        Ax_old = copy(Ax)
        Ax = copy(Ay)
        told = t
        t = (1+sqrt(4*t**2 +1))/2
        lambd = 1 + (told-1)/t
        A  = Ax_old + lambd*(Ax-Ax_old)

        i+=1


    return A

def non_unif_smoothing_mult_coeff_pos_cp_5_dirty(im,src,src_hr,tree,basis,alpha_init,theta=0.1,p_smth_mat_inv=None,to=None,eps=0.01,nb_iter=100,tol=0.01,Ainit=None,pos_en=False,reg_param=1000,spars_en=True,spars_perc=0.1):

    Atemp,mat,v = lsq_mult_coeff_stack(im,src)

    alpha = alpha_init*0

    shap = basis.shape
    supp_size = np.int(shap[1]*spars_perc)

    res = copy(Atemp)
    Aout = Atemp*0
    basis_stack = ones((shap[0],shap[1],Atemp.shape[0]))

    for i in range(0,Atemp.shape[0]):
        basis_stack[:,:,i] = copy(basis)

    for i in range(0,supp_size):
        for j in range(0,Atemp.shape[0]):
            res_vec = reshape(copy(res[j,:]),[shap[1],1])
            proj = basis_stack[:,:,j].dot(res_vec)
            coeff = abs(proj).max()
            ind = where(abs(proj)==coeff)
            id = ind[0]
            if len(id)>1:
                id = id[0]
            alpha[j,id] = proj[id]
            Aout[j,:] = Aout[j,:]+proj[id]*reshape(basis_stack[id,:,j],[shap[1],])
            vect_select = reshape(copy(basis_stack[id,:,j]),[shap[1],1])

            res[j,:] = res[j,:] - proj[id]*reshape(basis_stack[id,:,j],[shap[1],])
            basis_stack[:,:,j]-= basis_stack[:,:,j].dot(vect_select.dot(transpose(vect_select)))
            for f in range(0,shap[1]):
                n = sqrt(sum(basis_stack[f,:,j]**2))
                if n>10**-31:
                    basis_stack[f,:,j]/=n
            print basis.shape
        print "residual: ",sum(res**2)

    mat_out = alpha.dot(basis)

    return mat_out,alpha


def non_unif_interp_mult_coeff_BT(Aref,comp_ref,posi_ref,posi_target,e,p,tol=10**-5,nb_iter=200): # e,p: notch filters paramters

    shapA = Aref.shape
    nb_ref = shapA[1]
    posi_in = zeros((nb_ref+1,2))
    posi_in[0:nb_ref,:] = posi_ref
    posi_in[nb_ref,:] = posi_target
    dists_unsorted = utils.feat_dist_mat(posi_in)
    dist_med = np.median(dists_unsorted)
    dist_weights_in = zeros((nb_ref+1,nb_ref,shapA[0]))
    for ind in range(0,shapA[0]):
        dist_weights_in[:,:,ind] = (dist_med/dists_unsorted)**p[ind]
        dist_weights_in[:,:,ind] = dist_weights_in[:,:,ind]/sqrt((dist_weights_in[:,:,ind]**2).sum())

    basis = zeros(((nb_ref+1)*shapA[0],nb_ref+1))
    for i in range(0,shapA[0]):
        mati = non_uniform_smoothing_mat_2(dist_weights_in[:,:,i],e[i])
        U, s, Vt = linalg.svd(mati,full_matrices=True)
        basis[i*(nb_ref+1):(i+1)*(nb_ref+1),:] = Vt

    M = transpose(basis[:,:-1]).dot(basis[:,:-1])
    U, s, Vt = linalg.svd(M,full_matrices=True)
    spec_norm = s.max()
    cost = 1
    cost_old=0

    i=0
    alpha = zeros((shapA[0],(nb_ref+1)*shapA[0]))
    t = 1
    alphax = copy(alpha)
    while (i < nb_iter and 100*abs((cost-cost_old)/cost)>0.01) or cost>tol :
        res = Aref-alpha.dot(basis[:,:-1])
        cost_old = cost
        cost = sum(res**2)
        print "--------- mse: ",cost," -------------"
        grad = -res.dot(transpose(basis[:,:-1]))
        alphay = alpha - grad/spec_norm
        alphax_old = copy(alphax)
        alphax = utils.lineskthresholding(alphay,np.int(floor(sqrt(i)))+1)
        told = t
        t = (1+sqrt(4*t**2 +1))/2
        lambd = 1 + (told-1)/t
        alpha  = alphax_old + lambd*(alphax-alphax_old)
        i+=1

    Aout = alpha.dot(basis)
    im_interp = 0*comp_ref[:,:,0]
    for i in range(0,shapA[0]):
        im_interp+=comp_ref[:,:,i]*Aout[i,-1]

    return im_interp

def non_unif_interp_mult_coeff_pos_cp(Aref,comp_ref,posi_ref,posi_target,e,p,tol=10**-5,eps=0.8,theta=0.1,nb_iter=200): # e,p: notch filters paramters
    shapA = Aref.shape
    nb_ref = shapA[1]
    posi_in = zeros((nb_ref+1,2))
    posi_in[0:nb_ref,:] = posi_ref
    posi_in[nb_ref,:] = posi_target
    dists_unsorted = utils.feat_dist_mat(posi_in)
    dist_med = np.median(dists_unsorted)
    dist_weights_in = zeros((nb_ref+1,nb_ref,shapA[0]))
    for ind in range(0,shapA[0]):
        dist_weights_in[:,:,ind] = (dist_med/dists_unsorted)**p[ind]
        dist_weights_in[:,:,ind] = dist_weights_in[:,:,ind]/sqrt((dist_weights_in[:,:,ind]**2).sum())

    basis = zeros(((nb_ref+1)*shapA[0],nb_ref+1))
    for i in range(0,shapA[0]):
        mati = non_uniform_smoothing_mat_2(dist_weights_in[:,:,i],e[i])
        U, s, Vt = linalg.svd(mati,full_matrices=True)
        basis[i*(nb_ref+1):(i+1)*(nb_ref+1),:] = Vt

    M = transpose(basis[:,:-1]).dot(basis[:,:-1])
    U, s, Vt = linalg.svd(M,full_matrices=True)
    spec_norm_fid = s.max()
    lsq_mat = zeros((shapA[0],shapA[0]))
    for i in range(0,shapA[0]):
        for j in range(0,shapA[0]):
            lsq_mat[i,j] = sum(comp_ref[:,:,i]*comp_ref[:,:,j])
    U, s, Vt = linalg.svd(lsq_mat,full_matrices=True)
    spec_norm_pos = s.max()*sum(basis[:,-1]**2)

    spec_norm = sqrt(spec_norm_fid+spec_norm_pos)
    sig = eps/spec_norm
    to = eps/(sig*spec_norm**2)
    cost = 1
    cost_old=0

    i=0

    # Dual variables
    Y1 = copy(Aref)*0
    Y2 = copy(comp_ref[:,:,0])*0

    # Primal variable
    X = zeros((shapA[0],(nb_ref+1)*shapA[0]))

    # Main variable
    alpha = zeros((shapA[0],(nb_ref+1)*shapA[0]))
    res = sum(Aref**2)
    while (i < nb_iter and 100*abs((cost-cost_old)/cost)>0.01) or res>tol :

        # Dual variable update

        # Data attachement
        temp1 = Y1+sig*alpha.dot(basis[:,:-1])
        Y1 = (temp1-sig*Aref)/(1+sig)

        # Positivity
        coeff_temp = sig*alpha.dot(basis[:,-1].reshape(((nb_ref+1)*shapA[0],1)))
        temp2 = comp_ref[:,:,0]*0
        for j in range(0,shapA[0]):
            temp2+=coeff_temp[j]*comp_ref[:,:,j]

        Y2 = temp2-sig*pos_proj_mat(temp2/sig)

        # Primal variable update
        coeff_temp = zeros((shapA[0],1))
        for j in range(0,shapA[0]):
            coeff_temp[j] = sum(comp_ref[:,:,j]*Y2)

        temp31 = coeff_temp.dot(basis[:,-1].reshape((1,(nb_ref+1)*shapA[0])))
        temp3 = temp31 + Y1.dot(transpose(basis[:,:-1]))

        Xold = copy(X)
        X = utils.lineskthresholding(X-to*temp3,np.int(floor(sqrt(i)))+1)

        # Main variable update
        alpha = X+theta*(X-Xold)

        i+=1
        cost_old = cost
        res = cost
        cost = sum((alpha.dot(basis[:,:-1]) - Aref)**2)
        # Sanity check
        print "-------- res: ",cost,"-------- Min val: ",temp2.min()/sig," ---------"

    Aout = alpha.dot(basis)
    im_interp = 0*comp_ref[:,:,0]
    for i in range(0,shapA[0]):
        im_interp+=comp_ref[:,:,i]*Aout[i,-1]

    return im_interp,Aout,basis


def non_unif_interp_mult_coeff(Aref,comp_ref,posi_ref,posi_target,e,p,tol=10**-5,eps=0.8,theta=0.1,nb_iter=200,pos_en=True):

    shap1 = posi_target.shape
    shap2 = comp_ref.shape

    im_interp = zeros((shap2[0],shap2[1],shap1[0]))

    for i in range(0,shap1[0]):
        print "---------- Interpolation ",i+1,"/",shap1[0],"------------"
        if pos_en:
            im_interp[:,:,i],Aouti,basis_i = non_unif_interp_mult_coeff_pos_cp(Aref,comp_ref,posi_ref,posi_target[i,:],e,p,tol=tol,eps=eps,theta=theta,nb_iter=nb_iter)
        else:
            im_interp[:,:,i],Aouti,basis_i = non_unif_interp_mult_coeff_BT(Aref,comp_ref,posi_ref,posi_target[i,:],e,p,tol=tol,eps=eps,nb_iter=nb_iter)

    return im_interp


"""def non_unif_smoothing_mult_coeff_pos_cp_4(im,src,src_hr,tree,weights,l1_rad,weights_thresh,opt=opt,mr_file=mr_file,theta=0.8,p_smth_mat_inv=None,to=None,eps=0.8,nb_iter=100,tol=0.01,Ainit=None,pos_en=0): # to is the weight related to the primal variable

    shap = src.shape
    shap1 = src_hr.shape
    src_mat = zeros((shap[2],shap[2]))
    src_mat = utils.cube_gram_mat(src_hr)
    U, s, Vt = linalg.svd(src_mat,full_matrices=False)
    spec_rad_pos = s.max()
    nb_neigh = tree.shape[1]
    Atemp,mat,v = lsq_mult_coeff_stack(im,src)
    nb_src = Atemp.shape[0]
    if Ainit is None:
        if pos_en==1:
            if nb_src==1:
                Ainit = pos_proj(Atemp)
            elif nb_src>1:
                Ainit = pos_proj_mat(Atemp)
        else:
            Ainit = Atemp

    w_max = 0
    spec_rad_spars = sqrt(spec_rad_pos)*weights_thresh.max()*(l1_rad.min()**(-1))*shap[3]



    spec_rad = zeros((shap[3]))
    for k in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()
    spec_norm=None
    if pos_en==1:
        spec_norm = sqrt(spec_rad.sum()+spec_rad_pos+spec_rad_spars**2)
    else:
        spec_norm = sqrt(spec_rad.sum()+spec_rad_spars**2)
    if to is None:
        a = sum(weights,axis=1)
    #to = (2*a.max())**(-1)
    to = 1/(spec_norm)
    print '----smoothing cp spec_rad-----: ',spec_norm
    if p_smth_mat_inv is None:
        p_smth_mat_inv = zeros((shap[3],shap[3],nb_src))
        p_smth_mat_stack = zeros((shap[3],shap[3],nb_src))
        spec_rad_stack = zeros((nb_src,))
        for ind in range(0,nb_src):
            p_smth_mat = non_uniform_smoothing_mat(weights[:,:,ind])
            #p_smth_mat_stack[:,:,ind] = transpose(p_smth_mat).dot(p_smth_mat)
            p_smth_mat_stack[:,:,ind] = copy(p_smth_mat)
            U, s, Vt = linalg.svd(p_smth_mat_stack[:,:,ind],full_matrices=False)
            spec_rad_stack[ind] = s.max()
        #to = 1.0/spec_rad_stack.max()
        for ind in range(0,nb_src):
            mat_temp = eye(shap[3])+to*p_smth_mat_stack[:,:,ind]
            p_smth_mat_inv[:,:,ind] = LA.inv(mat_temp)

    sig = eps/(to*spec_norm**2) # Weight related to the dual variables
    # Dual variables
    #y1 = copy(Ainit)
    y1 = zeros((shap1[0],shap1[1],shap[3]))
    y2 = im*0
    o = y2*0
    y3 = weights_thresh*0
    rad = zeros((shap[3],))
    for k in range(0,shap[3]):
        res = copy(im[:,:,k])
        for l in range(0,shap[2]):
            res-=Ainit[l,k]*src[:,:,l,k]
        y2[:,:,k] = -res
        rad[k] = (res**2).sum()
    print "--->> ref res: <<---",rad.sum()
    # Primal variable
    v = copy(Ainit)

    # Main variable
    x = copy(Ainit)

    cost=1
    cost_old=0
    iter=0
    temp6 = None
    r = array(range(0,shap[3]))


    while iter<nb_iter and 100*abs(cost-cost_old)/abs(cost)>tol :

        iter+=1
        # Dual variables update
        # Positivity
        #temp1 = y1+sig*x
        temp1 = copy(y1)
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp1[:,:,k] +=sig*x[l,k]*src_hr[:,:,l]
        if shap[2]==1:
            y1 = temp1-sig*pos_proj_mat(temp1/sig)
        else:
            y1 = temp1-sig*pos_proj_cube(temp1/sig)

        # Residual constraint
        temp2 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp2[:,:,k] +=sig*x[l,k]*src[:,:,l,k]
        temp2+=y2-sig*im
        y2 = temp2-sig*proj_sphere_cube(temp2/sig,o,sqrt(rad)) # Moreau id

        # Primal variable update
        temp3 = x*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp3[l,k]+=to*((src[:,:,l,k]*y2[:,:,k]).sum()+(src_hr[:,:,l]*y1[:,:,k]).sum())
        temp4 = v - temp3
        vold = copy(v)
        v = smoothing_prox(p_smth_mat_inv,temp4)
        x = v+theta*(v-vold)
        # Sanity check
        cost_smooth = 0
        for k in range(0,shap[3]):
            ind_k = where(r!=k)
            for l in range(0,shap[3]-1):
                cost_smooth+=(((x[:,k]-x[:,ind_k[0][l]])*sqrt(weights[k,l,:])).sum())**2
        # Objective value
        temp5 = y2*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp5[:,:,k] +=x[l,k]*src[:,:,l,k]
        temp6 = y1*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp6[:,:,k] +=x[l,k]*src_hr[:,:,l]
        max_comp = (temp5*y2).sum()+(y1*temp6).sum()
        temp5-=im
        min_comp = cost_smooth
        cost_old = cost
        cost = min_comp+max_comp
        print "residual: ",(temp5**2).sum(), " smoothness: ",cost_smooth," linear part: ",max_comp," objective val: ",cost," min val: ",temp6.min()

    min_val_map = temp6[:,:,0]
    for i in range(0,shap1[0]):
        for j in range(0,shap1[0]):
            min_val_map[i,j] = temp6[i,j,:].min()
    print "median min val: ",np.median(min_val_map)

    return x,min_val_map"""


def non_uniform_smoothing_mat(weights_mat): # the ith line of the weight matrix contains the weights relating the ith feature to the n-1 other features
    shap = weights_mat.shape

    mat_out = zeros((shap[0],shap[0]))
    mat_temp = zeros((shap[0],shap[0]))
    temp_col = zeros((shap[0],1))
    temp_col_2 = zeros((shap[0],1))
    ind = array(range(0,shap[0]))
    for i in range(0,shap[0]):
        ai = sum(weights_mat[i,:])
        temp_col = temp_col*0
        ik = where(ind!=i)
        temp_col[ik[0]] = reshape(-weights_mat[i,:],(shap[0]-1,1))
        temp_col_2[ik[0]] = reshape(-weights_mat[i,:],(shap[0]-1,1))
        temp_col_2[i] = ai
        mat_temp[:,i] = ndarray.flatten(temp_col_2*ai)
        mat_out += temp_col_2.dot(transpose(temp_col))
    mat_out+= mat_temp

    return mat_out

def non_uniform_smoothing_bas_mat(weights):
    """ **[???]**
    
    """
    shap = weights.shape
    A = zeros((shap[0],shap[0]))
    B = zeros((shap[0],shap[0]))

    range_ref = array(range(0,shap[0]))

    for i in range(0,shap[0]):
        ik = where(range_ref!=i)
        A[ik[0],i] = -weights[i,:]
        B[i,i] = sum(weights[i,:])

    return A,B

def non_uniform_smoothing_mat_2(weights,e): # e>=0
    """ **[???]**
    
    Calls:
    
    * :func:`optim_utils.non_uniform_smoothing_bas_mat`
    """
    A,B = non_uniform_smoothing_bas_mat(weights)
    mat_out = A.dot(transpose(A))+ e*(A.dot(B)+ B.dot(transpose(A))) + (e**2)*B.dot(B) # B is diagonal matrix
    return mat_out

def non_uniform_smoothing_mat_dist_1(dist,expo_range,e):
    """ **[???] Computes some distance matrix, *again*.**
    
    Calls:
    
    * :func:`optim_utils.non_uniform_smoothing_mat_2`
    """
    dist_med = np.median(dist)
    nb_samp = len(expo_range)
    nb_im = dist.shape[0]
    mat_stack = zeros((nb_im,nb_im,nb_samp))
    for i in range(0,nb_samp):
        dist_weights = (dist_med/dist)**expo_range[i]
        dist_weigths = dist_weights/dist_weights.max()
        mat_stack[:,:,i] = non_uniform_smoothing_mat_2(dist_weights,e)
    return mat_stack

def non_uniform_smoothing_mat_dist_2(dist,expo,e_range):
    """ Same as :func:`optim_utils.non_uniform_smoothing_mat_dist_1`, but with
    constant ``expo`` and varying ``e`` (as opposed to constant ``e`` and 
    varying ``expo``.
    
    Calls:
    
    * :func:`optim_utils.non_uniform_smoothing_mat_2`
    """
    dist_med = np.median(dist)
    nb_samp = len(e_range)
    nb_im = dist.shape[0]
    mat_stack = zeros((nb_im,nb_im,nb_samp))
    for i in range(0,nb_samp):
        dist_weights = (dist_med/dist)**expo
        dist_weigths = dist_weights/dist_weights.max()
        mat_stack[:,:,i] = non_uniform_smoothing_mat_2(dist_weights,e_range[i])
    return mat_stack

def notch_filt_optim(test_mat,dist,expo_range,e_range,nb_iter=3,tol=0.01):
    expo_out = None
    e_out = 0.5
    for i in range(0,nb_iter):
        mat_stack = non_uniform_smoothing_mat_dist_1(dist,expo_range,e_out)
        loss = utils.kernel_testmat_stack(mat_stack,test_mat,tol=tol)
        j = where(loss==loss.max())
        expo_out = expo_range[j[0][0]]
        mat_stack = non_uniform_smoothing_mat_dist_2(dist,expo_out,e_range)
        loss = utils.kernel_testmat_stack(mat_stack,test_mat,tol=tol)
        j = where(loss==loss.max())
        e_out = e_range[j[0][0]]
        print "optimal loss: ",loss[i]
        print " Optimum values: ",expo_out," ",e_out#," ",expo_range," ",e_range
    return expo_out,e_out

def notch_filt_optim_2(test_mat,dist,expo_range,e_range,nb_iter=2,tol=0.01):
    """**[???]** Finds notch filter hyperparameters I guess?
    
    Calls:
    
    * :func:`optim_utils.non_uniform_smoothing_mat_dist_1`
    * :func:`utils.kernel_mat_stack_test_unit`
    * :func:`optim_utils.non_uniform_smoothing_mat_dist_2`
    """
    expo_out = None
    e_out = 0.5
    loss = None
    vect = None
    ker = None
    j2 = None
    for i in range(0,nb_iter):
        mat_stack = non_uniform_smoothing_mat_dist_1(dist,expo_range,e_out)
        vect,j,loss,ker,j2 = utils.kernel_mat_stack_test_unit(mat_stack,test_mat,tol=tol)
        print "=========== Loss out ==========: ",loss
        expo_out = expo_range[j]
        mat_stack = non_uniform_smoothing_mat_dist_2(dist,expo_out,e_range)
        vect,j,loss,ker,j2 = utils.kernel_mat_stack_test_unit(mat_stack,test_mat,tol=tol)
        print "=========== Loss out ==========: ",loss
        e_out = e_range[j]

    return expo_out,e_out,loss,vect,ker,j2

def analysis(cube,sig,field_dist,p_min = 0.01,e_min=0.01,e_max=1.99,nb_max=30,tol=0.01):
    """Computes graph-constraint related values, see RCA paper sections 5.2 and (especially) 5.5.3.
    
    
    Calls:
    
    * :func:`utils.knn_interf`
    * :func:`optim_utils.pow_law_select`
    * :func:`utils.feat_dist_mat`
    * :func:`utils.log_sampling`
    * :func:`optim_utils.notch_filt_optim_2`
    * :func:`utils.mat_to_cube`
    """
    nb_samp_opt = 10
    shap = cube.shape
    nb_neighs = shap[2]-1
    neigh,dists = utils.knn_interf(field_dist,nb_neighs)
    p_max = pow_law_select(dists,nb_neighs)
    print "power max = ",p_max

    print "Done..."
    dists_unsorted = utils.feat_dist_mat(field_dist)
    e_range = utils.log_sampling(e_min,e_max,nb_samp_opt)
    p_range = utils.log_sampling(p_min,p_max,nb_samp_opt)
    res_mat = copy(transpose(cube.reshape((shap[0]*shap[1],shap[2]))))

    list_comp = list()
    list_e = list()
    list_p = list()
    list_ind = list()
    list_ker = list()
    err = 1e20
    nb_iter = 0
    while nb_iter<nb_max: # err>sig and
        expo_out,e_out,loss,vect,ker,j = notch_filt_optim_2(res_mat,dists_unsorted,p_range,e_range,nb_iter=3,tol=tol)
        list_e.append(e_out)
        list_p.append(expo_out)
        list_comp.append(vect)
        list_ind.append(j)
        list_ker.append(ker)
        nb_iter+=1
        res_mat = res_mat-transpose(vect).dot(vect.dot(res_mat))
        print "nb_comp: ",nb_iter," residual: ",loss," e: ",e_out," p: ",expo_out,"chosen index: ",j,"/",shap[2]
        err = sum(res_mat**2)

    e_vect = zeros((nb_iter,))
    p_vect = zeros((nb_iter,))
    weights = zeros((nb_iter,shap[2]))
    ker = zeros((nb_iter*shap[2],shap[2]))
    ind = zeros((nb_iter,nb_iter*shap[2]))
    for i in range(0,nb_iter):
        e_vect[i] = list_e[i]
        p_vect[i] = list_p[i]
        weights[i,:] = list_comp[i].reshape((shap[2],))
        ker[i*shap[2]:(i+1)*shap[2],:] = list_ker[i]
        ind[i,i*shap[2]+list_ind[i]] = 1


    res_mat = copy(transpose(cube.reshape((shap[0]*shap[1],shap[2]))))
    proj_coeff = weights.dot(res_mat)
    comp = utils.mat_to_cube(proj_coeff,shap[0],shap[1])

    proj_data = transpose(weights).dot(proj_coeff)
    proj_data = utils.mat_to_cube(proj_data,shap[0],shap[1])

    return e_vect,p_vect,weights,comp,proj_data,ker,ind



def weight_init(weights_mat,data_cube,nb_iter=10,winit=None):
    mat = non_uniform_smoothing_mat(weights_mat)
    siz = data_cube.shape
    w=winit
    if winit is None:
        w = ((data_cube.sum(axis=0))).sum(axis=0)/sqrt(siz[0]*siz[1])
        w = w.reshape((siz[2],1))
    U, s, Vt = linalg.svd(mat,full_matrices=False)
    lip = abs(s).max()
    for i in range(0,nb_iter):
        z = mat.dot(w)/lip
        w = w - z
        #print (z**2).sum()
    w = w.reshape((siz[2],))
    return w

def weights_mat_test(weights,nb_iter=100):
    mse = zeros((nb_iter,))
    shap = weights.shape
    w = np.random.randn(shap[0])

    mat = non_uniform_smoothing_mat(weights)
    U, s, Vt = linalg.svd(mat,full_matrices=False)
    lip = s.max()
    r = array(range(0,shap[0]))
    for i in range(0,nb_iter):
        cost = 0
        for k in range(0,shap[0]):
            ind_k = where(r!=k)
            a=0
            for l in range(0,shap[0]-1):
                a+= (w[k]-w[ind_k[0][l]])*weights[k,l]
            cost+=a**2
        grad = mat.dot(w)
        w = w - grad/lip

    return w

def non_unif_smoothing_mult_coeff_pos_dr(im,src_hr,src_lr,tree,weights,upfact,ker,ker_adj,sig_est,flux_est,p_smth_mat_inv=None,to=None,eps=0.8,nb_iter=100,tol=0.01,Ainit=None,pos_en=1):
    shap = src_lr.shape
    nb_neigh = tree.shape[1]
    Atemp,mat,v = lsq_mult_coeff_stack(im,src_lr)
    nb_src = Atemp.shape[0]
    if Ainit is None:
        Ainit = Atemp
    spec_rad = zeros((shap[3]))
    for k in range(0,shap[3]):
        U, s, Vt = linalg.svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()
    spec_norm = sqrt(spec_rad.sum())
    gam = 1
    if p_smth_mat_inv is None:
        p_smth_mat_inv = zeros((shap[3],shap[3],nb_src))
        for ind in range(0,nb_src):
            p_smth_mat = zeros((shap[3],shap[3]))
            for k in range(0,shap[3]):
                i1,i2 = where(tree==k)
                for l in range(0,nb_neigh):
                    p_smth_mat[k,tree[k,l]]=-weights[k,l,ind]
                w=weights[k,:,ind].sum()
                for j in range(0,size(i1)):
                    p = where((tree[k,:]==i1[j]))
                    if size(p)==0:
                        w+=weights[i1[j],i2[j],ind]
                        p_smth_mat[k,i1[j]] = -weights[i1[j],i2[j],ind]
                p_smth_mat[k,k] = w
            mat_temp = eye(shap[3])+gam*p_smth_mat
            p_smth_mat_inv[:,:,ind] = LA.inv(mat_temp)


    # Inequality constraint radius
    rad = zeros((shap[3],))
    for k in range(0,shap[3]):
        res = copy(im[:,:,k])
        for l in range(0,shap[2]):
            res-=Ainit[l,k]*src_lr[:,:,l,k]
        rad[k] = (res**2).sum()
    print "--->>ref res: <<---",rad.sum()

    # Kernels setting
    ker_in = copy(ker)
    ker_adj_in = copy(ker_adj)
    for k in range(0,shap[3]):
        ker_in[:,:,k] = flux_est[k]*ker_in[:,:,k]/sig_est[k]
        ker_adj_in[:,:,k] = flux_est[k]*ker_adj_in[:,:,k]/sig_est[k]

    # Optimization variables init
    Atemp = copy(Ainit)
    Xtemp = zeros((shap[0]*upfact,shap[1]*upfact,shap[3]))
    for k in range(0,shap[3]):
        for l in range(0,shap[2]):
            Xtemp[:,:,k]+=src_hr[:,:,l]*Ainit[l,k]

    A = None
    X = None
    i=0
    cost=1
    cost_old=0
    iter=0
    corr_coeff=None
    lambd = 1
    print "Orthogonality check: ",(src_hr[:,:,0]**2).sum()," ",(src_hr[:,:,1]**2).sum()," ",(src_hr[:,:,0]*src_hr[:,:,1]).sum()
    while iter<nb_iter and 100*abs(cost-cost_old)/abs(cost)>tol :
        iter+=1
        A,X,corr_coeff = proj_dict_equal_pos_dyks_stack(Atemp,Xtemp,src_hr,corr_coeff=corr_coeff,nb_iter=500,tol=10**(-10))
        Atemp = Atemp+lambd*(smoothing_prox(p_smth_mat_inv,2*A-Atemp)-A)
        Xtemp = Xtemp+lambd*(proj_sphere_decim_conv_stack(2*X-Xtemp,im,sqrt(rad),upfact,ker_in,ker_adj_in)-X)
        # Sanity check
        cost_smooth = 0
        for k in range(0,shap[3]):
            for l in range(0,nb_neigh):
                cost_smooth+=(((A[:,k]-A[:,tree[k,l]])*sqrt(weights[k,l,:]))**2).sum()
        # Objective value
        temp1 = im*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp1[:,:,k] +=A[l,k]*src_lr[:,:,l,k]
        temp1-=im
        # Positivity check
        temp2 = X*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp2[:,:,k]+=src_hr[:,:,l]*A[l,k]
        cost_old = cost
        cost = cost_smooth
        print "residual: ",(temp1**2).sum()," smoothness: ",cost_smooth," min val: ",temp2.min()

    return A


def pt_wise_bound(x,a1,a2):

    x_proj = copy(x)
    i1 = where(x>(np.maximum(a1,a2)).reshape(x.shape))
    i2 = where(x<(np.minimum(a1,a2)).reshape(x.shape))
    i3 = where(a1==a2)
    x_proj[i1[0]] = np.maximum(a1[i1[0]],a2[i1[0]])
    x_proj[i2[0]] = np.minimum(a1[i2[0]],a2[i2[0]])
    x_proj[i3[0]] = a1[i3[0]]
    return x_proj

def grad_nn_unif_smooth(x,tree,weights,wsum):
    grad = x*wsum
    shap = x.shape
    for i in range(0,shap[0]):
        grad[i] = x[i]*(weights[i,:].sum()) - (x[tree[i,:]]*weights[i,:]).sum()

    return grad

def grad_nn_unif_smooth_imp(input):
    x = input[0]
    tree = input[1]
    weights = input[2]
    wsum = sum(weights,axis=1)
    grad = grad_nn_unif_smooth(x,tree,weights,wsum)
    return grad



def proj_affine_pos(x,a,u): # Projects x onto the set {y,ak*y+uk>=0, k=1..n}; a is entry-wise positive
    proj=copy(x)
    id = where(a>0)
    shap = x.shape
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            cons_vec = -u[i,j,id]/a[id]
            proj[i,j] = max(x[i,j],cons_vec.max())
    return proj

def proj_affine_pos2(x,a,u,min_val_map=None): # Projects x onto the set {y,ak*y+uk>=min_val_map, k=1..n};
    proj=copy(x)
    id1 = where(a>0)
    id2 = where(a<0)
    shap = x.shape
    c1 = -10**15
    c2 = 10**15
    if min_val_map is None:
        min_val_map = zeros((shap[0],shap[1]))
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            if size(id1)>0:
                cons_vec_pos = (-u[i,j,id1]+min_val_map[i,j])/a[id1]
                c1 = cons_vec_pos.max()
            if size(id2)>0:
                cons_vec_neg = (-u[i,j,id2]+min_val_map[i,j])/a[id2]
                c2 = cons_vec_neg.min()
            if x[i,j]<=c1:
                proj[i,j]=c1
            elif x[i,j]>=c2:
                proj[i,j]=c2
    return proj

def pow_law_select(dist_weights,nb_neigh,min_val=10**(-15)):
    """ **[???] but related to proximity constrains hyperparameters**
    """

    a = dist_weights[:,0]/dist_weights[:,nb_neigh-1]
    r_med = a.min()
    print "r_med: ",r_med,nb_neigh
    p = log(min_val)/log(r_med)
    return p

def low_rank_comp_wise(psf_stack,upfact,shifts=None,nb_iter=5,nb_subiter=20,nb_subiter_min=7,mu=0.7,nb_comp_max=10,tol=0.1):

    if shifts is None:
        shifts = utils.shift_est(psf_stack)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)
    shap = psf_stack.shape
    weights = zeros((shap[2],nb_comp_max))
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max))
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(w)
    siz_in = upfact*array(shap[0:-1])
    eig_init,spec_rad_init = pow_meth(sr_op_trans_stack,input,siz_in)
    res = copy(psf_stack)
    comp_lr = copy(psf_stack)
    i = 0
    buff = zeros((nb_subiter_min))
    comp_var = zeros((nb_comp_max,))
    while i < nb_comp_max and w.max()>0:
        w = ones((shap[2],))
        spec_rad = spec_rad_init
        eig = eig_init
        resk = res*0
        for k in range(0,nb_iter):
            x = copy(comp[:,:,i])
            t=1
            # kth component estimation
            cur_res=100
            res_old=1
            j=0
            while j < nb_subiter and 100*abs(cur_res-res_old)/res_old>0.1:

                res_old = cur_res
                # Gradient computation
                grad = x*0
                for l in range(0,shap[2]):
                    resk[:,:,l] = res[:,:,l]-w[l]*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                    grad = grad -w[l]*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')

                buff[j%nb_subiter_min] = (resk**2).sum()

                print "mse: ", buff[j%nb_subiter_min]
                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]

                y = comp[:,:,i] - mu*grad/spec_rad
                # Positivity constraint
                xnew = proj_affine_pos(y,w,im_hr)
                #xnew = pos_proj_mat(y)
                tnew  = (1+sqrt(4*t**2+1))/2
                lambd = 1 + (t-1)/tnew
                t = tnew
                # Acceleration
                comp[:,:,i] = x+lambd*(xnew-x)
                x = xnew
                j+=1
            for l in range(0,shap[2]):
                comp_lr[:,:,l] = utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)

            # kth component estimation coeff estimation
            w = pos_proj(lsq_coeff_stack(comp_lr,res))
            print "weights: ",w
            input[4]=w
            eig,spec_rad = pow_meth(sr_op_trans_stack,input,siz_in,ainit=eig)
        comp_var[i] = std(w)*sqrt(((comp[:,:,i])**2).sum())
        print "---------- ",i,"th component variability ----------- :",comp_var[i]
        # Residual update and HR images update
        if w.max()>0:
            for l in range(0,shap[2]):
                res[:,:,l] = res[:,:,l]-w[l]*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                im_hr[:,:,l] = im_hr[:,:,l]+w[l]*comp[:,:,i]
            weights[:,i] = w
        i+=1

    return im_hr,comp,weights,res



def low_rank_comp_wise_sparse(psf_stack,upfact,opt,nsig,shifts=None,nb_iter=5,nb_subiter=20,nb_subiter_min=7,mu=0.7,nb_comp_max=10,tol=0.1):

    if shifts is None:
        shifts = utils.shift_est(psf_stack)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)
    shap = psf_stack.shape
    weights = zeros((shap[2],nb_comp_max))
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max))
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(w)
    input_trans = cp.deepcopy(input)


    siz_in = upfact*array(shap[0:-1])
    eig_max = None
    eig_min_init = None
    eig_vect_min,gam = min_eig_val(sr_op_trans_stack,input,siz_in,eig_max=eig_max,eig_min_init=eig_min_init)
    eig_min_init = copy(eig_vect_min)



    # Accelerated Chamb-Pock variables
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    x = zeros((upfact*shap[0],upfact*shap[1])) # Primal variable
    xold = zeros((upfact*shap[0],upfact*shap[1]))
    u,mr_file = isap.mr_trans(comp[:,:,0],opt=opt) # First dual variable
    u = u*0
    v = 0*comp[:,:,0] # Second dual variable
    grad = x*0

    shap2 = u.shape
    weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
    res = copy(psf_stack)
    comp_lr = copy(psf_stack)
    i = 0
    buff = zeros((nb_subiter_min))
    comp_var = zeros((nb_comp_max,))
    rho = 0.9 # to*rho<L**2
    to = 0.9
    to_old = ones((2,))*to
    theta=0.8
    input_trans.append(to)

    input_trans_prox = cp.deepcopy(input_trans)
    input_trans_prox.append(copy(x))
    input_trans_prox.append(copy(psf_stack))

    nb_iter_noisest = 20

    thresh_type = 1
    tol_conj_grad=10**(-10)
    while i < nb_comp_max and w.max()>0:
        if i==1:
            tol_conj_grad = 10**(-10)
        w = ones((shap[2],))
        #spec_rad = spec_rad_init
        #eig = eig_init
        resk = res*0
        input_trans_prox[-1] = copy(res)
        weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
        input[4]=w
        input_trans[4]=w
        input_trans_prox[4]=w

        for k in range(0,nb_iter):
            x = copy(comp[:,:,i])
            t=1
            # kth component estimation
            cur_res=100
            res_old=1
            j=0
            while j < nb_subiter and 100*abs(cur_res-res_old)/res_old>0.1:
                res_old = cur_res
                # Residual computation
                grad = 0*grad
                for l in range(0,shap[2]):
                    resk[:,:,l] = res[:,:,l]-w[l]*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                    grad = grad -w[l]*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')

                # Noise estimation
                input_trans[5] = to_old[k%2]
                input_trans[0] = copy(grad)
                noise_vect_1 = sr_op_trans_stack_translate_inv(input_trans,nb_iter_noisest)
                sig_map = utils.res_sig_map(noise_vect_1*to_old[k%2]*rho,opt=opt)
                thresh_map = nsig*sig_map

                # Dual variables update
                # Positivity
                t1 = v+rho*comp[:,:,i]
                v = t1-rho*proj_affine_pos(t1/rho,w,im_hr)
                # Sparsity
                t2,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
                t3 = u+rho*t2
                u = utils.l_inf_ball_proj_3D(t3,thresh_map*weights_sp,thresh_type)

                # Primal variable update
                t4 = isap.mr_recons_coeff(u,mr_file)+v
                t5 = x - to_old[(k+1)%2]*t4
                xold = copy(x)
                input_trans_prox[-3] = to_old[(k+1)%2]
                input_trans_prox[-2] = t5
                # Ref value
                prox_cost=0
                for l in range(0,shap[2]):
                    prox_cost = prox_cost + ((res[:,:,l]-w[l]*utils.decim(scisig.fftconvolve(t5,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0))**2).sum()
                print "ref prox cost value: ",to_old[(k+1)%2]*prox_cost
                x = sr_op_trans_stack_prox(input_trans_prox,nb_iter_noisest,tol=tol_conj_grad)
                # Optimal value
                prox_opt = 0
                for l in range(0,shap[2]):
                    prox_cost = prox_cost + ((res[:,:,l]-w[l]*utils.decim(scisig.fftconvolve(x,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0))**2).sum()
                print "optimal prox cost value: ",to_old[(k+1)%2]*prox_cost+((x-t5)**2).sum()

                # Hyper parameters update
                #theta = 1.0/sqrt(1+2*gam*to_old[(k+1)%2])
                #to_old[k%2] = theta*to_old[(k+1)%2]
                #rho = rho/theta

                # Acceleration
                comp[:,:,i] = x+theta*(x-xold)
                j+=1



            for l in range(0,shap[2]):
                comp_lr[:,:,l] = utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)

            # kth component estimation coeff estimation
            w = lsq_coeff_stack(comp_lr,res)
            print "weights: ",w
            input[4]=w
            eig_vect_min,gam = min_eig_val(sr_op_trans_stack,input,siz_in,eig_max=eig_max,eig_min_init=eig_min_init)
            eig_min_init = copy(eig_vect_min)
            input_trans[4]=w
            input_trans_prox[4]=w
            # Reweighting
            coeffx,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
            weights_sp  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)

        comp_var[i] = std(w)*sqrt(((comp[:,:,i])**2).sum())
        #print "---------- ",i,"th component variability ----------- :",comp_var[i]
        # Residual update and HR images update
        if w.max()>0:
            for l in range(0,shap[2]):
                res[:,:,l] = res[:,:,l]-w[l]*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                im_hr[:,:,l] = im_hr[:,:,l]+w[l]*comp[:,:,i]
            weights[:,i] = w
        i+=1

    return im_hr,comp,weights,res


def low_rank_comp_wise_sparse_dist(psf_stack_in,field_dist,upfact,opt,nsig,neigh_frac=0.5,dist_weight_deg=1,shifts=None,opt_shift_est=None,nsig_shift_est=None,sig_est=None,flux_est=None,nb_iter=5,nb_subiter=20,nb_subiter_min=7,mu=0.7,nb_comp_max=10,tol=0.1,wvl_transp_cor=None,accel_en=False,decay_fact=1):
    psf_stack = copy(psf_stack_in)

    # Degradation operator parameters estimation
    if shifts is None:
        #shifts = utils.shift_est(psf_stack)*upfact
        shifts = utils.wvl_shift_est(psf_stack,opt_shift_est,nsig_shift_est)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)
    if sig_est is None:
        sig_est = utils.im_gauss_nois_est_cube(psf_stack_in,opt=opt_shift_est)
    sig_min = sig_est.min()
    sig_est = sig_est/sig_min
    shap = psf_stack.shape
    for k in range(0,shap[2]):
        psf_stack[:,:,k] = psf_stack[:,:,k]/sig_est[k]
    if flux_est is None:
        flux_est = utils.flux_estimate_stack(psf_stack_in,rad=4)
    flux_ref = np.median(flux_est)
    weights = zeros((shap[2],nb_comp_max))
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max))
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(w)
    input.append(sig_est)
    input.append(flux_est)
    input_trans = cp.deepcopy(input)

    # Distances settings
    print "Contructing PSF tree..."
    neigh,dists = utils.knn_interf(field_dist,int(shap[2]*neigh_frac))
    print "Done..."
    dist_med = np.median(dists)
    dist_weights = (dist_med/dists)**dist_weight_deg
    # Spatial smoothing operator gradient lip constant estimation
    spec_rad_smooth = None

    """input_dist_op = list()
    input_dist_op.append(w)
    input_dist_op.append(neigh)
    input_dist_op.append(dist_weights)
    eig_smooth,spec_rad_smooth = pow_meth(grad_nn_unif_smooth_imp,input_dist_op,shap[2])"""

    siz_in = upfact*array(shap[0:-1])
    eig_max = None
    eig_min_init = None
    gam=None
    if accel_en==True:
        eig_vect_min,gam = min_eig_val(sr_op_trans_stack,input,siz_in,eig_max=eig_max,eig_min_init=eig_min_init)
        gam = gam*0.9
        eig_min_init = copy(eig_vect_min)


    # Accelerated Chamb-Pock variables
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    comp_old = zeros((upfact*shap[0],upfact*shap[1]))
    x = zeros((upfact*shap[0],upfact*shap[1])) # Primal variable
    u,mr_file = isap.mr_trans(comp[:,:,0],opt=opt) # First dual variable
    u = u*0
    v = 0*comp[:,:,0] # Second dual variable
    grad = x*0

    shap2 = u.shape
    weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
    res = copy(psf_stack)
    comp_lr = copy(psf_stack)
    i = 0
    buff = zeros((nb_subiter_min))
    comp_var = zeros((nb_comp_max,))
    rho = 0.9/sqrt(2) # to*rho<L**(-2)
    to = 0.9/sqrt(2)
    to_old = ones((2,))*to
    theta=1
    input_trans.append(to)

    input_trans_prox = cp.deepcopy(input_trans)
    input_trans_prox.append(copy(x))
    input_trans_prox.append(copy(psf_stack))

    nb_iter_noisest = 20
    thresh_type = 1
    noise_vect_1 = None
    sig_map = None
    cur_res=100000000.0
    res_old=1000000000.0
    while i < nb_comp_max:
        if i>0:
            w = utils.abs_val_reverting(weights[:,i-1])
        #spec_rad = spec_rad_init
        #eig = eig_init
        resk = res*0
        input_trans_prox[-1] = copy(res)
        weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
        input[4]=w
        input_trans[4]=w
        input_trans_prox[4]=w

        for k in range(0,nb_iter):
            t=1
            # ith component estimation
            j=0
            #print '--------------- ',nb_subiter,' ----------------'
            res_old = cur_res*2

            while j < nb_subiter and 100*abs(cur_res-res_old)/res_old>0.1:# and cur_res<res_old:
                res_old = cur_res
                # Residual computation
                grad = grad*0
                if j==0 and k==0:
                    for l in range(0,shap[2]):
                        resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                else:
                    grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')

                buff[j%nb_subiter_min] = (resk**2).sum()


                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]

                # Noise estimation
                input_trans[7] = to_old[k%2]
                input_trans[0] = copy(grad)
                if j==0:
                    noise_vect_1 = sr_op_trans_stack_translate_inv(input_trans,nb_iter_noisest)
                    sig_map = utils.res_sig_map(noise_vect_1*to_old[k%2]*rho,opt=opt)
                if j==0:
                    print "Estimated noise at the first scales: ",sig_map[0,0,:]
                thresh_map = nsig*sig_map

                # Dual variables update
                # Positivity
                t1 = v+rho*comp[:,:,i]
                v = t1-rho*proj_affine_pos2(t1/rho,w,im_hr)
                # Sparsity
                t2,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
                if i==0 and k==0 and (comp[:,:,i]**2).max()>0:
                    tempy = isap.mr_recons_coeff(t2,mr_file)
                    a = (comp[:,:,i]*tempy).sum()
                    b = (t2**2).sum()
                    wvl_transp_cor = b/a
                    print '--------- correction coeff: ',wvl_transp_cor,' --------------------'

                t3 = u+rho*t2
                u = utils.l_inf_ball_proj_3D(t3,thresh_map*weights_sp,thresh_type)

                # Primal variable update
                t4 = isap.mr_recons_coeff(u,mr_file)
                if wvl_transp_cor is None:
                    t5 = t4+v
                else:
                    t5 = wvl_transp_cor*t4+v
                t6 = x - to_old[(k+1)%2]*t5
                xold = copy(x)
                input_trans_prox[-3] = to_old[(k+1)%2]
                input_trans_prox[-2] = t6
                # Ref value
                prox_cost=0
                for l in range(0,shap[2]):
                    prox_cost = prox_cost + ((res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(t6,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0))**2).sum()
                print "ref prox cost value: ",to_old[(k+1)%2]*prox_cost
                x = sr_op_trans_stack_prox(input_trans_prox,nb_iter_noisest)
                # Optimal value
                prox_opt = 0
                for l in range(0,shap[2]):
                    prox_opt = prox_opt + ((res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(x,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0))**2).sum()
                print "optimal prox cost value: ",to_old[(k+1)%2]*prox_opt+((x-t6)**2).sum()

                # Acceleration
                # Hyper parameters update
                if accel_en==True:
                    theta = 1.0/sqrt(1+2*gam*to_old[(k+1)%2])
                    to_old[k%2] = theta*to_old[(k+1)%2]
                    rho = rho/theta
                else:
                    theta = theta*exp((-j/decay_fact))
                comp_old = copy(comp[:,:,i])
                comp[:,:,i] = x+theta*(x-xold)
                j+=1

                for l in range(0,shap[2]):
                    resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                buff[j%nb_subiter_min] = (resk**2).sum()

                print "mse: ", buff[j%nb_subiter_min]
                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]

                if i==0 and k==0 and wvl_transp_cor is not None:
                    rho = 0.9/sqrt(1+wvl_transp_cor)
                    to_old = 0.9*ones((2,))/sqrt(1+wvl_transp_cor)

                if j==1 and cur_res>res_old:
                    res_old = 2*cur_res        # Might tolerate a degration of the residual in the first iteration of the estimation loop of a component
            if cur_res>res_old:
                comp[:,:,i]=copy(comp_old)
            for l in range(0,shap[2]):
                comp_lr[:,:,l] = (flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)


            if k<nb_iter-1:
                if accel_en==True:
                    eig_vect_min,gam = min_eig_val(sr_op_trans_stack,input,siz_in,eig_max=eig_max,eig_min_init=eig_min_init)
                    gam = 0.9*gam
                    eig_min_init = copy(eig_vect_min)

                # Reweighting
                coeffx,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
                weights_sp  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)
            # ith component coeff estimation
            if k==0:
                w = lsq_coeff_stack(comp_lr,res)
            else:
                w = non_unif_smoothing(neigh,dist_weights,w,res,comp_lr,mu=1,spec_rad=spec_rad_smooth,nb_iter=100)
            print "weights: ",w
            input[4]=w
            input_trans[4]=w
            input_trans_prox[4]=w

            # Hyper parameters reset
            rho = 0.9/sqrt(1+wvl_transp_cor)
            to_old = 0.9*ones((2,))/sqrt(1+wvl_transp_cor)

        # Residual update and HR images update
        if abs(w).max()>0:
            for l in range(0,shap[2]):
                res[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                im_hr[:,:,l] = im_hr[:,:,l]+w[l]*comp[:,:,i]
            weights[:,i] = w
        i+=1
        # Optim variables resetting
        x = x*0
        u = u*0
        v = v*0

    return im_hr,comp,weights,res,sig_est*sig_min,flux_est,shifts

def low_rank_comp_wise_sparse_dist_recond(psf_stack_in,field_dist,upfact,opt,nsig,neigh_frac=0.5,dist_weight_deg=1,shifts=None,opt_shift_est=None,nsig_shift_est=None,sig_est=None,flux_est=None,nb_iter=5,nb_subiter=20,nb_subiter_min=7,mu=0.7,nb_comp_max=10,tol=0.1,wvl_transp_cor=None,accel_en=False,decay_fact=1,cond_fact=0.1,weights_bal=0.05):
    print "---------------- Reconditioned RCA -----------------"
    psf_stack = copy(psf_stack_in)

    # Degradation operator parameters estimation
    if sig_est is None:
        sig_est = utils.im_gauss_nois_est_cube(psf_stack_in,opt=opt_shift_est)
    if shifts is None:
        #shifts = utils.shift_est(psf_stack)*upfact
        #shifts = utils.wvl_shift_est(psf_stack,opt_shift_est,nsig_shift_est)*upfact
        shifts = utils.thresh_shift_est(im_stack,sig_thresh=sig_est,opt=opt_shift_est)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)

    sig_min = sig_est.min()
    sig_est = sig_est/sig_min
    shap = psf_stack.shape
    for k in range(0,shap[2]):
        psf_stack[:,:,k] = psf_stack[:,:,k]/sig_est[k]
    if flux_est is None:
        flux_est = utils.flux_estimate_stack(psf_stack_in,rad=4)
    flux_ref = np.median(flux_est)
    weights = zeros((shap[2],nb_comp_max))
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max))
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(w)
    input.append(sig_est)
    input.append(flux_est)
    input_trans = cp.deepcopy(input)

    # Distances settings
    print "Contructing PSF tree..."
    neigh,dists = utils.knn_interf(field_dist,int(shap[2]*neigh_frac))
    print "Done..."
    dist_med = np.median(dists)
    dist_weights = (dist_med/dists)**dist_weight_deg
    # Spatial smoothing operator gradient lip constant estimation
    spec_rad_smooth = None

    """input_dist_op = list()
        input_dist_op.append(w)
        input_dist_op.append(neigh)
        input_dist_op.append(dist_weights)
        eig_smooth,spec_rad_smooth = pow_meth(grad_nn_unif_smooth_imp,input_dist_op,shap[2])"""

    siz_in = upfact*array(shap[0:-1])
    eig_max = None
    eig_min_init = None
    gam=None
    if accel_en==True:
        eig_vect_min,gam = min_eig_val(sr_op_trans_stack,input,siz_in,eig_max=eig_max,eig_min_init=eig_min_init)
        gam = gam*0.9
        eig_min_init = copy(eig_vect_min)



    # Accelerated Chamb-Pock variables
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    comp_old = zeros((upfact*shap[0],upfact*shap[1]))
    x = zeros((upfact*shap[0],upfact*shap[1])) # Primal variable
    u,mr_file = isap.mr_trans(comp[:,:,0],opt=opt) # First dual variable
    u = u*0
    v = 0*comp[:,:,0] # Second dual variable
    grad = x*0

    shap2 = u.shape
    weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
    res = copy(psf_stack)
    comp_lr = copy(psf_stack)
    i = 0
    buff = zeros((nb_subiter_min))
    comp_var = zeros((nb_comp_max,))

    rho = weights_bal*0.9/sqrt(3) # to*rho<L**(-2)
    to = 0.9/(3*rho)
    to_old = ones((2,))*to
    theta=1
    input_trans.append(to)

    input_trans_prox = cp.deepcopy(input_trans)
    input_trans_prox.append(copy(x))
    input_trans_prox.append(copy(psf_stack))

    nb_iter_noisest = 20
    thresh_type = 1
    noise_vect_1 = None
    sig_map = None
    cur_res=100000000.0
    res_old=1000000000.0
    while i < nb_comp_max:
        if i>0:
            w = utils.vect_recond(utils.abs_val_reverting(weights[:,i-1]),cond_fact)
        #spec_rad = spec_rad_init
        #eig = eig_init
        resk = res*0
        input_trans_prox[-1] = copy(res)
        weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
        input[4]=w
        input_trans[4]=w
        input_trans_prox[4]=w

        for k in range(0,nb_iter):
            t=1
            # ith component estimation
            j=0
            #print '--------------- ',nb_subiter,' ----------------'
            res_old = cur_res*2

            while j < nb_subiter and 100*abs(cur_res-res_old)/res_old>0.1:# and cur_res<res_old:
                res_old = cur_res
                # Residual computation
                grad = grad*0
                if j==0 and k==0:
                    for l in range(0,shap[2]):
                        resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                else:
                    grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')

                buff[j%nb_subiter_min] = (resk**2).sum()


                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]

                # Noise estimation
                input_trans[7] = to_old[k%2]
                input_trans[0] = copy(grad)
                if j==0:
                    noise_vect_1 = sr_op_trans_stack_translate_inv(input_trans,nb_iter_noisest)
                    sig_map = utils.res_sig_map(noise_vect_1*to_old[k%2]*rho,opt=opt)
                if j==0:
                    print "Estimated noise at the first scales: ",sig_map[0,0,:]
                thresh_map = nsig*sig_map

                # Dual variables update
                # Positivity
                t1 = v+rho*comp[:,:,i]
                v = t1-rho*proj_affine_pos2(t1/rho,w,im_hr)
                # Sparsity
                t2,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
                if i==0 and k==0 and (comp[:,:,i]**2).max()>0:
                    tempy = isap.mr_recons_coeff(t2,mr_file)
                    a = (comp[:,:,i]*tempy).sum()
                    b = (t2**2).sum()
                    wvl_transp_cor = b/a
                    print '--------- correction coeff: ',wvl_transp_cor,' --------------------'

                t3 = u+rho*t2
                u = utils.l_inf_ball_proj_3D(t3,thresh_map*weights_sp,thresh_type)

                # Primal variable update
                t4 = isap.mr_recons_coeff(u,mr_file)
                if wvl_transp_cor is None:
                    t5 = t4+v
                else:
                    t5 = wvl_transp_cor*t4+v
                t6 = x - to_old[(k+1)%2]*t5
                xold = copy(x)
                input_trans_prox[-3] = to_old[(k+1)%2]
                input_trans_prox[-2] = t6
                # Ref value
                prox_cost=0
                for l in range(0,shap[2]):
                    prox_cost = prox_cost + ((res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(t6,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0))**2).sum()
                print "ref prox cost value: ",to_old[(k+1)%2]*prox_cost
                x = sr_op_trans_stack_prox(input_trans_prox,nb_iter_noisest)
                # Optimal value
                prox_opt = 0
                for l in range(0,shap[2]):
                    prox_opt = prox_opt + ((res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(x,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0))**2).sum()
                print "optimal prox cost value: ",to_old[(k+1)%2]*prox_opt+((x-t6)**2).sum()

                # Acceleration
                # Hyper parameters update
                if accel_en==True:
                    theta = 1.0/sqrt(1+2*gam*to_old[(k+1)%2])
                    to_old[k%2] = theta*to_old[(k+1)%2]
                    rho = rho/theta
                else:
                    theta = theta*exp((-j/decay_fact))
                comp_old = copy(comp[:,:,i])
                comp[:,:,i] = x+theta*(x-xold)
                j+=1

                for l in range(0,shap[2]):
                    resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                buff[j%nb_subiter_min] = (resk**2).sum()

                print "mse: ", buff[j%nb_subiter_min]
                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]

                if i==0 and k==0 and wvl_transp_cor is not None:
                    rho = 0.9/sqrt(1+wvl_transp_cor)
                    to_old = 0.9*ones((2,))/sqrt(1+wvl_transp_cor)

                #if j==1 and cur_res>res_old:
                #   res_old = 2*cur_res        # Might tolerate a degration of the residual in the first iteration of the estimation loop of a component
            """if cur_res>res_old:
                comp[:,:,i]=copy(comp_old)"""
            for l in range(0,shap[2]):
                comp_lr[:,:,l] = (flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)


            if k<nb_iter-1:
                if accel_en==True:
                    eig_vect_min,gam = min_eig_val(sr_op_trans_stack,input,siz_in,eig_max=eig_max,eig_min_init=eig_min_init)
                    gam = 0.9*gam
                    eig_min_init = copy(eig_vect_min)

                # Reweighting
                coeffx,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
                weights_sp  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)
            # ith component coeff estimation
            if k==0:
                w = lsq_coeff_stack(comp_lr,res)
            else:
                w = non_unif_smoothing(neigh,dist_weights,w,res,comp_lr,mu=1,spec_rad=spec_rad_smooth,nb_iter=100)
            print "weights: ",w
            if k<nb_iter-1:
                w = utils.vect_recond(w,cond_fact)
                print "weights balanced: ",w
            input[4]=w
            input_trans[4]=w
            input_trans_prox[4]=w

            # Hyper parameters reset
            rho = weights_bal*0.9/sqrt(1+wvl_transp_cor)
            to_old = 0.9*ones((2,))/((1+wvl_transp_cor)*rho)

        # Residual update and HR images update
        if abs(w).max()>0:
            for l in range(0,shap[2]):
                res[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                im_hr[:,:,l] = im_hr[:,:,l]+w[l]*comp[:,:,i]
            weights[:,i] = w
        i+=1
        # Optim variables resetting
        x = x*0
        u = u*0
        v = v*0

    return im_hr,comp,weights,res,sig_est*sig_min,flux_est,shifts

def low_rank_comp_wise_sparse_dist_recond_2(psf_stack_in,field_dist,upfact,opt,nsig,neigh_frac=0.5,dist_weight_deg=1,shifts=None,opt_shift_est=None,nsig_shift_est=None,sig_est=None,flux_est=None,nb_iter=5,nb_subiter=20,nb_subiter_min=7,mu=0.7,nb_comp_max=10,tol=0.1,wvl_transp_cor=None,accel_en=False,decay_fact=1,cond_fact=1.0):
    print "---------------- Reconditioned RCA 2 -----------------"
    psf_stack = copy(psf_stack_in)

    # Degradation operator parameters estimation
    if sig_est is None:
        sig_est = utils.im_gauss_nois_est_cube(psf_stack_in,opt=opt_shift_est)
    if shifts is None:
        #shifts = utils.shift_est(psf_stack)*upfact
        #shifts = utils.wvl_shift_est(psf_stack,opt_shift_est,nsig_shift_est)*upfact
        shifts = utils.thresh_shift_est(psf_stack,sig_thresh=sig_est,opt=opt_shift_est)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)

    sig_min = sig_est.min()
    sig_est = sig_est/sig_min
    shap = psf_stack.shape
    for k in range(0,shap[2]):
        psf_stack[:,:,k] = psf_stack[:,:,k]/sig_est[k]
    if flux_est is None:
        flux_est = utils.flux_estimate_stack(psf_stack_in,rad=4)
    flux_ref = np.median(flux_est)
    weights = zeros((shap[2],nb_comp_max))
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max))
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(w)
    input.append(sig_est)
    input.append(flux_est)
    input_trans = cp.deepcopy(input)

    # Distances settings
    print "Contructing PSF tree..."
    neigh,dists = utils.knn_interf(field_dist,int(shap[2]*neigh_frac))
    print "Done..."
    dist_med = np.median(dists)
    dist_weights = (dist_med/dists)**dist_weight_deg
    # Spatial smoothing operator gradient lip constant estimation
    spec_rad_smooth = None

    """input_dist_op = list()
        input_dist_op.append(w)
        input_dist_op.append(neigh)
        input_dist_op.append(dist_weights)
        eig_smooth,spec_rad_smooth = pow_meth(grad_nn_unif_smooth_imp,input_dist_op,shap[2])"""

    siz_in = upfact*array(shap[0:-1])
    eig_max = None
    eig_min_init = None
    gam=None
    if accel_en==True:
        eig_vect_min,gam = min_eig_val(sr_op_trans_stack,input,siz_in,eig_max=eig_max,eig_min_init=eig_min_init)
        gam = gam*0.9
        eig_min_init = copy(eig_vect_min)

    # Accelerated Chamb-Pock variables
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    comp_old = zeros((upfact*shap[0],upfact*shap[1]))
    x = zeros((upfact*shap[0],upfact*shap[1])) # Primal variable
    u,mr_file = isap.mr_trans(comp[:,:,0],opt=opt) # First dual variable
    u = u*0
    v = 0*comp[:,:,0] # Second dual variable
    grad = x*0

    shap2 = u.shape
    weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
    res = copy(psf_stack)
    comp_lr = copy(psf_stack)
    i = 0
    buff = zeros((nb_subiter_min))
    comp_var = zeros((nb_comp_max,))

    to = None
    rho = None # to*rho<L**(-2)
    to_old = None
    theta=1
    input_trans.append(to)

    input_trans_prox = cp.deepcopy(input_trans)
    input_trans_prox.append(copy(x))
    input_trans_prox.append(copy(psf_stack))

    nb_iter_noisest = 20
    thresh_type = 1
    noise_vect_1 = None
    sig_map = None
    cur_res=100000000.0
    res_old=1000000000.0
    abs_count=1
    while i < nb_comp_max:
        if i>0:
            w = utils.abs_val_reverting(weights[:,i-1])
        #spec_rad = spec_rad_init
        #eig = eig_init
        resk = res*0
        input_trans_prox[-1] = copy(res)
        weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
        input[4]=w
        input_trans[4]=w
        input_trans_prox[4]=w
        eig_max,spec_rad = pow_meth(sr_op_trans_stack,input,siz_in,ainit=eig_max)
        print "spectral radius: ",spec_rad
        # Hyper parameters reset
        to = cond_fact/spec_rad
        if wvl_transp_cor is None:
            rho = 0.9/((3*to)*abs_count) # to*rho<L**(-2)
        else:
            rho = 0.9/(((1+wvl_transp_cor)*to)*abs_count) # to*rho<L**(-2)
        to_old = ones((2,))*to
        input_trans_prox[-3] = to_old[(k+1)%2]
        input_trans[7] = to_old[(k+1)%2]
        for k in range(0,nb_iter):
            t=1
            # ith component estimation
            j=0
            #print '--------------- ',nb_subiter,' ----------------'
            res_old = cur_res*2

            while j < nb_subiter and 100*abs(cur_res-res_old)/res_old>0.1:# and cur_res<res_old:
                abs_count+=1
                res_old = cur_res
                # Residual computation
                grad = grad*0
                if j==0 and k==0:
                    for l in range(0,shap[2]):
                        resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                else:
                    grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')

                buff[j%nb_subiter_min] = (resk**2).sum()


                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]

                # Noise estimation
                input_trans[7] = to_old[k%2]
                input_trans[0] = copy(grad)
                if j==0:
                    noise_vect_1 = sr_op_trans_stack_translate_inv(input_trans,nb_iter_noisest)
                    sig_map = utils.res_sig_map(noise_vect_1*to_old[k%2]*rho,opt=opt)
                if j==0:
                    print "Estimated noise at the first scales: ",sig_map[0,0,:]
                thresh_map = nsig*sig_map

                # Dual variables update
                if i==0 and k==0 and wvl_transp_cor is not None:
                    to = cond_fact/spec_rad
                    rho = 0.9/((1+wvl_transp_cor)*to*abs_count) # to*rho<L**(-2)
                    to_old = ones((2,))*to
                    input_trans_prox[-3] = to_old[(k+1)%2]
                elif wvl_transp_cor is not None:
                    rho = 0.9/((1+wvl_transp_cor)*to*abs_count)
                # Positivity
                t1 = v+rho*comp[:,:,i]
                v = t1-rho*proj_affine_pos2(t1/rho,w,im_hr)
                # Sparsity
                t2,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
                if i==0 and k==0 and (comp[:,:,i]**2).max()>0:
                    tempy = isap.mr_recons_coeff(t2,mr_file)
                    a = (comp[:,:,i]*tempy).sum()
                    b = (t2**2).sum()
                    wvl_transp_cor = a/b
                    print '--------- correction coeff: ',wvl_transp_cor,' --------------------'

                t3 = u+rho*t2
                u = utils.l_inf_ball_proj_3D(t3,thresh_map*weights_sp,thresh_type)

                # Primal variable update
                t4 = isap.mr_recons_coeff(u,mr_file)
                if wvl_transp_cor is None:
                    t5 = t4+v
                else:
                    t5 = wvl_transp_cor*t4+v
                t6 = x - to_old[(k+1)%2]*t5
                xold = copy(x)
                input_trans_prox[-3] = to_old[(k+1)%2]
                input_trans_prox[-2] = t6
                # Ref value
                prox_cost=0
                for l in range(0,shap[2]):
                    prox_cost = prox_cost + ((res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(t6,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0))**2).sum()
                print "ref prox cost value: ",to_old[(k+1)%2]*prox_cost
                x = sr_op_trans_stack_prox(input_trans_prox,nb_iter_noisest)
                # Optimal value
                prox_opt = 0
                for l in range(0,shap[2]):
                    prox_opt = prox_opt + ((res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(x,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0))**2).sum()
                print "optimal prox cost value: ",to_old[(k+1)%2]*prox_opt+((x-t6)**2).sum()

                # Acceleration
                # Hyper parameters update
                if accel_en==True:
                    theta = 1.0/sqrt(1+2*gam*to_old[(k+1)%2])
                    to_old[k%2] = theta*to_old[(k+1)%2]
                    rho = rho/theta
                #else:
                    #theta = theta*exp((-j/decay_fact))
                comp_old = copy(comp[:,:,i])
                comp[:,:,i] = x+theta*(x-xold)
                j+=1

                for l in range(0,shap[2]):
                    resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                buff[j%nb_subiter_min] = (resk**2).sum()

                print "mse: ", buff[j%nb_subiter_min]
                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]

                 # to*rho<L**(-2)
                    #input_trans[8] = to_old[(k+1)%2]
            #if j==1 and cur_res>res_old:
            #   res_old = 2*cur_res        # Might tolerate a degration of the residual in the first iteration of the estimation loop of a component
            """if cur_res>res_old:
                comp[:,:,i]=copy(comp_old)"""
            for l in range(0,shap[2]):
                comp_lr[:,:,l] = (flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)


            if k<nb_iter-1:
                if accel_en==True:
                    eig_vect_min,gam = min_eig_val(sr_op_trans_stack,input,siz_in,eig_max=eig_max,eig_min_init=eig_min_init)
                    gam = 0.9*gam
                    eig_min_init = copy(eig_vect_min)
                # Reweighting
                coeffx,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
                weights_sp  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)
                # ith component coeff estimation
            if k==0:
                w = lsq_coeff_stack(comp_lr,res)
            else:
                w = non_unif_smoothing(neigh,dist_weights,w,res,comp_lr,mu=1,spec_rad=spec_rad_smooth,nb_iter=100)
            print "weights: ",w
            if k<nb_iter-1:
                input[4]=w
                input_trans[4]=w
                input_trans_prox[4]=w
                eig_max,spec_rad = pow_meth(sr_op_trans_stack,input,siz_in,ainit=eig_max)
                print "spectral radius: ",spec_rad
                # Hyper parameters reset
                to = cond_fact/spec_rad
                rho = 0.9/((1+wvl_transp_cor)*to*abs_count) # to*rho<L**(-2)
                to_old = ones((2,))*to
                input_trans_prox[-3] = to_old[(k+1)%2]
                input_trans[7] = to_old[(k+1)%2]
        # Residual update and HR images update
        if abs(w).max()>0:
            for l in range(0,shap[2]):
                res[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                im_hr[:,:,l] = im_hr[:,:,l]+w[l]*comp[:,:,i]
            weights[:,i] = w
        i+=1
        # Optim variables resetting
        x = x*0
        u = u*0
        v = v*0

    return im_hr,comp,weights,res,sig_est*sig_min,flux_est,shifts



def low_rank_comp_wise_sparse_dist_GFB(psf_stack_in,field_dist,upfact,opt,nsig,\
    neigh_frac=0.5,dist_weight_deg=1,shifts=None,opt_shift_est=None,nsig_shift_est=None,\
    sig_est=None,flux_est=None,nb_iter=5,nb_subiter=20,nb_subiter_min=7,mu=1.5,\
    nb_comp_max=10,tol=0.1,wvl_transp_cor=1,cv_proof_en=0,line_search_en=0):
    psf_stack = copy(psf_stack_in)
    print "Minimiser: GFB"
    # Degradation operator parameters estimation
    if shifts is None:
        #shifts = utils.shift_est(psf_stack)*upfact
        shifts = utils.wvl_shift_est(psf_stack,opt_shift_est,nsig_shift_est)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)
    if sig_est is None:
        sig_est = utils.im_gauss_nois_est_cube(psf_stack_in,opt=opt_shift_est)
    sig_min = sig_est.min()
    sig_est = sig_est/sig_min
    shap = psf_stack.shape
    for k in range(0,shap[2]):
        psf_stack[:,:,k] = psf_stack[:,:,k]/sig_est[k]
    if flux_est is None:
        flux_est = utils.flux_estimate_stack(psf_stack_in,rad=4)
    flux_ref = np.median(flux_est)
    weights = zeros((shap[2],nb_comp_max))
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))

    # Distances settings
    print "Contructing PSF tree..."
    neigh,dists = utils.knn_interf(field_dist,int(shap[2]*neigh_frac))
    print "Done..."
    dist_med = np.median(dists)
    dist_weights = (dist_med/dists)**dist_weight_deg
    # Spatial smoothing operator gradient lip constant estimation
    spec_rad_smooth = None


    siz_in = upfact*array(shap[0:-1])
    eig_max = None


    # GFB variables
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(w)
    input.append(sig_est)
    input.append(flux_est)
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    comp_old = zeros((upfact*shap[0],upfact*shap[1]))
    z1 = zeros((upfact*shap[0],upfact*shap[1])) # Sparse variable
    z2 = zeros((upfact*shap[0],upfact*shap[1])) # Positive variable
    compz1 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    compz2 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    coeff_init = None
    w1 = 0.8
    w2 = 0.2
    lambd = 1
    grad = zeros((upfact*shap[0],upfact*shap[1]))

    u,mr_file = isap.mr_trans(comp[:,:,0],opt=opt)
    shap2 = u.shape
    weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
    res = copy(psf_stack)
    comp_lr = copy(psf_stack)
    i = 0
    buff = zeros((nb_subiter_min))

    nb_iter_noisest = 20
    nb_subiter_prox = 50
    thresh_type = 1
    noise_vect_1 = None
    sig_map = None
    cur_res=100000000.0
    res_old=1000000000.0
    print " -------------------- Debugging warning: constraints switched off; reweighting off; no coeff smoothing; positivity,fw-bw ------------------- "
    while i < nb_comp_max:
        if i>0:
            w = utils.abs_val_reverting(weights[:,i-1])
        #spec_rad = spec_rad_init
        #eig = eig_init
        resk = res*0
        weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
        input[4]=w
        eig_max,spec_rad = pow_meth(sr_op_trans_stack,input,siz_in,ainit=eig_max)
        """spec_rad=0
            for f in range(0,shap[2]):
            spec_rad = spec_rad+(w[f]*flux_ref/(sig_est[f]*flux_est[f]))**2*(abs(shift_ker_stack[:,:,f]).sum())**2
            print "spectral radius: ",spec_rad"""

        nb_subiter_prox = 50
        print "nb_iter: ",nb_iter
        for k in range(0,nb_iter):
            t=1
            # ith component estimation
            j=0
            #print '--------------- ',nb_subiter,' ----------------'
            res_old = cur_res*2

            while j < nb_subiter and 100*abs(cur_res-res_old)/res_old>0.1:
                res_old = cur_res
                # Residual computation
                grad = grad*0
                if j==0:
                    for l in range(0,shap[2]):
                        resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                else:
                    for l in range(0,shap[2]):
                        grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')

                buff[j%nb_subiter_min] = (resk**2).sum()
                if j==0:
                    print "---- Ref mse ---- ",buff[j%nb_subiter_min]

                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]

                # Exact line search
                muj = mu/spec_rad
                if line_search_en==1:
                    c1=0
                    c2=0
                    c3=0
                    for l in range(0,shap[2]):
                        u1 = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(2*comp[:,:,i]-z1,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        u2 = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(2*comp[:,:,i]-z2,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        u3 = -(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(grad,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        c1+= (u1*u3).sum()
                        c2+= (u2*u3).sum()
                        c3+= (u3**2).sum()
                    muj = min(c1/c3,c2/c3)
                if cv_proof_en ==1:
                    muj = min(muj,mu/spec_rad)
                    print "Optimal step: ",muj," Max for cv proof: ",2/spec_rad
                # ---- Analysis constraint ---- #
                # Wavelet noise estimation
                sig_map = utils.res_sig_map(muj*grad,opt=opt)
                temp1 = 2*comp[:,:,i] - z1 - muj*grad
                thresh_map = nsig*sig_map
                if weights_sp is not None:
                    thresh_map = thresh_map*weights_sp
                # Sparsity
                t2,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
                if i==0 and k==0 and (comp[:,:,i]**2).max()>0:
                    tempy = isap.mr_recons_coeff(t2,mr_file)
                    a = (comp[:,:,i]*tempy).sum()
                    b = (t2**2).sum()
                    wvl_transp_cor = a/b
                    print '--------- correction coeff: ',wvl_transp_cor,' --------------------'

                #result,mr_file,n,coeff_init = wvl_analysis_op(temp1,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter_prox,transp_coeff=wvl_transp_cor)
                #z1 = z1+lambd*(result-comp[:,:,i])
                z1 = z1+lambd*(temp1-comp[:,:,i])
                nb_subiter_prox = 10
                # ---- Positivity constraint ---- #
                #z2 = z2+lambd*(proj_affine_pos2(2*comp[:,:,i] - z2 - muj*grad,w,im_hr)-comp[:,:,i])
                z2 = z2+lambd*(comp[:,:,i] - z2 - muj*grad)

                # ---- Main variable update ---- #
                comp[:,:,i] = proj_affine_pos2(w1*z1 + w2*z2,w,im_hr)
                #comp[:,:,i] = w1*z1 + w2*z2

                if j==0:
                    print "Estimated noise at the first scales: ",sig_map[0,0,:]

                j+=1

                for l in range(0,shap[2]):
                    resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                buff[j%nb_subiter_min] = (resk**2).sum()

                print "mse: ", buff[j%nb_subiter_min]
                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]


            for l in range(0,shap[2]):
                comp_lr[:,:,l] = (flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)

            # ith component coeff estimation
            if k>=0:
                w = lsq_coeff_stack(comp_lr,res)
            else:
                w = non_unif_smoothing(neigh,dist_weights,w,res,comp_lr,mu=1,spec_rad=spec_rad_smooth,nb_iter=100)
            print "------------------- Component ",i," ------------------"
            print "weights: ",w
            input[4]=w
            if k<nb_iter-1:
                # Reweighting
                #coeffx,mr_file = isap.mr_trans(comp[:,:,i],opt=opt)
                #weights_sp  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)

                # Spectral radius
                eig_max,spec_rad = pow_meth(sr_op_trans_stack,input,siz_in,ainit=eig_max)
                """spec_rad=0
                    for f in range(0,shap[2]):
                    spec_rad = spec_rad+(w[f]*flux_ref/(sig_est[f]*flux_est[f]))**2*(abs(shift_ker_stack[:,:,f]).sum())**2
                    print "spectral radius: ",spec_rad"""


        # Residual update and HR images update
        if abs(w).max()>0:
            for l in range(0,shap[2]):
                res[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                im_hr[:,:,l] = im_hr[:,:,l]+w[l]*comp[:,:,i]
            weights[:,i] = w
            print "---- Updated mse ---- ",(res**2).sum()
        compz1[:,:,i] = z1
        compz2[:,:,i] = z2
        i+=1
        # Optim variables resetting
        z1 = z1*0
        z2 = z2*0
    # coeff_init = coeff_init*0

    return im_hr,comp,weights,res,sig_est*sig_min,flux_est,shifts,compz1,compz2

def low_rank_global_src_est_cp(input,weights,y,ksig=4,eps=0.9,ainit=None,nb_iter=100,\
    tol=0.01,nb_rw=0,pos_en=True,opt='coif5',nb_scale=None,wav_en=False,wavr_en=False,\
    optr = None,nsig_hub=1000000000000000000,Y1=None,Y2=None,Y3=None,V=None,\
    only_noise_est_en=False,select_en=True,thresh_perc = 0.01,rad=None):

    S = copy(input[0]) # (Main variable)
    ref_mse = ((y-sr_stack_op(input))**2).sum()
    l1_norm = abs(S*weights).sum()
    print "--->>ref res: <<---",ref_mse
    print "--->>ref l1 norm: <<---",l1_norm

    ker_adj = input[3]
    shap = ker_adj.shape
    A = input[4]
    nb_im = shap[2]
    shapS = S.shape

    ind_select = range(0,shapS[2])


    # Spectral radius setting
    spec_rad = None
    if pos_en:
        eig_max,spec_rad = pow_meth(sr_stack_trans_op_cons_src,input,shapS,ainit=ainit)
    else:
        eig_max,spec_rad = pow_meth(sr_stack_trans_op_src,input,shapS,ainit=ainit)
    corr_coeff = None
    rad = None

    if wavr_en:
        dirac = zeros((shapS[0],shapS[1]))
        dirac[int(shapS[0]/2),int(shapS[1]/2)] = 1
        test,mr_file = isap.mr_trans(dirac,opt=optr)
        corr_coeff = sum(test**2)
        print "correction coeff: ",corr_coeff
        spec_rad = sqrt(spec_rad+sqrt(corr_coeff))
        os.remove(mr_file)
    else:
        spec_rad = sqrt(spec_rad)
    sig = 100
    to = eps/(sig*spec_rad) # Weight related to the dual variables
    print "--------------------- sig = ",sig," ------------------ to = ",to,"------------------------- spec rad = ",spec_rad," --------------------- "
    theta = 0.0

    # Dual variables
    # Data attachement
    if Y1 is None:
        Y1 = y*0
    input1 = cp.deepcopy(input)
    O = y*0
    # Positivity
    if Y2 is None:
        Y2 = zeros((shapS[0],shapS[1],nb_im))
    rweights_an = list()
    # Analysis constraint
    data_trans = None
    if wavr_en:
        if Y3 is None:
            Y3 = isap.mr_trans_stack(S*0,opt=optr,clean_en=True)
            data_trans = isap.mr_trans_stack(S,opt=optr,clean_en=True)

    rad = ones((shapS[2],))



    # Primal variable (Weighted L1 constraint)
    input2 = cp.deepcopy(input)
    if V is None:
        V = S*0
    rweights = ones((shapS[0],shapS[1],shapS[2]))

    cost=0
    cost_old=1
    iter=0

    shapy = y.shape
    ones_mat = ones((shapS[0],shapS[1],shapS[2]))

    wav_weights = None
    wav_rweights = None
    wav_weights_in = None
    if wav_en:
        print "Wavelet noise estimation..."
        wav_weights = utils.pywt_ksig_noise_2_stack(to*weights,opt=opt,nb_scale=nb_scale,nb_montecarlo=1000)
        print "Done."
        wav_weights_in = cp.deepcopy(wav_weights)

    weights_noise = ones(shapS)
    weights_spars = None
    temp3 = None

    # Transpose testing

    for l in range(0,nb_rw+1):
        print l+1,"th pass/",nb_rw+1
        if l>0:
            if wav_en:
                wav_weights_in = list()
                for i in range(0,len(wav_weights)):
                    a=list()
                    a.append(wav_weights[i][0]*wav_rweights[i][0])
                    for j in range(0,len(wav_weights[i])-1):
                        a.append([wav_weights[i][j+1][0]*wav_rweights[i][j+1][0],wav_weights[i][j+1][1]*wav_rweights[i][j+1][1],wav_weights[i][j+1][2]*wav_rweights[i][j+1][2]])
                    wav_weights_in.append(a)

        while iter<nb_iter and 100*abs(cost-cost_old)/abs(cost_old)>0.001:
            print "tol: ",100*abs(cost-cost_old)/abs(cost_old)
            iter+=1
            # Dual variables update
                        # -- Analysis variable update
            if wavr_en:
                #if iter<10:
                if l==0:
                    rweights_an = list()
                temp4 = cp.deepcopy(Y3)
                temp5,files = isap.mr_trans_stack(S,opt=optr)

                for i in range(0,len(temp4)):


                    temp4[i]+=sig*temp5[i]
                    temp6 = None
                        #if iter<10:
                    if l==0:
                        temp6,weights_temp = utils.thresholding_perc(temp4[i][:,:,:-1]/sig,thresh_perc,1)
                        rweights_an.append(weights_temp)

                    else:
                        temp6 = utils.thresholding_3D(temp4[i][:,:,:-1]/sig,rweights_an[i],1)
                    temp4bis = copy(temp4[i])
                    temp4bis[:,:,:-1] = temp6
                    Y3[i][:,:,:-1] = temp4[i][:,:,:-1] - sig*temp6
            # -- Data attachment variable update
            input1[0] = copy(S)
            temp1 = Y1+sig*sr_stack_op(input1)
            Y1 = (temp1-sig*y)/(1+sig)

            #temp1 = Y1+sig*sr_stack_op(input1)
            #Y1 = temp1 - sig*y - sig*prox_hub_stack((temp1-sig*y)/sig,nsig_hub*ones((y.shape[2],)),1.0/sig)

            # -- Positivity variable update
            if pos_en:
                temp2 = Y2*0
                for i in range(0,nb_im):
                    for j in range(0,shapS[2]):
                        temp2[:,:,i]+=S[:,:,j]*A[j,i]
                temp2 = sig*temp2+Y2
                Y2 = -pos_proj_cube(-temp2)




            # Primal variable update
            Vold = copy(V)
            temp30 = V*0
            if pos_en:
                for i in range(0,shapS[2]):
                    for j in range(0,nb_im):
                        temp30[:,:,i]+=Y2[:,:,j]*A[i,j]
            input2[0] = Y1
            temp31 = sr_stack_trans(input2)
            temp32 = None
            temp3 = None
            if wavr_en and pos_en:
                temp32 = corr_coeff*isap.mr_recons_stack(files,shapS)
                for i in range(0,len(files)):
                    os.remove(files[i])
                temp3 = V - to*(temp30+temp31+temp32)
            elif pos_en:
                temp3 = V - to*(temp30+temp31)
            elif wavr_en:
                temp3 = V - to*(temp30+temp32)
            else:
                temp3 = V - to*temp31
            #for id in range(0,shapS[2]):
                #print "Src ",id," PNR: ",ksig*sqrt(sum(V[:,:,id]**2))/sqrt(sum(weights[:,:,id]**2))," Comp noise:",sqrt(sum(weights[:,:,id]**2))

            #temp3 = V - to*temp31
            if wav_en:
                V = utils.pywt_filter_stack(temp3,ksig_map=wav_weights_in,opt=opt,nb_scale=nb_scale,thresh_type=1)
            elif wavr_en:
                V = proj_sphere_cube(temp3,temp3*0,rad)
            else:
                V = utils.thresholding_3D(temp3,to*weights*rweights,1)
                #V = proj_sphere_cube(temp3,temp3*0,rad)
                """if l==0:
                    V = temp3
                else:
                    V = utils.thresholding_3D(temp3,to*weights*rweights,1)
                    #V = utils.thresholding_3D(temp3,maximum(weights_noise*rweights,weights_spars*rweights),1)"""





            # Main variable update

            S = V + theta*(V-Vold)

            # Sanity check
            temp2 = Y2*0
            for i in range(0,nb_im):
                for j in range(0,shapS[2]):
                    temp2[:,:,i]+=S[:,:,j]*A[j,i]

            cost_old = cost
            input1[0] = copy(S)
            r = ((y-sr_stack_op(input1))**2).sum()
            lw = abs(S*weights).sum()
            if wavr_en:
                d = r-cost
                cost = cost+d/iter
            else:
                d = r+lw-cost
                cost = cost+d/iter
            print "residual: ",r," min val: ",temp2.min()," l1 term: ",lw
            # Thresholds estimation
        """if l==0:
            sig_temp = utils.im_gauss_nois_est_cube(S)
            k = 4
            for i in range(0,shap[2]):
                weights[:,:,i]*=sig_temp[i]*k
            perc = 0.2
            temp,weights_spars = utils.thresholding_perc(temp3,perc,1)"""


        # Reweighting

        if wav_en:
            wcoeff = utils.pywt_stack(S,opt=opt,nb_scale=nb_scale)
            wav_rweights = list()
            for i in range(0,len(wcoeff)):
                a=list()
                a.append((1+(abs(wcoeff[i][0])/wav_weights[i][0]))**(-1))
                for j in range(0,len(wcoeff[i])-1):
                    a.append([(1+(abs(wcoeff[i][j+1][0])/wav_weights[i][j+1][0]))**(-1),(1+(abs(wcoeff[i][j+1][1])/wav_weights[i][j+1][1]))**(-1),(1+(abs(wcoeff[i][j+1][2])/wav_weights[i][j+1][2]))**(-1)])
                wav_rweights.append(a)
        elif wavr_en:
            data_trans = isap.mr_trans_stack(S,opt=optr,clean_en=True)
            for i in range(0,len(rweights_an)):
                print i,"th source L1 norm's dual",sum(Y3[i]*data_trans[i]) - (rweights_an[i]*Y3[i][:,:,:-1]).max()
            for i in range(0,len(rweights_an)):
                rweights_an[i] *= (1+(abs(data_trans[i][:,:,:-1])/rweights_an[i]))**(-1)
        else:
            rweights  = (1+(abs(S)/weights))**(-1)
            #rweights  = (1+(abs(S)/maximum(weights_noise,weights_spars)))**(-1)

        iter = 0
        cost=2
        cost_old=1

    list_ind = list()

    for id in range(0,shapS[2]):
        a = ksig*sqrt(sum(V[:,:,id]**2))/sqrt(sum(weights[:,:,id]**2))
        print "Src ",id," PNR: ",a
        if a>=1:
            list_ind.append(id)

    if select_en and len(list_ind)>0:
        ind_select = tuple(list_ind)
        S = S[:,:,ind_select]
        V = V[:,:,ind_select]


    return S,Y1,Y2,Y3,V,rad,ind_select


def low_rank_global_src_est_comb(input,weights,y,ksig=4,eps=0.9,ainit=None,nb_iter=100,tol=0.01,nb_rw=0,pos_en=True,opt='coif5',nb_scale=None,wav_en=False,wavr_en=False,optr = None,nsig_hub=1000000000000000000,Y2=None,Y3=None,V=None,cY3=None,only_noise_est_en=False,select_en=True,thresh_perc = 0.01,rad=None,filters=None,filters_rot=None,iter_min=10,mu=0.1,curv=True,nb_sc = 4,nb_dir=8,curv_obj = None,Scurl=None):
    if wavr_en is False:
        curv = False

    S = copy(input[0]) # (Main variable)
    ref_mse = ((y-sr_stack_op(input))**2).sum()
    l1_norm = abs(S*weights).sum()
    print "--->>ref res: <<---",ref_mse
    print "--->>ref l1 norm: <<---",l1_norm

    ker_adj = input[3]
    shap = ker_adj.shape
    A = input[4]
    nb_im = shap[2]
    shapS = S.shape
    if Scurl is None:
        Scurl = S*0
    Scurv = S*0
    ind_select = range(0,shapS[2])

    spec_rad = 0
    # Spectral radius setting
    eig_max,spec_rad1 = pow_meth(sr_stack_trans_op_src,input,shapS,ainit=ainit)
    U, s, Vt = linalg.svd(A.dot(transpose(A)),full_matrices=False)
    spec_rad2 = sqrt(s[0])


    # Dual variables
    # Data attachement
    # Positivity
    if Y2 is None:
        Y2 = zeros((shapS[0],shapS[1],nb_im))

    rweights_an = None
    weights_an = None
    crweights_an = None
    cweights_an = None
    # Analysis constraint

    trans_data = None
    ctrans_data = None
    spec_rad3 = 0
    corr_coeff = 0

    if wavr_en:
        if Y3 is None or filters is None:
            Y3,filters = isap.mr_trans_stack_2(S*0,opt=optr)
            filters_rot = utils.rot90_stack(filters)
            Y3[:,:,-1,:]*=0 # Puts the coarse scales to 0
        for i in range(0,filters.shape[2]):
            spec_rad3 +=sum(abs(filters[:,:,i]))**2
        spec_rad3 = sqrt(spec_rad3)
        weights_an_temp,filters_temp = isap.mr_trans_stack_2(weights,filters=filters**2)
        weights_an = 4*sqrt(weights_an_temp[:,:,:-1,:])
        rweights_an = copy(Y3[:,:,:-1,:])
        if curv :

            cY3,curv_obj,corr_coeff = isap.stack_pyct_fwd(S*0,curv_obj=curv_obj,nb_sc=3,nb_dir=8,corr_en=True)
            print "Curvelets scaling coefficient: ",corr_coeff
            shapc = cY3.shape
            cweights_an = ones((shapc[0],shapc[1]))
            crweights_an = ones((shapc[0],shapc[1]))
            for i in range(0,shapc[0]):
                cweights_an[i,:] *= weights[:,:,i].max()
            spec_rad3 = spec_rad3+sqrt(corr_coeff)


    if wavr_en and pos_en:
        spec_rad = spec_rad1+spec_rad2+spec_rad3
    elif pos_en:
        spec_rad = spec_rad1+spec_rad2
    elif wavr_en:
        spec_rad = spec_rad1+spec_rad3
    else:
        spec_rad = spec_rad1

    rweights = ones((shapS[0],shapS[1],shapS[2]))

    cost=0
    cost_old=1
    iter=0

    shapy = y.shape
    ones_mat = ones((shapS[0],shapS[1],shapS[2]))

    wav_weights = None
    wav_rweights = None
    wav_weights_in = None
    weights_noise = ones(shapS)

    weights_spars = None
    temp3 = None

    input1 = cp.deepcopy(input)
    input2 = cp.deepcopy(input)
    filters_noise = None
    Ycurv = None
    for l in range(0,nb_rw+1):
        print l+1,"th pass/",nb_rw+1

        while (iter<nb_iter and 100*abs(cost-cost_old)/abs(cost_old)>0.001) or iter<iter_min:
            print "tol: ",100*abs(cost-cost_old)/abs(cost_old)
            # Checking transpose
            """xtest = np.random.randn(shapS[0],shapS[1],shapS[2])
            tx,filters = isap.mr_trans_stack_2(xtest,filters=filters)
            shapY = tx.shape
            ytest = np.random.randn(shapY[0],shapY[1],shapY[2],shapY[3])
            ty = isap.mr_transf_transp_stack(ytest,filters_rot)
            print "Transpose accuracy -------------",(xtest*ty).sum(),"----------------",(ytest*tx).sum()"""

            iter+=1
            input1[0] = copy(S+Scurv)
            est = sr_stack_op(input1)
            res = est - y
            input2[0] = res
            gradS = sr_stack_trans(input2)
            temp30 = None
            if pos_en:
                temp30 = S*0
                for i in range(0,shapS[2]):
                    for j in range(0,nb_im):
                        temp30[:,:,i]+=Y2[:,:,j]*A[i,j]
            temp31 = None
            ctemp31 = None
            if wavr_en:
                temp31 = isap.mr_transf_transp_stack(Y3,filters_rot)
                if curv:
                    ctemp31= corr_coeff*isap.stack_pyct_inv(cY3,curv_obj=curv_obj)
            temp3 = None
            ctemp3 = None
            if wavr_en and pos_en:
                temp3 = temp30+temp31
                if curv:
                    ctemp3 = temp30+ctemp31

            elif pos_en:
                temp3 = temp30
            elif wavr_en:
                temp3 = temp31
                if curv:
                    ctemp3 = ctemp31
            else:
                temp3 = S*0
            P = S - mu*(temp3+gradS)/spec_rad
            if curv:
                Pcurv = Scurv - mu*(ctemp3+gradS)/spec_rad
                Ycurv = 2*Pcurv-Scurv
                Scurv = Scurv+(1-1.0/(iter+1))*(Pcurv-Scurv)
                #Scurv = copy(Pcurv)

            if wavr_en is not True:
                P = utils.thresholding_3D(P,mu*weights*rweights/spec_rad,1)

            Y = 2*P-S
            S = S+(1-1.0/(iter+1))*(P-S)
            #S = copy(P)
            if pos_en:
                temp2 = Y2*0
                for i in range(0,nb_im):
                    for j in range(0,shapS[2]):
                        temp2[:,:,i]+=Y[:,:,j]*A[j,i]
                tY2 = -pos_proj_cube(-Y2-temp2/spec_rad)
                Y2 = Y2+(1-1.0/(iter+1))*(tY2-Y2)
            if wavr_en:
                temp3,filters = isap.mr_trans_stack_2(Y,filters=filters)

                for k in range(0,shapS[2]):
                    temp4 = None
                    if l==0:
                        temp4 = utils.thresholding_3D(Y3[:,:,:-1,k]+temp3[:,:,:-1,k]*spec_rad,mu*weights_an[:,:,:,k],1)
                    else:
                        temp4 = utils.thresholding_3D(Y3[:,:,:-1,k]+temp3[:,:,:-1,k]*spec_rad,mu*rweights_an[:,:,:,k]*weights_an[:,:,:,k],1)
                    tY3 = Y3[:,:,:-1,k]+temp3[:,:,:-1,k] - temp4/spec_rad
                    #Y3[:,:,:-1,k] = copy(tY3)
                    Y3[:,:,:-1,k] = Y3[:,:,:-1,k]+(1-1.0/(iter+1))*(tY3-Y3[:,:,:-1,k])
                if curv:
                    ctemp3,curv_obj = isap.stack_pyct_fwd(Ycurv,curv_obj=curv_obj)
                    ctemp4 = utils.thresholding(cY3+ctemp3*spec_rad,mu*cweights_an*crweights_an,1)
                    ctY3 = cY3+ctemp3 - ctemp4/spec_rad
                    #cY3 = copy(ctY3)
                    cY3 = cY3+(1-1.0/(iter+1))*(ctY3-cY3)


            cost_old = cost
            cost = sum(res**2)
            # ----------- Sanity check ----------- #
            print "Mini val: ",est.min(),"; Residual: ",cost," Gradient's norm: ",sum(gradS**2)," spectral norm: ",spec_rad
            if wavr_en is not True:
                print "Weighted l1 norm direct domain: ",sum(abs(weights*rweights*S))
            else:
                trans_data,filters = isap.mr_trans_stack_2(S,filters=filters)
                if l==0:
                    print "Weighted l1 norm analysis: ",sum(abs(trans_data[:,:,:-1,:]))
                else:
                    print "Weighted l1 norm analysis: ",sum(abs(trans_data[:,:,:-1,:]*rweights_an))
                if curv:
                    ctrans_data,curv_obj = isap.stack_pyct_fwd(Scurv,curv_obj=curv_obj)
                    print "Weighted l1 norm analysis (curvelets block): ",sum(abs(ctrans_data*crweights_an))


        if wavr_en:
            rweights_an = (1+(abs(trans_data[:,:,:-1,:])/weights_an))**(-1)
            print "Weight max: ",rweights_an.max()," Weight min: ",rweights_an.min()
            if curv:
                crweights_an = (1+(abs(ctrans_data)/cweights_an))**(-1)
        else:
            rweights  = (1+(abs(S)/weights))**(-1)

        iter = 0
        cost=2
        cost_old=1

    list_ind = list()

    for id in range(0,shapS[2]):
        a = ksig*sqrt(sum((S[:,:,id]+Scurv[:,:,id])**2))/sqrt(sum(weights[:,:,id]**2))
        print "Src ",id," PNR: ",a
        if a>=1:
            list_ind.append(id)

    if select_en and len(list_ind)>0:
        ind_select = tuple(list_ind)
        S = S[:,:,ind_select]
        Scurv = Scurv[:,:,ind_select]



    return filters,filters_rot,Y2,Y3,cY3,S,Scurv,ind_select,curv_obj




def robust_low_rank_global_src_est_cp(input,weights,y,eps=0.8,ainit=None,nb_iter=100,tol=1,nsig=4,sig_im=None):
    S = copy(input[0]) # (Main variable)
    upfact = input[1]
    ker_adj = input[3]
    shap = ker_adj.shape
    A = input[4]
    nb_im = shap[2]
    shapS = S.shape
    ref_mse = ((y-sr_stack_op(input))**2).sum()
    l1_norm = abs(S*weights).sum()
    print "--->>ref res: <<---",ref_mse
    print "--->>ref l1 norm: <<---",l1_norm
    scal = 0.5*ref_mse/abs(y-sr_stack_op(input)).sum()

    shap = S.shape
    # Spectral radius setting
    eig_max,spec_rad = pow_meth(sr_stack_trans_op_cons_src,input,shapS,ainit=ainit)
    spec_rad = sqrt(spec_rad)
    to = spec_rad**(-1)
    sig = eps/(to*spec_rad**2) # Weight related to the dual variables
    theta = 0.9

    # Dual variables
    # Data attachement
    Y1 = y*0
    input1 = cp.deepcopy(input)
    w = ones((shapS[0]/upfact,shapS[1]/upfact,nb_im))
    if sig_im is None:
        sig_im = ones((nb_im,))

    # Positivity
    Y2 = zeros((shapS[0],shapS[1],nb_im))
    # Primal variable (Weighted L1 constraint)
    input2 = cp.deepcopy(input)
    V = S*0
    ref_thresh = ones((shapS[0],shapS[1],shapS[2]))
    for i in range(0,shapS[2]):
        ref_thresh[:,:,i] *= median(weights[:,:,i])
    cost=2
    cost_old=1
    iter=0
    sig_map=None
    while iter<nb_iter:# and 100*abs(cost-cost_old)/abs(cost_old)>tol:
        iter+=1

        # Dual variables update
        # -- Data attachment variable update
        input1[0] = copy(S)
        temp1 = Y1+sig*sr_stack_op(input1)
        Y1 = temp1 - sig*y - sig*prox_hub_stack((temp1-sig*y)/sig,nsig*sig_im,1.0/sig)

        #Y1 = utils.l_inf_ball_proj_3D((temp1-sig*y)/scal,w,1)
        # -- Positivity variable update
        temp2 = Y2*0
        for i in range(0,nb_im):
            for j in range(0,shapS[2]):
                temp2[:,:,i]+=S[:,:,j]*A[j,i]
        temp2 = sig*temp2+Y2
        Y2 = -pos_proj_cube(-temp2)
        # Primal variable update
        Vold = copy(V)
        temp30 = V*0
        for i in range(0,shapS[2]):
            for j in range(0,nb_im):
                temp30[:,:,i]+=Y2[:,:,j]*A[i,j]
        input2[0] = Y1
        temp31 = sr_stack_trans(input2)
        temp3 = V - to*(temp30+temp31)
        V = utils.thresholding_3D(temp3,to*weights,1)
        # Main variable update

        S = V + theta*(V-Vold)
        # Sanity check
        temp2 = Y2*0
        for i in range(0,nb_im):
            for j in range(0,shapS[2]):
                temp2[:,:,i]+=S[:,:,j]*A[j,i]

        input1[0] = copy(S)
        cost_old = cost
        r = abs(y-sr_stack_op(input1)).sum()
        l = abs(S*weights).sum()
        cost = r+l
        print "residual: ",r*scal," min val: ",temp2.min()," l1 term: ",l

    return V

def low_rank_comp_wise_sparse_dist_dyn_coeff_GFB(psf_stack_in,field_dist,upfact,opt,nsig,global_sparsity_en=True,\
    sr_en=True,sparsity_en=True,pix_sparsity=True,redun_fact=2,neigh_frac=0.5,dist_weight_deg=0,shifts=None,\
    opt_shift_est=None,nsig_shift_est=5,sig_est=None,flux_est=None,nb_iter=5,nb_subiter=20,nb_subiter_min=7,\
    mu=1.5,nb_comp_max=10,tol=0.1,wvl_transp_cor=1,cv_proof_en=0,line_search_en=0,cond_fact=0.1,pos_relax=0,\
    nb_subiter_min_2=5,dyn_upload=False,positivity_en=False,score=90,refine=True,robust_refine=True,nsig_hub=10,\
    res_check_en=False,tracking_en=False): # Representations coefficietns are updated in all the components loops

    psf_stack = copy(psf_stack_in)
    shap = psf_stack.shape
    siz_in = upfact*array(shap[0:-1])
    print "Minimiser: GFB"
    if sparsity_en is False:
        print "----- Sparsity disable -----"
    # Degradation operator parameters estimation
    centroids = None
    if sig_est is None:
        sig_est = utils.im_gauss_nois_est_cube(psf_stack_in,opt=opt_shift_est)
    if shifts is None:
        map = ones((shap[0],shap[1],shap[2]))
        for i in range(0,shap[2]):
            map[:,:,i] *= nsig_shift_est*sig_est[i]
        print 'Shifts estimation...'
        psf_stack_shift = utils.thresholding_3D(psf_stack_in,map,0)
        shifts,centroids = utils.shift_est(psf_stack_shift)
        print 'Done...'
    else:
        print "------------ /!\ Warning: shifts provided /!\ ---------"

    #shifts = utils.wvl_shift_est(psf_stack,opt_shift_est,nsig_shift_est)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)
    sig_min = sig_est.min()
    sig_min_vect = ones((shap[2],))*sig_min
    sig_est = sig_est/sig_min
    muj = None
    nb_im = shap[2]
    for k in range(0,shap[2]):
        psf_stack[:,:,k] = psf_stack[:,:,k]/sig_est[k]
    print " ------ ref energy: ",(psf_stack**2).sum()," ------- "
    if flux_est is None:
        flux_est = utils.flux_estimate_stack(psf_stack_in,rad=4)
    flux_ref = np.median(flux_est)

    weights = None
    ref_weights = None
    if dyn_upload is False:
        weights = zeros((nb_comp_max,shap[2]))
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    ref_im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    # Distances settings
    print "Contructing PSF tree..."
    #nb_neighs = redun_fact*upfact**2
    nb_neighs = shap[2]-1
    neigh,dists = utils.knn_interf(field_dist,nb_neighs)
    p_max = pow_law_select(dists,nb_neighs)
    print "power max = ",p_max
    p_min = 0.01
    print "Done..."
    e_min = 0.01
    e_max = 1.99
    nb_samp_opt = nb_comp_max*10
    dists_unsorted = utils.feat_dist_mat(field_dist)
    dist_med = np.median(dists)
    dist_weights = (dist_med/dists_unsorted)**dist_weight_deg
    #dist_weights = (dist_med/dists)**dist_weight_deg
    dist_weigths = dist_weights/dist_weights.max()
    dist_weights_in = None
    # Spatial smoothing operator gradient lip constant estimation
    spec_rad_smooth = None


    siz_in = upfact*array(shap[0:-1])
    eig_max = None


    # GFB variables
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    w_old = ones((shap[2],))
    l1_norms = zeros((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    im_hr_dev = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(w)
    input.append(sig_est)
    input.append(flux_est)
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    ref_comp = None
    comp_old = zeros((upfact*shap[0],upfact*shap[1]))
    z1 = zeros((upfact*shap[0],upfact*shap[1])) # Sparse variable
    z2 = zeros((upfact*shap[0],upfact*shap[1])) # Positive variable
    compz1 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    compz2 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    min_val_map = zeros((upfact*shap[0],upfact*shap[1]))
    res_inter_comp = zeros((nb_comp_max,))
    coeff_init = None
    w1 = 0.5
    w2 = 0.5
    if positivity_en is not True :
        w2=0
        w1 = 1

    lambd = 1
    grad = zeros((upfact*shap[0],upfact*shap[1]))

    u,mr_file = isap.mr_trans(comp[:,:,0],opt=opt)
    shap2 = u.shape
    weights_sp = None
    if pix_sparsity is not True:
        weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
    else:
        weights_sp = ones((upfact*shap[0],upfact*shap[1]))

    thresh_save = None

    if pix_sparsity is not True:
        thresh_save = ones((shap2[0],shap2[1],shap2[2]-1,nb_comp_max))
    else:
        thresh_save = ones((upfact*shap[0],upfact*shap[1],nb_comp_max))

    g_weights_sp = ones((shap2[0],shap2[1],shap2[2],shap[2]))
    g_weights_sp_2 = ones((shap2[0],shap2[1],shap2[2],shap[2]))
    g_weights_sp[:,:,shap2[2]-1,:] = 0

    dev = copy(g_weights_sp)*0
    thresh_map = weights_sp*0
    g_thresh_map = g_weights_sp*0
    g_thresh_map_2 = g_weights_sp*0
    res = copy(psf_stack)
    ref_res = copy(psf_stack)*0

    comp_lr = zeros((shap[0],shap[1],nb_comp_max,shap[2]))
    i = 0
    buff = zeros((nb_subiter_min))

    nb_iter_noisest = 20
    nb_subiter_prox = 50
    thresh_type = 1
    noise_vect_1 = None
    sig_map = None
    cur_res=100000000.0
    res_old=1000000000.0
    cur_res_comp=100000000.0
    res_old_comp=1000000000.0
    to=None
    weights_init=None
    p_smth_mat_inv=None
    print " ------ ref energy: ",(psf_stack**2).sum()," ------- "
    pows = zeros((nb_comp_max))
    e_opt = zeros((nb_comp_max))
    e_range = utils.log_sampling(e_min,e_max,nb_samp_opt)
    p_range = utils.log_sampling(p_min,p_max,nb_samp_opt)
    score_res = 100
    detect_flag = None
    thresh_max = None
    comp_tracking = zeros((shap[0]*upfact,shap[1]*upfact,nb_comp_max,nb_subiter))

    while i < nb_comp_max and score_res>100-score:
        to=None
        p_smth_mat_inv=None
        #res_mean = ((res.sum(axis=0))).sum(axis=0)/sqrt(shap[0]*shap[1])
        #res_mean = res_mean.reshape((shap[2],1))
        res_mat = transpose(res.reshape((shap[0]*shap[1],shap[2])))
        p_out,e_out = notch_filt_optim(res_mat,dists_unsorted,p_range,e_range,nb_iter=4,tol=0.0001)
        pows[i] = p_out
        e_opt[i] = e_out
        """if i==0:
            pows[i]=p_min # =1
        else:
            pows[i]= (pows[i-1]+p_max)/2"""
        dist_weights_in = zeros((nb_im,nb_neighs,i+1))
        for ind in range(0,i+1):
            #dist_weights_in[:,:,ind] = (dist_med/dists)**pows[ind]
            dist_weights_in[:,:,ind] = (dist_med/dists_unsorted)**pows[ind]
            dist_weights_in[:,:,ind] = dist_weights_in[:,:,ind]/sqrt((dist_weights_in[:,:,ind]**2).sum())

        #if i>0:
        #w = weight_init(dist_weights_in[:,:,i],res,nb_iter=0)
        #w = utils.abs_val_reverting(weights[i-1,:])
        coeff_res,comp_res,est = utils.cube_svd(utils.rect_crop_c(res,int(0.9*shap[0]),int(0.9*shap[1]),centroids))

        #coeff_res,comp_res,est = utils.cube_svd(res)
        i0,j0 = where(abs(comp_res[:,:,0])==abs(comp_res[:,:,0]).max())
        w = coeff_res[0,:]*sign(comp_res[i0,j0,0])
        #w = weight_init(dist_weights_in[:,:,i],res,nb_iter=10,winit=w)
        #print "warning coeff debug"
        #w = w*0+1
        resk = res*0
        if pix_sparsity is not True:
            weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
        else:
            weights_sp = ones((upfact*shap[0],upfact*shap[1]))

        nb_subiter_prox = 50
        count = 0
        weights_init=None
        for k in range(0,nb_iter):
            # Direction initialization
            t=1

            print " ------ ref energy: ",(psf_stack**2).sum()," ------- "

            # Global residual calculation
            for l in range(0,shap[2]):
                res[:,:,l] = psf_stack[:,:,l]-(flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(im_hr[:,:,l],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)

            nsig_hub_k = abs(res).max()/sig_min
            print "------------------- Component ",i," ------------------"
            print "weights: ",w
            input[4]=w
            if i>0:
                for i_l in range(0,i):
                    print "weights ",i_l," comp:",weights[i_l,:]

            # Spectral radius
            eig_max,spec_rad = pow_meth(sr_op_trans_stack,input,siz_in,ainit=eig_max)


            # ith component estimation
            j=0
            print '--------------- ',nb_subiter,' ----------------'
            res_old = cur_res*2
            while (j < nb_subiter and 100*abs(cur_res-res_old)/res_old>0.01):# or j<nb_subiter_min_2:

                res_old = cur_res
                # Residual computation
                grad = grad*0
                nsig_hubj = nsig_hub
                if nsig_hub_k>nsig_hub:
                    atemp = double((nsig_hub_k-nsig_hub)*nb_subiter)/(nb_subiter-1)
                    btemp = nsig_hub_k-atemp
                    nsig_hubj = atemp/(j+1) + btemp
                    print "Huber k sig: ",nsig_hubj
                if j==0:

                    for l in range(0,shap[2]):
                        resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        s = sign(resk[:,:,l])
                        ones_mat = ones((shap[0],shap[1]))


                        ind1,ind2 = where((abs(resk[:,:,l]))<nsig_hubj*sig_min)
                        mask = ones((shap[0],shap[1]))
                        mask[ind1,ind2] = 0
                        grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim((ones_mat-mask)*resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')-(w[l]*nsig_hubj*sig_min*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(mask*s,upfact),shift_ker_stack_adj[:,:,l],mode='same')
                else:
                    for l in range(0,shap[2]):
                        s = sign(resk[:,:,l])
                        ones_mat = ones((shap[0],shap[1]))
                        ind1,ind2 = where((abs(resk[:,:,l]))<nsig_hubj*sig_min)
                        mask = ones((shap[0],shap[1]))
                        mask[ind1,ind2] = 0
                        grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim((ones_mat-mask)*resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')-(w[l]*nsig_hubj*sig_min*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(mask*s,upfact),shift_ker_stack_adj[:,:,l],mode='same')


                buff[j%nb_subiter_min] = (resk**2).sum()
                if j==0:
                    print "---- Ref mse ---- ",buff[j%nb_subiter_min]

                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]
                # Exact line search
                muj = mu/spec_rad
                if line_search_en==1:
                    c1=0
                    c2=0
                    c3=0
                    for l in range(0,shap[2]):
                        u1 = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(2*comp[:,:,i]-z1,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        u2 = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(2*comp[:,:,i]-z2,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        u3 = -(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(grad,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        c1+= (u1*u3).sum()
                        c2+= (u2*u3).sum()
                        c3+= (u3**2).sum()
                        muj = min(c1/c3,c2/c3)
                if cv_proof_en ==1:
                    muj = min(muj,mu/spec_rad)
                    print "Optimal step: ",muj," Max for cv proof: ",2/spec_rad
                # ---- Sparsity constraint ---- #
                # Wavelet noise estimation
                #temp1 = 2*comp[:,:,i] - z1 - muj*grad
                temp1 = comp[:,:,i] - muj*grad
                if j==0:
                    thresh_max = 0.9*abs(temp1).max()
                #result = copy(temp1)
                if sparsity_en:
                    if pix_sparsity:
                        if j==0:
                            sig_map = muj*utils.acc_sig_map(shap,shift_ker_stack_adj,sig_est,flux_est,flux_ref,upfact,w,sig_data=sig_min_vect)
                            sig_map = utils.sig_map(muj*grad)
                            thresh_map = nsig*sig_map
                            thresh_map = thresh_map*weights_sp
                        #thresh_map = ones((shap[0]*upfact,shap[1]*upfact))*((thresh_max-nsig*sig_map.max())/((j+1)**2)) + nsig*sig_map
                        result = utils.thresholding(temp1,thresh_map,1)
                    else:
                        if k==0 and j==nb_subiter_min_2-1:
                            sig_map = utils.res_sig_map(muj*grad,opt=opt)
                            if global_sparsity_en:
                                for l in range(0,shap[2]):
                                    g_thresh_map[:,:,:-1,l] = w[l]*sig_map
                                    g_thresh_map_2[:,:,:-1,l] = sig_map
                                g_thresh_map = nsig*g_thresh_map*g_weights_sp
                                g_thresh_map_2 = nsig*g_thresh_map*g_weights_sp_2
                                if i>0:
                                    for l in range(0,shap[2]):
                                        u,mr_file_temp = isap.mr_trans(im_hr_dev[:,:,l],opt=opt)
                                        os.remove(mr_file_temp)
                                        dev[:,:,:,l] = g_thresh_map[:,:,:,l]*u/w[l]
                            else:
                                thresh_map = nsig*sig_map
                                thresh_map = thresh_map*weights_sp
                        # Sparsity
                        if j>=nb_subiter_min_2-1:
                            if global_sparsity_en:
                                result,mr_file,n,coeff_init = wvl_analysis_op_src(temp1,dev,g_thresh_map/shap[2],w,mu=1,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter_prox)
                            else:
                                result,mr_file,n,coeff_init = wvl_analysis_op(temp1,thresh_map,opt=opt,coeff_init=coeff_init,mr_file=mr_file,nb_iter=nb_subiter_prox)
                            nb_subiter_prox = 10

                #z1 = z1+lambd*(result-comp[:,:,i])
                # ---- Positivity constraint ---- #
                temp2 = 2*comp[:,:,i] - z2 - muj*grad
                if positivity_en:
                    z2 = z2+lambd*(proj_affine_pos2(temp2,w,im_hr,min_val_map=min_val_map)-comp[:,:,i])
                else:
                    z2 = z2+lambd*(temp2-comp[:,:,i])
                # ---- Main variable update ---- #
                #comp[:,:,i] = w1*z1 + w2*z2
                comp[:,:,i] = result
                if k==0:
                    comp_tracking[:,:,i,j] = copy(comp[:,:,i])
                count+=1
                if sparsity_en:
                    if pix_sparsity:
                        if j==0:
                            print "========== Estimated back-progated noise in pixel domain (average): =========",mean(sig_map[0,0])
                    else:
                        if k==0 and j==nb_subiter_min_2-1:
                            print "========== Estimated noise at the first scales: =========",sig_map[0,0,:]

                j+=1
                for l in range(0,shap[2]):
                    resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                buff[j%nb_subiter_min] = (resk**2).sum()

                print "mse: ", buff[j%nb_subiter_min]
                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]
            res_inter_comp[i] = buff[j%nb_subiter_min]
            buff=buff*0
            # First HR images update
            im_hr = 0*im_hr
            for l in range(0,shap[2]):
                for p in range(0,i):
                    im_hr[:,:,l] = im_hr[:,:,l]+weights[p,l]*comp[:,:,p]
                if global_sparsity_en:
                    coeffx,mr_file_2 = isap.mr_trans(im_hr[:,:,l],opt=opt)
                    os.remove(mr_file_2)
            # Reweighting
            if sparsity_en:
                if global_sparsity_en:
                    for l in range(0,shap[2]):
                        coeffx=None
                        if i==0:
                            coeffx,mr_file_2 = isap.mr_trans(im_hr[:,:,l],opt=opt)
                            os.remove(mr_file_2)
                        else:
                            coeffx,mr_file_2 = isap.mr_trans(im_hr_dev[:,:,l],opt=opt)
                            os.remove(mr_file_2)

                        g_weights_sp[:,:,:-1,l]  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map*wold[l]))**(-1)
                        g_weights_sp_2[:,:,:-1,l]  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)
                elif pix_sparsity is not True:
                    coeffx,mr_file_2 = isap.mr_trans(comp[:,:,i],opt=opt)
                    os.remove(mr_file_2)
                    weights_sp  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)
                """else:
                    weights_sp  = (1+abs(comp[:,:,i])/(nsig*sig_map))**(-1)"""

            # Component normalization
            a = sqrt((comp[:,:,i]**2).sum())
            comp[:,:,i] = comp[:,:,i]/a
            if k>0:
                weights[i,:] = a*weights[i,:]
            for l in range(0,shap[2]):
                comp_lr[:,:,i,l] = (flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
            # Weights update
            if k==0:
                if dyn_upload:
                    weights,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_3(psf_stack,comp_lr[:,:,0:i+1,:],comp[:,:,0:i+1],neigh,dist_weights_in,e_opt[0:i+1],p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=300,tol=0.1,pos_en=positivity_en)
                    wold = copy(w)
                    w = copy(weights[i,:])
                else:
                    wold = copy(w)
                    w,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_3(res,comp_lr[:,:,i:i+1,:],comp[:,:,i:i+1],neigh,dist_weights_in[:,:,i:i+1],e_opt[i:i+1],p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=300,tol=0.1,pos_en=positivity_en)
                    w = w.reshape((shap[2],))
                    weights[i,:]  =copy(w)
                #if global_sparsity_en:
                    #weights,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_4(psf_stack,comp_lr[:,:,0:i+1,:],comp[:,:,0:i+1],neigh,dist_weights_in,opt=opt,mr_file=mr_file,l1_norms,g_thresh_map_2,p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=30,tol=0.1)
                #else:

            else:
                if dyn_upload:
                    weights,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_3(psf_stack,comp_lr[:,:,0:i+1,:],comp[:,:,0:i+1],neigh,dist_weights_in,e_opt[0:i+1],p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=300,tol=0.1,Ainit=weights,pos_en=positivity_en)
                    wold = copy(w)
                    w = copy(weights[i,:])
                else:
                    wold = copy(w)
                    w,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_3(res,comp_lr[:,:,i:i+1,:],comp[:,:,i:i+1],neigh,dist_weights_in[:,:,i:i+1],e_opt[i:i+1],p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=300,tol=0.1,Ainit = weights[i:i+1,:],pos_en=positivity_en)
                    w = w.reshape((shap[2],))
                    weights[i,:]  =copy(w)

            im_hr = 0*im_hr
            im_hr_dev = 0*im_hr_dev
            for l in range(0,shap[2]):
                for p in range(0,i):
                    im_hr[:,:,l] = im_hr[:,:,l]+weights[p,l]*comp[:,:,p]
                    if i>1 and p>0 and p<i-1:
                        im_hr_dev[:,:,l] = im_hr_dev[:,:,l]+weights[p,l]*comp[:,:,p]

        # Residual update and HR images update
        res_old_comp = cur_res_comp
        cur_res_comp = cur_res
        im_hr = 0*im_hr
        im_hr_dev = 0*im_hr_dev
        for l in range(0,shap[2]):
            for p in range(0,i+1):
                im_hr[:,:,l] = im_hr[:,:,l]+weights[p,l]*comp[:,:,p]
                if p>0:
                    im_hr_dev[:,:,l] = im_hr_dev[:,:,l]+weights[p,l]*comp[:,:,p]
            res[:,:,l] = psf_stack[:,:,l]-(flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(im_hr[:,:,l],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)

        compz1[:,:,i] = z1
        compz2[:,:,i] = z2

        # Optim variables resetting
        z1 = z1*0
        z2 = z2*0
        if sparsity_en:
            if pix_sparsity is not True:
                thresh_save[:,:,:,i] = thresh_map/muj
                i1,i2,i3 = where(weights_sp!=1)
                weights_sp[i1,i2,i3]=1
            else:
                thresh_save[:,:,i] = thresh_map/muj
                i1,i2 = where(weights_sp!=1)
                weights_sp[i1,i2]=1
            thresh_map = 0*thresh_map
            i1,i2,i3,i4 = where(g_weights_sp[:,:,:-1,:]!=1)
            g_weights_sp[i1,i2,i3]=1
            g_thresh_map = 0*g_thresh_map

        # Residual check
        if res_check_en:
            detect_flag,score_res = utils.autocorrel_detection_stack(res)
            print "residual score: ",score_res
        i+=1
    if nsig==0 or sparsity_en is False:
        print "Warning: no sparsity constraint"

    # Refining

    if refine:
        ainit = None
        positivity_en = True
        input_ref = list()
        input_ref.append(copy(comp[:,:,0:i]))
        input_ref.append(upfact)
        input_ref.append(shift_ker_stack)
        input_ref.append(shift_ker_stack_adj)
        input_ref.append(weights[0:i,:])
        ref_weights = copy(weights[0:i,:])
        input_ref.append(sig_est)
        input_ref.append(flux_est)
        survivors = ones((i,))
        for k in range(0,nb_iter):
            # Sources estimation
            if robust_refine is not True:
                ref_comp = low_rank_global_src_est_cp(input_ref,thresh_save[:,:,0:i]/survivors.sum(),psf_stack,eps=0.8,ainit=ainit,nb_iter=nb_subiter,tol=1)
            else:
                ref_comp = robust_low_rank_global_src_est_cp(input_ref,thresh_save[:,:,0:i]/survivors.sum(),psf_stack,eps=0.8,ainit=ainit,nb_iter=nb_subiter,tol=1,sig_im=sig_min_vect,nsig=nsig_hub)
            survivors = zeros((i,))
            for l in range(0,i):
                a = sqrt((ref_comp[:,:,l]**2).sum())
                if a>0:
                    survivors[l] = 1
                    ref_comp[:,:,l] /= a

                for p in range(0,shap[2]):
                    comp_lr[:,:,l,p] = (flux_ref/(sig_est[p]*flux_est[p]))*utils.decim(scisig.fftconvolve(ref_comp[:,:,l],shift_ker_stack[:,:,p],mode='same'),upfact,av_en=0)

            id0 = where((survivors==1))
            id = id0[0]
            # Weights estimation
            ref_weights *= 0
            print type(p_smth_mat_inv)
            ref_weights_k,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_3(psf_stack,comp_lr[:,:,id,:],ref_comp[:,:,id],neigh,dist_weights_in[:,:,id],e_opt[id],p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=300,tol=0.1,Ainit=ref_weights[id,:],pos_en=positivity_en)
            ref_weights[id,:] = ref_weights_k
            input_ref[0] = ref_comp
            input_ref[4] = ref_weights

        for l in range(0,shap[2]):
            for p in range(0,i):
                ref_im_hr[:,:,l] = ref_im_hr[:,:,l]+ref_weights[p,l]*ref_comp[:,:,p]
            ref_res[:,:,l] = psf_stack[:,:,l]-(flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(ref_im_hr[:,:,l],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
    if tracking_en:
        return im_hr,ref_im_hr,comp,ref_comp,weights,ref_weights,res,ref_res,sig_est*sig_min,flux_est,shifts,compz1,compz2,res_inter_comp,detect_flag,comp_tracking
    else:
        return im_hr,ref_im_hr,comp,ref_comp,weights,ref_weights,res,ref_res,sig_est*sig_min,flux_est,shifts,compz1,compz2,res_inter_comp,detect_flag

def low_rank_comp_wise_sparse_dist_dyn_coeff_GFB_joint(psf_stack_in,field_dist,upfact,opt,nsig,global_sparsity_en=True,sr_en=True,sparsity_en=True,pix_sparsity=True,redun_fact=2,neigh_frac=0.5,dist_weight_deg=0,shifts=None,opt_shift_est=None,nsig_shift_est=5,sig_est=None,flux_est=None,nb_iter=2,nb_subiter=300,nb_subiter_min=7,mu=1.5,nb_comp_max=10,tol=0.1,wvl_transp_cor=1,cv_proof_en=0,line_search_en=0,cond_fact=0.1,pos_relax=0,nb_subiter_min_2=5,dyn_upload=False,positivity_en=False,score=90,refine=True,robust_refine=True,nsig_hub=10,res_check_en=False,tracking_en=False,nb_rw=1,lsq_en=False,shifts_regist = True,wavr_en=False,curv=False):

    if wavr_en is False:
        curv = False
    psf_stack = copy(psf_stack_in)
    shap = psf_stack.shape
    if nb_comp_max>shap[2]:
        print "/!\ Warning: number of components higher than the number of images; reduced to ",shap[2]
    siz_in = upfact*array(shap[0:-1])
    print "Minimiser: GFB"
    if sparsity_en is False:
        print "----- Sparsity disable -----"
    # Degradation operator parameters estimation
    centroids = None
    if sig_est is None:
        print 'Noise level estimation...'
        sig_est,filters_lr = utils.im_gauss_nois_est_cube(psf_stack_in,opt=opt_shift_est)
        print 'Done.'
    if shifts_regist:
        if shifts is None:
            map = ones((shap[0],shap[1],shap[2]))
            for i in range(0,shap[2]):
                nsig_shifts = min(nsig_shift_est,0.8*psf_stack_in[:,:,i].max()/sig_est[i])
                map[:,:,i] *= nsig_shifts*sig_est[i]
            print 'Shifts estimation...'
            psf_stack_shift = utils.thresholding_3D(psf_stack_in,map,0)
            shifts,centroids = utils.shift_est(psf_stack_shift)
            print 'Done.'

        else:
            print "------------ /!\ Warning: shifts provided /!\ ---------"
    else:
        print "------------ /!\ Warning: no registration /!\ ---------"
        shifts = zeros((shap[2],2))
    if flux_est is None:
        flux_est = utils.flux_estimate_stack(psf_stack,rad=4)
    flux_ref = np.median(flux_est)

    #shifts = utils.wvl_shift_est(psf_stack,opt_shift_est,nsig_shift_est)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)
    sig_min = sig_est.min()
    sig_min_vect = ones((shap[2],))*sig_min
    sig_est = sig_est/sig_min
    muj = None
    nb_im = shap[2]
    for k in range(0,shap[2]):
        psf_stack[:,:,k] = psf_stack[:,:,k]/sig_est[k]
    print " ------ ref energy: ",(psf_stack**2).sum()," ------- "

    print "flux min: ",flux_est.min()," flux max: ",flux_est.max(),"sig min: ",sig_est.min()," sig max: ",sig_est.max()
    weights = None
    ref_weights = None
    if dyn_upload is False:
        weights = zeros((nb_comp_max,shap[2]))
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    ref_im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    # Distances settings
    print "Contructing PSF tree..."
    #nb_neighs = redun_fact*upfact**2
    nb_neighs = shap[2]-1
    neigh,dists = utils.knn_interf(field_dist,nb_neighs)
    p_max = pow_law_select(dists,nb_neighs)
    print "power max = ",p_max
    p_min = 0.01
    print "Done..."
    e_min = 0.01
    e_max = 1.99
    nb_samp_opt = nb_comp_max*10
    dists_unsorted = utils.feat_dist_mat(field_dist)
    dist_med = np.median(dists)
    dist_weights = (dist_med/dists_unsorted)**dist_weight_deg
    #dist_weights = (dist_med/dists)**dist_weight_deg
    dist_weigths = dist_weights/dist_weights.max()
    # Spatial smoothing operator gradient lip constant estimation
    spec_rad_smooth = None
    siz_in = upfact*array(shap[0:-1])
    eig_max = None
    # GFB variables
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    w_old = ones((shap[2],))
    l1_norms = zeros((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    im_hr_curv = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    im_hr_dev = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(w)
    input.append(sig_est)
    input.append(flux_est)
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    comp_curl = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max))
    ref_comp = None
    comp_old = zeros((upfact*shap[0],upfact*shap[1]))
    z1 = zeros((upfact*shap[0],upfact*shap[1])) # Sparse variable
    z2 = zeros((upfact*shap[0],upfact*shap[1])) # Positive variable
    compz1 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    compz2 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    min_val_map = zeros((upfact*shap[0],upfact*shap[1]))
    res_inter_comp = zeros((nb_comp_max,))
    coeff_init = None
    w1 = 0.5
    w2 = 0.5
    if positivity_en is not True :
        w2=0
        w1 = 1

    lambd = 1
    grad = zeros((upfact*shap[0],upfact*shap[1]))
    u,mr_file = isap.mr_trans(comp[:,:,0],opt=opt)
    shap2 = u.shape
    weights_sp = None
    if pix_sparsity is not True:
        weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
    else:
        weights_sp = ones((upfact*shap[0],upfact*shap[1]))
    thresh_save = None
    if pix_sparsity is not True:
        thresh_save = ones((shap2[0],shap2[1],shap2[2]-1,nb_comp_max))
    else:
        thresh_save = ones((upfact*shap[0],upfact*shap[1],nb_comp_max))
    g_weights_sp = ones((shap2[0],shap2[1],shap2[2],shap[2]))
    g_weights_sp_2 = ones((shap2[0],shap2[1],shap2[2],shap[2]))
    g_weights_sp[:,:,shap2[2]-1,:] = 0
    dev = copy(g_weights_sp)*0
    thresh_map = weights_sp*0
    g_thresh_map = g_weights_sp*0
    g_thresh_map_2 = g_weights_sp*0
    res = copy(psf_stack)
    ref_res = copy(psf_stack)*0
    comp_lr = zeros((shap[0],shap[1],nb_comp_max,shap[2]))
    i = 0
    buff = zeros((nb_subiter_min))
    nb_iter_noisest = 20
    nb_subiter_prox = 50
    thresh_type = 1
    noise_vect_1 = None
    sig_map = None
    cur_res=100000000.0
    res_old=1000000000.0
    cur_res_comp=100000000.0
    res_old_comp=1000000000.0
    to=None
    weights_init=None
    p_smth_mat_inv=None
    print " ------ ref energy: ",(psf_stack**2).sum()," ------- "
    pows = None
    e_opt = None
    e_range = None
    p_range = None
    if lsq_en is not True and shap[2]>2:
        pows = zeros((nb_comp_max))
        e_opt = zeros((nb_comp_max))
        e_range = utils.log_sampling(e_min,e_max,nb_samp_opt)
        p_range = utils.log_sampling(p_min,p_max,nb_samp_opt)
    elif shap[2]==2:
        print "Number of sources insufficient to use the spatial constraint; activating the lsq"
        lsq_en = True
    score_res = 100
    detect_flag = None
    thresh_max = None
    comp_tracking = zeros((shap[0]*upfact,shap[1]*upfact,nb_comp_max,nb_subiter))
    # Weights init
    p_opt = None
    comp_temp = None
    data = None
    ker = None
    alph_ref = None
    alph = None
    cube_est =  None
    if lsq_en:
        if shifts_regist:
            weights,coeff_res,cube_est = utils.cube_svd(utils.rect_crop_c(res,int(0.9*shap[0]),int(0.9*shap[1]),centroids),nb_comp=nb_comp_max)
        else:
            weights,coeff_res,cube_est = utils.cube_svd(res,nb_comp=nb_comp_max)
        #coeff_res,comp_res,est = utils.cube_svd(res)
        weights = weights[0:nb_comp_max,:]
        for l in range(0,nb_comp_max):
            a = sqrt((weights[l,:]**2).sum())
            if a>0:
                weights[l,:] /= a
    else:
        if shifts_regist:
            e_opt,p_opt,weights,comp_temp,data,ker,alph_ref  = analysis(utils.rect_crop_c(res,int(0.9*shap[0]),int(0.9*shap[1]),centroids),int(0.9*shap[0])*int(0.9*shap[1])*sig_min**2,field_dist,tol=0,nb_max=nb_comp_max)
            #if wavr_en:
            #weights,coeff_res,est = utils.cube_svd(utils.rect_crop_c(res,int(0.9*shap[0]),int(0.9*shap[1]),centroids),nb_comp=nb_comp_max)
        else:
            e_opt,p_opt,weights,comp_temp,data,ker,alph_ref  = analysis(res,nb_im*int(0.9*shap[0])*int(0.9*shap[1])*sig_min**2,field_dist,tol=0,nb_max=nb_comp_max)
            #if wavr_en:
            #weights,coeff_res,cube_est = utils.cube_svd(res,nb_comp=nb_comp_max)

        alph = alph_ref
        print "Mat constraint parameters:",e_opt,p_opt

        for l in range(0,nb_comp_max):
            a = sqrt((weights[l,:]**2).sum())
            if a>0:
                weights[l,:] /= a


    weights_temp,coeff_res_temp,cube_est = utils.cube_svd(res,nb_comp=nb_comp_max)
    print "============================>>>> min res: ",sum((cube_est-res)**2)," <<<<============================"
    #print sum(abs(weights-alph.dot(ker)))
    ainit = None
    input_ref = list()
    input_ref.append(copy(comp))
    input_ref.append(upfact)
    input_ref.append(shift_ker_stack)
    input_ref.append(shift_ker_stack_adj)
    input_ref.append(weights)
    ref_weights = copy(weights)
    input_ref.append(sig_est)
    input_ref.append(flux_est)
    survivors = ones((nb_comp_max,))
    Y1 = None
    Y2 = None
    Y3 = None
    cY3 = None
    curv_obj = None
    V = None
    filters=None
    filters_rot = None
    rad = None
    select_en = False
    for k in range(0,nb_iter):
        # Sources estimation
        thresh = nsig*utils.acc_sig_maps(shap,shift_ker_stack_adj,sig_est,flux_est,flux_ref,upfact,weights,sig_data=sig_min_vect)
        if k==1:
            select_en = True
        #comp,Y1,Y2,Y3,V,rad,ind_select = low_rank_global_src_est_cp(input_ref,thresh,psf_stack,ksig=nsig,eps=0.8,ainit=ainit,nb_iter=nb_subiter,tol=1,nb_rw=nb_rw,Y1=Y1,Y2=Y2,V=V,rad=None,select_en=select_en,wavr_en=wavr_en,optr=opt,pos_en=False)
        filters,filters_rot,Y2,Y3,cY3,comp,comp_curv,ind_select,curv_obj = low_rank_global_src_est_comb(input_ref,\
        thresh,psf_stack,ksig=nsig,eps=0.8,ainit=ainit,nb_iter=nb_subiter,tol=1,nb_rw=nb_rw,Y2=Y2,V=V,rad=None,\
        select_en=select_en,wavr_en=wavr_en,optr=opt,pos_en=positivity_en,filters=filters,filters_rot=filters_rot,\
        curv=curv,curv_obj=curv_obj)

        #print "============= ",comp.shape,rad.shape," =============="
        comp_lr = zeros((shap[0],shap[1],comp.shape[2],shap[2]))
        survivors = zeros((nb_comp_max,))
        for l in range(0,comp.shape[2]):
            """a = sqrt((comp[:,:,l]**2).sum())
            if a>0:
                survivors[l] = 1
                comp[:,:,l] /= a
                weights[l,:] *= a"""

            for p in range(0,shap[2]):
                comp_lr[:,:,l,p] = (flux_est[p]/(sig_est[p]*flux_ref))*utils.decim(scisig.fftconvolve(comp[:,:,l]+comp_curv[:,:,l],shift_ker_stack[:,:,p],mode='same'),upfact,av_en=0)
        id0 = where((survivors==1))
        id = id0[0]
        # Weights estimation
        #weights_k,alph,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_4(psf_stack,comp_lr[:,:,id,:],comp[:,:,id],neigh,ker,alph,to=to,nb_iter=2000,tol=0.1,Ainit=weights[id,:],pos_en=positivity_en)
        #weights_k,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_3(psf_stack,comp_lr[:,:,id,:],comp[:,:,id],neigh,dist_weights_in,e_opt,ker,alph,to=to,nb_iter=nb_subiter,p_smth_mat_inv=p_smth_mat_inv,tol=0.1,Ainit=weights[id,:],pos_en=positivity_en)
        #weights_k,alph = non_unif_smoothing_mult_coeff_pos_cp_5(psf_stack,comp_lr[:,:,id,:],comp[:,:,id],neigh,ker,alph,to=to,nb_iter=nb_subiter,tol=0.1,Ainit=weights[id,:],pos_en=positivity_en)
        n_max = nb_iter-1
        if k < n_max:
            weights_k = None
            if lsq_en:
                weights_k = non_unif_smoothing_mult_coeff_pos_cp_6(psf_stack,comp_lr,nb_iter=nb_subiter*2)
                #weights_k,mat,v = lsq_mult_coeff_stack(psf_stack,comp_lr)
            else:
                weights_k,alph,supports = non_unif_smoothing_mult_coeff_pos_cp_5(psf_stack,comp_lr,comp+comp_curv,neigh,ker,alph[ind_select,:],to=to,nb_iter=nb_subiter*2,tol=0.1,Ainit=weights[ind_select,:],pos_en=positivity_en)#,spars_perc=1)#*double((k+1))/nb_iter)
            #weights[id,:] = weights_k
            list_surv = list()
            for l in range(0,comp.shape[2]):
                a = sqrt((weights_k[l,:]**2).sum())
                if a>0:
                    list_surv.append(l)
                    if wavr_en:
                        comp[:,:,l] *= a
                        comp_curl[:,:,l] *= a
                        weights_k[l,:] /= a
            ind_select = tuple(list_surv)
            weights = weights_k[ind_select,:]
            input_ref[0] = comp[:,:,ind_select]+comp_curv[:,:,ind_select]
            input_ref[4] = weights
            if curv:
                cY3 = cY3[ind_select,:]
            #V = V[:,:,ind_select]
            #rad = rad[ind_select,]

    for l in range(0,shap[2]):
        for p in range(0,comp.shape[2]):
            if curv:
                im_hr_curv[:,:,l] = im_hr_curv[:,:,l]+weights[p,l]*comp_curv[:,:,p]
            im_hr[:,:,l] = im_hr[:,:,l]+weights[p,l]*(comp[:,:,p]+comp_curv[:,:,p])
            res[:,:,l] = psf_stack[:,:,l]-(flux_est[l]/(sig_est[l]*flux_ref))*utils.decim(scisig.fftconvolve(im_hr[:,:,l],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)

    return im_hr,im_hr_curv,comp,comp_curv,weights,res,sig_est*sig_min,flux_est,shifts,alph,alph_ref,e_opt,p_opt,ker,supports

def transport_plan_update(P_stack_init,im_stack,supp,neighbors_graph,weights_neighbors,\
                            spectrums,A,flux,sig,ker,D,ker_rot,nb_iter=100,mu=0.3,tol = 0.1):

    # Spectral radius calculation
    shap_obs = im_stack.shape
    shap = (shap_obs[0]*D,shap_obs[1]*D)

    input_op = list()
    input_op.append(copy(P_stack_init))
    input_op.append(shap)
    input_op.append(supp)
    input_op.append(neighbors_graph)
    input_op.append(weights_neighbors)
    input_op.append(spectrums)
    input_op.append(A)
    input_op.append(flux)
    input_op.append(sig)
    input_op.append(ker)
    input_op.append(D)
    input_op.append(ker_rot)

    eig_max,spec_rad = pow_meth(psf_learning_utils.transport_plan_projections_field_grad_mat,input_op,\
                        P_stack_init.shape)
    #spec_rad = 4.77321241379
    t = 1
    told = t

    z = copy(P_stack_init)
    x = copy(P_stack_init)

    i = 0
    var = 100
    res_old = 100
    while(i<nb_iter and var>tol):
        res,grad = psf_learning_utils.transport_plan_projections_field_gradient(im_stack,z,supp,\
                    neighbors_graph,weights_neighbors,spectrums,A,flux,sig,ker,ker_rot,D)
        res_en = sum(res**2)
        print "Residual energy: ",res_en
        var = 100*abs(res_en-res_old)/res_old
        y = z - mu*grad/spec_rad
        x_old = copy(x)
        x = copy(y) # Prox = id
        t_old = t
        t = (1+sqrt(1+4*t**2))/2
        lambd = 1+(t_old-1)/t
        z = x_old + lambd*(x-x_old)
        i+=1
        gc.collect()
    return z

def transport_plan_coeff_update(P_stack,im_stack,supp,neighbors_graph,weights_neighbors,\
                            spectrums,A_init,flux,sig,ker,D,ker_rot,nb_iter=100,mu=0.3,tol = 0.1):

    shap_obs = im_stack.shape
    shap = (shap_obs[0]*D,shap_obs[1]*D)

    # Spectral radius calculation
    input_op = list()
    input_op.append(A_init)
    input_op.append(shap)
    input_op.append(supp)
    input_op.append(neighbors_graph)
    input_op.append(weights_neighbors)
    input_op.append(spectrums)
    input_op.append(copy(P_stack))
    input_op.append(flux)
    input_op.append(sig)
    input_op.append(ker)
    input_op.append(D)
    input_op.append(ker_rot)


    eig_max,spec_rad = pow_meth(psf_learning_utils.transport_plan_projections_field_coeff_grad_mat,input_op,\
                        A_init.shape)
    t = 1
    told = t

    z = copy(A_init)
    x = copy(A_init)

    i = 0
    var = 100
    res_old = 100
    while(i<nb_iter and var>tol):
        res,grad = psf_learning_utils.transport_plan_projections_field_coeff_gradient(im_stack,P_stack,supp,\
                    neighbors_graph,weights_neighbors,spectrums,z,flux,sig,ker,ker_rot,D)
        res_en = sum(res**2)
        print "Residual energy: ",res_en
        var = 100*abs(res_en-res_old)/res_old
        y = z - mu*grad/spec_rad
        x_old = copy(x)
        x = copy(y) # Prox = id
        t_old = t
        t = (1+sqrt(1+4*t**2))/2
        lambd = 1+(t_old-1)/t
        z = x_old + lambd*(x-x_old)
        i+=1

    return z

def polychromatic_psf_field_est(im_stack,spectrums,wvl,D,opt_shift_est,nb_comp,nb_iter=4,nb_subiter=100,mu=0.3,\
                        tol = 0.1,sig_supp = 3,sig=None,shifts=None,flux=None,nsig_shift_est=4):

    print "--------------- Transport architecture setting ------------------"
    shap_obs = im_stack.shape
    shap = (shap_obs[0]*D,shap_obs[1]*D)
    P_stack = utils.diagonally_dominated_mat_stack(shap,nb_comp,sig=sig_supp,thresh_en=True)
    i,j = where(P_stack[:,:,0]>0)
    supp = transpose(array([i,j]))
    t = (wvl-wvl.min())/(wvl.max()-wvl.min())
    neighbors_graph,weights_neighbors,cent,coord_map,knn = psf_learning_utils.full_displacement(shap,supp,t,\
    pol_en=True,cent=None,theta_param=1,pol_mod=True,coord_map=None,knn=None)

    print "--------------------- Forward operator parameters estimation ------------------------"
    centroids = None
    if sig is None:
        sig,filters = utils.im_gauss_nois_est_cube(copy(im_stack),opt=opt_shift_est)

    if shifts is None:
        map = ones(im_stack.shape)
        for i in range(0,shap_obs[2]):
            map[:,:,i] *= nsig_shift_est*sig[i]
        print 'Shifts estimation...'
        psf_stack_shift = utils.thresholding_3D(copy(im_stack),map,0)
        shifts,centroids = utils.shift_est(psf_stack_shift)
        print 'Done...'
    else:
        print "------------ /!\ Warning: shifts provided /!\ -----------"
    ker,ker_rot = utils.shift_ker_stack(shifts,D)
    sig /=sig.min()
    for k in range(0,shap_obs[2]):
        im_stack[:,:,k] = im_stack[:,:,k]/sig[k]
    print " ------ ref energy: ",(im_stack**2).sum()," ------- "
    if flux is None:
        flux = utils.flux_estimate_stack(copy(im_stack),rad=4)
    flux_ref = np.median(flux)

    print "------------- Coeff init ------------"
    A,comp,cube_est = utils.cube_svd(im_stack,nb_comp=nb_comp)

    i=0
    print " --------- Main loop ---------- "
    for i in range(0,nb_iter):
        print "----------------Iter ",i+1,"/",nb_iter,"-------------------"
        print "------------------- Transport plans estimation ------------------"
        P_stack = transport_plan_update(P_stack,im_stack,supp,neighbors_graph,weights_neighbors,\
                                    spectrums,A,flux,sig,ker,D,ker_rot,nb_iter=nb_subiter,mu=0.3,tol = 0.1)
        print "------------------- Coefficients estimation ----------------------"
        A = transport_plan_coeff_update(P_stack,im_stack,supp,neighbors_graph,weights_neighbors,\
                                    spectrums,A,flux,sig,ker,D,ker_rot,nb_iter=nb_subiter,mu=0.3,tol = 0.1)
        # Normalization
        for j in range(0,nb_comp):
            l1_P = sum(abs(P_stack[:,:,j]))
            P_stack[:,:,j]/= l1_P
            A[j,:] *= l1_P

    psf_est = psf_learning_utils.field_reconstruction(P_stack,shap,supp,neighbors_graph,weights_neighbors,A)

    return psf_est,P_stack,A

def polychromatic_psf_field_est_2(im_stack_in,spectrums,wvl,D,opt_shift_est,nb_comp,field_pos=None,nb_iter=4,nb_subiter=100,mu=0.3,\
                        tol = 0.1,sig_supp = 3,sig=None,shifts=None,flux=None,nsig_shift_est=4,pos_en = True,simplex_en=False,\
                        wvl_en=True,wvl_opt=None,nsig=3,graph_cons_en=False):
    """ Main LambdaRCA function.
    
    Calls:
    
    * :func:`utils.get_noise_arr`
    * :func:`utils.diagonally_dominated_mat_stack` 
    * :func:`psf_learning_utils.full_displacement` 
    * :func:`utils.im_gauss_nois_est_cube` 
    * :func:`utils.thresholding_3D` 
    * :func:`utils.shift_est` 
    * :func:`utils.shift_ker_stack` 
    * :func:`utils.flux_estimate_stack` 
    * :func:`optim_utils.analysis` 
    * :func:`utils.cube_svd`
    * :func:`grads.polychrom_eigen_psf`
    * :func:`grads.polychrom_eigen_psf_coeff_graph`
    * :func:`grads.polychrom_eigen_psf_coeff`
    * [SAM's] :func:`linear.transport_plan_lin_comb_wavelet`
    * [SAM's] :func:`linear.transport_plan_marg_wavelet`
    * [SAM's] :func:`linear.transport_plan_lin_comb`
    * [SAM's] :func:`linear.transport_plan_lin_comb_coeff`
    * [SAM's] :func:`proximity.simplex_threshold`
    * :func:`psf_learning_utils.field_reconstruction`
    
    Pure "Sam" imports: #TODO: replace with ModOpt import or something
    
    * :func:`linear.Identity`
    * :func:`proximity.Threshold`
    * :func:`proximity.Simplex`
    * :func:`proximity.Positive`
    * :func:`proximity.KThreshold`
    * :func:`cost.costFunction`
    
    """

    im_stack = copy(im_stack_in)
    if wvl_en:
        from utils import get_noise_arr

    print "--------------- Transport architecture setting ------------------"
    nb_im = im_stack.shape[-1]
    shap_obs = im_stack.shape
    shap = (shap_obs[0]*D,shap_obs[1]*D)
    P_stack = utils.diagonally_dominated_mat_stack(shap,nb_comp,sig=sig_supp,thresh_en=True)
    i,j = where(P_stack[:,:,0]>0)
    supp = transpose(array([i,j]))
    t = (wvl-wvl.min()).astype(float)/(wvl.max()-wvl.min())

    neighbors_graph,weights_neighbors,cent,coord_map,knn = psf_learning_utils.full_displacement(shap,supp,t,\
    pol_en=True,cent=None,theta_param=1,pol_mod=True,coord_map=None,knn=None)

    print "------------------- Forward operator parameters estimation ------------------------"
    centroids = None
    if sig is None:
        sig,filters = utils.im_gauss_nois_est_cube(copy(im_stack),opt=opt_shift_est)

    if shifts is None:
        map = ones(im_stack.shape)
        for i in range(0,shap_obs[2]):
            map[:,:,i] *= nsig_shift_est*sig[i]
        print 'Shifts estimation...'
        psf_stack_shift = utils.thresholding_3D(copy(im_stack),map,0)
        shifts,centroids = utils.shift_est(psf_stack_shift)
        print 'Done...'
    else:
        print "---------- /!\ Warning: shifts provided /!\ ---------"
    ker,ker_rot = utils.shift_ker_stack(shifts,D)
    sig /=sig.min()
    for k in range(0,shap_obs[2]):
        im_stack[:,:,k] = im_stack[:,:,k]/sig[k]
    print " ------ ref energy: ",(im_stack**2).sum()," ------- "
    if flux is None:
        flux = utils.flux_estimate_stack(copy(im_stack),rad=4)

    if graph_cons_en:
        print "-------------------- Spatial constraint setting -----------------------"
        e_opt,p_opt,weights,comp_temp,data,basis,alph  = analysis(im_stack,0.1*prod(shap_obs)*sig.min()**2,field_pos,nb_max=nb_comp)

    print "------------- Coeff init ------------"
    A,comp,cube_est = utils.cube_svd(im_stack,nb_comp=nb_comp)

    i=0
    print " --------- Optimization instances setting ---------- "

    # Data fidelity related instances
    polychrom_grad = grad.polychrom_eigen_psf(im_stack, supp, neighbors_graph, \
                weights_neighbors, spectrums, A, flux, sig, ker, ker_rot, D)

    if graph_cons_en:
        polychrom_grad_coeff = grad.polychrom_eigen_psf_coeff_graph(im_stack, supp, neighbors_graph, \
                weights_neighbors, spectrums, P_stack, flux, sig, ker, ker_rot, D, basis)
    else:
        polychrom_grad_coeff = grad.polychrom_eigen_psf_coeff(im_stack, supp, neighbors_graph, \
                weights_neighbors, spectrums, P_stack, flux, sig, ker, ker_rot, D)


    # Dual variable related linear operators instances
    dual_var_coeff = zeros((supp.shape[0],nb_im))
    if wvl_en and pos_en:
        lin_com = sams_linear.transport_plan_lin_comb_wavelet(A,supp,weights_neighbors,neighbors_graph,shap,wavelet_opt=wvl_opt)
    else:
        if wvl_en:
            lin_com = sams_linear.transport_plan_marg_wavelet(supp,weights_neighbors,neighbors_graph,shap,wavelet_opt=wvl_opt)
        else:
            lin_com = sams_linear.transport_plan_lin_comb(A, supp,shap)

    if not graph_cons_en:
        lin_com_coeff = sams_linear.transport_plan_lin_comb_coeff(P_stack, supp)

    # Proximity operators related instances
    id_prox = sams_linear.Identity()
    if wvl_en and pos_en:
        noise_map = get_noise_arr(lin_com.op(polychrom_grad.MtX(im_stack))[1])
        dual_var_plan = np.array([zeros((supp.shape[0],nb_im)),zeros(noise_map.shape)])
        dual_prox_plan = sams_prox.simplex_threshold(nsig*noise_map,pos_en=(not simplex_en))
    else:
        if wvl_en:
            # Noise estimation
            noise_map = get_noise_arr(lin_com.op(polychrom_grad.MtX(im_stack)))
            dual_var_plan = zeros(noise_map.shape)
            dual_prox_plan = sams_prox.Threshold(nsig*noise_map)
        else:
            dual_var_plan = zeros((supp.shape[0],nb_im))
            if simplex_en:
                dual_prox_plan = sams_prox.Simplex()
            else:
                dual_prox_plan = sams_prox.Positive()

    if graph_cons_en:
        iter_func = lambda x: floor(sqrt(x))
        prox_coeff = sams_prox.KThreshold(iter_func)
    else:
        if simplex_en:
            dual_prox_coeff = sams_prox.Simplex()
        else:
            dual_prox_coeff = sams_prox.Positive()
    #dual_prox_coeff = sams_linear.Identity()

    # ---- (Re)Setting hyperparameters
    delta  = (polychrom_grad.inv_spec_rad**(-1)/2)**2 + 4*lin_com.mat_norm**2
    w = 0.9
    sigma_P = w*(np.sqrt(delta)-polychrom_grad.inv_spec_rad**(-1)/2)/(2*lin_com.mat_norm**2)
    tau_P = sigma_P
    rho_P = 1

    # Cost function instance
    cost_op = costObj([polychrom_grad])
    '''sams_cost.costFunction(im_stack, polychrom_grad, wavelet=None, weights=None,\
                 lambda_reg=None, mode='grad',\
                 positivity=True, tolerance=1e-4, window=1, print_cost=True,\
                 residual=False, output=None)'''

    condat_min = optimalg.Condat(P_stack, dual_var_plan, polychrom_grad, id_prox, dual_prox_plan, lin_com, cost=cost_op,\
                 rho=rho_P,  sigma=sigma_P, tau=tau_P, rho_update=None, sigma_update=None,
                 tau_update=None, auto_iterate=False)
    print "------------------- Transport plans estimation ------------------"

    condat_min.iterate(max_iter=nb_subiter) # ! actually runs optimisation
    P_stack = condat_min.x_final
    dual_var_plan = condat_min.y_final

    obs_est = polychrom_grad.MX(P_stack)
    res = im_stack - obs_est

    for i in range(0,nb_iter):
        print "----------------Iter ",i+1,"/",nb_iter,"-------------------"

        # Parameters update
        polychrom_grad_coeff.set_P(P_stack)
        if not graph_cons_en:
            lin_com_coeff.set_P_stack(P_stack)
            # ---- (Re)Setting hyperparameters
            delta  = (polychrom_grad_coeff.inv_spec_rad**(-1)/2)**2 + 4*lin_com_coeff.mat_norm**2
            w = 0.9
            sigma_coeff = w*(np.sqrt(delta)-polychrom_grad_coeff.inv_spec_rad**(-1)/2)/(2*lin_com_coeff.mat_norm**2)
            tau_coeff = sigma_coeff
            rho_coeff = 1

        # Coefficients cost function instance
        cost_op_coeff = costObj([polychrom_grad_coeff])
        '''sams_cost.costFunction(im_stack, polychrom_grad_coeff, wavelet=None, weights=None,\
                     lambda_reg=None, mode='grad',\
                     positivity=True, tolerance=1e-4, window=1, print_cost=True,\
                     residual=False, output=None)'''

        if graph_cons_en:
            beta_param = polychrom_grad_coeff.inv_spec_rad# set stepsize to inverse spectral radius of coefficient gradient
            min_coeff = optimalg.ForwardBackward(alph, polychrom_grad_coeff, prox_coeff, beta_param=beta_param, 
                                                 cost=cost_op_coeff,auto_iterate=False)
        else:
            min_coeff = optimalg.Condat(A, dual_var_coeff, polychrom_grad_coeff, id_prox, dual_prox_coeff, lin_com_coeff, cost=cost_op_coeff,\
                                            rho=rho_coeff,  sigma=sigma_coeff, tau=tau_coeff, rho_update=None, sigma_update=None,\
                                            tau_update=None, auto_iterate=False)

        print "------------------- Coefficients estimation ----------------------"
        min_coeff.iterate(max_iter=nb_subiter) # ! actually runs optimisation
        if graph_cons_en:
            prox_coeff.reset_iter()
            alph = min_coeff.x_final
            A = alph.dot(basis)
        else:
            A = min_coeff.x_final
            dual_var_coeff = min_coeff.y_final

        # Parameters update
        polychrom_grad.set_A(A)
        if not wvl_en:
            lin_com.set_A(A)
        if wvl_en:
            # Noise estimate update
            noise_map = get_noise_arr(lin_com.op(polychrom_grad.MtX(im_stack))[1])
            dual_prox_plan.update_weights(noise_map)

        # ---- (Re)Setting hyperparameters
        delta  = (polychrom_grad.inv_spec_rad**(-1)/2)**2 + 4*lin_com.mat_norm**2
        w = 0.9
        sigma_P = w*(np.sqrt(delta)-polychrom_grad.inv_spec_rad**(-1)/2)/(2*lin_com.mat_norm**2)
        tau_P = sigma_P
        rho_P = 1

        # Cost function instance
        condat_min = optimalg.Condat(P_stack, dual_var_plan, polychrom_grad, id_prox, dual_prox_plan, lin_com, cost=cost_op,\
                     rho=rho_P,  sigma=sigma_P, tau=tau_P, rho_update=None, sigma_update=None,
                     tau_update=None, auto_iterate=False)
        print "------------------- Transport plans estimation ------------------"

        condat_min.iterate(max_iter=nb_subiter) # ! actually runs optimisation
        P_stack = condat_min.x_final
        dual_var_plan = condat_min.y_final

        # Normalization
        for j in range(0,nb_comp):
            l1_P = sum(abs(P_stack[:,:,j]))
            P_stack[:,:,j]/= l1_P
            A[j,:] *= l1_P
            if graph_cons_en:
                alph[j,:] *= l1_P
        polychrom_grad.set_A(A)
        # Flux update
        obs_est = polychrom_grad.MX(P_stack)
        err_ref = 0.5*sum((obs_est-im_stack)**2)
        flux_new = (obs_est*im_stack).sum(axis=(0,1))/(obs_est**2).sum(axis=(0,1))
        print "Flux correction: ",flux_new
        polychrom_grad.set_flux(polychrom_grad.get_flux()*flux_new)
        polychrom_grad_coeff.set_flux(polychrom_grad_coeff.get_flux()*flux_new)

        obs_est = polychrom_grad.MX(P_stack)
        res = im_stack - obs_est
        err_rec = 0.5*sum(res**2)
        print "err_ref : ",err_ref," ; err_rec : ", err_rec
        # Computing residual


    psf_est = psf_learning_utils.field_reconstruction(P_stack,shap,supp,neighbors_graph,weights_neighbors,A)

    return psf_est,P_stack,A,res


def test_lsq(Y,A,nb_iter=1000):
    shap1 = Y.shape
    shap2 = A.shape
    S  = zeros((shap1[0],shap1[1],shap2[0]))
    U, s, Vt = linalg.svd(A.dot(transpose(A)),full_matrices=False)
    spec_rad = s[0]
    cur_est  =None
    for k in range(0,nb_iter):
        cur_est = Y*0
        for l in range(0,shap1[2]):
            for p in range(0,shap2[0]):
                cur_est[:,:,l] += A[p,l]*S[:,:,p]
        res = Y-cur_est
        print 'residual: ',sum(res**2)
        grad = S*0
        for i in range(0,shap2[0]):
            for j in range(0,shap1[2]):
                grad[:,:,i]-=res[:,:,j]*A[i,j]

        S = S - grad/spec_rad

    return S,cur_est


def low_rank_comp_wise_sparse_dist_dyn_coeff_GFB_m(psf_stack_in,field_dist,upfact,opt,nsig,global_sparsity_en=True,sr_en=True,sparsity_en=True,pix_sparsity=True,redun_fact=3,neigh_frac=0.5,dist_weight_deg=0,shifts=None,opt_shift_est=None,nsig_shift_est=None,sig_est=None,flux_est=None,nb_iter=2,nb_subiter=200,nb_subiter_min=7,mu=1.5,nb_comp_max=10,tol=0.1,wvl_transp_cor=1,cv_proof_en=0,line_search_en=0,cond_fact=0.1,pos_relax=0,nb_subiter_min_2=5,dyn_upload=False,positivity_en=False,score=90,refine=False,robust_refine=True,nsig_hub=5,nb_rw=1,lsq_en=False,shifts_regist = True,wavr_en=False,curv=True):

    shap = psf_stack_in.shape
    est = zeros((shap[0]*upfact,shap[1]*upfact,shap[2],shap[3],shap[4]))
    resm =  zeros((shap[0],shap[1],shap[2],shap[3],shap[4]))
    ref_est = None
    if refine:
        ref_est = zeros((shap[0]*upfact,shap[1]*upfact,shap[2],shap[3],shap[4]))
    no_shifts = True
    if shifts is not None:
        no_shifts = False
    for k in range(0,shap[4]):

        for i in range(0,shap[3]):
            print "==============================================SNR: ",i,"/",shap[4],"Field: ",k,"/",shap[3],"======================================================="
            im_hr,im_hr_curv,comp,comp_curv,weights,res,sig_out,flux_est,shifts,alph,alph_ref,e_opt,p_opt,ker,supports = low_rank_comp_wise_sparse_dist_dyn_coeff_GFB_joint(psf_stack_in[:,:,:,i,k],field_dist,upfact,opt,nsig,global_sparsity_en=global_sparsity_en,sr_en=sr_en,sparsity_en=sparsity_en,pix_sparsity=pix_sparsity,redun_fact=redun_fact,neigh_frac=neigh_frac,dist_weight_deg=dist_weight_deg,shifts=shifts,opt_shift_est=opt_shift_est,nsig_shift_est=nsig_shift_est,sig_est=sig_est,flux_est=flux_est,nb_iter=nb_iter,nb_subiter=nb_subiter*2,nb_subiter_min=nb_subiter_min,mu=mu,nb_comp_max=nb_comp_max,tol=tol,wvl_transp_cor=wvl_transp_cor,cv_proof_en=cv_proof_en,line_search_en=line_search_en,cond_fact=cond_fact,pos_relax=pos_relax,nb_subiter_min_2=nb_subiter_min_2,dyn_upload=dyn_upload,positivity_en=positivity_en,score=score,refine=refine,robust_refine=robust_refine,nsig_hub=nsig_hub,nb_rw=nb_rw,lsq_en=lsq_en,shifts_regist = shifts_regist,wavr_en=wavr_en,curv=curv)
            if no_shifts:
                shifts = None
            est[:,:,:,i,k] = im_hr
            resm[:,:,:,i,k] = res



    return est,resm

def low_rank_comp_wise_sparse_dist_dyn_coeff_GFB_m2(psf_stack_in,field_dist,upfact,opt,nsig,global_sparsity_en=True,sr_en=True,sparsity_en=True,pix_sparsity=True,redun_fact=3,neigh_frac=0.5,dist_weight_deg=0,shifts_arr=None,opt_shift_est=None,nsig_shift_est=None,sig_est=None,flux_est=None,nb_iter=5,nb_subiter=20,nb_subiter_min=7,mu=1.5,nb_comp_max=10,tol=0.1,wvl_transp_cor=1,cv_proof_en=0,line_search_en=0,cond_fact=0.1,pos_relax=0,nb_subiter_min_2=5,dyn_upload=False,positivity_en=False,score=90,refine=True,robust_refine=True,nsig_hub=5):

    shap = psf_stack_in.shape
    shap2 = shifts_arr.shape
    est = zeros((shap[0]*upfact,shap[1]*upfact,shap[2],shap2[2]))

    ref_est = None
    if refine:
        ref_est = zeros((shap[0]*upfact,shap[1]*upfact,shap[2],shap2[2]))


    for i in range(0,shap2[2]):
        print "==============================================",i,"/",shap2[2],"======================================================="
        im_hr,ref_im_hr,comp,ref_comp,weights,ref_weights,res,ref_res,sig_out,flux_est,shifts,compz1,compz2,res_inter_comp,detect_flag = low_rank_comp_wise_sparse_dist_dyn_coeff_GFB(psf_stack_in,field_dist,upfact,opt,nsig,global_sparsity_en=global_sparsity_en,sr_en=sr_en,sparsity_en=sparsity_en,pix_sparsity=pix_sparsity,redun_fact=redun_fact,neigh_frac=neigh_frac,dist_weight_deg=dist_weight_deg,shifts=shifts_arr[:,:,i],opt_shift_est=opt_shift_est,nsig_shift_est=nsig_shift_est,sig_est=sig_est,flux_est=flux_est,nb_iter=nb_iter,nb_subiter=nb_subiter,nb_subiter_min=nb_subiter_min,mu=mu,nb_comp_max=nb_comp_max,tol=tol,wvl_transp_cor=wvl_transp_cor,cv_proof_en=cv_proof_en,line_search_en=line_search_en,cond_fact=cond_fact,pos_relax=pos_relax,nb_subiter_min_2=nb_subiter_min_2,dyn_upload=dyn_upload,positivity_en=positivity_en,score=score,refine=refine,robust_refine=robust_refine,nsig_hub=nsig_hub)

        est[:,:,:,i] = im_hr
        if refine:
            ref_est[:,:,:,i] = ref_im_hr

    return est,ref_est

def low_rank_comp_wise_sparse_dist_dyn_coeff_GFB_fast(psf_stack_in,field_dist,upfact,opt,nsig,neigh_frac=0.5,dist_weight_deg=1,shifts=None,opt_shift_est=None,nsig_shift_est=None,sig_est=None,flux_est=None,nb_iter=5,redun_fact=1,nb_subiter=20,nb_subiter_min=5,mu=1.5,nb_comp_max=10,tol=0.1,wvl_transp_cor=1,cv_proof_en=0,line_search_en=0,cond_fact=0.1,pos_relax=0,ortho_en=False,ortho_en_strong=True): # Representations coefficietns are updated in all the components loops
    if ortho_en and ortho_en_strong:
        ortho_en=False
    psf_stack = copy(psf_stack_in)
    print "Minimiser: GFB"
    # Degradation operator parameters estimation
    if shifts is None:
        #shifts = utils.shift_est(psf_stack)*upfact
        shifts = utils.wvl_shift_est(psf_stack,opt_shift_est,nsig_shift_est)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)
    if sig_est is None:
        sig_est = utils.im_gauss_nois_est_cube(psf_stack_in,opt=opt_shift_est)
    sig_min = sig_est.min()
    sig_est = sig_est/sig_min
    shap = psf_stack.shape
    nb_im = shap[2]
    for k in range(0,shap[2]):
        psf_stack[:,:,k] = psf_stack[:,:,k]/sig_est[k]
    print " ------ ref energy: ",(psf_stack**2).sum()," ------- "
    if flux_est is None:
        flux_est = utils.flux_estimate_stack(psf_stack_in,rad=4)
    flux_ref = np.median(flux_est)
    weights = None
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))

    # Distances settings
    print "Contructing PSF tree..."
    nb_neighs = redun_fact*upfact**2
    neigh,dists = utils.knn_interf(field_dist,nb_neighs)
    p_max = pow_law_select(dists,nb_neighs)
    print "power max = ",p_max
    p_min = 1
    print "Done..."
    dists = dists
    dist_med = np.median(dists)
    dist_weights = (dist_med/dists)**dist_weight_deg
    dist_weigths = dist_weights/dist_weights.max()
    # Spatial smoothing operator gradient lip constant estimation
    spec_rad_smooth = None


    siz_in = upfact*array(shap[0:-1])
    eig_max = None


    # GFB variables
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(copy(w))
    input.append(sig_est)
    input.append(flux_est)
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    comp_sp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    comp_old = zeros((upfact*shap[0],upfact*shap[1]))
    z11 = zeros((upfact*shap[0],upfact*shap[1])) # Positive variable
    z12,mr_file = isap.mr_trans(z11,opt=opt) # Sparse variable
    os.remove(mr_file)
    z21 = zeros((upfact*shap[0],upfact*shap[1])) # Bonding variables
    z22,mr_file = isap.mr_trans(z11,opt=opt)
    os.remove(mr_file)
    x2 = copy(z22) # Main variable second group of components
    compz1 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    compz2 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    w1 = 0.5
    w2 = 0.5
    w_pos = 0.5
    w_sp = 1-w_pos
    lambd = 1
    grad = zeros((upfact*shap[0],upfact*shap[1]))
    gradsp_temp = zeros((upfact*shap[0],upfact*shap[1]))
    grad_sp = z22*0
    u,mr_file = isap.mr_trans(comp[:,:,0],opt=opt)
    shap2 = u.shape
    weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
    res = copy(psf_stack)

    comp_lr = zeros((shap[0],shap[1],nb_comp_max,shap[2]))
    comp_hr_i = zeros((shap[0]*upfact,shap[1]*upfact,nb_comp_max,shap[2]))
    comp_lr_i_stack = zeros((shap[0]*upfact,shap[1]*upfact,nb_comp_max*shap[2]))
    i = 0
    buff = zeros((nb_subiter_min))
    buffsp = zeros((nb_subiter_min))
    nb_iter_noisest = 20
    thresh_type = 1
    noise_vect_1 = None
    sig_map = None
    cur_res=100000000.0
    res_old=1000000000.0
    cur_res_comp=100000000.0
    res_old_comp=1000000000.0
    res_inter_comp = zeros(nb_comp_max)
    weights_init=None
    p_smth_mat_inv=None
    to=None
    print " ------ ref energy: ",(psf_stack**2).sum()," ------- "
    pows = zeros((nb_comp_max))
    min_val_map = None
    ortho_mat = None
    while i < nb_comp_max :#and 100*abs(cur_res_comp-res_old_comp)/cur_res_comp>0.001:
        if i==0:
            pows[i]=p_min # =1
        else:
            pows[i]= (pows[i-1]+p_max)/2
        dist_weights_in = zeros((nb_im,nb_neighs,i+1))
        for ind in range(0,i+1):
            dist_weights_in[:,:,ind] = (dist_med/dists)**pows[ind]
            dist_weights_in[:,:,ind] = dist_weights_in[:,:,ind]/(dist_weights_in[:,:,ind].sum())

        if i>0:
            w = utils.abs_val_reverting(weights[i-1,:])

        resk = res*0
        resk_sp = res*0
        weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))

        count = 0
        weights_init=None
        p_smth_mat_inv=None
        for k in range(0,nb_iter):
            # Direction initialization
            if k==0:
                if i==0:
                    comp[:,:,i] = pos_proj_mat(sr_op_trans_stack_pseudo_inv(res,shift_ker_stack,shift_ker_stack_adj,upfact,utils.vect_recond(w,cond_fact),utils.vect_recond(sig_est,cond_fact),utils.vect_recond(flux_est,cond_fact),15))
                else:
                    comp[:,:,i] = sr_op_trans_stack_pseudo_inv(res,shift_ker_stack,shift_ker_stack_adj,upfact,utils.vect_recond(w,cond_fact),utils.vect_recond(sig_est,cond_fact),utils.vect_recond(flux_est,cond_fact),15)
            t=1
            comp[:,:,i] = comp[:,:,i]/sqrt((comp[:,:,i]**2).sum())

            for l in range(0,shap[2]):
                comp_lr[:,:,i,l] = (flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)


            print " ------ ref energy: ",(psf_stack**2).sum()," ------- "
            if k==0:
                weights,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_2(psf_stack,comp_lr[:,:,0:i+1,:],comp[:,:,0:i+1],neigh,dist_weights_in,p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=30,tol=0.1)
            else:
                weights,min_val_map = non_unif_smoothing_mult_coeff_pos_cp_2(psf_stack,comp_lr[:,:,0:i+1,:],comp[:,:,0:i+1],neigh,dist_weights_in,p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=30,tol=0.1,Ainit=weights)

            w = copy(weights[i,:])
            im_hr = 0*im_hr
            for l in range(0,shap[2]):
                for p in range(0,i):
                    im_hr[:,:,l] = im_hr[:,:,l]+weights[p,l]*comp[:,:,p]
                res[:,:,l] = psf_stack[:,:,l]-(flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(im_hr[:,:,l],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)


            print "------------------- Component ",i," ------------------"
            print "weights: ",w
            input[4]=w
            # Spectral radius
            eig_max,spec_rad = pow_meth(sr_op_trans_stack,input,siz_in,ainit=eig_max)


            # ith component estimation
            j=0
            #print '--------------- ',nb_subiter,' ----------------'
            res_old = cur_res*2
            while j < nb_subiter and 100*abs(cur_res-res_old)/res_old>0.001:
                res_old = cur_res
                # Residual computation
                grad = grad*0
                gradsp_temp = gradsp_temp*0
                if j==0:
                    for l in range(0,shap[2]):
                        resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        grad = grad-w_pos*(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                        comp_sp[:,:,i] = isap.mr_recons_coeff(x2,mr_file)
                        resk_sp[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp_sp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        gradsp_temp = gradsp_temp-w_sp*(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk_sp[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                else:
                    for l in range(0,shap[2]):
                        grad = grad-w_pos*(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                        gradsp_temp = gradsp_temp-w_sp*(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk_sp[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                buff[j%nb_subiter_min] = (resk**2).sum()
                if j==0:
                    print "---- Ref mse ---- ",buff[j%nb_subiter_min]

                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]
                # Exact line search
                muj = mu/spec_rad
                if line_search_en==1:
                    c1=0
                    c2=0
                    c3=0
                    for l in range(0,shap[2]):
                        u1 = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(2*comp[:,:,i]-z1,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        u2 = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(2*comp[:,:,i]-z2,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        u3 = -(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(grad,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        c1+= (u1*u3).sum()
                        c2+= (u2*u3).sum()
                        c3+= (u3**2).sum()
                        muj = min(c1/c3,c2/c3)
                if cv_proof_en ==1:
                    muj = min(muj,mu/spec_rad)
                    print "Optimal step: ",muj," Max for cv proof: ",2/spec_rad
                # ---- Analysis constraint ---- #
                if j==0:
                    print "---- Ref l1 norm ---- ",buffsp[j%nb_subiter_min]
                if k==0 and j==0:
                    # GFB variables init
                    sig_map = utils.res_sig_map(muj*grad,opt=opt)
                    thresh_map = nsig*sig_map
                    z11 = copy(comp[:,:,i])
                    z12,mr_file_temp = isap.mr_trans(z11,opt=opt) # Sparse variable
                    os.remove(mr_file_temp)
                    z12 = utils.thresholding(z12,thresh_map,1) # 1=> soft thresholding
                    x2 = copy(z12)
                    z21,z22 = proj_dict_equal_wvl(z11,z12,opt,mr_file,corr_coeff=wvl_transp_cor)



                # Sparsity
                t2,mr_file_temp = isap.mr_trans(comp[:,:,i],opt=opt)
                if i==0 and k==0 and (comp[:,:,i]**2).max()>0:
                    a = (comp[:,:,i]**2).sum()
                    b = (t2**2).sum()
                    wvl_transp_cor = a/b
                gradsp_temp = gradsp_temp*wvl_transp_cor
                gradsp,mr_file_temp = isap.mr_trans(gradsp_temp,opt=opt)
                temp1 = 2*x2 - z12 - muj*gradsp
                # Wavelet noise estimation

                sig_map = utils.res_sig_map(muj*gradsp_temp,opt=opt)
                thresh_map = nsig*sig_map
                thresh_map = thresh_map*weights_sp
                buffsp[j%nb_subiter_min] = abs(x2[:,:,0:-1]*thresh_map).sum()

                result = utils.thresholding_3D(temp1,thresh_map,1) # 1=> soft thresholding
                os.remove(mr_file_temp)
                z12 = z12+lambd*(result-x2)

                # ---- Positivity constraint ---- #
                z11 = z11+lambd*(proj_affine_pos2(2*comp[:,:,i] - z11 - muj*grad,w,im_hr,min_val_map=min_val_map)-comp[:,:,i])

                # ---- Bonding constraint ----- #
                temp1 = 2*comp[:,:,i] - z21 - muj*grad
                temp2 = 2*x2 - z22 - muj*gradsp
                proj1,proj2 = proj_dict_equal_wvl(temp1,temp2,opt,mr_file,corr_coeff=wvl_transp_cor)
                z21 = z21+lambd*(proj1-comp[:,:,i])
                z22 = z22+lambd*(proj2-x2)

                # ---- Main variable update ---- #
                comp[:,:,i] = w1*z11 + w2*z21
                x2 = w1*z12 + w2*z22


                # --- Sanity check --- #
                im_hr = 0*im_hr
                for l in range(0,shap[2]):
                    for p in range(0,i+1):
                        im_hr[:,:,l] = im_hr[:,:,l]+weights[p,l]*comp[:,:,p]

                print "min val: ",im_hr.min()

                count+=1
                if j==0:
                    print "Estimated noise at the first scales: ",sig_map[0,0,:]

                j+=1
                comp_sp[:,:,i] = isap.mr_recons_coeff(x2,mr_file)
                for l in range(0,shap[2]):
                    resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                    resk_sp[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp_sp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)

                buff[j%nb_subiter_min] = (resk**2).sum()
                buffsp[j%nb_subiter_min] = abs(x2[:,:,0:-1]*thresh_map).sum()

                print "mse: ", buff[j%nb_subiter_min]#," sparse term l1 norm: ",buffsp[j%nb_subiter_min]," total cost: ",buff[j%nb_subiter_min]+buffsp[j%nb_subiter_min]
                if j>=nb_subiter_min:
                    cur_res = buff.mean()#+buffsp.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]#+buffsp[j%nb_subiter_min]
                # Sparsity transfer check
                x2est,mr_file_temp = isap.mr_trans(comp[:,:,i],opt=opt)
                os.remove(mr_file_temp)
                misfit = 100*((x2est[:,:,0:-1]-x2[:,:,0:-1])**2).sum()/(x2[:,:,0:-1]**2).sum()
                print misfit,"% of sparsity not transfered"
            res_inter_comp[i] = buff[j%nb_subiter_min]
            buff=buff*0
            # Component normalization
            a = sqrt((comp[:,:,i]**2).sum())
            comp[:,:,i] = comp[:,:,i]/a
            weights[i,:] = a*weights[i,:]
            # Sparse component update
            comp_sp[:,:,i] = isap.mr_recons_coeff(z12,mr_file)
            comp_sp[:,:,i] = comp_sp[:,:,i]/sqrt((comp_sp[:,:,i]**2).sum())
            # Reweighting
            coeffx,mr_file_temp = isap.mr_trans(comp[:,:,i],opt=opt)
            os.remove(mr_file_temp)
            weights_sp  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)
        # Residual update and HR images update
        res_old_comp = cur_res_comp
        cur_res_comp = buff[j%nb_subiter_min]

        im_hr = 0*im_hr

        for l in range(0,shap[2]):
            for p in range(0,i+1):
                im_hr[:,:,l] = im_hr[:,:,l]+weights[p,l]*comp[:,:,p]
            res[:,:,l] = psf_stack[:,:,l]-(flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(im_hr[:,:,l],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
            comp_lr[:,:,i,l] = (flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)

        i+=1
        # Optim variables resetting
        z11 = z11*0
        z12 = z12*0
        z21 = z21*0
        z22 = z22*0
        i1,i2,i3 = where(weights_sp!=1)
        weights_sp[i1,i2,i3]=1
    os.remove(mr_file)
    if nsig==0:
        print "Warning: no sparsity constraint"
    return im_hr,comp,weights,res,sig_est*sig_min,flux_est,shifts,res_inter_comp

def low_rank_comp_wise_sparse_dist_dyn_coeff_GFB_fast_2(psf_stack_in,field_dist,upfact,opt,nsig,neigh_frac=0.5,dist_weight_deg=1,shifts=None,opt_shift_est=None,nsig_shift_est=None,sig_est=None,flux_est=None,nb_iter=5,redun_fact=1,nb_subiter=20,nb_subiter_min=5,mu=1.5,nb_comp_max=10,tol=0.1,wvl_transp_cor=1,cv_proof_en=0,line_search_en=0,cond_fact=0.6,pos_relax=0): # Representations coefficietns are updated in all the components loops
    psf_stack = copy(psf_stack_in)
    print "Minimiser: GFB"
    # Degradation operator parameters estimation
    if shifts is None:
        #shifts = utils.shift_est(psf_stack)*upfact
        shifts = utils.wvl_shift_est(psf_stack,opt_shift_est,nsig_shift_est)*upfact
    shift_ker_stack,shift_ker_stack_adj = utils.shift_ker_stack(shifts,upfact)
    if sig_est is None:
        sig_est = utils.im_gauss_nois_est_cube(psf_stack_in,opt=opt_shift_est)
    sig_min = sig_est.min()
    sig_est = sig_est/sig_min
    shap = psf_stack.shape
    nb_im = shap[2]
    for k in range(0,shap[2]):
        psf_stack[:,:,k] = psf_stack[:,:,k]/sig_est[k]
    print " ------ ref energy: ",(psf_stack**2).sum()," ------- "
    if flux_est is None:
        flux_est = utils.flux_estimate_stack(psf_stack_in,rad=4)
    flux_ref = np.median(flux_est)
    weights = None
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))

    # Distances settings
    print "Contructing PSF tree..."
    nb_neighs = redun_fact*upfact**2
    neigh,dists = utils.knn_interf(field_dist,nb_neighs)
    p_max = pow_law_select(dists,nb_neighs)
    print "power max = ",p_max
    p_min = 1
    print "Done..."
    dists = dists
    dist_med = np.median(dists)
    dist_weights = (dist_med/dists)**dist_weight_deg
    dist_weigths = dist_weights/dist_weights.max()
    # Spatial smoothing operator gradient lip constant estimation
    spec_rad_smooth = None


    siz_in = upfact*array(shap[0:-1])
    eig_max = None


    # GFB variables
    input = list()
    x = zeros((upfact*shap[0],upfact*shap[1]))
    input.append(x)
    input.append(upfact)
    input.append(shift_ker_stack)
    input.append(shift_ker_stack_adj)
    w = ones((shap[2],))
    im_hr = zeros((upfact*shap[0],upfact*shap[1],shap[2]))
    input.append(copy(w))
    input.append(sig_est)
    input.append(flux_est)
    comp = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    comp_old = zeros((upfact*shap[0],upfact*shap[1]))
    z11 = zeros((upfact*shap[0],upfact*shap[1])) # Positive variable
    z12,mr_file = isap.mr_trans(z11,opt=opt) # Sparse variable
    os.remove(mr_file)
    z21 = zeros((upfact*shap[0],upfact*shap[1])) # Bonding variables
    z22,mr_file = isap.mr_trans(z11,opt=opt)
    proj1=None
    proj2=None
    os.remove(mr_file)
    x2 = copy(z22) # Main variable second group of components
    compz1 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    compz2 = zeros((upfact*shap[0],upfact*shap[1],nb_comp_max)) # Main variable
    coeff_init = None
    w1 = 0.5
    w2 = 0.5
    lambd = 1
    grad = zeros((upfact*shap[0],upfact*shap[1]))

    u,mr_file = isap.mr_trans(comp[:,:,0],opt=opt)
    shap2 = u.shape
    weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
    res = copy(psf_stack)

    comp_lr = zeros((shap[0],shap[1],nb_comp_max,shap[2]))
    i = 0
    buff = zeros((nb_subiter_min))

    nb_iter_noisest = 20
    nb_subiter_prox = 50
    thresh_type = 1
    noise_vect_1 = None
    sig_map = None
    cur_res=100000000.0
    res_old=1000000000.0
    cur_res_comp=100000000.0
    res_old_comp=1000000000.0
    res_inter_comp = zeros(nb_comp_max)
    print " -------------------- Debugging warning: sparsity, no coeff smoothing ------------------- "
    weights_init=None
    p_smth_mat_inv=None
    to=None
    print " ------ ref energy: ",(psf_stack**2).sum()," ------- "
    pows = zeros((nb_comp_max))
    while i < nb_comp_max and 100*abs(cur_res_comp-res_old_comp)/cur_res_comp>0.1:
        if i==0:
            pows[i]=p_min # =1
        else:
            pows[i]= (pows[i-1]+p_max)/2
        dist_weights_in = zeros((nb_im,nb_neighs,i+1))
        for ind in range(0,i+1):
            dist_weights_in[:,:,ind] = (dist_med/dists)**pows[ind]
            dist_weights_in[:,:,ind] = dist_weights_in[:,:,ind]/(dist_weights_in[:,:,ind].sum())

        if i>0:
            w = utils.abs_val_reverting(weights[i-1,:])
        #spec_rad = spec_rad_init
        #eig = eig_init
        resk = res*0
        weights_sp = ones((shap2[0],shap2[1],shap2[2]-1))
        """spec_rad=0
            for f in range(0,shap[2]):
            spec_rad = spec_rad+(w[f]*flux_ref/(sig_est[f]*flux_est[f]))**2*(abs(shift_ker_stack[:,:,f]).sum())**2
            print "spectral radius: ",spec_rad"""

        nb_subiter_prox = 60
        count = 0
        weights_init=None
        p_smth_mat_inv=None
        for k in range(0,nb_iter):
            # Direction initialization
            if k==0:
                if i==0:
                    comp[:,:,i] = pos_proj_mat(sr_op_trans_stack_pseudo_inv(res,shift_ker_stack,shift_ker_stack_adj,upfact,utils.vect_recond(w,cond_fact),utils.vect_recond(sig_est,cond_fact),utils.vect_recond(flux_est,cond_fact),30))
                else:
                    comp[:,:,i] = sr_op_trans_stack_pseudo_inv(res,shift_ker_stack,shift_ker_stack_adj,upfact,utils.vect_recond(w,cond_fact),utils.vect_recond(sig_est,cond_fact),utils.vect_recond(flux_est,cond_fact),30)

            t=1
            if i>0:
                comp[:,:,0:i+1] = utils.gram_schmidt_cube(comp[:,:,0:i+1])

            for l in range(0,shap[2]):
                comp_lr[:,:,i,l] = (flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)

            #weights,mat = lsq_mult_coeff_stack(psf_stack,comp_lr[:,:,0:i+1,:])
            print " ------ ref energy: ",(psf_stack**2).sum()," ------- "
            if i==0:
                if k==0:
                    weights = non_unif_smoothing_mult_coeff_pos_cp(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights_in,p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=30,tol=0.1)
                    #weights = lsq_mult_coeff_stack_pos_2(psf_stack,comp_lr[:,:,0:i+1,:],nb_iter=30)
                    #weights = non_unif_smoothing_mult_coeff_pos(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights,nb_iter=30)
                else:
                    weights = non_unif_smoothing_mult_coeff_pos_cp(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights_in,p_smth_mat_inv=p_smth_mat_inv,to=to,nb_iter=30,tol=0.1,Ainit=weights)
                    #weights = non_unif_smoothing_mult_coeff_pos(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights,Ainit=weights,nb_iter=30)
                    """weights_init=None
                        if k>0:
                        weights_init = weights
                        weights = non_unif_smoothing_mult_coeff(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights,Ainit=weights_init)
                        else :
                        weights,mat = lsq_mult_coeff_stack(psf_stack,comp_lr[:,:,0:i+1,:])
                        weights_init = zeros((i+1,shap[2]))
                        weights_init[0:i,:] = weights
                        weights_init[i,:] = w
                        weights = non_unif_smoothing_mult_coeff(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights,Ainit=weights_init)"""
            else:

                if k==0:
                    weights = non_unif_smoothing_mult_coeff_pos_dr(psf_stack,comp[:,:,0:i+1],comp_lr[:,:,0:i+1,:],neigh,dist_weights_in,upfact,shift_ker_stack,shift_ker_stack_adj,sig_est,flux_est,p_smth_mat_inv=p_smth_mat_inv,eps=0.8,nb_iter=30,tol=0.01)
                #weights = lsq_mult_coeff_stack_pos_2(psf_stack,comp_lr[:,:,0:i+1,:],nb_iter=30)
                #weights = non_unif_smoothing_mult_coeff_pos(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights,nb_iter=30)
                else:
                    weights = non_unif_smoothing_mult_coeff_pos_dr(psf_stack,comp[:,:,0:i+1],comp_lr[:,:,0:i+1,:],neigh,dist_weights_in,upfact,shift_ker_stack,shift_ker_stack_adj,sig_est,flux_est,p_smth_mat_inv=p_smth_mat_inv,eps=0.8,nb_iter=30,tol=0.01,Ainit=weights)
                    #weights = non_unif_smoothing_mult_coeff_pos(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights,Ainit=weights,nb_iter=30)
                    """weights_init=None
                        if k>0:
                        weights_init = weights
                        weights = non_unif_smoothing_mult_coeff(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights,Ainit=weights_init)
                        else :
                        weights,mat = lsq_mult_coeff_stack(psf_stack,comp_lr[:,:,0:i+1,:])
                        weights_init = zeros((i+1,shap[2]))
                        weights_init[0:i,:] = weights
                        weights_init[i,:] = w
                        weights = non_unif_smoothing_mult_coeff(psf_stack,comp_lr[:,:,0:i+1,:],neigh,dist_weights,Ainit=weights_init)"""

            w = weights[i,:]
            im_hr = 0*im_hr
            for l in range(0,shap[2]):
                for p in range(0,i):
                    im_hr[:,:,l] = im_hr[:,:,l]+weights[p,l]*comp[:,:,p]
                res[:,:,l] = psf_stack[:,:,l]-(flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(im_hr[:,:,l],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
            print "------------------- Component ",i," ------------------"
            print "weights: ",w
            input[4]=w
            # Spectral radius
            eig_max,spec_rad = pow_meth(sr_op_trans_stack,input,siz_in,ainit=eig_max)

            # ith component estimation
            j=0
            #print '--------------- ',nb_subiter,' ----------------'
            res_old = cur_res*2
            while j < nb_subiter and 100*abs(cur_res-res_old)/res_old>0.001:
                res_old = cur_res
                # Residual computation
                grad = grad*0
                if j==0:
                    for l in range(0,shap[2]):
                        resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                    grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                else:
                    for l in range(0,shap[2]):
                        grad = grad-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*scisig.convolve(utils.transpose_decim(resk[:,:,l],upfact),shift_ker_stack_adj[:,:,l],mode='same')
                buff[j%nb_subiter_min] = (resk**2).sum()
                if j==0:
                    print "---- Ref mse ---- ",buff[j%nb_subiter_min]

                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]
                # Exact line search
                muj = mu/spec_rad
                if line_search_en==1:
                    c1=0
                    c2=0
                    c3=0
                    for l in range(0,shap[2]):
                        u1 = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(2*comp[:,:,i]-z1,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        u2 = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(2*comp[:,:,i]-z2,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        u3 = -(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(grad,shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                        c1+= (u1*u3).sum()
                        c2+= (u2*u3).sum()
                        c3+= (u3**2).sum()
                        muj = min(c1/c3,c2/c3)
                if cv_proof_en ==1:
                    muj = min(muj,mu/spec_rad)
                    print "Optimal step: ",muj," Max for cv proof: ",2/spec_rad
                # ---- Analysis constraint ---- #
                # Wavelet noise estimation
                sig_map = utils.res_sig_map(muj*grad,opt=opt)
                thresh_map = nsig*sig_map
                thresh_map = thresh_map*weights_sp
                if k==0 and j==0:
                    # GFB variables init
                    z11 = copy(comp[:,:,i])
                    z12,mr_file_temp = isap.mr_trans(z11,opt=opt) # Sparse variable
                    os.remove(mr_file_temp)
                    z12 = utils.thresholding(z12,thresh_map,1) # 1=> soft thresholding
                    x2 = copy(z12)
                    z21,z22 = proj_dict_equal_wvl(z11,z12,opt,mr_file,corr_coeff=wvl_transp_cor)
                temp1 = 2*x2 - z12

                # Sparsity
                t2,mr_file_temp = isap.mr_trans(comp[:,:,i],opt=opt)
                if i==0 and k==0 and (comp[:,:,i]**2).max()>0:
                    a = (comp[:,:,i]**2).sum()
                    b = (t2**2).sum()
                    wvl_transp_cor = a/b
                result = utils.thresholding(temp1,thresh_map,1) # 1=> soft thresholding
                os.remove(mr_file_temp)
                z12 = z12+lambd*(result-x2)

                # ---- Positivity constraint ---- #
                z11 = z11+lambd*(proj_affine_pos2(2*comp[:,:,i] - z11 - muj*grad,w,im_hr)-comp[:,:,i])

                # ---- Bonding constraint ----- #
                temp1 = 2*comp[:,:,i] - z21 - muj*grad
                temp2 = 2*x2 - z22
                if i==0:
                    proj1,proj2 = proj_dict_equal_wvl(temp1,temp2,opt,mr_file,corr_coeff=wvl_transp_cor)
                else:
                    proj1,proj2 = proj_dict_equal_wvl_ortho(temp1,temp2,comp[:,:,0:i+1],opt,mr_file,corr_coeff=wvl_transp_cor)
                z21 = z21+lambd*(proj1-comp[:,:,i])
                z22 = z22+lambd*(proj2-x2)

                # ---- Main variable update ---- #
                comp[:,:,i] = w1*z11 + w2*z21
                x2 = w1*z12 + w2*z22
                print "min val: ",comp[:,:,0:i+1].min()
                count+=1
                #comp[:,:,i] = z1
                if j==0:
                    print "Estimated noise at the first scales: ",sig_map[0,0,:]

                j+=1

                for l in range(0,shap[2]):
                    resk[:,:,l] = res[:,:,l]-(w[l]*flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
                buff[j%nb_subiter_min] = (resk**2).sum()

                print "mse: ", buff[j%nb_subiter_min]
                if j>=nb_subiter_min:
                    cur_res = buff.mean()
                else:
                    cur_res = buff[j%nb_subiter_min]
                # Sparsity transfer check
                x2est,mr_file_temp = isap.mr_trans(comp[:,:,i],opt=opt)
                os.remove(mr_file_temp)
                misfit = 100*((x2est[:,:,-1]-x2[:,:,-1])**2).sum()/(x2[:,:,-1]**2).sum()
                print misfit,"% of sparsity not transfered"

            buff=buff*0
            # Component normalization
            a = sqrt((comp[:,:,i]**2).sum())
            comp[:,:,i] = comp[:,:,i]/a
            weights[i,:] = a*weights[i,:]
            # Reweighting
            coeffx,mr_file_temp = isap.mr_trans(comp[:,:,i],opt=opt)
            os.remove(mr_file_temp)
            weights_sp  = (1+abs(coeffx[:,:,:-1])/(nsig*sig_map))**(-1)
        # Residual update and HR images update
        res_old_comp = cur_res_comp
        cur_res_comp = buff[j%nb_subiter_min]
        res_inter_comp[i] = buff[j%nb_subiter_min]
        im_hr = 0*im_hr
        for l in range(0,shap[2]):
            for p in range(0,i+1):
                im_hr[:,:,l] = im_hr[:,:,l]+weights[p,l]*comp[:,:,p]
            res[:,:,l] = psf_stack[:,:,l]-(flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(im_hr[:,:,l],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
            comp_lr[:,:,i,l] = (flux_ref/(sig_est[l]*flux_est[l]))*utils.decim(scisig.fftconvolve(comp[:,:,i],shift_ker_stack[:,:,l],mode='same'),upfact,av_en=0)
        i+=1
        # Optim variables resetting
        z11 = z11*0
        z12 = z12*0
        z21 = z21*0
        z22 = z22*0
        i1,i2,i3 = where(weights_sp!=1)
        weights_sp[i1,i2,i3]=1
    # coeff_init = coeff_init*0
    os.remove(mr_file)
    return im_hr,comp,weights,res,sig_est*sig_min,flux_est,shifts,res_inter_comp



def positive_cv_update(X,S,Ainit=None,pseudo_inv=None,nb_iter=100,tol=10**(-10)): # X and S are assumed to be positive; This routine calculates A so that to minimize KL(SA|X), with the constraint that the sum of A's rows are 1; the observations are stored in X lines
    Xpos = pos_proj_mat(X)
    Spos = pos_proj_mat(S)

    shap = [S.shape[1],X.shape[1]]
    if Ainit is None:
        ones_line = ones((1,shap[1]))
        pseudo_inv = LA.inv(transpose(S).dot(S)).dot(transpose(S))
        Alsq = pos_proj_mat(pseudo_inv.dot(X))
        Ainit = transpose(kl_marg_proj(transpose(Alsq),transpose(ones_line)))
    A = copy(Ainit)
    k=0
    Slmarg = Spos.sum(axis=0)
    id = where(Slmarg==0)
    A[id,:] = 0
    id1,id2 = where(Xpos>0)
    var = 100
    while k < nb_iter and var>tol :
        k+=1
        Xest = Spos.dot(A)
        kl_div = (Xest[id1,id2] - Xpos[id1,id2]*log(Xest[id1,id2]/Xpos[id1,id2])).sum()
        lsq = ((Xest-Xpos)**2).sum()
        Aold = copy(A)
        for j in range(0,shap[1]):
            for i in range(0,shap[0]):
                if Slmarg[i]>0:
                    id = where(Xest[:,j]>0)
                    num = (Spos[id,i]*Xpos[id,j]/Xest[id,j]).sum()
                    A[i,j] = A[i,j]*num/Slmarg[i]
        A = transpose(kl_marg_proj(transpose(A),transpose(ones_line)))
        var = 100*((A-Aold)**2).sum()/(Aold**2).sum()
    print "coeff update lsq: ",lsq
    return A

#def positive_cv_update_GFB(X,S,Ainit=None,pseudo_inv=None,nb_iter=100,tol=10**(-10)):


def positive_lsq(X,A,pseudo_inv=None,Sinit=None,nb_iter=100,tol=1e-15): # This routine minimizes ||X-SA||_2^2 with S positive; the observations are stored in X colmns
    if Sinit is None:
        if pseudo_inv is None:
            pseudo_inv = transpose(A).dot(LA.inv(A.dot(transpose(A))))
        Sinit = X.dot(pseudo_inv)
    S = copy(Sinit)
    T = copy(S)
    M = A.dot(transpose(A))
    U, s, Vt = linalg.svd(M,full_matrices=False)
    spec_rad = s.max()
    t = 1
    k = 0
    var = 100
    while k <nb_iter and var>tol:
        k+=1
        grad = S.dot(M)-X.dot(transpose(A))
        Y = S - grad/spec_rad
        Told = copy(T)
        #T = pos_proj_mat(Y)
        T = copy(Y)
        told = t
        t = (1+sqrt(4*t**2+1))/2
        lambd = 1+(told-1)/t
        Sold = copy(S)
        S = Told + lambd*(T-Told)
        var = 100*((S-Sold)**2).sum()/(Sold**2).sum()
        res = X-S.dot(A)

    print "SOurces update res: ",(res**2).sum()
    return S

def pos_convex_hull_est(X,nb_points,nb_iter=10,rand_disp_param=0.4,thresh=1,sparsity_pow=3,hull_tightening=True,Sinit=None,src_ind=None,Ainit=None): # The observations are stored in lines
    shap = X.shape
    if src_ind is None:
        src_ind = range(0,shap[1])
    U, s, Vt = linalg.svd(X[:,src_ind],full_matrices=False)
    weights = s[0:nb_points]
    basis = transpose(Vt[0:nb_points,:])
    #Sinit,centroid,label_ind,ret = utils.convex_hull_init(X,nb_points)
    centroid = None

    if Sinit is None:
        tSinit = utils.convex_hull_init_3(X,nb_points,nb_frac=0.1)
        Sinit = transpose(tSinit)
        #tSinit,centroid = utils.convex_hull_init_2(X,nb_points)


    S = copy(Sinit)
    Sdisp = copy(S)
    A = None
    if Ainit is not None:
        A = copy(Ainit)
    X2 = X.dot(transpose(X))
    U, s, Vt = linalg.svd(X2,full_matrices=False)
    ref_res = s[nb_points:].sum()
    ones_vect = ones((nb_points,1))
    if centroid is None:
        centroid = (X.sum(axis=0)).reshape(1,shap[1])/shap[0]
    centroidn = ones_vect.dot(centroid)

    print "ref_res: ",ref_res
    res_in=None
    thresh_map = None
    for k in range(0,nb_iter):
        #A = positive_cv_update(transpose(X),transpose(Sdisp),nb_iter=150)
        A = bar_coord_pb_field(transpose(Sdisp),transpose(X),tinit=A,nb_iter=8000,tol=1e-15)
        #A = sparse_bar_coord_pb_field(transpose(Sdisp),transpose(X),tinit=A,thresh_map=thresh_map,thresh_type=1,nb_iter=8000,tol=1e-15)
        S[:,src_ind] = transpose(positive_lsq(transpose(X[:,src_ind]),A,nb_iter=8000,Sinit=transpose(S[:,src_ind])))
        # Random perturbation
        dist = sqrt(((S-centroidn)**2).sum(axis=1))
        print "mean distance to data set barycenter: ",mean(dist)
        sig = np.random.randn(nb_points)*dist*rand_disp_param*(nb_iter-1-k)/(nb_iter-1)
        Sdisp[:,src_ind] = transpose(utils.tg_rand_disp_set(transpose(centroid[0,src_ind]),transpose(S[:,src_ind]),sig,basis,weights=weights))

        res = X - transpose(A).dot(S)
        print "global res: ", (res**2).sum()
        res_in = ((X - transpose(A).dot(Sdisp))**2).sum()
    res_in = (res**2).sum()
    Arough = copy(A)
    Srough = copy(S)
    if hull_tightening:
        for k in range(0,nb_iter):
            #A = positive_cv_update(transpose(X),transpose(Sdisp),nb_iter=150)
            #A = sparse_bar_coord_pb_field(transpose(Sdisp),transpose(X),tinit=A,thresh_map=thresh_map,thresh_type=1,nb_iter=8000,tol=1e-15)
            thresh_k = double(thresh)*(double(k+1)/nb_iter)**sparsity_pow
            thresh_map = thresh_k*((A/thresh_k)+1)**(-1)
            A = sparse_bar_coord_pb_field_cp(transpose(S),transpose(X),res_in,A,thresh_map,thresh_type=1,nb_iter=10000,tol=1e-15,eps=0.8)
            res = X - transpose(A).dot(S)
            if ((res**2).sum()<res_in):
                res_in = (res**2).sum()
            #S = transpose(positive_lsq(transpose(X),A,nb_iter=1000,Sinit=transpose(S)))
            S[:,src_ind] = transpose(spreading_constraint_src_update_cp(transpose(X[:,src_ind]),A,transpose(centroid),transpose(S[:,src_ind]),res_in,tol=1e-15,nb_iter=10000,accel_en=True))
            dist = sqrt(((S-centroidn)**2).sum(axis=1))
            print "mean distance to data set barycenter: ",mean(dist)

            res = X - transpose(A).dot(S)
            if ((res**2).sum()<res_in):
                res_in = (res**2).sum()
            print "global res: ",(res**2).sum()
    A = pos_proj(A)
    one_vect = ones((nb_points,1))
    A = A/(one_vect.dot((A.sum(axis=0)).reshape(1,shap[0])))

    return A,S,Arough,Srough

def im_cv_hull_interf(im_stack,nb_points,nb_iter=100,hull_tightening=False):
    X = utils.cube_to_mat(im_stack)
    A,S,Arough,Srough = pos_convex_hull_est(X,nb_points,nb_iter=nb_iter,hull_tightening=hull_tightening)
    shap = im_stack.shape
    Sim = utils.mat_to_cube(S,shap[0],shap[1])
    return A,Sim,Arough,Srough

def im_cv_hull_interf_2(im_stack,nb_points,coord,nb_alt=2,nb_iter=20,scaling_fact=0.5,hull_tightening=True):
    #X = utils.cube_to_mat(im_stack)
    X,scal_fact = psf_learning_utils.data_shaping_2(im_stack,coord,scaling_fact=scaling_fact)
    shap = X.shape

    range_2 = range(shap[1]-2,shap[1])
    range_1 = range(0,shap[1]-2)
    A=None
    S=None
    rand_disp_param=0.5
    for k in range(0,nb_alt):
        if k>0:
            rand_disp_param = 0
        A,S,Arough,Srough = pos_convex_hull_est(X,nb_points,rand_disp_param=rand_disp_param,nb_iter=nb_iter,hull_tightening=False,src_ind=range_1,Sinit=S,Ainit=A)
        A,S,Arough,Srough = pos_convex_hull_est(X,nb_points,rand_disp_param=rand_disp_param,nb_iter=nb_iter,hull_tightening=False,src_ind=range_2,Sinit=S,Ainit=A)
    A,S,Arough,Srough = pos_convex_hull_est(X,nb_points,rand_disp_param=0,nb_iter=nb_iter,hull_tightening=False,src_ind=range_1,Sinit=S,Ainit=A)
    if hull_tightening:
        A,S,Arough,Srough = pos_convex_hull_est(X,nb_points,rand_disp_param=0,nb_iter=nb_iter,hull_tightening=hull_tightening,src_ind=range_1,Sinit=S,Ainit=A)
    shap = im_stack.shape
    Sim = utils.mat_to_cube(S[:,0:shap[0]*shap[1]],shap[0],shap[1])
    coord_src = S[:,shap[0]*shap[1]:]/scal_fact
    return A,Sim,S,Arough,Srough,coord_src

def proj_dict_equal_wvl(x,y,opt,mr_file,corr_coeff=None): # DTD = corr_coeff*Id; ^D = DT/corr_coeff
    shap = x.shape
    if corr_coeff is None:
        dirac = zeros(shap)
        dirac[round(shap[0]/2),round(shap[0]/2)]=1
        t2,mr_file_temp = isap.mr_trans(dirac,opt=opt)
        a = (dirac**2).sum()
        b = (t2**2).sum()
        corr_coeff = a/b
        os.remove(mr_file_temp)
    x1 = (x+isap.mr_recons_coeff(y,mr_file)*corr_coeff)/(1+corr_coeff)
    y1,mr_file_temp = isap.mr_trans(x1,opt=opt)
    os.remove(mr_file_temp)
    return x1,y1

def proj_dict_equal_wvl_ortho(x,y,S,opt,mr_file,corr_coeff=None): # DTD = corr_coeff*Id; ^D = DT/corr_coeff
    shap = x.shape
    if corr_coeff is None:
        dirac = zeros(shap)
        dirac[round(shap[0]/2),round(shap[0]/2)]=1
        t2,mr_file_temp = isap.mr_trans(dirac,opt=opt)
        a = (dirac**2).sum()
        b = (t2**2).sum()
        corr_coeff = a/b
        os.remove(mr_file_temp)
    x1 = (x+isap.mr_recons_coeff(y,mr_file)*corr_coeff)/(1+corr_coeff)
    x1 = x1 - utils.proj_cube(x1,S,ortho=None)
    y1,mr_file_temp = isap.mr_trans(x1,opt=opt)
    os.remove(mr_file_temp)
    return x1,y1

def lin_cons_proj(A,b,y): # Project y on the constraint Ax = b
    yproj = y - (transpose(A).dot(LA.inv(A.dot(transpose(A))))).dot(A.dot(y)-b)
    return yproj

def pos_lin_cons(A,b,y,xinit,nb_iter=10,tol=1e-15,mu=1):
    w1 = 0.5
    w2 = 1-w1

    # Initialization
    x = None
    if xinit is not None:
        x = copy(xinit)
    else:
        x = y*0

    z1 = copy(x)
    z2 = copy(x)
    mse = zeros((nb_iter,))
    lambd = 1.5
    var = 100
    i=0
    while i<nb_iter and var>tol:

        res = x-y
        grad = copy(res)
        mse[i] = (res**2).sum()
        #print mse[i]
        temp1 = 2*x - z1 - mu*grad
        z1 = z1 + lambd*(lin_cons_proj(A,b,temp1)-x)
        temp2 = 2*x - z2 - mu*grad
        z2 = z2 + lambd*(pos_proj(temp2)-x)
        xold = copy(x)
        x = w1*z1+w2*z2
        var  = 100*((x-xold)**2).sum()/(xold**2).sum()
        i+=1


    x = lin_cons_proj(A,b,x)
    x = pos_proj(x)
    return x


def proj_dict_equal(x,y,S,corr_coeff=None): # Projects (x,y) onto the constraint y = Mx, knowing that M^TM = Id; this routine assumes that S is given as a stack of image vectors, y is an image and x is a one dimensional array
    shap = x.shape
    if corr_coeff is None:
        d = zeros((shap[0]))
        for i in range(0,shap[0]):
            d[i] = (S[:,:,i]**2).sum()
        corr_coeff = mean(d)
    xout = copy(x)
    yout = y*0
    for i in range(0,shap[0]):
        xout[i] = (x[i]+(y*S[:,:,i]).sum())/(1+corr_coeff)
        yout += yout + S[:,:,i]*xout[i]
    return xout,yout,corr_coeff

def proj_dict_equal_pos_dyks(x,y,S,corr_coeff=None,nb_iter=500,tol=10**(-10)):
    lambd = 1
    cost=100
    i=0
    xout = copy(x)
    yout = copy(y)
    p1=x*0
    p2=y*0
    q1=x*0
    q2=y*0
    shap = S.shape
    while i<nb_iter and cost>tol:
        i+=1
        xtemp,ytemp,corr_coeff = proj_dict_equal(xout+p1,yout+p2,S,corr_coeff=corr_coeff)
        p1 = xout+p1-xtemp
        p2 = yout+p2-ytemp
        xout = xtemp+q1
        yout = pos_proj_mat(ytemp+q2)
        q1 = xtemp+q1-xout
        q2 = ytemp+q2-yout
        yapprox = yout*0
        for k in range(0,shap[2]):
            yapprox += yapprox+S[:,:,k]*xout[k]
        cost = 100*((yapprox-yout)**2).sum()/(yout**2).sum()
    print "Dyks cv: cost = ",cost," tol = ",tol," min val: ",yapprox.min()
    """for k in range(0,shap[2]):
        xout[k] = (S[:,:,k]*yout).sum()/corr_coeff"""
    return xout,yout,corr_coeff

def proj_dict_equal_pos_dyks_stack(x,y,S,corr_coeff=None,nb_iter=500,tol=10**(-5)): # y is a stack of images and x is matrix, each column corresponding to an image in y
    xout = copy(x)
    yout = copy(y)
    shap = y.shape
    for i in range(0,shap[2]):
        xouti,youti,corr_coeff = proj_dict_equal_pos_dyks(x[:,i],y[:,:,i],S,corr_coeff=corr_coeff,nb_iter=nb_iter,tol=tol)
        xout[:,i] = xouti
        yout[:,:,i] = youti
    return xout,yout,corr_coeff

def laguerre_exp_proj_r(input):
    x = input[0]
    output = x*0
    dict = real(input[1])
    shap = dict.shape
    for i in range(0,shap[3]):
        for j in range(0,shap[2]):
            ai = (x*dict[:,:,j,i]).sum()
            output = output+ai*dict[:,:,j,i]
    return output

def laguerre_exp_proj_r_i(input): # Analysis in the imaginary dict, projection on the real dict
    x = input[0]
    output = x*0
    dict_r = real(input[1])
    dict_i = imag(input[1])
    shap = dict_r.shape
    for i in range(0,shap[3]):
        for j in range(0,shap[2]):
            ai = (x*dict_i[:,:,j,i]).sum()
            output = output+ai*dict_r[:,:,j,i]
    return output

def laguerre_exp_proj_i(input):
    x = input[0]
    output = x*0
    dict = imag(input[1])
    shap = dict.shape
    for i in range(0,shap[3]):
        for j in range(0,shap[2]):
            ai = (x*dict[:,:,j,i]).sum()
            output = output+ai*dict[:,:,j,i]
    return output

def laguerre_exp_proj_i_r(input): # Analysis in the real dict, projection on the imaginary dict
    x = input[0]
    output = x*0
    dict_r = real(input[1])
    dict_i = imag(input[1])
    shap = dict_r.shape
    for i in range(0,shap[3]):
        for j in range(0,shap[2]):
            ai = (x*dict_r[:,:,j,i]).sum()
            output = output+ai*dict_i[:,:,j,i]
    return output

def laguerre_exp_proj_r_r_i(input):
    return laguerre_exp_proj_r(input)+laguerre_exp_proj_r_i(input)

def laguerre_exp_proj_i_i_r(input):
    return laguerre_exp_proj_i(input)+laguerre_exp_proj_i_r(input)

def laguerre_exp_proj_r_r_i_i(input):
    return laguerre_exp_proj_r(input)+laguerre_exp_proj_i(input)

def laguerre_exp_r(dict,alpha):
    dict_r = real(dict)
    shap = dict_r.shape
    output = zeros((shap[0],shap[1]))
    for i in range(0,shap[3]):
        for j in range(0,shap[2]):
            output = output+alpha[j,i]*dict_r[:,:,j,i]
    return output

def laguerre_analys_r(dict,x):
    dict_r = real(dict)
    shap = dict_r.shape
    alpha = zeros((shap[2],shap[3]))
    for i in range(0,shap[3]):
        for j in range(0,shap[2]):
            alpha[j,i] = (x*dict_r[:,:,j,i]).sum()
    return alpha

def laguerre_exp_i(dict,beta):
    dict_i = imag(dict)
    shap = dict_i.shape
    output = zeros((shap[0],shap[1]))
    for i in range(0,shap[3]):
        for j in range(0,shap[2]):
            output = output+beta[j,i]*dict_i[:,:,j,i]
    return output

def laguerre_analys_i(dict,x):
    dict_i = imag(dict)
    shap = dict_i.shape
    beta = zeros((shap[2],shap[3]))
    for i in range(0,shap[3]):
        for j in range(0,shap[2]):
            beta[j,i] = (x*dict_i[:,:,j,i]).sum()
    return beta


def laguerre_exp_mat_r(dict):
    dict_r = real(dict)
    shap = dict.shape
    mat = zeros((shap[2]*shap[3],shap[2]*shap[3]))
    for i in range(1,shap[2]*shap[3]):
        for j in range(i,shap[2]*shap[3]):
            i1 = i%shap[2]
            j1 = floor(i/shap[2])
            i2 = j%shap[2]
            j2 = floor(j/shap[2])
            mat[i,j] = (dict_r[:,:,i1,j1]*dict_r[:,:,i2,j2]).sum()
    mat = mat+transpose(mat)
    for i in range(0,shap[2]*shap[3]):
        i1 = i%shap[2]
        j1 = floor(i/shap[2])
        i2 = j%shap[2]
        j2 = floor(j/shap[2])
        mat[i,i] = (dict_r[:,:,i1,j1]**2).sum()

    return mat

def laguerre_exp_mat_i(dict):
    dict_i = imag(dict)
    shap = dict.shape
    mat = zeros((shap[2]*shap[3],shap[2]*shap[3]))
    for i in range(1,shap[2]*shap[3]):
        for j in range(i,shap[2]*shap[3]):
            i1 = i%shap[2]
            j1 = floor(i/shap[2])
            i2 = j%shap[2]
            j2 = floor(j/shap[2])
            mat[i,j] = (dict_i[:,:,i1,j1]*dict_i[:,:,i2,j2]).sum()
    mat = mat+transpose(mat)
    for i in range(0,shap[2]*shap[3]):
        i1 = i%shap[2]
        j1 = floor(i/shap[2])
        i2 = j%shap[2]
        j2 = floor(j/shap[2])
        mat[i,i] = (dict_i[:,:,i1,j1]**2).sum()

    return mat

def laguerre_exp_mat_i_r(dict):
    dict_i = imag(dict)
    dict_r = real(dict)
    shap = dict.shape
    mat = zeros((shap[2]*shap[3],shap[2]*shap[3]))
    for i in range(0,shap[2]*shap[3]):
        for j in range(0,shap[2]*shap[3]):
            i1 = i%shap[2]
            j1 = floor(i/shap[2])
            i2 = j%shap[2]
            j2 = floor(j/shap[2])
            mat[i,j] = (dict_i[:,:,i1,j1]*dict_r[:,:,i2,j2]).sum()
    return mat

def sparse_laguerre_exp(x,dict,spec_rad_r=None,spec_rad_i=None,dec_coeff=1,nb_iter=100,mu=1):
    siz = dict.shape
    input_dict = list()
    input_dict.append(copy(x))
    input_dict.append(copy(dict))
    eig_dict_r = None
    eig_dict_i = None
    if spec_rad_r is None:
        eig_dict_r,spec_rad_r= pow_meth(laguerre_exp_proj_r,input_dict,[siz[0],siz[1]],ainit=None)
    if spec_rad_i is None:
        eig_dict_i,spec_rad_i= pow_meth(laguerre_exp_proj_i,input_dict,[siz[0],siz[1]],ainit=None)

    thresh_r_max = None
    thresh_i_max = None
    kappa = 0.8

    alpha_r = zeros((siz[2],siz[3]))
    alpha_i = zeros((siz[2],siz[3]))

    grad_r = zeros((siz[2],siz[3]))
    grad_i = zeros((siz[2],siz[3]))
    dict_i = imag(dict)
    dict_r = real(dict)

    res_r = 0
    res_i = 0
    thresh_type = 0
    t1 = None
    t2 = None

    for i in range(0,nb_iter):
        t1 = x - laguerre_exp_r(dict,alpha_r) + laguerre_exp_i(dict,alpha_i)
        t2 = laguerre_exp_i(dict,alpha_r) + laguerre_exp_r(dict,alpha_i)
        res_r = (t1**2).sum()/(x**2).sum()
        res_i = (t2**2).sum()/(x**2).sum()
        print "Real part residual: ",res_r
        grad_r = -laguerre_analys_r(dict,t1)+0*laguerre_analys_i(dict,t2)
        grad_i = laguerre_analys_i(dict,t1)+0*laguerre_analys_r(dict,t2)

        if i==0:
            thresh_r_max = kappa*(abs(grad_r).max())
            thresh_i_max = kappa*(abs(grad_i).max())
        alpha_r = alpha_r - mu*grad_r/spec_rad_r
        alpha_i = alpha_i - mu*grad_i/spec_rad_i

        thresh_r_i = ones((siz[2],siz[3]))*thresh_r_max*exp(-dec_coeff*i)/spec_rad_r
        thresh_i_i = ones((siz[2],siz[3]))*thresh_i_max*exp(-dec_coeff*i)/spec_rad_i

        alpha_r = utils.thresholding(alpha_r,thresh_r_i,thresh_type)
        alpha_i = utils.thresholding(alpha_i,thresh_i_i,thresh_type)

    r = sqrt(alpha_r**2+alpha_i**2)
    theta = zeros((siz[2],siz[3]))
    i,j = where(alpha_r!=0)
    theta[i,j] = np.arctan(alpha_i[i,j]/alpha_r[i,j])
    i,j = where(alpha_r!=0)
    theta[i,j] = sign(alpha_i[i,j])*pi/2

    return alpha_r,alpha_i,t1,t2,r,theta,spec_rad_r,spec_rad_i


def optim_poly_fit_2d(ref_val,target_val,deg_min=1,deg_max=30,kmad_min=1,kmad_max = 400,step_deg=3,step_kmad=20):
    from numpy import linspace,copy
    deg = linspace(deg_min,deg_max,step_deg)
    kmad = linspace(kmad_min,kmad_max,step_kmad)

    S = None
    w = None
    mean_err = 100
    deg_min = None
    kmad_min = None
    for deg_i in deg:
        for kmad_i in kmad:
            S_i,mean_err_i,w_i = poly_fit_2d(ref_val,target_val,deg=int(deg_i),kmad=kmad_i)
            if mean_err_i<mean_err:
                mean_err = mean_err_i
                S = copy(S_i)
                w = copy(w_i)
                deg_min = deg_i
                kmad_min = kmad_i
    print "optimal degree :", deg_min, " optimal kmad: ",kmad_min," modes dimension: ",S.shape
    return S,mean_err,w,int(deg_min)


def poly_fit_2d(ref_val,target_val,deg=5,kmad=1): # The lines represent the samples; this function finds the best fitting S in Y = SA, Y being a matrix of target vamues and A being calculed from the ref values
    from utils import poly_val,mad
    from numpy import transpose,ones,exp,diag
    from numpy.linalg import pinv,norm

    nb_monomials = None
    shap = ref_val.shape
    nb_pts = shap[0]
    dim_feat = target_val.shape[1]
    med_feat = median(target_val,axis=0)
    target_val_0 = target_val - ones((nb_pts,1)).dot(med_feat.reshape((1,dim_feat)))
    A = list()
    for i in range(0,nb_pts):
        coeff,nb_monomials = poly_val(ref_val[i,0],ref_val[i,1],deg)
        A.append(coeff)

    A = transpose(array(A))

    # Weights setting

    mads = zeros((dim_feat,))
    for i in range(0,dim_feat):
        mads[i] = mad(target_val[:,1])

    w = exp(-sum((target_val_0.dot(diag((kmad*mads)**(-1))))**2,axis=1))

    if nb_pts<nb_monomials:
        print("Warning: the fitting might be inaccurate because of the number of samples.")


    S = transpose(diag(w).dot(target_val_0)).dot(transpose(pinv(transpose(A.dot(diag(w))))))
    S[:,0] += median(target_val,axis=0)


    mean_err = 100*norm((transpose(target_val) - S.dot(A))/norm(target_val,axis=1))/nb_pts

    return S,mean_err,w
