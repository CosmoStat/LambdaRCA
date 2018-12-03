
import numpy as np 
import psf_toolkit as tk
import gaussfitter
from numpy import zeros,size,where,ones,copy,around,double,sinc,random,pi,arange,cos,sin,arccos,transpose,diag,sqrt,arange,floor,exp,array,mean,roots,float64,int,pi,median,rot90,argsort,tile,repeat,squeeze,log
import utils





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




def pow_law_select(dist_weights,nb_neigh,min_val=10**(-15)):
    """ **[???] but related to proximity constrains hyperparameters**
    """

    a = dist_weights[:,0]/dist_weights[:,nb_neigh-1]
    r_med = a.min()
    print "r_med: ",r_med,nb_neigh
    p = log(min_val)/log(r_med)
    return p






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