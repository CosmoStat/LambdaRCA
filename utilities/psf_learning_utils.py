import gc
import utils
import optim_utils
from numpy import *
import numpy as np
import scipy.signal as scisig
import sys
sys.path.append('../../Github/python_lib/python/psf')
sys.path.append('../utilities')

def full_displacement(shap,supp,t,pol_en=False,cent=None,theta_param=1,pol_mod=False,coord_map=None,knn=None,eps = 1.e-16):
    """Computes all quantities required to compute displacement interpolation at steps ``t``.
    
    Calls:
    
    * :func:`utils.polar_coord_cloud`
    """
    from numpy import ones,zeros,copy,array,pi,int,transpose,diag
    from utils import polar_coord_cloud
    from pyflann import FLANN

    if coord_map is None:
        coord_map = zeros((shap[0],shap[1],2))
        coord_map[:,:,0] = arange(0,shap[0]).reshape((shap[0],1)).dot(ones((1,shap[1])))
        coord_map[:,:,1] = ones((shap[0],1)).dot(arange(0,shap[1]).reshape((1,shap[1])))
        if pol_en:
            if cent is None:
                cent = array([shap[0]/2,shap[1]/2])
            cloud_in = zeros((2,shap[0]*shap[1]))
            cloud_in[0,:] = copy(coord_map[:,:,0].reshape((shap[0]*shap[1],)))
            cloud_in[1,:] = copy(coord_map[:,:,1].reshape((shap[0]*shap[1],)))
            cloud_out = polar_coord_cloud(cloud_in,cent)
            coord_map[:,:,0] = cloud_out[0,:].reshape((shap[0],shap[1]))
            coord_map[:,:,1] = theta_param*cloud_out[1,:].reshape((shap[0],shap[1]))/(2*pi)
            if pol_mod:
                coord_map[:,:,1] *= coord_map[:,:,0]
        knn = FLANN()
        cloud_in = zeros((shap[0]*shap[1],2))
        cloud_in[:,0] = copy(coord_map[:,:,0].reshape((shap[0]*shap[1],)))
        cloud_in[:,1] = copy(coord_map[:,:,1].reshape((shap[0]*shap[1],)))
        params = knn.build_index(array(cloud_in, dtype=float64))

    advection_points = zeros((supp.shape[0],2,size(t)))

    for i in range(0,supp.shape[0]):
        # Matching coordinates
        pos1_i = int(supp[i,0]/(shap[0]))
        pos1_j = int(supp[i,0]%(shap[0]))
        pos2_i = int(supp[i,1]/(shap[0]))
        pos2_j = int(supp[i,1]%(shap[0]))

        if size(t)==1:
            advection_points[i,0,0] = (1-t)*coord_map[pos1_i,pos1_j,0]+t*coord_map[pos2_i,pos2_j,0]
            advection_points[i,1,0] = (1-t)*coord_map[pos1_i,pos1_j,1]+t*coord_map[pos2_i,pos2_j,1]
        else:
            for j in range(0,size(t)):
                advection_points[i,0,j] = (1-t[j])*coord_map[pos1_i,pos1_j,0]+t[j]*coord_map[pos2_i,pos2_j,0]
                advection_points[i,1,j] = (1-t[j])*coord_map[pos1_i,pos1_j,1]+t[j]*coord_map[pos2_i,pos2_j,1]

    neighbors_graph = zeros((supp.shape[0],4,size(t)))
    neighbors_graph = zeros((supp.shape[0],2,4,size(t)))
    weights_neighbors = zeros((supp.shape[0],4,size(t)))

    if size(t)==1:
        neighbors_graph_temp,dist_neighbors = knn.nn_index(advection_points[:,:,0],4)
        neighbors_graph[:,0,:,0] = neighbors_graph_temp/shap[0]
        neighbors_graph[:,1,:,0] = neighbors_graph_temp%shap[0]
        inv_dist = (dist_neighbors+eps)**(-1)
        weights_neighbors[:,:,0] = inv_dist/(inv_dist.sum(axis=1).reshape((supp.shape[0],1)).dot(ones((1,4))))
    else:
        for j in range(0,size(t)):
            print "Wavelength ",j+1,"/",size(t)
            neighbors_graph_temp,dist_neighbors = knn.nn_index(advection_points[:,:,j],4)
            neighbors_graph[:,0,:,j] = neighbors_graph_temp/shap[0]
            neighbors_graph[:,1,:,j] = neighbors_graph_temp%shap[0]
            inv_dist = (dist_neighbors+eps)**(-1)
            weights_neighbors[:,:,j] = inv_dist/(inv_dist.sum(axis=1).reshape((supp.shape[0],1)).dot(ones((1,4))))
    gc.collect()

    return neighbors_graph.astype(int),weights_neighbors,cent,coord_map,knn

def transport_plan_projections(P,shap,supp,neighbors_graph,weights_neighbors,spectrum=None,indices=None):
    """ Computes monochromatic components (displacement interpolation steps) from transport plan.
    
    """
    from numpy import zeros,int,squeeze,add,ones

    if indices is None:
        nb_proj = neighbors_graph.shape[3]
        indices = arange(0,nb_proj)
    else:
        nb_proj = size(indices)
    if spectrum is None:
        spectrum = ones((nb_proj,))
    im_proj = zeros((shap[0],shap[1],nb_proj))
    siz_supp = neighbors_graph.shape[0]
    for i in range(0,nb_proj):
        add.at(im_proj[:,:,indices[i]],(neighbors_graph[:,0,:,indices[i]],neighbors_graph[:,1,:,indices[i]]),\
        spectrum[indices[i]]*weights_neighbors[:,:,indices[i]]*P[supp[:,0],supp[:,1]].reshape((siz_supp,1)).dot(ones((1,4))))

    return squeeze(im_proj)

def transport_plan_projections_transpose(im,supp,neighbors_graph,weights_neighbors,spectrum=None,indices=None):
    """ Adjoint operator to :func:`transport_plan_projections` 
    
    """
    from numpy import zeros,squeeze,sqrt,add,repeat

    shap = im.shape
    if indices is None:
        nb_proj = neighbors_graph.shape[3]
        indices = arange(0,nb_proj)
    else:
        nb_proj = size(indices)
    if spectrum is None:
        spectrum = ones((nb_proj,))
    P_out = zeros((shap[0]*shap[1],shap[0]*shap[1],nb_proj))
    siz_supp = neighbors_graph.shape[0]
    indx = repeat(supp[:,0],4).reshape((siz_supp,4))
    indy = repeat(supp[:,1],4).reshape((siz_supp,4))

    for i in range(0,nb_proj):
        add.at(P_out[:,:,indices[i]],(indx,indy),spectrum[indices[i]]*weights_neighbors[:,:,indices[i]]\
                                *im[neighbors_graph[:,0,:,indices[i]],neighbors_graph[:,1,:,indices[i]]])
    return squeeze(P_out)

def transport_plan_projections_field_marg(P_stack,shap,supp,neighbors_graph,weights_neighbors):
    """ Computes marginals of a set of transport plans.
    
    Calls:
    
    * :func:`psf_learning_utils.transport_plan_projections`
    
    """
    nb_plans = P_stack.shape[-1]
    output = zeros((shap[0],shap[1],nb_plans))
    for i in range(0,nb_plans):
        output[:,:,i] = transport_plan_projections(P_stack[:,:,i],shap,supp,neighbors_graph,weights_neighbors,indices=[0])

    return output

def transport_plan_projections_field_marg_transpose(im_stack,shap,supp,neighbors_graph,weights_neighbors):
    """ Adjoint operator to :func:`transport_plan_projections_field_marg`.
    
    Calls:
    
    * :func:`psf_learning_utils.transport_plan_projections_transpose`
    
    """
    nb_plans = im_stack.shape[-1]
    output = zeros((shap[0]*shap[1],shap[0]*shap[1],nb_plans))
    for i in range(0,nb_plans):
        output[:,:,i] = transport_plan_projections_transpose(im_stack[:,:,i],supp,neighbors_graph,weights_neighbors,indices=[0])
    return output

def transport_plan_projections_field(P_stack,shap,supp,neighbors_graph,weights_neighbors,spectrums,A,flux,sig,ker,D):
    """ Reconstruct all stars in the field from current eigen transport plans.
    
    Calls:
    
    * :func:`psf_learning_utils.transport_plan_projections`
    * :func:`utils.decim`
    """
    from numpy import zeros,ones,prod,median
    nb_comp = P_stack.shape[-1]
    nb_bands = spectrums.shape[0]
    nb_im = size(flux)
    multi_spec_comp_mat = zeros((shap[0]*shap[1],nb_comp,nb_bands))
    mono_chromatic_psf = zeros((shap[0]*shap[1],nb_im,nb_bands))
    ones_vect = ones((nb_bands,))

    for i in range(0,nb_comp):
        multi_spec_comp = transport_plan_projections(P_stack[:,:,i],shap,supp,neighbors_graph,weights_neighbors)
        multi_spec_comp_mat[:,i,:] = multi_spec_comp.reshape((prod(shap),nb_bands))

    for i in range(0,nb_bands):
        mono_chromatic_psf[:,:,i] = multi_spec_comp_mat[:,:,i].dot(A)

    stars_est = zeros((shap[0]/D,shap[1]/D,nb_im))
    for i in range(0,nb_im):
        stars_temp = (mono_chromatic_psf[:,i,:].dot(spectrums[:,i].reshape((nb_bands,1)))).reshape(shap)
        stars_est[:,:,i] = (flux[i]/sig[i])*utils.decim(scisig.fftconvolve(stars_temp,ker[:,:,i],mode='same'),D,av_en=0)
    gc.collect()

    return stars_est

def transport_plan_projections_flat_field(P_stack,supp,A):
    """ This method returns linear combinations of the slices of the input cube
        on the support, following the mixing matrix A
    """
    return P_stack[supp[:,0],supp[:,1],:].dot(A)

def transport_plan_projections_flat_field_transpose(P_mat,supp,A,shap):
    """ Adjoint operator to :func:`transport_plan_projections_flat_field` (with
    regards to transport plans).
    """
    temp_mat = P_mat.dot(transpose(A))
    P_stack = zeros((prod(shap),prod(shap),A.shape[0]))
    P_stack[supp[:,0],supp[:,1],:] = temp_mat
    return P_stack

def transport_plan_projections_flat_field_transpose_coeff(P_mat,P_stack,supp):
    """ Adjoint operator to :func:`transport_plan_projections_flat_field` (with
    regards to weights).
    """
    return transpose(P_stack[supp[:,0],supp[:,1],:]).dot(P_mat)

def field_reconstruction(P_stack,shap,supp,neighbors_graph,weights_neighbors,A):
    """ Computes monochromatic PSFs from eigenTransport plans.
    
    Calls:
    
    * :func:`psf_learning_utils.transport_plan_projections`
    """
    from numpy import zeros,ones,prod,median
    nb_comp = P_stack.shape[-1]
    nb_bands = neighbors_graph.shape[-1]
    nb_im = A.shape[1]
    multi_spec_comp_mat = zeros((shap[0]*shap[1],nb_comp,nb_bands))
    mono_chromatic_psf = zeros((shap[0],shap[1],nb_im,nb_bands))
    ones_vect = ones((nb_bands,))

    for i in range(0,nb_comp):
        multi_spec_comp = transport_plan_projections(P_stack[:,:,i],shap,supp,neighbors_graph,weights_neighbors)
        multi_spec_comp_mat[:,i,:] = multi_spec_comp.reshape((prod(shap),nb_bands))

    for i in range(0,nb_bands):
        mono_chromatic_psf_temp = multi_spec_comp_mat[:,:,i].dot(A)
        for j in range(0,nb_im):
            mono_chromatic_psf[:,:,j,i] = mono_chromatic_psf_temp[:,j].reshape((shap[0],shap[1]))

    return mono_chromatic_psf

def transport_plan_projections_field_transpose(im_stack,supp,neighbors_graph,weights_neighbors,spectrums,A,flux,sig,ker_rot,D):
    """ Adjoint operator to :func:`psf_learning_utils.transport_plan_projections_field`
    (with regards to transport plans).
    
    Calls:
    
    * :func:`utils.transpose_decim`
    * :func:`psf_learning_utils.transport_plan_projections_transpose_2`
    """

    from numpy import zeros,ones,prod,median,diag
    nb_comp = A.shape[0]
    nb_bands = spectrums.shape[0]
    nb_im = size(flux)
    shap_lr = im_stack.shape
    shap = (shap_lr[0]*D,shap_lr[1]*D)
    psf_est = zeros((shap[0]*shap[1],nb_im))
    for i in range(0,nb_im):
        psf_temp = (flux[i]/sig[i])*scisig.convolve(utils.transpose_decim(im_stack[:,:,i],D),ker_rot[:,:,i],mode='same')
        psf_est[:,i] = psf_temp.reshape((prod(shap),))
    comp_est = zeros((shap[0]*shap[1],nb_comp,nb_bands))
    for i in range(0,nb_bands):
        comp_est[:,:,i] = psf_est.dot(diag(spectrums[i,:]).dot(transpose(A)))
    ones_vect = ones((nb_bands,))
    P_stack = zeros((prod(shap),prod(shap),nb_comp))
    for i in range(0,nb_comp):
        P_stack[:,:,i] = transport_plan_projections_transpose_2(comp_est[:,i,:].reshape((shap[0],shap[1],nb_bands)),supp,neighbors_graph,weights_neighbors,ones_vect).sum(axis=2)
    gc.collect()
    return P_stack

def transport_plan_projections_field_coeff_transpose(im_stack,supp,neighbors_graph,weights_neighbors,spectrums,P_stack,flux,sig,ker_rot,D):
    """ Adjoint operator to :func:`psf_learning_utils.transport_plan_projections_field`
    (with regards to weights).
    
    Calls:
    
    * :func:`utils.transpose_decim`
    * :func:`psf_learning_utils.transport_plan_projections`
    
    """
    from numpy import zeros,ones,prod,median,diag

    nb_comp = P_stack.shape[2]
    nb_bands = spectrums.shape[0]
    nb_im = size(flux)
    shap_lr = im_stack.shape
    shap = (shap_lr[0]*D,shap_lr[1]*D)
    multi_spec_comp_mat = zeros((shap[0]*shap[1],nb_comp,nb_bands))
    psf_est = zeros((shap[0]*shap[1],nb_im))
    for i in range(0,nb_im):
        psf_temp = (flux[i]/sig[i])*scisig.convolve(utils.transpose_decim(im_stack[:,:,i],D),ker_rot[:,:,i],mode='same')
        psf_est[:,i] = psf_temp.reshape((prod(shap),))

    ones_vect = ones((nb_bands,))
    for i in range(0,nb_comp):
        multi_spec_comp = transport_plan_projections(P_stack[:,:,i],shap,supp,neighbors_graph,weights_neighbors)
        multi_spec_comp_mat[:,i,:] = multi_spec_comp.reshape((prod(shap),nb_bands))

    A = zeros((nb_comp,nb_im))
    for i in range(0,nb_im):
        Si = zeros((shap[0]*shap[1],nb_comp))
        for k in range(0,nb_comp):
            Si[:,k] = (multi_spec_comp_mat[:,k,:].dot(spectrums[:,i].reshape((nb_bands,1)))).reshape((prod(shap),))
        A[:,i] = (transpose(Si).dot(psf_est[:,i].reshape((prod(shap),1)))).reshape((nb_comp,))
    gc.collect()

    return A


def transport_plan_projections_transpose_2(im_stack,supp,neighbors_graph,weights_neighbors,spectrum):
    """ Computes the adjoint operator of the displacement inteerpolation for a
    given transport plan."""
    from numpy import zeros,squeeze,sqrt,add,repeat
    shap = im_stack.shape
    nb_proj = neighbors_graph.shape[3]
    P_out = zeros((shap[0]*shap[1],shap[0]*shap[1],nb_proj))
    siz_supp = neighbors_graph.shape[0]
    indx = repeat(supp[:,0],4).reshape((siz_supp,4))
    indy = repeat(supp[:,1],4).reshape((siz_supp,4))

    for i in range(0,nb_proj):
        add.at(P_out[:,:,i],(indx,indy),spectrum[i]*weights_neighbors[:,:,i]*im_stack[neighbors_graph[:,0,:,i],neighbors_graph[:,1,:,i],i])
    return P_out

def columns_wise_simplex_proj(mat,mass=None):
    """ Projects each column of input matrix onto simplex.
    
    Calls:
    
    * simplex_projection.euclidean_proj_simplex
    
    """

    from simplex_projection import euclidean_proj_simplex
    nb_columns = mat.shape[1]
    mat_out = zeros(mat.shape)
    if mass is None:
        mass = max(0,((mat*(mat>=0)).sum(axis=0)).mean())
    if mass>0:
        for i in range(0,nb_columns):
            mat_out[:,i] = euclidean_proj_simplex(mat[:,i],s=mass)

    return mat_out

