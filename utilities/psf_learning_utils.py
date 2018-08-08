import gc
import utils
import optim_utils
from numpy import *
import numpy as np
from scipy.interpolate import Rbf
from scipy.fftpack import dct,idct
import scipy.signal as scisig
from utils import bar_coord2d
import sys
sys.path.append('../../Github/python_lib/python/psf')
sys.path.append('../utilities')
import isap
from numpy.random import randn


def stack_wavelets_transform(stack,opt=['-t2','-n2'],kmad=5):
    from scipy.signal import fftconvolve
    from wavelet import get_mr_filters
    from utils import mad,thresholding_3D
    from numpy import ones
    shap = stack.shape
    filters = get_mr_filters(shap[0:2], levels=None, opt=opt, course=False)
    output = stack*0
    for i in range(0,shap[2]):
        output[:,:,i] = fftconvolve(stack[:,:,i],filters[0,:,:],mode='same')

    mad_data = mad(output)
    output = thresholding_3D(output,ones(shap)*kmad*mad_data,0)

    return output

def data_shaping(psf_arr,coord):
    n1 = psf_arr.shape[0]
    n2 = psf_arr.shape[1]
    n3 = psf_arr.shape[2]
    mat_learning = zeros((n3,n1*n2+2))
    i=0
    #coordx = coord[:,0]
    #coordy = coord[:,1]
    #coordx = (coordx-coordx.min())/(coordx.max()-coordx.min())
    #coordy = (coordy-coordy.min())/(coordy.max()-coordy.min())
    for i in range(0,n3):
        psf_i = psf_arr[:,:,i]
        mat_learning[i,0:n1*n2] = psf_i.reshape(1,n1*n2)#/abs(psf_i.max())
        mat_learning[i,n1*n2] = coord[i,0]
        mat_learning[i,n1*n2+1] = coord[i,1]
    return mat_learning

def data_shaping_2(psf_arr,coord,scaling_fact=1):
    n1 = psf_arr.shape[0]
    n2 = psf_arr.shape[1]
    n3 = psf_arr.shape[2]
    mat_learning = zeros((n3,n1*n2+2))

    i=0
    norm_mean = 0
    for i in range(0,n3):
        psf_i = psf_arr[:,:,i]
        norm_mean+=sqrt((psf_i**2).sum())
        mat_learning[i,0:n1*n2] = psf_i.reshape(1,n1*n2)
        mat_learning[i,n1*n2] = coord[i,0]
        mat_learning[i,n1*n2+1] = coord[i,1]
    norm_mean /=n3
    range_coord = sqrt((coord[:,0].min()-coord[:,0].max())**2+(coord[:,1].min()-coord[:,1].max())**2)
    scal_fact = norm_mean*scaling_fact/range_coord
    mat_learning[:,n1*n2:] = mat_learning[:,n1*n2:]*scal_fact

    return mat_learning,scal_fact


def data_shaping_lack(psf_arr,coord,n):
    psf_arr_lack,ind_rm,ind_kpt = util.cube_sampling(psf_arr,n)
    coord_lack = coord[ind_kpt,:]
    mat_learning = data_shaping(psf_arr_lack,coord_lack)
    return mat_learning,ind_rm,ind_kpt

def interp_field_surf(cells_means,dict,posxy):
    nb_cell = cells_means.shape[0]
    n = cells_means.shape[1]
    distxy = zeros((nb_cell,))
    i=0
    ind=0
    min_val = 1e14
    for i in range(0,nb_cell):
        distxy[i] = sqrt((cells_means[i,n-2]-posxy[0])**2+(cells_means[i,n-1]-posxy[1])**2)
        if distxy[i]<min_val:
            min_val = distxy[i]
            ind = i
    O = cells_means[ind,:]
    n1 = dict[:,0,ind]
    n2 = dict[:,1,ind]
    interp_val = util.affine_surf_eval(O,n1,n2,posxy[0],posxy[1])
    interp_val = interp_val.reshape(1,n)
    output = interp_val[0,0:n-2]
    pos_err = abs(array([interp_val[0,n-2]-posxy[0],interp_val[0,n-1]-posxy[1]]))
    return output,pos_err

def interp_field_surf_arr(cells_means,dict,posxy_arr,n1,n2):
    nb_psf = posxy_arr.shape[0]
    psf_interp = zeros((n1,n2,nb_psf))
    i=0
    for i in range(0,nb_psf):
        posxy = posxy_arr[i,:]
        psf_interp_i,pos_erri = interp_field_surf(cells_means,dict,posxy)
        psf_interp[:,:,i] = psf_interp_i.reshape(n1,n2)
    return psf_interp

def interp_field_surf_err(psf_field,dico,cells_means,ell_theta,posxy,ind_rmv):
    pos_in = posxy[ind_rmv,:]

    n1 = psf_field.shape[0]
    n2 = psf_field.shape[1]
    psf_interp = interp_field_surf_arr(cells_means,dico,pos_in,n1,n2)
    ell_interp,theta_interp = util.mk_ellipticity_arr(psf_interp,1e6,niter_cent=1)
    nb_interp = ind_rmv.shape[0]
    mse = zeros((nb_interp,))
    error_map = zeros((n1,n2,nb_interp))
    error_ell_theta = zeros((nb_interp,3))
    rerror_ell_theta = zeros((nb_interp,3))
    i = 0
    for i in range(0,nb_interp):
        error_map[:,:,i] = psf_interp[:,:,i]-psf_field[:,:,ind_rmv[i]]
        mse[i] = (error_map[:,:,i]**2).sum()
        error_ell_theta[i,0] = abs(ell_theta[ind_rmv[i],0]-ell_interp[i,0])
        error_ell_theta[i,1] = abs(ell_theta[ind_rmv[i],1]-ell_interp[i,1])
        error_ell_theta[i,2] = abs(ell_theta[ind_rmv[i],2]-theta_interp[i])
        rerror_ell_theta[i,0] = 100*error_ell_theta[i,0]/abs(ell_theta[ind_rmv[i],0])
        rerror_ell_theta[i,1] = 100*error_ell_theta[i,1]/abs(ell_theta[ind_rmv[i],1])
        rerror_ell_theta[i,2] = 100*error_ell_theta[i,2]/abs(ell_theta[ind_rmv[i],1])
    return psf_interp,error_map,mse,ell_interp,theta_interp,error_ell_theta,rerror_ell_theta

def interp_field_inv_weight_pca(ref_psf,mean_psf,p_comp,coeff_data,data_coord,coord_interp,nb_neighb,pw=5):
    nb_data = coeff_data.shape[0]
    dist = zeros((nb_data,))
    i = 0
    for i in range(0,nb_data):
        dist[i]= sqrt(((data_coord[i,:]-coord_interp)**2).sum())
    ind = argsort(dist)
    weights = zeros((nb_neighb,))
    output = copy(mean_psf)
    shap = ref_psf.shape
    output = zeros((shap[0],shap[1]))
    wout = zeros((nb_data,))
    for i in range(0,nb_neighb):
        weights[i]=1/(dist[ind[i]])**pw
        #print weights[i]
        output = output+weights[i]*ref_psf[:,:,ind[i]]

        coeff_i = coeff_data[ind[i],:]
    #output = output + weights[i]*coeff_i.dot(nnspose(p_comp))
    tt_weight = weights.sum()
    output=output/tt_weight
    wout[ind[0:nb_neighb]] = weights/tt_weight
    return output,wout

def inv_weigthed_coeff(mat_ref,pos_ref,target_pos,nb_neigh,pw=5): # Each sample coeff is in a column of mat_ref
    nb_data = mat_ref.shape[1]
    dist = zeros((nb_data,))
    i = 0
    for i in range(0,nb_data):
        dist[i]= sqrt(((pos_ref[i,:]-target_pos)**2).sum())

    print dist.min()
    ind = argsort(dist)
    weights = zeros((nb_data,1))
    for i in range(0,nb_neigh):
        weights[ind[i]] = 1/(dist[ind[i]])**pw

    interp_coeff = mat_ref.dot(weights)
    return interp_coeff

def inv_weigthed_coeff_stack(mat_ref,pos_ref,target_pos,nb_neigh,pw=5):
    shap1 = target_pos.shape
    shap2 = mat_ref.shape
    nb_target = shap1[0]
    weights_interp = zeros((shap2[0],nb_target))
    for i in range(0,nb_target):
        weights_i = inv_weigthed_coeff(mat_ref,pos_ref,target_pos[i,:],nb_neigh,pw=5)
        weights_interp[:,i] = weights_i.reshape((shap2[0],))

    return weights_interp

def inv_weigthed_cube_svd(cube_ref,pos_ref,target_pos,nb_neigh,nb_comp,pw=5):
    shap = cube_ref.shape
    data_mat = utils.cube_to_mat(cube_ref)
    U,s,Vt = linalg.svd(data_mat,full_matrices=False)
    basis = Vt[0:nb_comp,:]
    mat_ref = basis.dot(transpose(data_mat))
    weights_interp = inv_weigthed_coeff_stack(mat_ref,pos_ref,target_pos,nb_neigh,pw=5)

    data_interp = transpose(weights_interp).dot(basis)
    cube_interp =  utils.mat_to_cube(data_interp,shap[0],shap[1])

    return cube_interp

def interp_field_inv_weight_cv_hull(coeff_data,data_coord,coord_interp,nb_neighb,pw=1):
    nb_data = coeff_data.shape[1]
    dist = zeros((nb_data,))
    i = 0
    for i in range(0,nb_data):
        dist[i]= sqrt(((data_coord[i,:]-coord_interp)**2).sum())
    ind = argsort(dist)
    weights = zeros((nb_neighb,))
    output = zeros((coeff_data.shape[0],))
    for i in range(0,nb_neighb):
        weights[i]=1/(dist[ind[i]])**pw
        output = output+weights[i]*coeff_data[:,ind[i]]
    tt_weight = weights.sum()
    output=output/tt_weight
    return output

def interp_field_inv_weight_cv_hull_arr(coeff_data,data_coord,coord_interp_arr,nb_neighb):
    nb_psf = coord_interp_arr.shape[0]
    coeff_interp = zeros((coeff_data.shape[0],nb_psf))
    for i in range(0,nb_psf):
        coord_interp = coord_interp_arr[i,:]
        coeff_interp[:,i] = interp_field_inv_weight_cv_hull(coeff_data,data_coord,coord_interp,nb_neighb)
    return coeff_interp


def interp_field_inv_weight_pca_arr(ref_psf,mean_psf,p_comp,coeff_data,data_coord,coord_interp_arr,nb_neighb,n1,n2):
    nb_psf = coord_interp_arr.shape[0]
    psf_interp = zeros((n1,n2,nb_psf))
    i=0
    weights = zeros((data_coord.shape[0],nb_psf))
    for i in range(0,nb_psf):
        coord_interp = coord_interp_arr[i,:]
        psf_interp_i,wout = interp_field_inv_weight_pca(ref_psf,mean_psf,p_comp,coeff_data,data_coord,coord_interp,nb_neighb)
        psf_interp[:,:,i] = psf_interp_i
        weights[:,i] = wout
    return psf_interp,weights


def interp_field_inv_weight_pca_err(ref_psf,psf_field,mean_psf,p_comp,coeff_data,data_coord,ind_rmv,ind_kpt,nb_neighb):
    coord_interp_arr = data_coord[ind_rmv,:]
    coord_interp_learning = data_coord[ind_kpt,:]
    n1 = psf_field.shape[0]
    n2 = psf_field.shape[1]
    psf_interp,weights = interp_field_inv_weight_pca_arr(ref_psf,mean_psf,p_comp,coeff_data,coord_interp_learning,coord_interp_arr,nb_neighb,n1,n2)
    ell_interp,theta_interp = utils.mk_ellipticity_arr(psf_interp,1e6,niter_cent=1)
    ell,theta = utils.mk_ellipticity_arr(psf_field,1e6,niter_cent=1)
    nb_interp = ind_rmv.shape[0]
    mse = zeros((nb_interp,))
    error_map = zeros((n1,n2,nb_interp))
    error_ell_theta = zeros((nb_interp,3))
    rerror_ell_theta = zeros((nb_interp,3))
    i = 0
    for i in range(0,nb_interp):
        error_map[:,:,i] = psf_interp[:,:,i]-psf_field[:,:,ind_rmv[i]]
        mse[i] = (error_map[:,:,i]**2).sum()
        error_ell_theta[i,0] = abs(ell[ind_rmv[i],0]-ell_interp[i,0])
        error_ell_theta[i,1] = abs(ell[ind_rmv[i],1]-ell_interp[i,1])
        error_ell_theta[i,2] = abs(theta[ind_rmv[i]]-theta_interp[i])
        rerror_ell_theta[i,0] = 100*error_ell_theta[i,0]/abs(ell[ind_rmv[i],0])
        rerror_ell_theta[i,1] = 100*error_ell_theta[i,1]/abs(ell[ind_rmv[i],1])
        rerror_ell_theta[i,2] = 100*error_ell_theta[i,2]/abs(theta[ind_rmv[i]])
    return psf_interp,error_map,mse,ell_interp,theta_interp,error_ell_theta,rerror_ell_theta,weights


def interp_field_err(psf_field,psf_interp,ind_rmv,ind_kpt):
    n1 = psf_field.shape[0]
    n2 = psf_field.shape[1]

    ell_interp,theta_interp = utils.mk_ellipticity_arr(psf_interp,1e6,niter_cent=1)
    ell,theta = utils.mk_ellipticity_arr(psf_field,1e6,niter_cent=1)
    nb_interp = ind_rmv.shape[0]
    mse = zeros((nb_interp,))
    error_map = zeros((n1,n2,nb_interp))
    error_ell_theta = zeros((nb_interp,3))
    rerror_ell_theta = zeros((nb_interp,3))
    i = 0
    for i in range(0,nb_interp):
        error_map[:,:,i] = psf_interp[:,:,i]-psf_field[:,:,ind_rmv[i]]
        mse[i] = (error_map[:,:,i]**2).sum()
        error_ell_theta[i,0] = abs(ell[ind_rmv[i],0]-ell_interp[i,0])
        error_ell_theta[i,1] = abs(ell[ind_rmv[i],1]-ell_interp[i,1])
        error_ell_theta[i,2] = abs(theta[ind_rmv[i]]-theta_interp[i])
        rerror_ell_theta[i,0] = 100*error_ell_theta[i,0]/abs(ell[ind_rmv[i],0])
        rerror_ell_theta[i,1] = 100*error_ell_theta[i,1]/abs(ell[ind_rmv[i],1])
        rerror_ell_theta[i,2] = 100*error_ell_theta[i,2]/abs(theta[ind_rmv[i]])

    return error_map,mse,error_ell_theta,rerror_ell_theta

def deg_to_micron_euclid(pos,up_fact): # up_fact must match the original data resolution (the data containing the angular information in their fits header)
    pos_micron = 12*pos*pi/(up_fact*180*4.03e-8)
    return pos_micron

def euclid_pixels_position_map(cent_pos,siz,up_fact): # cent_pos: central pixel center coordinates in microns (we assume odd sizes); outputs pixels positions in microns; up_fact = upsampling factor compared to Euclid resolution
    map = zeros((siz[0],siz[1],2))
    i=0
    j=0
    lx = (siz[1]-1)/2 # Half the number of columns
    ly = (siz[0]-1)/2
    for i in range(0,siz[0]):
        for j in range(0,siz[1]):
            map[i,j,0] = 12*(j-lx)/up_fact+cent_pos[0]
            map[i,j,1] = 12*(i-ly)/up_fact+cent_pos[1]
    return map

def cross_distances_map(map1,map2): # A line of cross_map contains distances from one point in map1 to all the points in map2; the two maps are spanned lines after lines
    shap1 = map1.shape
    shap2 = map2.shape
    siz1 = shap1[0]*shap1[1]
    siz2 = shap2[0]*shap2[1]
    cross_map = zeros((siz1,siz2))
    i=0
    j=0
    for i in range(0,siz1):
        i1 = np.int(i/shap1[1]) # Line index in map1
        j1 = mod(i,shap1[1])
        for j in range(0,siz2):
            i2 = np.int(j/shap2[1]) # Line index in map2
            j2 = mod(j,shap2[1])
            cross_map[i,j]=(map1[i1,j1,0]-map2[i2,j2,0])**2+(map1[i1,j1,1]-map2[i2,j2,1])**2
    return cross_map


def cross_distances_cloud(pos): # A line of cross_map contains distances from one point in map1 to all the points in map2; the two maps are spanned lines after lines
    nb_pts = pos.shape[0]
    cross_map = zeros((nb_pts,nb_pts))

    for i in range(0,nb_pts):
        for j in range(i+1,nb_pts):
            cross_map[i,j]=(pos[i,0]-pos[j,0])**2+(pos[i,1]-pos[j,1])**2
    cross_map+=transpose(cross_map)
    return cross_map


def pix_map(field_cent,res_fact,shap):
    if field_cent is None:
        field_cent_pix = [0,0]
    else:
        field_cent_pix = [0,0]
        field_cent_pix[0] = utils.arcsec2pix_euclid(field_cent[0],res_fact)
        field_cent_pix[1] = utils.arcsec2pix_euclid(field_cent[1],res_fact)

    mapx = ones((shap[0],1)).dot(array(range(-np.int(shap[1]/2),shap[1]-np.int(shap[1]/2))).reshape(1,shap[1])) + field_cent_pix[0]*ones(shap)
    mapy = (array(range(-np.int(shap[0]/2),shap[0]-np.int(shap[0]/2))).reshape(shap[0],1)).dot(ones((1,shap[1]))) + field_cent_pix[1]*ones(shap)

    map = zeros((shap[0],shap[1],2))
    map[:,:,0] = mapx
    map[:,:,1] = mapy

    return map

def dist_map(map1,map2,same=True):
    map_in = copy(map1)
    if same is not True:
        shap = map1.shape
        map_in = zeros((shap[0],shap[1]*2,2))
        map_in[:,0:shap[0],:] = copy(map1)
        map_in[:,shap[0]:,:] = copy(map2)

    dist_map = cross_distances_map(map_in,map_in)
    return dist_map

def psf_interp_opt_transp(psf_cubes,positions_map,transport_map,nb_iter_bar=50,nb_iter_transp=100): # PSF interpolation using Wasserstein barycenter

    # Barycenter coordinates calculation
    shap = psf_cubes.shape
    nb_psf = shap[2]
    A = zeros((2,nb_psf))
    k=0
    for k in range(0,nb_psf):
        A[0,k] = positions_map[(shap[0]-1)/2,(shap[1]-1)/2,0,k]
        A[1,k] = positions_map[(shap[0]-1)/2,(shap[1]-1)/2,1,k]
    B = zeros((2,))
    B[0] = transport_map[(shap[0]-1)/2,(shap[1]-1)/2,0]
    B[1] = transport_map[(shap[0]-1)/2,(shap[1]-1)/2,1]
    T,mse_bar = optim_utils.bar_coord_pb(A,B,nb_iter_bar)
    print "T = ",T
    # Distances map setting
    cross_maps = zeros((shap[0]*shap[1],shap[0]*shap[1],nb_psf))
    for k in range(0,nb_psf):
        mapk = squeeze(positions_map[:,:,:,k])
        cross_maps[:,:,k] = cross_distances_map(transport_map,mapk)

    # Distribution matrix setting
    distrib_mat = zeros((shap[0]*shap[1],nb_psf))
    for k in range(0,nb_psf):
        psf_k = psf_cubes[:,:,k]
        psf_k=psf_k/psf_k.sum()
        distrib_mat[:,k] = psf_k.reshape((shap[0]*shap[1],))

    thresh=0.6
    P,cost = optim_utils.displacement_interp(distrib_mat,cross_maps,T,thresh,nb_iter_transp)

    P1 = P[:,:,0]
    P2 = P[:,:,1]
    ones_vect = ones((shap[0]*shap[1],1))
    psf_interp_vect = P1.dot(ones_vect)
    psf_interp_vect_2 = P2.dot(ones_vect)
    err_vect = psf_interp_vect-psf_interp_vect_2
    err = err_vect.reshape((shap[0],shap[1]))
    psf_interp = psf_interp_vect.reshape((shap[0],shap[1]))

    return psf_interp,cost,P,err


def wasserstein_geodes(psf_1,psf_2,position_map_1,position_map_2,t,nb_iter_coupling=100): # t should be picked between 0 and 1

    # Distance map setting
    shap = psf_1.shape
    nb_psf = 2
    cross_map = zeros((shap[0]*shap[1],shap[0]*shap[1]))
    cross_map = cross_distances_map(position_map_1,position_map_2)

    # Distribution matrix setting
    psf_1_temp = psf_1/psf_1.sum()
    psf_2_temp = psf_2/psf_2.sum()

    # Optimal coupling estiamtion
    P,grad=psf_opt_coupling(psf_1_temp,psf_2_temp,position_map_1,position_map_2,nb_iter_coupling=100)

    # Interpolation
    tol = 1
    min_val = tol*min(psf_1_temp.min(),psf_2_temp.min())/(shap[0]*shap[1])
    interp_pos = t*position_map_1+(1-t)*position_map_2
    psf_interp_vect = zeros((shap[0]*shap[1]))
    psf_interp = 0*psf_1
    i=0
    j=0
    for i in range(0,shap[0]*shap[1]):
        i1 = np.int(i/shap[1]) # Line index in position_map_1
        j1 = mod(i,shap[1])
        for j in range(0,shap[0]*shap[1]):
            if (P[i,j] >= min_val):
                i2 = np.int(j/shap[1]) # Line index in postion_map_2
                j2 = mod(j,shap[1])
                if (i1==i2 and j1==j2):
                    psf_interp[i1,j1] = psf_interp[i1,j1] + P[i,j]
                else :
                    pos = t*position_map_1[i1,j1,:]+(1-t)*position_map_2[i2,j2,:]
                    imin = int(min(i1,i2))
                    imax = int(max(i1,i2))
                    jmin = int(min(j1,j2))
                    jmax = int(max(j1,j2))
                    min_dist = 1e15
                    iopt=0
                    jopt=0
                    k=0
                    l=0
                    for k in range(imin,imax+1):
                        for l in range(jmin,jmax+1):
                            dist = sqrt((pos[0]-interp_pos[k,l,0])**2+(pos[1]-interp_pos[k,l,1])**2)
                            if (dist < min_dist):
                                min_dist=dist
                                iopt = k
                                jopt = l
                    psf_interp[iopt,jopt] = psf_interp[iopt,jopt]+P[i,j]

    return psf_interp,P

def psf_opt_coupling(psf_1,psf_2,position_map_1,position_map_2,nb_iter_coupling=100): # t should be picked between 0 and 1

    # Distance map setting
    shap = psf_1.shape
    nb_psf = 2
    cross_map = zeros((shap[0]*shap[1],shap[0]*shap[1]))
    cross_map = cross_distances_map(position_map_1,position_map_2)

    # Distribution matrix setting
    distrib_1 = psf_1.reshape((shap[0]*shap[1],1))/psf_1.sum()
    distrib_2 = psf_2.reshape((shap[0]*shap[1],1))/psf_2.sum()
    thresh=0.05

    # Optimal coupling estiamtion
    P,cost,grad = optim_utils.opt_coupling(distrib_1,distrib_2,cross_map,thresh,nb_iter_coupling)

    return P,grad


def curved_wasserstein_geodes(psf_1,psf_2,position_map_1,position_map_2,t,theta,nb_iter_coupling=100,nb_iter_rot=100): # t should be picked between 0 and 1
    # Distance map setting
    print 'nb_iter_coupling = ',nb_iter_coupling
    shap = psf_1.shape
    nb_psf = 2
    cross_map = zeros((shap[0]*shap[1],shap[0]*shap[1]))
    cross_map = cross_distances_map(position_map_1,position_map_2)

    # Distribution matrix setting
    psf_1_temp = psf_1/psf_1.sum()
    psf_2_temp = psf_2/psf_2.sum()


    # Optimal coupling estiamtion
    psf_1_temp_rot = util.lanczos_rot(psf_1_temp,theta)
    psf_1_temp_rot = psf_1_temp_rot/psf_1_temp_rot.sum()
    rot_map = util.rotate_positions_field(position_map_1,theta)
    P,grad=psf_opt_coupling(psf_1_temp_rot,psf_2_temp,position_map_1,position_map_2,nb_iter_coupling=nb_iter_coupling)

    # Optimal rotation estimation
    #theta,rot_map = optim_utils.opt_rotation(position_map_1,position_map_2,P,nb_iter=nb_iter_rot)


    # Interpolation
    tol = 1
    min_val = tol*min(psf_1_temp_rot.min(),psf_2_temp.min())/(shap[0]*shap[1])
    #interp_pos = t*rot_map+(1-t)*position_map_2
    interp_pos = t*position_map_1+(1-t)*position_map_2
    psf_interp_vect = zeros((shap[0]*shap[1]))
    psf_interp = 0*psf_1
    i=0
    j=0
    for i in range(0,shap[0]*shap[1]):
        i1 = np.int(i/shap[1]) # Line index in position_map_1
        j1 = mod(i,shap[1])
        for j in range(0,shap[0]*shap[1]):
            if (P[i,j] >= min_val):
                i2 = np.int(j/shap[1]) # Line index in postion_map_2
                j2 = mod(j,shap[1])
                if (i1==i2 and j1==j2):
                    psf_interp[i1,j1] = psf_interp[i1,j1] + P[i,j]
                else :
                    pos = t*position_map_1[i1,j1,:]+(1-t)*position_map_2[i2,j2,:]
                    imin = int(min(i1,i2))
                    imax = int(max(i1,i2))
                    jmin = int(min(j1,j2))
                    jmax = int(max(j1,j2))
                    min_dist = 1e15
                    iopt=0
                    jopt=0
                    k=0
                    l=0
                    for k in range(imin,imax+1):
                        for l in range(jmin,jmax+1):
                            dist = sqrt((pos[0]-interp_pos[k,l,0])**2+(pos[1]-interp_pos[k,l,1])**2)
                            if (dist < min_dist):
                                min_dist=dist
                                iopt = k
                                jopt = l
                    psf_interp[iopt,jopt] = psf_interp[iopt,jopt]+P[i,j]

    theta_t = t*theta
    psf_interp_theta = util.lanczos_rot(psf_interp,theta_t)
    print "Cost = ",(P*cross_map).sum()
    return psf_interp_theta,psf_interp,P,theta,grad,psf_1_temp_rot


def partial_transport(shap,mapping,t):
    nb_pair = mapping.shape[0]
    psf_interp = zeros((shap[0],shap[1]))
    k=0
    coord = zeros((nb_pair,2))
    for k in range(0,nb_pair):
        i = mapping[k,0]
        j = mapping[k,1]
        i1 = np.int(i/shap[1]) # Line index in position_map_1
        j1 = mod(i,shap[1])
        i2 = np.int(j/shap[1]) # Line index in postion_map_2
        j2 = mod(j,shap[1])
        if (i1==i2 and j1==j2):
            psf_interp[i1,j1] = psf_interp[i1,j1] + mapping[k,2]
            coord[k,0]=i1
            coord[k,1]=j1
        else :
            iopt = round(t*i1+(1-t)*i2)
            jopt = round(t*j1+(1-t)*j2)
            psf_interp[iopt,jopt] = psf_interp[iopt,jopt]+mapping[k,2]
            coord[k,0]=iopt
            coord[k,1]=jopt

    return psf_interp,coord


def part_transp_gridding(mapping,shap,t,lanczos_rad=4,nb_iter=50,step_size=1):

    psf_init,coord = partial_transport(shap,mapping,t)
    data_val = mapping[:,2]
    psf_interp,mse,supp_cols,supp_lines,mat = optim_utils.lanczos_gridding(coord,data_val,shap,lanczos_rad=lanczos_rad,nb_iter=nb_iter,step_size=step_size,im_init=psf_init)

    return psf_interp,supp_cols,supp_lines,mat,psf_init,coord


def psf_ground_transport(psf_1,psf_2,exe_path="../../CPP/NOT/build/gnd_tp_solve"):

    psf_1_in = psf_1/psf_1.sum()
    psf_2_in = psf_2/psf_2.sum()
    shap = psf_1_in.shape
    N1 = array(range(0,shap[1]))
    N1 = N1.reshape((1,shap[1]))
    N2 = array(range(0,shap[0]))
    N2 = N2.reshape((shap[0],1))
    ones1 = ones((shap[0],1))
    ones2 = ones((1,shap[1]))
    Xs = ones1.dot(N1)
    Ys = N2.dot(ones2)
    pos_map = zeros((shap[0],shap[1],2))
    pos_map[:,:,0]=Xs
    pos_map[:,:,1]=Ys
    mapping = utils.gnd_tp_solver_wrp(psf_1_in,psf_2_in,pos_map,pos_map,exe_path=exe_path)

    return mapping



def psf_sinkh_transport(psf_1,psf_2,dist_map,inf_val=None,beta=100,nb_iter=10):
    # Distance map setting
    shap = psf_1.shape

    # Distribution matrix setting

    distrib_1 = psf_1.reshape((shap[0]*shap[1],1))
    distrib_2 = psf_2.reshape((shap[0]*shap[1],1))*psf_1.sum()/psf_2.sum()

    P,dist = optim_utils.beg_proj_transp(dist_map,distrib_1,distrib_2,beta=beta,nb_iter=nb_iter)
    return P,dist

def psf_sinkh_transport_clust(psf_1,psf_2,dist_map,cart_prods,w,inf_val=None,beta=100,nb_iter=10):
    # Distance map setting
    shap = psf_1.shape

    # Distribution matrix setting

    distrib_1 = psf_1.reshape((shap[0]*shap[1],1))
    distrib_2 = psf_2.reshape((shap[0]*shap[1],1))*psf_1.sum()/psf_2.sum()

    P,dist = optim_utils.beg_proj_transp_clust(dist_map,distrib_1,distrib_2,cart_prods,w,inf_val=None,beta=beta,nb_iter=nb_iter)
    return P,dist



def psf_sinkh_transport_2(psf_1,psf_2,pos1,pos2,res_fact,same=False,cart_prods=None,w=None,inf_val=None,beta=100,nb_iter=10,centering=False,dist_map=None,pol_en=True,scale_factt=1,scale_factr=1,clust_en=True):
    shap = psf_1.shape
    psf_2 = psf_2*psf_1.sum()/psf_2.sum()
    cent = None
    map_1 = None
    dyn_coeff = None
    if pol_en:
        cent1 = utils.compute_centroid_2(psf_1,ones(shap))
        cent2 = utils.compute_centroid_2(psf_2,ones(shap))
        cent = (cent1+cent2)/2
        map_1 = pol_coord_map_2(shap,scale_factt,scale_factr,cent=cent)
    else:
        map_1 = pix_map(pos1,res_fact,shap)

    map_2 = None
    if same is False and pol_en is False:
        map_2 = pix_map(pos2,res_fact,shap)
    elif same is True:
        map_2 = map_1
    else:
        print "Warning: if pol_en is True, same has to be True!!"

    map_in = None
    psf_in_1 = None
    psf_in_2 = None
    psf_in_10 = copy(psf_1)
    psf_in_20 = copy(psf_2)
    inf_im = None
    if centering:
        stack = zeros((shap[0],shap[1],2))
        stack[:,:,0] = psf_in_10
        stack[:,:,1] = psf_in_20
        stack_cent,inf_im = optim_utils.positive_centering(stack)
        psf_in_10 = stack_cent[:,:,0]
        psf_in_20 = stack_cent[:,:,1]

    if same:
        psf_in_1 = copy(psf_in_10)
        psf_in_2 = copy(psf_in_20)
        map_in = copy(map_1)
    else:
        psf_in_1 = zeros((shap[0]*2,shap[1]))
        psf_in_2 = zeros((shap[0]*2,shap[1]))
        psf_in_1[0:shap[0],:] = psf_in_10
        psf_in_2[shap[0]:,:] = psf_in_20
        map_in = zeros((shap[0]*2,shap[1],2))
        map_in[0:shap[0],:,:] = copy(map_1)
        map_in[shap[0]:,:,:] = copy(map_2)



    dist_map = cross_distances_map(map_in,map_in)
    P = None
    if clust_en:
        P,dist = psf_sinkh_transport_clust(psf_in_1,psf_in_2,dist_map,cart_prods,w,inf_val=None,beta=beta,nb_iter=nb_iter)
    else:
        P,dist = psf_sinkh_transport(psf_in_1,psf_in_2,dist_map,inf_val=None,beta=beta,nb_iter=nb_iter)
    if same is False:
        P = P[shap[0]*shap[1]:,0:shap[0]*shap[1]]
    if centering:
        return P,map_1,map_2,dist_map,inf_im,dyn_coeff,cent
    else:
        return P,map_1,map_2,dist_map,dyn_coeff,cent

def psf_sinkh_transport_3(psf_1,psf_2,pos1,pos2,res_fact,nb_iter=100,centering=False,pol_en=True,scale_factt=1,scale_factr=1):

    shap = psf_1.shape
    psf_2 = psf_2*psf_1.sum()/psf_2.sum()
    cent = None
    map_1 = None
    dyn_coeff = None
    if pol_en:
        cent1 = utils.compute_centroid_2(psf_1,ones(shap))
        cent2 = utils.compute_centroid_2(psf_2,ones(shap))
        cent = (cent1+cent2)/2
        map_1 = pol_coord_map_2(shap,scale_factt,scale_factr,cent=cent)
    else:
        map_1 = pix_map(pos1,res_fact,shap)

    psf_in_1 = copy(psf_1)
    psf_in_2 = copy(psf_2)


    if centering:
        stack = zeros((shap[0],shap[1],2))
        stack[:,:,0] = copy(psf_1)
        stack[:,:,1] = copy(psf_2)
        stack_cent,inf_im = optim_utils.positive_centering(stack)
        psf_in_1 = stack_cent[:,:,0]
        psf_in_2 = stack_cent[:,:,1]

    # Dynamic normization
    m1 = max(map_1[:,:,0].std(),map_1[:,:,1].std())*0.0001000
    m2 = max(psf_in_1.std(),psf_in_2.std())

    f = zeros((3,size(psf_1)))
    f[0,:] = psf_in_1.reshape((size(psf_in_1)))
    f[1,:] = map_1[:,:,0].reshape((size(psf_in_2)))*m2/m1
    f[2,:] = map_1[:,:,1].reshape((size(psf_in_2)))*m2/m1

    g = zeros((3,size(psf_1)))
    g[0,:] = psf_in_2.reshape((size(psf_in_1)))
    g[1,:] = map_1[:,:,0].reshape((size(psf_in_2)))*m2/m1
    g[2,:] = map_1[:,:,1].reshape((size(psf_in_2)))*m2/m1

    pf,dist_sig,i_opt = optim_utils.sliced_transport(f,g,nb_iter=nb_iter)
    pf[1:,:] *= m1/m2
    interp = pf[0,:].reshape(shap)

    final_pos = zeros((shap[0],shap[1],2))
    final_pos[:,:,0] = pf[1,:].reshape(shap)
    final_pos[:,:,1] = pf[2,:].reshape(shap)

    return interp,map_1,final_pos,cent


def cloud_sinkh_transport_2(w_1,w_2,pos1,pos2,same=False,inf_val=None,beta=100,nb_iter=10,centering=False,dist_map=None,pol_en=True,scale_factt=1,scale_factr=1):

    cent = None
    nb_pts_1 = pos1.shape[0]
    nb_pts_2 = pos2.shape[0]
    pos_in = zeros((nb_pts_1+nb_pts_2,2))
    pos_in[0:nb_pts_1,:] = copy(pos1)
    pos_in[nb_pts_1:,:] = copy(pos2)
    dyn_coeff = None
    if pol_en:
        w = zeros((nb_pts_1+nb_pts_2,))
        w[0:nb_pts_1] = copy(w_1)
        w[nb_pts_1:] = copy(w_2)

        cent = utils.cloud_centroid(pos_in,w)
        pos_in = pol_coord_map_cloud(pos_in,scale_factt,scale_factr,cent)

    w1_in = zeros((nb_pts_1+nb_pts_2,))
    w2_in = zeros((nb_pts_1+nb_pts_2,))
    w1_in[0:nb_pts_1] = copy(w_1)
    w2_in[nb_pts_1:] = copy(w_2)


    dist_map = cross_distances_cloud(pos_in)
    P,dist = optim_utils.beg_proj_transp(dist_map,w1_in.reshape((nb_pts_1+nb_pts_2,1)),w2_in.reshape((nb_pts_1+nb_pts_2,1)),inf_val=None,beta=beta,nb_iter=nb_iter)
    P = P[nb_pts_2:,0:nb_pts_1]
    return P,dist_map,dyn_coeff,cent


def psf_sinkh_transport_bar(psf_stack,dist_maps,weights,inf_val=None,beta=100,nb_iter=10):
    # Distance map setting
    shap = psf_stack.shape
    distribs = zeros((shap[2],shap[0]*shap[1]))
    for i in range(0,shap[2]):
        # Distribution matrix setting
        distribs[i,:] = psf_stack[:,:,i].reshape((shap[0]*shap[1],))/psf_stack[:,:,i].sum()

    P,bar = optim_utils.beg_proj_transp_bar(dist_maps,distribs,weights,inf_val=inf_val,beta=beta,nb_iter=nb_iter)
    bar = bar.reshape((shap[0],shap[1]))
    return P,bar

#def psf_sinkh_transport_bar(psf_stack,dist_maps,weights,inf_val=None,beta=100,nb_iter=10):

def sinkh_transport_bar(distribs,dist_maps,data_pos,target_pos,beta=100,nb_iter=10,n=1): # The distribution are assumed to be normalized
    shap = distribs.shape
    weights = zeros((shap[0],))
    for i in range(0,shap[0]):
        # Distribution nromalization
        distribs[i,:] = distribs[i,:]/(distribs[i,:].sum())
        weights[i] = sqrt(((data_pos[i,:]-target_pos)**2).sum())**(-n)
    weights = weights/(weights.sum())
    #print weights
    P,bar = optim_utils.beg_proj_transp_bar(dist_maps,distribs,weights,beta=beta,nb_iter=nb_iter)
    return P,bar

def opt_assig_psf_bar(psf,data_pos,target_pos,n=1):
    shap = psf.shape
    weights = zeros((shap[2],))
    distribs = zeros((shap[0]*shap[1],shap[2]))
    lin_bar = zeros(shap[0:2])
    for i in range(0,shap[2]):
        weights[i] = sqrt(((data_pos[i,:]-target_pos)**2).sum())**(-n)
        # Distribution matrix setting
        distribs[:,i] = psf[:,:,i].reshape((shap[0]*shap[1],))
        lin_bar+= weights[i]*psf[:,:,i]
    lin_bar/=weights.sum()
    weights = weights/(weights.sum())
    bar_init = lin_bar.reshape((shap[0]*shap[1],1))
    #bar_init = distribs[0,:].reshape((shap[0]*shap[1],1))
    bar = optim_utils.sliced_transport_bar(distribs,weights,bar_init=bar_init)
    assign_bar = bar.reshape(shap[0:2])
    return assign_bar,lin_bar


def opt_assig_psf_bar2(psf,data_pos,target_pos,nb_iter=1500,n=2,embedding_en=False):
    shap = psf.shape
    nb_obs = psf.shape[2]
    weights = zeros((nb_obs,))

    lin_bar = zeros(shap[0:2])


    for i in range(0,nb_obs):
        weights[i] = sqrt(((data_pos[i,:]-target_pos)**2).sum())**(-n)
        lin_bar+= weights[i]*psf[:,:,i]

    lin_bar/=weights.sum()
    weights = weights/(weights.sum())


    assign_bar,hess_obj,flann_obj = LLE_sliced_transport_bar(psf,weights,cent=None,coord_map=None,log_param=1.5,zeros_inc=True,nb_neigh_emb=None,nb_comp=3,nb_neigh_inv_map=3,nb_samp=8,embedding_en=embedding_en)


    return assign_bar,lin_bar


def opt_assig_psf_bar2_stack(psf_in,data_pos,target_pos,n=1,pol_en=False,scale_factt=1,scale_factr=1,res_fact=6,w=1,nb_iter=50,nb_real=1,centering=True,nb_neighb=5,remapping=True,map_fact=3,embedding_en=False):
    nb_est = target_pos.shape[0]
    shap = psf_in.shape
    assign_bar = zeros((shap[0],shap[1],nb_est))
    assign_bar_2 = zeros((shap[0],shap[1],nb_est))
    lin_bar = zeros((shap[0],shap[1],nb_est))

    for i in range(0,nb_est):
        ind = utils.knn_pos(target_pos[i,:],data_pos,nb_neighb)
        assign_bar_i,lin_bar_i = opt_assig_psf_bar2(psf_in[:,:,ind],data_pos[ind,:],target_pos[i,:],n=n,embedding_en=embedding_en)
        assign_bar[:,:,i] = assign_bar_i

        lin_bar[:,:,i] = lin_bar_i

    return assign_bar,lin_bar


def opt_assig_psf_bar2_stack2(psf_in,data_pos,dist,data_tree,min_dist,n=1,pol_en=False,scale_factt=1,scale_factr=1,res_fact=6,w=1,nb_iter=50,nb_real=1,centering=True,nb_neighb=5,remapping=True,map_fact=3,embedding_en=False):
    nb_est = data_pos.shape[0]
    shap = psf_in.shape
    assign_bar = zeros((shap[0],shap[1],nb_est))
    assign_bar_2 = zeros((shap[0],shap[1],nb_est))
    lin_bar = zeros((shap[0],shap[1],nb_est))
    rmin = 0
    for i in range(0,nb_est):
        print "psf ",i+1,"/",nb_est
        ind0 = where(dist[i,:]>=min_dist[i])
        ind = data_tree[i,ind0[0][0:nb_neighb]]
        rmin += dist[i,ind0[0][0]]
        assign_bar_i,lin_bar_i = opt_assig_psf_bar2(psf_in[:,:,ind],data_pos[ind,:],data_pos[i,:],n=n,embedding_en=embedding_en)

        assign_bar[:,:,i] = assign_bar_i
        print "Sum val: ",assign_bar[:,:,i].sum()

        lin_bar[:,:,i] = lin_bar_i
    rmin/=nb_est
    return assign_bar,lin_bar,rmin




def opt_assig_psf_bar2_stackm(psf_in,data_pos,target_pos,n=1,pol_en=False,scale_factt=1,scale_factr=1,res_fact=6,w=1,nb_iter=50,nb_real=1,centering=True,nb_neighb=5,remapping=True,map_fact=3,embedding_en=False):
    nb_realizations = len(psf_in)
    assign_bar = list()
    assign_bar_2 = list()
    lin_bar = list()

    for i in range(0,nb_realizations):
        print "realization ",i+1,"/",nb_realizations
        assign_bar_i,lin_bar_i = opt_assig_psf_bar2_stack(psf_in[i],data_pos[i],target_pos[i],n=n,pol_en=pol_en,scale_factt=scale_factt,scale_factr=scale_factr,res_fact=res_fact,w=w,nb_iter=nb_iter,nb_real=nb_real,centering=centering,nb_neighb=nb_neighb,remapping=remapping,map_fact=map_fact,embedding_en=embedding_en)
        assign_bar.append(assign_bar_i)

        lin_bar.append(lin_bar_i)

    return assign_bar,lin_bar


def opt_assig_psf_bar2_stackm2(psf_in,data_pos,dist,data_tree,dist_range,n=2,pol_en=False,scale_factt=1,scale_factr=1,res_fact=6,w=1,nb_iter=20,nb_real=1,centering=True,nb_neighb=5,remapping=True,map_fact=3,embedding_en=False):
    nb_realizations = dist_range.shape[1]
    assign_bar = list()
    assign_bar_2 = list()
    lin_bar = list()
    av_dist = zeros((nb_realizations,))
    rmin = zeros((nb_realizations,))
    for i in range(0,nb_realizations):
        print "realization ",i+1,"/",nb_realizations
        assign_bar_i,lin_bar_i,rmini = opt_assig_psf_bar2_stack2(psf_in,data_pos,dist,data_tree,dist_range[:,i],n=n,pol_en=pol_en,scale_factt=scale_factt,scale_factr=scale_factr,res_fact=res_fact,w=w,nb_iter=nb_iter,nb_real=nb_real,centering=centering,nb_neighb=nb_neighb,remapping=remapping,map_fact=map_fact,embedding_en=embedding_en)
        assign_bar.append(assign_bar_i)
        lin_bar.append(lin_bar_i)
        rmin[i] = rmini

    return assign_bar,lin_bar,rmin

def opt_assig_psf_bar2_stackm2bis(psf_in,data_pos,dist,data_tree,dist_range,n=2,pol_en=False,scale_factt=1,scale_factr=1,res_fact=6,w=1,nb_iter=20,nb_real=1,centering=True,nb_neighb=None,remapping=True,map_fact=3,embedding_en=False):
    nb_realizations = size(nb_neighb)
    assign_bar = list()
    assign_bar_2 = list()
    lin_bar = list()

    for i in range(0,nb_realizations):
        print "realization ",i+1,"/",nb_realizations
        assign_bar_i,lin_bar_i = opt_assig_psf_bar2_stack2(psf_in,data_pos,dist,data_tree,dist_range,n=n,pol_en=pol_en,scale_factt=scale_factt,scale_factr=scale_factr,res_fact=res_fact,w=w,nb_iter=nb_iter,nb_real=nb_real,centering=centering,nb_neighb=nb_neighb[i],remapping=remapping,map_fact=map_fact,embedding_en=embedding_en)
        assign_bar.append(assign_bar_i)
        lin_bar.append(lin_bar_i)

    return assign_bar,lin_bar




def sinkh_transport_bar_located(distribs,dist_maps,data_pos,target_pos,src_pos,beta=100,nb_iter=10,n=20): # The distribution are assumed to be normalized
    shap = distribs.shape
    weights = zeros((shap[0],))
    for i in range(0,shap[0]):
        # Distribution nromalization
        distribs[i,:] = distribs[i,:]/(distribs[i,:].sum())
        weights[i] = sqrt(((data_pos[i,:]-target_pos)**2).sum())**(-n)
    weights = weights/(weights.sum())
    #print weights

    P,bar = optim_utils.beg_proj_transp_bar_located(dist_maps,distribs,weights,target_pos,src_pos[:,0],src_pos[:,1],inf_val=None,beta=beta,nb_iter=nb_iter)
    return P,bar


def sinkh_transport_ptbar(distribs,next_mat,data_pos,res_fact,target_pos,beta=100,nb_iter=10): # The distribution are assumed to be normalized
    shap = distribs.shape
    weights = zeros((shap[0],))
    for i in range(0,shap[0]):
        # Distribution nromalization
        distribs[i,:] = distribs[i,:]/(distribs[i,:].sum())
        weights[i] = sqrt(((data_pos[i,:]-target_pos)**2).sum())**(-1)
    weights = weights/(weights.sum())
    bar = optim_utils.beg_proj_transp_ptbar(dist_mat,next_mat,distribs,weights,inf_val=None,beta=500,nb_iter=100)
    return bar

def psf_sinkh_transport_ptbar(psf_stack,data_pos,target_pos,beta=50,nb_iter=50,centering=True,pol_en=True,same=True,res_fact=6,adaptive_gm=True,p=3):

    shap = psf_stack.shape

    # -------- barycentric coodinates calculation --------- #
    x = optim_utils.bar_coord_pb_dijkstra(data_pos,target_pos,nb_iter=10)
    nb_data = len(x)
    x = x.reshape((nb_data,))
    ind = argsort(x)
    cur_est = copy(psf_stack[:,:,ind[-1]])
    w = x[ind[-1]]
    cur_pos = data_pos[ind[-1],:]
    cur_est_stack = zeros((shap[0],shap[1],nb_data))
    cur_est_stack[:,:,0] = copy(cur_est)
    list_im = list()
    inf_pos_stack = None
    if centering:
        inf_pos_stack = zeros((shap[0],shap[1],nb_data-1))
    for i in range(0,nb_data-1):
        print i+1, "interpolation/",nb_data-1
        # ------- Partial barycenters estimation ------- #
        inf_pos = None
        im_in = zeros((shap[0],shap[1],2))
        im_in[:,:,0] = copy(cur_est)
        im_in[:,:,1] = copy(psf_stack[:,:,ind[-i-2]])

        if centering:
            im_in,inf_pos = optim_utils.positive_centering(im_in)
            list_im.append(im_in)
            inf_pos_stack[:,:,i] = copy(inf_pos)
        # ------- Local ground metric parameters computation ------- #
        scale_factr = 1
        scale_factt = 1
        if adaptive_gm:
            if pol_en:
                scale = im_in.max()
                U1,s1,map1 = utils.im_shape(im_in[:,:,0],scale=scale)
                U2,s2,map2 = utils.im_shape(im_in[:,:,1],scale=scale)
                scale_factr = min([s1[0]/s2[0],s2[0]/s1[0]])
                scale_factt = (abs(sum(U1[:,0]*U2[:,0])))**p
                print scale_factr,scale_factt
        # ------- Transport plan computation ------- #
        P,map_1,map_2,dist_mat,dyn_coeff,cent = psf_sinkh_transport_2(im_in[:,:,0],im_in[:,:,1],cur_pos,data_pos[ind[-i-2],:],res_fact,same=same,inf_val=None,beta=beta,nb_iter=nb_iter,pol_en=pol_en,scale_factr=scale_factr,scale_factt = scale_factt)
        #P,map_1,map_2,dist_mat,dyn_coeff,cent = psf_sinkh_transport_2(im_in[:,:,0],im_in[:,:,1],cur_pos,data_pos[ind[-i-2],:],res_fact,same=same,inf_val=None,beta=beta,nb_iter=nb_iter,pol_en=pol_en,scale_factt = 0.00001)

        # ------- Displacement interpolation ------- #
        t = w/(x[ind[-i-2]]+w)

        if pol_en:
            cur_est = displacement_interp_pol_2(P,t,[shap[0],shap[1]],map_1,map_2,cent,scale_factt,scale_factr,tol=0.001)
            #cur_est = displacement_interp_pol(P,t,[shap[0],shap[1]],map_1,map_2,cent,1,tol=0.001)
        else:
            cur_est = displacement_interp(P,t,[shap[0],shap[1]],map_1,map_2,tol=0.001)
        cur_est_stack[:,:,i+1] = copy(cur_est)
        if centering:
            cur_est += inf_pos
        cur_pos = (w*cur_pos+x[ind[-i-2]]*data_pos[ind[-i-2],:])/(w+x[ind[-i-2]])
        w += x[ind[-i-2]]

    return cur_est,inf_pos_stack,x,cur_est_stack,list_im


def psf_sinkh_transport_ptbar2(psf_stack,data_pos,target_pos,beta=50,nb_iter=50,centering=True,pol_en=True,same=True,res_fact=6,adaptive_gm=True,p=3):

    shap = psf_stack.shape

    # -------- barycentric coodinates calculation --------- #
    x = optim_utils.bar_coord_pb_dijkstra(data_pos,target_pos,nb_iter=10)
    nb_data = len(x)
    x = x.reshape((nb_data,))
    ind = argsort(x)

    linear_bar = zeros((shap[0],shap[1]))
    for i in range(0,shap[2]):
        linear_bar+=psf_stack[:,:,i]*x[i]

    im_in = zeros((shap[0],shap[1],2))
    im_in[:,:,0] = copy(psf_stack[:,:,ind[-1]])
    im_in[:,:,1] = copy(linear_bar)
    inf_pos = None
    if centering:
        im_in,inf_pos = optim_utils.positive_centering(im_in)

    # ------- Local ground metric parameters computation ------- #
    scale_factr = 1
    scale_factt = 1
    if adaptive_gm:
        if pol_en:
            scale = im_in.max()
            U1,s1,map1 = utils.im_shape(im_in[:,:,0],scale=scale)
            U2,s2,map2 = utils.im_shape(im_in[:,:,1],scale=scale)
            scale_factr = min([s1[0]/s2[0],s2[0]/s1[0]])
            scale_factt = (abs(sum(U1[:,0]*U2[:,0])))**p
            print scale_factr,scale_factt

    # ------- Transport plan computation ------- #
    P,map_1,map_2,dist_mat,dyn_coeff,cent = psf_sinkh_transport_2(im_in[:,:,0],im_in[:,:,1],data_pos[ind[-1],:],target_pos,res_fact,same=same,inf_val=None,beta=beta,nb_iter=nb_iter,pol_en=pol_en,scale_factr=scale_factr,scale_factt = scale_factt)

    # ------- Displacement interpolation ------- #
    t = x[ind[-1]]/(2*sum(x))

    transp_bar = None
    if pol_en:
        transp_bar = displacement_interp_pol_2(P,t,[shap[0],shap[1]],map_1,map_2,cent,scale_factt,scale_factr,tol=0.001)
    #cur_est = displacement_interp_pol(P,t,[shap[0],shap[1]],map_1,map_2,cent,1,tol=0.001)
    else:
        transp_bar = displacement_interp(P,t,[shap[0],shap[1]],map_1,map_2,tol=0.001)

    return transp_bar,linear_bar,inf_pos,x


def psf_sinkh_transport_ptbar_stack(psf_stack,data_pos,target_pos,nb_points,beta=50,nb_iter=50,centering=True,pol_en=True,same=True,res_fact=6,adaptive_gm=True,p=3):

    shap = target_pos.shape

    shap_im = psf_stack.shape
    est_transp = zeros((shap_im[0],shap_im[1],nb_points))
    est_lin = zeros((shap_im[0],shap_im[1],nb_points))

    for i in range(0,shap[0]):
        ind = utils.knearest(target_pos[i,:],data_pos,nb_points)
        print ind,nb_points
        cur_est,inf_pos_stack,x,cur_est_stack,list_im = psf_sinkh_transport_ptbar(psf_stack[:,:,ind],data_pos[ind,:],target_pos[i,:],beta=beta,nb_iter=nb_iter,centering=centering,pol_en=pol_en,same=same,res_fact=res_fact,adaptive_gm=adaptive_gm,p=p)
        est_transp[:,:,i] = cur_est
        for k in range(0,nb_points):
            est_lin[:,:,i]+=x[k]*psf_stack[:,:,ind[k]]

    return est_transp,est_lin

def psf_sinkh_transport_ptbar_stackm(psf_stack,data_pos,target_pos,nb_points,beta=50,nb_iter=50,centering=True,pol_en=True,same=True,res_fact=6,adaptive_gm=True,p=3):
    list_trans = list()
    list_lin = list()
    nb_realizations = len(psf_stack)

    for i in range(0,nb_realizations):
        est_transp,est_lin = psf_sinkh_transport_ptbar_stack(psf_stack[i],data_pos[i],target_pos[i],nb_points,beta=beta,nb_iter=nb_iter,centering=centering,pol_en=pol_en,same=same,res_fact=res_fact,adaptive_gm=adaptive_gm,p=p)
        list_trans.append(est_transp)
        list_lin.append(est_lin)
    return list_trans,list_lin


def psf_field_sinkh_transport_bar(psf_stack,dist_maps,data_pos,target_pos,nb_psf_interp,beta=100,nb_iter=500):
    shap = psf_stack.shape
    distribs = zeros((shap[2],shap[0]*shap[1]))
    for i in range(0,shap[2]):
        # Distribution matrix setting
        distribs[i,:] = psf_stack[:,:,i].reshape((shap[0]*shap[1],))/psf_stack[:,:,i].sum()
    bar_distribs,P = field_sinkh_transport_bar(distribs,dist_maps,data_pos,target_pos,nb_psf_interp,beta=beta,nb_iter=nb_iter)
    output = zeros((shap[0],shap[1],bar_distribs.shape[0]))
    for i in range(0,bar_distribs.shape[0]):
        output[:,:,i] = bar_distribs[i,:].reshape((shap[0],shap[1]))
    return output

def field_sinkh_transport_bar(distribs,dist_maps,data_pos,target_pos,nb_psf_interp,beta=100,nb_iter=500):
    shap1 = distribs.shape
    shap2 = target_pos.shape
    shap3 = data_pos.shape
    bar_distribs = zeros((shap2[0],shap1[1]))
    ones_vect = ones((shap3[0],1))
    P=None
    for i in range(0,shap2[0]):
        # Neighbors computation
        u = target_pos[i,:].reshape((1,2))
        delta = data_pos - ones_vect.dot(u)
        dist = sqrt((delta**2).sum(axis=1))
        ind = np.argsort(dist)
        P,bar = sinkh_transport_bar(distribs[ind[0:nb_psf_interp],:],dist_maps,data_pos[ind[0:nb_psf_interp],:],target_pos[i,:],beta=beta,nb_iter=nb_iter)
        bar_distribs[i,:] = bar
    return bar_distribs,P

def field_sinkh_transport_bar_reg(distribs,next_mat,lin_bar,dist_maps,data_pos,target_pos,nb_psf_interp,beta=100,nb_iter=100, trust=0.5,tol=5000,pw=1): # This function's purpose is to adjust the linear barycenter
    shap1 = distribs.shape
    shap2 = target_pos.shape
    shap3 = data_pos.shape
    bar_distribs = zeros((shap2[0],shap1[1]))
    ones_vect = ones((shap3[0],1))
    Pmax = zeros((shap1[1],shap1[1]))
    input = zeros((nb_psf_interp+1,shap1[1]))
    weights = zeros((nb_psf_interp+1,))*(1-trust)/nb_psf_interp

    print "Contructing PSF tree..."
    neigh,dists = utils.knn_interf(data_pos,shap3[0]-1)
    print "Done..."
    dist_med = np.median(dists)
    for i in range(0,shap2[0]):
        # Neighbors computation
        u = target_pos[i,:].reshape((1,2))
        delta = data_pos - ones_vect.dot(u)
        dist = sqrt((delta**2).sum(axis=1))
        ind = np.argsort(dist)
        input[0:nb_psf_interp,:] = distribs[ind[0:nb_psf_interp],:]
        input[nb_psf_interp,:] = lin_bar[i,:]
        trust = 1/(1+(median(dist[ind[0:nb_psf_interp]])/(tol*dist_med)))
        print "trust index: ",trust
        #weights = ones((nb_psf_interp+1,))*(1-trust)/nb_psf_interp
        #weights[nb_psf_interp] = trust
        for k in range(0,nb_psf_interp):
            weights[k] = dist[ind[k]]**(pw)
        weights[0:nb_psf_interp] = (1-trust)*weights[0:nb_psf_interp]/(weights[0:nb_psf_interp].sum())
        weights[nb_psf_interp] = trust
        #weights = ones((nb_psf_interp+1,))/(nb_psf_interp+1)
        # Transport upper bound maps computation
        """Pmax = zeros((shap1[1],shap1[1]))
        for k in range(0,nb_psf_interp):

            P,dist = optim_utils.beg_proj_transp(dist_maps[:,:,ind[k]],lin_bar[i,:].reshape((shap1[1],1)),distribs[ind[k],:].reshape((shap1[1],1)),beta=beta,nb_iter=nb_iter)
            i1,i2 = where(P>Pmax)
            Pmax[i1,i2] = P[i1,i2]"""
        #P,bar = optim_utils.beg_proj_transp_bar(dist_maps[:,:,ind[0:nb_psf_interp+1]],input,weights,beta=beta,nb_iter=nb_iter)
        bar = optim_utils.beg_proj_transp_ptbar(dist_maps[:,:,0],next_mat,input,weights,inf_val=None,beta=beta,nb_iter=nb_iter)
        #P,bar = optim_utils.beg_proj_transp_capacity(dist_maps[:,:,ind[0]],Pmax,lin_bar[i,:].reshape((shap1[1],1)),beta=beta,nb_iter=800)
        #P,bar = sinkh_transport_bar(distribs[ind[0:nb_psf_interp],:],dist_maps[:,:,ind[0:nb_psf_interp]],data_pos[ind[0:nb_psf_interp],:],target_pos[i,:],beta=beta,nb_iter=nb_iter)
        bar_distribs[i,:] = bar.reshape((size(bar),))
    return bar_distribs,Pmax

def field_sinkh_transport_bar_located(distribs,dist_maps,data_pos,target_pos,src_pos,nb_psf_interp,beta=100,nb_iter=500):
    shap1 = distribs.shape
    shap2 = target_pos.shape
    shap3 = data_pos.shape
    bar_distribs = zeros((shap2[0],shap1[1]))
    ones_vect = ones((shap3[0],1))
    P=None
    for i in range(0,shap2[0]):
        # Neighbors computation
        u = target_pos[i,:].reshape((1,2))
        delta = data_pos - ones_vect.dot(u)
        dist = sqrt((delta**2).sum(axis=1))
        ind = np.argsort(dist)
        P,bar = sinkh_transport_bar_located(distribs[ind[0:nb_psf_interp],:],dist_maps[:,:,ind[0:nb_psf_interp]],data_pos[ind[0:nb_psf_interp],:],target_pos[i,:],src_pos,beta=beta,nb_iter=nb_iter)
        bar_distribs[i,:] = bar
    return bar_distribs,P


def field_sinkh_transport_ptbar(distribs,dist_mat,next_mat,data_pos,target_pos,nb_psf_interp,beta=500,nb_iter=100):
    shap1 = distribs.shape
    shap2 = target_pos.shape
    shap3 = data_pos.shape
    bar_distribs = zeros((shap2[0],shap1[1]))
    ones_vect = ones((shap3[0],1))
    P=None
    for i in range(0,shap2[0]):
        # Neighbors computation
        u = target_pos[i,:].reshape((1,2))
        delta = data_pos - ones_vect.dot(u)
        dist = sqrt((delta**2).sum(axis=1))
        ind = np.argsort(dist)
        bar = sinkh_transport_ptbar(distribs[ind[0:nb_psf_interp],:],dist_mat,next_mat,data_pos[ind[0:nb_psf_interp],:],target_pos[i,:],beta=beta,nb_iter=nb_iter)
        bar_distribs[i,:] = bar.reshape((shap1[1],))
    return bar_distribs

def field_transport_ptbar(distribs,dist_mat,next_mat,target_pos,pos_scaling,nb_psf_interp,siz,beta=60):
    shap1 = distribs.shape
    anchor_ind = [shap1[0]-2,shap1[0]-1]
    shap2 = target_pos.shape
    bar_distribs = zeros((siz[0],siz[1],shap2[0]))
    pos_interp = zeros((shap2[0],2))
    ones_vect = ones((1,shap1[1]))
    weights = zeros((shap1[1],shap2[0]))
    P=None
    for i in range(0,shap2[0]):
        # Neighbors computation
        print i
        u = target_pos[i,:].reshape((2,1))
        delta = distribs[anchor_ind,:] - u.dot(ones_vect)*pos_scaling
        dist = sqrt((delta**2).sum(axis=0))
        ind = np.argsort(dist)
        #S = distribs[:,ind[0:nb_psf_interp]]
        target_feat = pos_scaling*target_pos[i,:].reshape((2,1))
        bar,weights_i = optim_utils.graph_transport_ptbar(dist_mat,next_mat,ind[0:nb_psf_interp],distribs,target_feat,anchor_ind,nb_loops=2,tol=0.000001,beta=beta)
        bar_distribs[:,:,i] = bar[:-2,0].reshape((siz[0],siz[1]))
        pos_interp[i,:] = bar[-2:,0]
        weights[:,i] = weights_i.reshape((shap1[1],))
    return bar_distribs,pos_interp,weights


def r_theta_dist_map(siz,alpha,cent=None):
    if cent is None:
        cent = [siz[0]/2,siz[1]/2]
    map = zeros((siz[0]*siz[1],siz[0]*siz[1]))
    i,j=0,0
    dyn_coeff = sqrt((siz[0]/2)**2+(siz[1]/2)**2)/2*pi
    for i in range(0,siz[0]*siz[1]):
        i1 = np.int(i/siz[1]) # Line index in map1
        j1 = mod(i,siz[1])
        r1,theta1 = utils.polar_coord([i1,j1],cent)
        for j in range(0,siz[0]*siz[1]):
            i2 = np.int(j/siz[1]) # Line index in map2
            j2 = mod(j,siz[1])
            r2,theta2 = utils.polar_coord([i2,j2],cent)
            map[i,j]=sqrt((r1-r2)**2+(dyn_coeff*alpha*(theta1-theta2))**2)

    return map,dyn_coeff

def r_theta_rig_dist_map(siz,alpha,theta,tol,inf_val=1e64,cent=None):
    if cent is None:
        cent = [siz[0]/2,siz[1]/2]
    map = zeros((siz[0]*siz[1],siz[0]*siz[1]))
    i,j=0,0
    dyn_coeff = sqrt((siz[0]/2)**2+(siz[1]/2)**2)/2*pi
    for i in range(0,siz[0]*siz[1]):
        i1 = np.int(i/siz[1]) # Line index in map1
        j1 = mod(i,siz[1])
        r1,theta1 = utils.polar_coord([i1,j1],cent)
        for j in range(0,siz[0]*siz[1]):
            i2 = np.int(j/siz[1]) # Line index in map2
            j2 = mod(j,siz[1])
            r2,theta2 = utils.polar_coord([i2,j2],cent)

            if r1==0 and r2==0:
                map[i,j]=0
            else:
                a = None
                if r2==0:
                    a = arcsin(r1**(-1))
                elif r1==0:
                    a = arcsin(r2**(-1))
                else:
                    a = arcsin(min(r1**(-1),r2**(-1)))
                tol_rel = tol*a
                if abs(theta1-theta2-theta)<tol_rel:
                    map[i,j]=sqrt((r1-r2)**2+(dyn_coeff*alpha*(theta1-theta2))**2)
                else:
                    map[i,j] = inf_val

    return map,dyn_coeff

def r_theta_rig_dist_map_2(siz,alpha,theta,tol,inf_val=1e64,cent=None):
    if cent is None:
        cent = [siz[0]/2,siz[1]/2]
    rmax = int(round(sqrt(siz[0]**2+siz[1]**2)/2))+1
    dyn_coeff = 1#rmax/2*pi
    map = ones((siz[0]*siz[1],siz[0]*siz[1]))*inf_val
    for i in range(0,siz[0]*siz[1]):
        i1 = np.int(i/siz[1]) # Line index in map1
        j1 = mod(i,siz[1])
        r1,theta1 = utils.polar_coord([i1,j1],cent)
        rk=0
        thetak = theta1+theta
        for rk in range(0,rmax):
            ik  = np.int(rk*cos(thetak)+cent[0])
            jk  = np.int(rk*sin(thetak)+cent[1])
            if (ik>=0 and ik<siz[0] and jk>=0 and jk<siz[1]):
                map[i,ik*siz[1]+jk]=sqrt((r1-rk)**2+(dyn_coeff*alpha*theta)**2)

    for j in range(0,siz[0]*siz[1]):
        i1 = np.int(j/siz[1]) # Line index in map1
        j1 = mod(j,siz[1])
        r1,theta1 = utils.polar_coord([i1,j1],cent)
        rk=0
        thetak = theta1-theta
        for rk in range(0,rmax):
            ik  = np.int(rk*cos(thetak)+cent[0])
            jk  = np.int(rk*sin(thetak)+cent[1])
            if (ik>=0 and ik<siz[0] and jk>=0 and jk<siz[1]) and map[ik*siz[1]+jk,j]==inf_val:
                map[ik*siz[1]+jk,j]=sqrt((r1-rk)**2+(dyn_coeff*alpha*theta)**2)


    return map, dyn_coeff

def r_theta_rig_dist_map_3(siz,theta,inf_val=1e64,cent=None):
    if cent is None:
        cent = [siz[0]/2,siz[1]/2]
    rmax = int(round(sqrt(siz[0]**2+siz[1]**2)/2))+1
    mapr = ones((siz[0]*siz[1],siz[0]*siz[1]))*inf_val
    maptheta = ones((siz[0]*siz[1],siz[0]*siz[1]))*inf_val
    for i in range(0,siz[0]*siz[1]):
        i1 = np.int(i/siz[1]) # Line index in map1
        j1 = mod(i,siz[1])
        r1,theta1 = utils.polar_coord([i1,j1],cent)
        rk=0
        thetak = theta1+theta
        for rk in range(0,rmax*2):
            ik  = round((double(rk)/2)*cos(thetak)+cent[0])
            jk  = round((double(rk)/2)*sin(thetak)+cent[1])
            if (ik>=0 and ik<siz[0] and jk>=0 and jk<siz[1]):
                mapr[i,ik*siz[1]+jk]=abs(r1-(double(rk)/2))
                maptheta[i,ik*siz[1]+jk]=abs(theta)

    for j in range(0,siz[0]*siz[1]):
        i1 = np.int(j/siz[1]) # Line index in map1
        j1 = mod(j,siz[1])
        r1,theta1 = utils.polar_coord([i1,j1],cent)
        rk=0
        thetak = theta1-theta
        for rk in range(0,2*rmax):
            ik  = round((double(rk)/2)*cos(thetak)+cent[0])
            jk  = round((double(rk)/2)*sin(thetak)+cent[1])
            if (ik>=0 and ik<siz[0] and jk>=0 and jk<siz[1]) and mapr[ik*siz[1]+jk,j]==inf_val:
                mapr[ik*siz[1]+jk,j]=abs(r1-(double(rk)/2))
                maptheta[ik*siz[1]+jk,j]=abs(theta)
    #mapr = (mapr+transpose(mapr))/2

    return mapr, maptheta


def pol_coord_map(siz,alphat,alphar,cent=None):
    coord = zeros((siz[0],siz[1],2))
    if cent is None:
        cent = [siz[0]/2,siz[1]/2]
    dyn_coeff = sqrt((siz[0]/2)**2+(siz[1]/2)**2)/2*pi
    i,j=0,0
    for i in range(0,siz[0]):
        for j in range(0,siz[1]):
            r,theta = utils.polar_coord([i,j],cent)
            coord[i,j,0] = r*alphar
            coord[i,j,1] = dyn_coeff*alphat*theta

    return coord,dyn_coeff

def pol_coord_map_2(siz,alphat,alphar,cent=None):
    coord = zeros((siz[0],siz[1],2))
    if cent is None:
        cent = [siz[0]/2,siz[1]/2]
    i,j=0,0
    for i in range(0,siz[0]):
        for j in range(0,siz[1]):
            r,theta = utils.polar_coord([i,j],cent)
            coord[i,j,0] = r*alphar
            coord[i,j,1] = r*alphat*theta

    return coord


def pol_coord_map_cloud(pos,alphat,alphar,cent):
    nb_points = pos.shape[0]
    coord = zeros((nb_points,2))
    for i in range(0,nb_points):
        r,theta = utils.polar_coord(pos[i,:],cent)
        coord[i,0] = r*alphar
        coord[i,1] = r*alphat*theta

    return coord


def intensity_coord(im,cent=None):
    siz = im.shape
    dyn_coeff = sqrt((siz[0]/2)**2+(siz[1]/2)**2)/2*pi
    coord_map = zeros((3,siz[0]*siz[1]))
    coord_map[0,:] = copy(im.reshape((siz[0]*siz[1],)))


    for i in range(0,siz[0]):
        for j in range(0,siz[1]):
            r,theta = utils.polar_coord([i,j],cent)
            coord_map[1,j+i*siz[1]] = r
            coord_map[2,j+i*siz[1]] = theta*dyn_coeff


    return coord_map,dyn_coeff

def psf_dist_map_setting(coord_map,r=3,a1=1,a2=0.75): # a2<0
    siz = coord_map.shape
    dist_map = zeros((siz[1],siz[1]))
    b = None
    c = None

    for i in range(0,siz[1]):
        for j in range(0,i):
            intensity_dist = utils.log_norm(coord_map[0,i],coord_map[0,j],u=a2)
            rdist2,b,c =  utils.l2_exp((coord_map[1,i]-coord_map[1,j])**2,a1,r,b=b,c=c)
            theta_dist2 = (coord_map[2,i]-coord_map[2,j])**2
            dist_map[i,j] = intensity_dist**2+rdist2+theta_dist2

    dist_map+=transpose(dist_map)

    return dist_map

#def mds(dist_mat,):


def partial_transport_2(P,t,shap,position_map_1,position_map_2,scale_coeff,tol=0,cent=None):
    psf_interp = zeros(shap)
    min_val = tol*P.max()
    i=0
    j=0
    if cent is None:
        cent = [shap[0]/2,shap[1]/2]
    for i in range(0,shap[0]*shap[1]):
        i1 = np.int(i/shap[1]) # Line index in position_map_1
        j1 = mod(i,shap[1])
        for j in range(0,shap[0]*shap[1]):
            if (P[i,j] >= min_val):
                i2 = np.int(j/shap[1]) # Line index in postion_map_2
                j2 = mod(j,shap[1])
                if (i1==i2 and j1==j2):
                    pos = position_map_1[i1,j1,:]
                    iopt = round(pos[0]*cos(pos[1]/scale_coeff)+cent[0])
                    jopt = round(pos[0]*sin(pos[1]/scale_coeff)+cent[1])
                    psf_interp[iopt,jopt] = psf_interp[iopt,jopt] + P[i,j]
                else :
                    pos = ((1-t)*position_map_1[i1,j1,:]+t*position_map_2[i2,j2,:])
                    iopt = round(pos[0]*cos(pos[1]/scale_coeff)+cent[0])
                    jopt = round(pos[0]*sin(pos[1]/scale_coeff)+cent[1])

                    if iopt < shap[0] and iopt>-1 and jopt < shap[1] and jopt>-1:
                        psf_interp[iopt,jopt] = psf_interp[iopt,jopt]+P[i,j]
    return psf_interp

def displacement_interp(P,t,shap,position_map_1,position_map_2,tol=0.001,var_speed=False):

    total = P.sum()
    res = total
    ind = utils.argsort2D(P)
    psf_interp = zeros(shap)
    l = -1
    target_pos = (1-t)*position_map_1[0,0,:]+t*position_map_2[0,0,:]+array([shap[1]/2,shap[0]/2])
    while l>-(shap[0]*shap[1])**2 and res>total*tol:
        indP = ind[l]
        l-=1
        indPl = int(indP/(shap[0]*shap[1]))

        indPc = int(indP%(shap[0]*shap[1]))

        i1 = np.int(indPl/shap[1]) # Line index in position_map_1
        j1 = mod(indPl,shap[1])
        i2 = np.int(indPc/shap[1]) # Line index in position_map_2
        j2 = mod(indPc,shap[1])
        interp_pos = None
        if var_speed is False:
            interp_pos = (1-t)*position_map_1[i1,j1,:]+t*position_map_2[i2,j2,:]
        else:
            t1 = utils.stretch_sigmoid(t)
            interp_pos = (1-t1)*position_map_1[i1,j1,:]+t1*position_map_2[i2,j2,:]
        ind_im = interp_pos+array([shap[1],shap[0]])/2-target_pos

        ind_im[0]= np.int(round(min(max(0,ind_im[0]),shap[1]-1)))
        ind_im[1]= np.int(round(min(max(0,ind_im[1]),shap[0]-1)))
        psf_interp[ind_im[1],ind_im[0]]+=P[indPl,indPc]
        res -= P[indPl,indPc]
    print abs(l),"/",(shap[0]*shap[1])**2
    return psf_interp


def displacement_interp_2(im_ref,im_proj,t,position_map_1,position_map_2,tol=0.001):

    shap = im_ref.shape
    total = im_ref.sum()
    res = total
    ind = utils.argsort2D(im_ref)
    psf_interp = zeros(shap)
    l = -1
    target_pos = (1-t)*position_map_1[0,0,:]+t*position_map_2[0,0,:]+array([shap[1]/2,shap[0]/2])
    while l>-(shap[0]*shap[1]) and res>total*tol:
        indP = ind[l]
        l-=1
        indPl = int(indP/shap[0])

        indPc = int(indP%shap[0])


        interp_pos = (1-t)*position_map_1[indPl,indPc,:]+t*position_map_2[indPl,indPc,:]
        ind_im = interp_pos+array([shap[1],shap[0]])/2-target_pos

        ind_im[0]= np.int(round(min(max(0,ind_im[0]),shap[1]-1)))
        ind_im[1]= np.int(round(min(max(0,ind_im[1]),shap[0]-1)))
        psf_interp[ind_im[1],ind_im[0]]+= (1-t)*im_ref[indPl,indPc]+t*im_proj[indPl,indPc]
        res -= (1-t)*im_ref[indPl,indPc]+t*im_proj[indPl,indPc]
    print abs(l),"/",(shap[0]*shap[1])
    return psf_interp


def displacement_interp_pol(P,t,shap,position_map_1,position_map_2,cent,scale_coefft,scale_coeffr,tol=0.001,var_speed=False):

    total = P.sum()
    res = total
    ind = utils.argsort2D(P)
    psf_interp = zeros(shap)
    l = -1
    while l>-(shap[0]*shap[1])**2 and res>total*tol:
        indP = ind[l]
        l-=1
        indPl = int(indP/(shap[0]*shap[1]))

        indPc = int(indP%(shap[0]*shap[1]))

        i1 = np.int(indPl/shap[1]) # Line index in position_map_1
        j1 = mod(indPl,shap[1])
        i2 = np.int(indPc/shap[1]) # Line index in position_map_2
        j2 = mod(indPc,shap[1])
        interp_pos = None
        if var_speed is False:
            interp_pos = (1-t)*position_map_1[i1,j1,:]+t*position_map_2[i2,j2,:]
        else:
            t1 = utils.stretch_sigmoid(t)
            interp_pos = (1-t1)*position_map_1[i1,j1,:]+t1*position_map_2[i2,j2,:]
        iopt = (interp_pos[0]/scale_coeffr)*cos(interp_pos[1]/scale_coefft)+cent[0]
        iopt = min(max(0,iopt),shap[0]-1)
        jopt = (interp_pos[0]/scale_coeffr)*sin(interp_pos[1]/scale_coefft)+cent[1]
        jopt = min(max(0,jopt),shap[1]-1)

        iopt1 = min(max(0,np.int(iopt)),shap[0]-1)
        jopt1 = min(max(0,np.int(jopt)),shap[0]-1)

        if iopt1==iopt and jopt1==jopt:
            psf_interp[iopt1,jopt1]+=P[indPl,indPc]

        else:
            iopt2 = min(max(0,np.int(iopt1)+1),shap[0]-1)
            jopt2 = min(max(0,np.int(jopt1)+1),shap[0]-1)
            if iopt==iopt1:
                d1 = abs(jopt-jopt1)**(-1)
                d2 = abs(jopt-jopt2)**(-1)
                d = d1+d2
                psf_interp[iopt1,jopt1]+=d1*P[indPl,indPc]/d
                psf_interp[iopt1,jopt2]+=d2*P[indPl,indPc]/d
            elif jopt==jopt1:
                d1 = abs(iopt-iopt1)**(-1)
                d2 = abs(iopt-iopt2)**(-1)
                d = d1+d2
                psf_interp[iopt1,jopt1]+=d1*P[indPl,indPc]/d
                psf_interp[iopt2,jopt1]+=d2*P[indPl,indPc]/d

            else:
                d1 = sqrt((iopt-iopt1)**2+(jopt-jopt1)**2)**(-1)
                d2 = sqrt((iopt-iopt1)**2+(jopt-jopt2)**2)**(-1)
                d3 = sqrt((iopt-iopt2)**2+(jopt-jopt1)**2)**(-1)
                d4 = sqrt((iopt-iopt2)**2+(jopt-jopt2)**2)**(-1)
                d = d1+d2+d3+d4
                psf_interp[iopt1,jopt1]+=d1*P[indPl,indPc]/d
                psf_interp[iopt1,jopt2]+=d2*P[indPl,indPc]/d
                psf_interp[iopt2,jopt1]+=d3*P[indPl,indPc]/d
                psf_interp[iopt2,jopt2]+=d4*P[indPl,indPc]/d

        res -= P[indPl,indPc]

    print abs(l),"/",(shap[0]*shap[1])**2
    return psf_interp


def displacement_interp_pol_2(P,t,shap,position_map_1,position_map_2,cent,scale_coefft,scale_coeffr,tol=0.001,var_speed=False):

    total = P.sum()
    res = total
    ind = utils.argsort2D(P)
    psf_interp = zeros(shap)
    l = -1
    while l>-(shap[0]*shap[1])**2 and res>total*tol:
        indP = ind[l]
        l-=1
        indPl = int(indP/(shap[0]*shap[1]))

        indPc = int(indP%(shap[0]*shap[1]))

        i1 = np.int(indPl/shap[1]) # Line index in position_map_1
        j1 = mod(indPl,shap[1])
        i2 = np.int(indPc/shap[1]) # Line index in position_map_2
        j2 = mod(indPc,shap[1])
        interp_pos = None
        if var_speed is False:
            interp_pos = (1-t)*position_map_1[i1,j1,:]+t*position_map_2[i2,j2,:]
        else:
            t1 = utils.stretch_sigmoid(t)
            interp_pos = (1-t1)*position_map_1[i1,j1,:]+t1*position_map_2[i2,j2,:]
        iopt = cent[0]
        jopt = cent[1]
        if interp_pos[0]>0:
            iopt = (interp_pos[0]/scale_coeffr)*cos(scale_coeffr*interp_pos[1]/(scale_coefft*interp_pos[0]))+cent[0]
            iopt = min(max(0,iopt),shap[0]-1)
            jopt = (interp_pos[0]/scale_coeffr)*sin(scale_coeffr*interp_pos[1]/(scale_coefft*interp_pos[0]))+cent[1]
            jopt = min(max(0,jopt),shap[1]-1)

        iopt1 = min(max(0,np.int(iopt)),shap[0]-1)
        jopt1 = min(max(0,np.int(jopt)),shap[0]-1)

        if iopt1==iopt and jopt1==jopt:
            psf_interp[iopt1,jopt1]+=P[indPl,indPc]

        else:
            iopt2 = min(max(0,np.int(iopt1)+1),shap[0]-1)
            jopt2 = min(max(0,np.int(jopt1)+1),shap[0]-1)
            if iopt==iopt1:
                d1 = abs(jopt-jopt1)**(-1)
                d2 = abs(jopt-jopt2)**(-1)
                d = d1+d2
                psf_interp[iopt1,jopt1]+=d1*P[indPl,indPc]/d
                psf_interp[iopt1,jopt2]+=d2*P[indPl,indPc]/d
            elif jopt==jopt1:
                d1 = abs(iopt-iopt1)**(-1)
                d2 = abs(iopt-iopt2)**(-1)
                d = d1+d2
                psf_interp[iopt1,jopt1]+=d1*P[indPl,indPc]/d
                psf_interp[iopt2,jopt1]+=d2*P[indPl,indPc]/d

            else:
                d1 = sqrt((iopt-iopt1)**2+(jopt-jopt1)**2)**(-1)
                d2 = sqrt((iopt-iopt1)**2+(jopt-jopt2)**2)**(-1)
                d3 = sqrt((iopt-iopt2)**2+(jopt-jopt1)**2)**(-1)
                d4 = sqrt((iopt-iopt2)**2+(jopt-jopt2)**2)**(-1)
                d = d1+d2+d3+d4
                psf_interp[iopt1,jopt1]+=d1*P[indPl,indPc]/d
                psf_interp[iopt1,jopt2]+=d2*P[indPl,indPc]/d
                psf_interp[iopt2,jopt1]+=d3*P[indPl,indPc]/d
                psf_interp[iopt2,jopt2]+=d4*P[indPl,indPc]/d

        res -= P[indPl,indPc]

    print abs(l),"/",(shap[0]*shap[1])**2
    return psf_interp


def displacement_interp_pol_3(im_ref,im_proj,t,pos_map,final_pos,cent,scale_coefft,scale_coeffr,tol=0.001):
    shap = im_proj.shape
    total = im_proj.sum()
    res = total
    ind = utils.argsort2D(im_ref)
    psf_interp = zeros(shap)
    l = -1
    while l>-(shap[0]*shap[1]) and res>total*tol:
        indP = ind[l]
        l-=1
        indPl = int(indP/(shap[0]))

        indPc = int(indP%(shap[0]))

        interp_pos = (1-t)*pos_map[indPl,indPc,:]+t*final_pos[indPl,indPc,:]

        iopt = cent[0]
        jopt = cent[1]

        if interp_pos[0]>0:
            iopt = (interp_pos[0]/scale_coeffr)*cos(scale_coeffr*interp_pos[1]/(scale_coefft*interp_pos[0]))+cent[0]
            iopt = min(max(0,iopt),shap[0]-1)
            jopt = (interp_pos[0]/scale_coeffr)*sin(scale_coeffr*interp_pos[1]/(scale_coefft*interp_pos[0]))+cent[1]
            jopt = min(max(0,jopt),shap[1]-1)

        iopt1 = min(max(0,np.int(iopt)),shap[0]-1)
        jopt1 = min(max(0,np.int(jopt)),shap[0]-1)
        mass = (1-t)*im_ref[indPl,indPc]+t*im_proj[indPl,indPc]
        if iopt1==iopt and jopt1==jopt:
            psf_interp[iopt1,jopt1]+=mass

        else:
            iopt2 = min(max(0,np.int(iopt1)+1),shap[0]-1)
            jopt2 = min(max(0,np.int(jopt1)+1),shap[0]-1)
            if iopt==iopt1:
                d1 = abs(jopt-jopt1)**(-1)
                d2 = abs(jopt-jopt2)**(-1)
                d = d1+d2
                psf_interp[iopt1,jopt1]+=d1*mass/d
                psf_interp[iopt1,jopt2]+=d2*mass/d
            elif jopt==jopt1:
                d1 = abs(iopt-iopt1)**(-1)
                d2 = abs(iopt-iopt2)**(-1)
                d = d1+d2
                psf_interp[iopt1,jopt1]+=d1*mass/d
                psf_interp[iopt2,jopt1]+=d2*mass/d

            else:
                d1 = sqrt((iopt-iopt1)**2+(jopt-jopt1)**2)**(-1)
                d2 = sqrt((iopt-iopt1)**2+(jopt-jopt2)**2)**(-1)
                d3 = sqrt((iopt-iopt2)**2+(jopt-jopt1)**2)**(-1)
                d4 = sqrt((iopt-iopt2)**2+(jopt-jopt2)**2)**(-1)
                d = d1+d2+d3+d4
                psf_interp[iopt1,jopt1]+=d1*mass/d
                psf_interp[iopt1,jopt2]+=d2*mass/d
                psf_interp[iopt2,jopt1]+=d3*mass/d
                psf_interp[iopt2,jopt2]+=d4*mass/d

        res -= mass

    print abs(l),"/",shap[0]*shap[1]
    return psf_interp

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


#def transport_plan_approx(supp,P_stack,shap):




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
    nb_plans = P_stack.shape[-1]
    output = zeros((shap[0],shap[1],nb_plans))
    for i in range(0,nb_plans):
        output[:,:,i] = transport_plan_projections(P_stack[:,:,i],shap,supp,neighbors_graph,weights_neighbors,indices=[0])

    return output

def transport_plan_projections_field_marg_transpose(im_stack,shap,supp,neighbors_graph,weights_neighbors):
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
    return P_stack[supp[:,0],supp[:,1],:].dot(A)

def transport_plan_projections_flat_field_transpose(P_mat,supp,A,shap):
    temp_mat = P_mat.dot(transpose(A))
    P_stack = zeros((prod(shap),prod(shap),A.shape[0]))
    P_stack[supp[:,0],supp[:,1],:] = temp_mat
    return P_stack

def transport_plan_projections_flat_field_transpose_coeff(P_mat,P_stack,supp):
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

def transport_plan_projections_field_grad_mat(input_op):

    P_stack = input_op[0]
    shap = input_op[1]
    supp = input_op[2]
    neighbors_graph = input_op[3]
    weights_neighbors = input_op[4]
    spectrums = input_op[5]
    A = input_op[6]
    flux = input_op[7]
    sig = input_op[8]
    ker = input_op[9]
    D = input_op[10]
    ker_rot = input_op[11]

    star_est = transport_plan_projections_field(P_stack,shap,supp,neighbors_graph,weights_neighbors,spectrums,A,flux,sig,ker,D)
    P = transport_plan_projections_field_transpose(star_est,supp,neighbors_graph,weights_neighbors,spectrums,A,flux,sig,ker_rot,D)

    return P

def transport_plan_projections_field_gradient(obs_stars,P_est,supp,neighbors_graph,weights_neighbors,spectrums,A_est,flux,sig,ker,ker_rot,D):

    shap_obs = obs_stars.shape
    shap = (shap_obs[0]*D,shap_obs[1]*D)
    star_est = transport_plan_projections_field(P_est,shap,supp,neighbors_graph,weights_neighbors,spectrums,A_est,flux,sig,ker,D)
    res = obs_stars-star_est
    grad = -transport_plan_projections_field_transpose(res,supp,neighbors_graph,weights_neighbors,spectrums,A_est,flux,sig,ker_rot,D)

    return res,grad

def transport_plan_projections_field_coeff_grad_mat(input_op):

    A = input_op[0]
    P_stack = input_op[6]
    shap = input_op[1]
    supp = input_op[2]
    neighbors_graph = input_op[3]
    weights_neighbors = input_op[4]
    spectrums = input_op[5]
    flux = input_op[7]
    sig = input_op[8]
    ker = input_op[9]
    D = input_op[10]
    ker_rot = input_op[11]

    star_est = transport_plan_projections_field(P_stack,shap,supp,neighbors_graph,weights_neighbors,spectrums,A,flux,sig,ker,D)
    A_out = transport_plan_projections_field_coeff_transpose(star_est,supp,neighbors_graph,\
                                    weights_neighbors,spectrums,P_stack,flux,sig,ker_rot,D)

    return A_out


def transport_plan_projections_field_coeff_gradient(obs_stars,P_est,supp,neighbors_graph,weights_neighbors,spectrums,A_est,flux,sig,ker,ker_rot,D):

    shap_obs = obs_stars.shape
    shap = (shap_obs[0]*D,shap_obs[1]*D)
    star_est = transport_plan_projections_field(P_est,shap,supp,neighbors_graph,weights_neighbors,spectrums,A_est,flux,sig,ker,D)
    res = obs_stars-star_est
    grad = -transport_plan_projections_field_coeff_transpose(res,supp,neighbors_graph,\
                                                            weights_neighbors,spectrums,P_est,flux,sig,ker_rot,D)

    return res,grad


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


def pwr_mth_transpose(input):
    P = input[0]
    shap = input[1]
    supp = input[2]
    neighbors_graph = input[3]
    weights_neighbors = input[4]
    spectrum = input[5]

    return transport_plan_projections_transpose(transport_plan_projections(P,shap,\
    supp,neighbors_graph,weights_neighbors,spectrum).sum(axis=2),supp,neighbors_graph,weights_neighbors,spectrum=spectrum).sum(axis=2)

def transport_plan_noise_map(spectrum,supp,neighbors_graph,weights_neighbors,shap):
    from numpy.linalg import norm
    from numpy import add,repeat,transpose

    nb_proj = neighbors_graph.shape[3]
    P_out = zeros((shap[0]*shap[1],shap[0]*shap[1]))
    rep_spec = transpose(repeat(spectrum,4).reshape((nb_proj,4)))

    for j in range(0,supp.shape[0]):
        temp_im = zeros(shap)
        add.at(temp_im,(neighbors_graph[j,0,:,:],neighbors_graph[j,1,:,:]),rep_spec*weights_neighbors[j,:,:])
        """for i in range(0,nb_proj):
            if dist_neighbors[j,0,i]==0:
                i_im = int(neighbors_graph[j,0,i]/shap[0])
                j_im = int(neighbors_graph[j,0,i]%shap[0])
                temp_im[i_im,j_im]+=spectrum[i]
            else:
                for k in range(0,dist_neighbors.shape[1]):
                    i_im = int(neighbors_graph[j,k,i]/shap[0])
                    j_im = int(neighbors_graph[j,k,i]%shap[0])
                    temp_im[i_im,j_im]+=spectrum[i]*(dist_neighbors[j,k,i]**(-1))/sum(dist_neighbors[j,:,i]**(-1))"""
        P_out[supp[j,0],supp[j,1]] = norm(temp_im)

    return P_out

def noise_estimation(im_stack,shap,supp,neighbors_graph,weights_neighbors,spectrums,\
    A,flux,sig,ker_rot,D,stack_coeff,filters,opt_wvlth,ind_i,nb_comp = 50):
    from utils import correlated_noise_est
    P_stack = transport_plan_projections_field_transpose(im_stack,shap,supp,neighbors_graph,weights_neighbors,spectrums,A,flux,sig,ker_rot,D)

    # Transport plans noise maps
    shap_P = P_stack.shape
    plans_noise_map = zeros(shap_P)
    for i in range(0,shap_P[2]):
        plans_noise_map[supp[:,0],supp[:,1],i]+= correlated_noise_est(P_stack[supp[:,0],supp[:,1],i])

    nb_bands = spectrums.shape[0]
    wavelets_coeff,spec_coeff_dct,spec_coeff_wav = analysis_op_3(P_stack,stack_coeff,filters,opt_wvlth,shap,supp,neighbors_graph,ind_i,weights_neighbors,nb_bands,nb_comp = 50)

    wavelets_coeff_noise = zeros(wavelets_coeff.shape)
    spec_coeff_dct_noise = zeros(spec_coeff_dct.shape)
    spec_coeff_wav_noise = zeros(spec_coeff_wav.shape)

    """for i in range(0,nb_bands):
        spec_coeff_dct_noise[:,:,i]+="""



def transpose_check(shap,sig=3.):
    from numpy import zeros,where,transpose,array
    from numpy.random import random,randn
    from utils import diagonally_dominated_mat
    from numpy.linalg import norm

    P = diagonally_dominated_mat(shap,sig=sig,thresh_en=True)
    i,j = where(P>0)
    P[i,j] = randn(size(i))
    supp = transpose(array([i,j]))
    t = random(1)
    neighbors_graph,dist_neighbors,cent,coord_map,knn = full_displacement(shap,supp,t,pol_en=True,cent=None,theta_param=1,pol_mod=True,coord_map=None,knn=None)
    im = randn(shap[0],shap[1])

    im_proj = transport_plan_projections(P,shap,supp,neighbors_graph,dist_neighbors)
    P_proj = transport_plan_projections_transpose(im,supp,neighbors_graph,dist_neighbors)

    a1 = sum(im*im_proj)
    a2 = sum(P*P_proj)

    check = 100*abs(a1-a2)/(norm(im)*norm(P))

    print "Transposition accuracy: ",100 - check,"%"
    return None

def transpose_field_check(shap,nb_comp,wvl,flux,sig,ker,ker_rot,D,sig_supp=3.):
    from utils import diagonally_dominated_mat_stack
    from numpy.linalg import norm

    PX = diagonally_dominated_mat_stack(shap,nb_comp,sig=sig_supp,thresh_en=True)
    nb_im = size(flux)
    imY = randn(shap[0]/D,shap[1]/D,nb_im)
    i,j,k = where(PX>0)
    PX[i,j,k] = randn(size(i))
    i,j = where(PX[:,:,0]>0)
    supp = transpose(array([i,j]))
    t = (wvl-wvl.min())/(wvl.max()-wvl.min())
    neighbors_graph,dist_neighbors,cent,coord_map,knn = full_displacement(shap,supp,t,\
    pol_en=True,cent=None,theta_param=1,pol_mod=True,coord_map=None,knn=None)
    spectrums = abs(randn(size(wvl),size(flux)))
    A = randn(nb_comp,size(flux))

    imX = transport_plan_projections_field(PX,shap,supp,neighbors_graph,dist_neighbors,spectrums,A,flux,sig,ker,D)
    PY = transport_plan_projections_field_transpose(imY,supp,neighbors_graph,dist_neighbors,spectrums,A,flux,sig,ker_rot,D)

    a1 = sum(imX*imY)
    a2 = sum(PX*PY)
    print a1,a2
    check = 100*abs(a1-a2)/(norm(imY)*norm(PY))
    print "Transposition accuracy: ",100 - check,"%"
    return None

def transpose_field_check_coeff(shap,nb_comp,wvl,flux,sig,ker,ker_rot,D,sig_supp=3.):
    from utils import diagonally_dominated_mat_stack
    from numpy.linalg import norm

    AX = randn(nb_comp,size(flux))
    P = diagonally_dominated_mat_stack(shap,nb_comp,sig=sig_supp,thresh_en=True)
    nb_im = size(flux)
    imY = randn(shap[0]/D,shap[1]/D,nb_im)
    i,j,k = where(P>0)
    P[i,j,k] = randn(size(i))
    i,j = where(P[:,:,0]>0)
    supp = transpose(array([i,j]))
    t = (wvl-wvl.min())/(wvl.max()-wvl.min())
    neighbors_graph,dist_neighbors,cent,coord_map,knn = full_displacement(shap,supp,t,\
    pol_en=True,cent=None,theta_param=1,pol_mod=True,coord_map=None,knn=None)
    spectrums = abs(randn(size(wvl),size(flux)))

    AY = transport_plan_projections_field_coeff_transpose(imY,supp,neighbors_graph,dist_neighbors,spectrums,P,flux,sig,ker_rot,D)
    imX = transport_plan_projections_field(P,shap,supp,neighbors_graph,dist_neighbors,spectrums,AX,flux,sig,ker,D)

    a1 = sum(imX*imY)
    a2 = sum(AX*AY)
    print a1,a2
    check = 100*abs(a1-a2)/(norm(imY)*norm(AY))
    print "Transposition accuracy: ",100 - check,"%"
    return None




def wavelengths_separation_init(obs_psf,wavelengths,spectrum,supp_rad,thresh_en=False,coord_map=None,pol_en=False,cent=None,theta_param=1,pol_mod=False,knn=None,nsig=3.,sig=None,radial_cons=True,tol_deg=15.,opt_space=None,opt_wvlth = None):
    from isap import mr_trans_1d,mr_trans_2
    from utils import diagonally_dominated_mat,im_gauss_nois_est,radial_support
    from numpy import sqrt,newaxis
    nb_bands = size(spectrum)
    shap = obs_psf.shape
    print "Initializing the transport plan..."
    Px = diagonally_dominated_mat(shap,sig=supp_rad,thresh_en=thresh_en,coord_map=coord_map,pol_en=pol_en,cent=cent,theta_param=theta_param,pol_mod=pol_mod)
    Px = radial_support(Px,shap,cent=None,tol_deg=10.,coord_map=None)
    print "Done."
    t = (wavelengths-wavelengths.min())/(wavelengths.max()-wavelengths.min())
    supp = transpose(array(where(Px>0)))
    print "Creating the displacement architecture..."
    neighbors_graph,weights_neighbors,cent,coord_map,knn = full_displacement(shap,supp,t,pol_en=pol_en,cent=cent,theta_param=theta_param,pol_mod=pol_mod,coord_map=coord_map,knn=knn)
    print "Done."
    print "Noise map setting..."
    # Transport plan space noise
    if sig is None:
        sig,filt = im_gauss_nois_est(obs_psf)
    noise_map = transport_plan_noise_map(spectrum,supp,neighbors_graph,weights_neighbors,shap)*sig

    # Spectral direction noise
    spec_coeff_wav,info_file = isap.mr_trans_1d(spectrum,opt=opt_wvlth,path='../data/',save_info_en=True)
    spec_coeff_dct = dct(spectrum,norm='ortho')

    weights_spec = zeros((nb_bands,2))
    weights_spec[:,0] = sig*abs(spec_coeff_wav)
    weights_spec[:,1] = sig*abs(spec_coeff_dct)

    # Spatial noise
    output,filters = isap.mr_trans_2(obs_psf,filters=None,opt=opt_space)
    weights_space = zeros((obs_psf.shape[0],obs_psf.shape[1],filters.shape[2]-1))
    ones_mat = ones(obs_psf.shape)
    for i in range(0,filters.shape[2]-1):
        weights_space[:,:,i] = sig*sqrt(scisig.fftconvolve(ones_mat,filters[:,:,i]**2,mode='same')) # account for the sed while thresholding!
    print "Done."


    return Px,supp,neighbors_graph,weights_neighbors,t,noise_map,coord_map,knn,weights_spec,weights_space[...,newaxis]*abs(spectrum[newaxis,...]),filters[:,:,:-1],info_file

#def displacement_times_cost(obs_psf,spectrum)
def spectral_weights(siz,nb_bands,fwhm,thresh_vect_dct,thresh_vect_wav,cent=None,k=1):
    from numpy.linalg import norm
    from numpy import exp,newaxis
    if cent is None:
        cent = array(siz)/2
    ones_col = ones((siz[0],1))
    ones_line = ones((1,siz[1]))
    arange_line = arange(0,siz[1]).reshape((1,siz[1]))
    arange_col = arange(0,siz[0]).reshape((siz[0],1))
    i_coord = arange_col.dot(ones_line)
    j_coord = ones_col.dot(arange_line)

    coord = concatenate((i_coord.reshape((1,prod(siz))),j_coord.reshape((1,prod(siz)))),axis=0) - (cent.reshape((2,1))).dot(ones((1,prod(siz))))
    dist_mat = norm(coord,axis=0).reshape(siz)

    smooth_comp_weight = exp(-dist_mat/k*fwhm)
    smooth_comp_weight/= smooth_comp_weight.max()

    osci_comp_weight = 1-smooth_comp_weight


    return smooth_comp_weight[...,newaxis]*thresh_vect_wav[newaxis,...],osci_comp_weight[...,newaxis]*thresh_vect_dct[newaxis,...]


def wavelengths_separation_gradient(obs,Px,shap,supp,neighbors_graph,ind_i,dist_neighbors,spectrum):

    err = obs - transport_plan_projections(Px,shap,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum=spectrum).sum(axis=2)
    grad = -transport_plan_projections_transpose(err,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum=spectrum).sum(axis=2)

    return grad,err

def prox_pos(Px,mass=1,simplex_en=True):

    from optim_utils import pos_proj_mat
    from simplex_projection import euclidean_proj_simplex

    if simplex_en:
        Pout = euclidean_proj_simplex(Px,s=mass)
    else:
        Pout = pos_proj_mat(Px)

    return Pout

def columns_wise_simplex_proj(mat,mass=None):

    from simplex_projection import euclidean_proj_simplex
    nb_columns = mat.shape[1]
    mat_out = zeros(mat.shape)
    if mass is None:
        mass = max(0,((mat*(mat>=0)).sum(axis=0)).mean())
    if mass>0:
        for i in range(0,nb_columns):
            mat_out[:,i] = euclidean_proj_simplex(mat[:,i],s=mass)

    return mat_out

def prox_pos_dual_stack(Px_stack,ind_i,mass=1,simplex_en=True):
    Px_stack_out = zeros(Px_stack.shape)

    for i in range(0,size(ind_i)):
        Px_stack_out[:,:,ind_i[i]] = Px_stack[:,:,ind_i[i]] - sigma*prox_pos(Px_stack[:,:,ind_i[i]]/sigma,mass=mass,simplex_en=simplex_en)

    return Px_stack_out


def prox_sparsity(P,stack_wav,stack_dct,stack_spec_wav,mono_direct_weights,space_weights,dct_weights,spec_wav_weights,sigma,thresh_type=1):

    P_thresh = P-sigma*utils.thresholding(P/sigma,mono_direct_weights/sigma,thresh_type)
    stack_wav_thresh = stack_wav*0
    for i in range(0,stack_wav.shape[-1]):
        stack_wav_thresh[:,:,:,i] = stack_wav[:,:,:,i]-sigma*utils.thresholding_3D(stack_wav[:,:,:,i]/sigma,space_weights[:,:,:,i]/sigma,thresh_type)
    stack_dct_thresh = stack_dct-sigma*utils.thresholding_3D(stack_dct/sigma,dct_weights/sigma,thresh_type)
    stack_spec_wav_thresh = stack_spec_wav-sigma*utils.thresholding_3D(stack_spec_wav/sigma,spec_wav_weights/sigma,thresh_type)

    return P_thresh,stack_wav_thresh,stack_dct_thresh,stack_spec_wav_thresh

def prox_sparsity_2(stack_wav,stack_dct,stack_spec_wav,space_weights,dct_weights,spec_wav_weights,sigma,thresh_type=1):

    stack_wav_thresh = stack_wav*0
    for i in range(0,stack_wav.shape[-1]):
        stack_wav_thresh[:,:,:,i] = utils.thresholding_3D(stack_wav[:,:,:,i],space_weights[:,:,:,i]*sigma,thresh_type)
    stack_dct_thresh = utils.thresholding_3D(stack_dct,dct_weights*sigma,thresh_type)
    stack_spec_wav_thresh = utils.thresholding_3D(stack_spec_wav,spec_wav_weights*sigma,thresh_type)

    return stack_wav_thresh,stack_dct_thresh,stack_spec_wav_thresh


def analysis_op(Px,filters,opt_wvlth,shap,supp,neighbors_graph,ind_i,dist_neighbors,spectrum,nb_comp = 50):
    from numpy.linalg import svd

    mono_chrom = transport_plan_projections(Px,shap,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum=spectrum)

    # Spatial wavelet coeff
    shap1 = mono_chrom.shape
    shap2 = filters.shape
    wavelets_coeff = zeros((shap1[0],shap1[1],shap2[2],shap1[2]))
    for i in range(0,shap1[2]):
        temp,filters = isap.mr_trans_2(mono_chrom[:,:,i],filters=filters)
        wavelets_coeff[:,:,:,i] = temp

    spec_mono_chrom = mono_chrom.reshape((prod(shap1[0:2]),shap1[2]))
    U,s,V = svd(spec_mono_chrom, full_matrices=0)
    U = U[:,0:nb_comp]
    s = s[0:nb_comp]
    V = V[0:nb_comp,:]

    # Spectral DCT coeff
    spec_coeff_dct_V = dct(V,norm='ortho')
    spec_coeff_dct = U.dot(diag(s).dot(spec_coeff_dct_V))

    # Wavelets coeff
    spec_coeff_wav_V = zeros((nb_comp,shap1[2]))
    for i in range(0,nb_comp):
        spec_coeff_wav_V[i,:],info_file = isap.mr_trans_1d(V[i,:],opt=opt_wvlth,path='../data/',save_info_en=False)
    spec_coeff_wav = U.dot(diag(s).dot(spec_coeff_wav_V))

    return


def analysis_op_2(mono_chrom,filters,opt_wvlth,shap,spectrum,nb_comp = 50):
    from numpy.linalg import svd

    # Spatial wavelet coeff
    shap1 = mono_chrom.shape
    shap2 = filters.shape
    wavelets_coeff = zeros((shap1[0],shap1[1],shap2[2],shap1[2]))
    for i in range(0,shap1[2]):
        temp,filters = isap.mr_trans_2(mono_chrom[:,:,i],filters=filters)
        wavelets_coeff[:,:,:,i] = temp

    spec_mono_chrom = mono_chrom.reshape((prod(shap1[0:2]),shap1[2]))
    U,s,V = svd(spec_mono_chrom, full_matrices=0)
    U = U[:,0:nb_comp]
    s = s[0:nb_comp]
    V = V[0:nb_comp,:]

    # Spectral DCT coeff
    spec_coeff_dct_V = dct(V,norm='ortho')
    spec_coeff_dct = U.dot(diag(s).dot(spec_coeff_dct_V))

    # Wavelets coeff
    spec_coeff_wav_V = zeros((nb_comp,shap1[2]))
    for i in range(0,nb_comp):
        spec_coeff_wav_V[i,:],info_file = isap.mr_trans_1d(V[i,:],opt=opt_wvlth,path='../data/',save_info_en=False)
    spec_coeff_wav = U.dot(diag(s).dot(spec_coeff_wav_V))

    return wavelets_coeff,spec_coeff_dct.reshape((shap1[0],shap1[1],size(spectrum))),spec_coeff_wav.reshape((shap1[0],shap1[1],size(spectrum)))

def analysis_op_3(Px_stack,stack_coeff,filters,opt_wvlth,shap,supp,neighbors_graph,ind_i,dist_neighbors,nb_bands,nb_comp = 50):
    from numpy import multiply

    Px = multiply(Px_stack,stack_coeff).sum(axis=2)

    return analysis_op(Px,filters,opt_wvlth,shap,supp,neighbors_graph,ind_i,dist_neighbors,ones((nb_bands,)),nb_comp = 50)

def analysis_op_transpose(wavelets_coeff,spec_coeff_dct,spec_coeff_wav,filters_rot,info_filename,shap,supp,neighbors_graph,ind_i,dist_neighbors,spectrum,nb_comp = 50):
    from numpy.linalg import svd

    shap1 = wavelets_coeff.shape
    # Spatial wavelet transp
    recons_stack = zeros((shap[0],shap[1],shap1[-1]))
    for i in range(0,shap1[-1]):
        recons_stack[:,:,i] = isap.mr_transf_transp(wavelets_coeff[:,:,:,i],filters_rot)

    # Spectral DCT coeff
    U,s,V = svd(spec_coeff_dct.reshape((prod(shap),size(spectrum))), full_matrices=0)
    U = U[:,0:nb_comp]
    s = s[0:nb_comp]
    V = V[0:nb_comp,:]

    spec_mono_chrom_dct_V = idct(V,norm='ortho')
    spec_mono_chrom_dct = U.dot(diag(s).dot(spec_mono_chrom_dct_V))
    mono_chrom_dct = spec_mono_chrom_dct.reshape((shap[0],shap[1],shap1[-1]))

    # Wavelets coeff
    U,s,V = svd(spec_coeff_wav.reshape((prod(shap),size(spectrum))), full_matrices=0)
    U = U[:,0:nb_comp]
    s = s[0:nb_comp]
    V = V[0:nb_comp,:]

    spec_mono_chrom_wav_V = zeros((nb_comp,shap1[3]))
    for i in range(0,nb_comp):
        spec_mono_chrom_wav_V[i,:] = isap.mr_recons_1d(V[i,:],info_filename)
    spec_mono_chrom_wav = U.dot(diag(s).dot(spec_mono_chrom_wav_V))
    mono_chrom_wav = spec_mono_chrom_wav.reshape((shap[0],shap[1],shap1[-1]))

    P = transport_plan_projections_transpose_2(recons_stack+mono_chrom_dct+mono_chrom_wav,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum).sum(axis=2)

    return P

def analysis_op_transpose_2(wavelets_coeff,spec_coeff_dct,spec_coeff_wav,filters_rot,info_filename,shap,spectrum,nb_comp = 50):
    from numpy.linalg import svd

    shap1 = wavelets_coeff.shape
    # Spatial wavelet transp
    recons_stack = zeros((shap[0],shap[1],shap1[-1]))
    for i in range(0,shap1[-1]):
        recons_stack[:,:,i] = isap.mr_transf_transp(wavelets_coeff[:,:,:,i],filters_rot)

    # Spectral DCT coeff
    U,s,V = svd(spec_coeff_dct.reshape((prod(shap),size(spectrum))), full_matrices=0)
    U = U[:,0:nb_comp]
    s = s[0:nb_comp]
    V = V[0:nb_comp,:]

    spec_mono_chrom_dct_V = idct(V,norm='ortho')
    spec_mono_chrom_dct = U.dot(diag(s).dot(spec_mono_chrom_dct_V))
    mono_chrom_dct = spec_mono_chrom_dct.reshape((shap[0],shap[1],shap1[-1]))

    # Wavelets coeff
    U,s,V = svd(spec_coeff_wav.reshape((prod(shap),size(spectrum))), full_matrices=0)
    U = U[:,0:nb_comp]
    s = s[0:nb_comp]
    V = V[0:nb_comp,:]

    spec_mono_chrom_wav_V = zeros((nb_comp,shap1[3]))
    for i in range(0,nb_comp):
        spec_mono_chrom_wav_V[i,:] = isap.mr_recons_1d(V[i,:],info_filename)
    spec_mono_chrom_wav = U.dot(diag(s).dot(spec_mono_chrom_wav_V))
    mono_chrom_wav = spec_mono_chrom_wav.reshape((shap[0],shap[1],shap1[-1]))

    return recons_stack+mono_chrom_dct+mono_chrom_wav

def analysis_op_transpose_3(stack_coeff,wavelets_coeff,spec_coeff_dct,spec_coeff_wav,filters_rot,info_filename,shap,supp,neighbors_graph,ind_i,dist_neighbors,nb_bands,nb_comp = 50):
    from numpy import newaxis

    P = analysis_op_transpose(wavelets_coeff,spec_coeff_dct,spec_coeff_wav,filters_rot,info_filename,shap,supp,neighbors_graph,ind_i,dist_neighbors,ones((nb_bands,)),nb_comp = 50)

    return P[...,newaxis]*stack_coeff[newaxis,...]


def wavelengths_separation_cv(obs_psf,wavelengths,spectrum,supp_rad,noise_map,weights_spec,weights_space,filters,info_file,opt_wvlth,tol = 0.01,nb_iter=500,knn=None,nsig=3.,Px=None,supp=None,neighbors_graph=None,dist_neighbors=None,t=None,spec_rad=None,pwr_meth_en=False,line_search=True,rand_frac = 0.2,simplex_en=False,mass=1,ksig=3,fwhm=3,nb_rw=1,eps=1e-15):

    from utils import diagonally_dominated_mat,im_gauss_nois_est,thresholding
    from numpy.linalg import norm
    from numpy import copy
    from numpy.random import choice
    from optim_utils import pow_meth

    # Init
    shap = obs_psf.shape
    if Px is None:
        print "Initializing the transport plan..."
        Px = diagonally_dominated_mat(shap,sig=supp_rad,thresh_en=thresh_en,coord_map=coord_map,pol_en=pol_en,cent=cent,theta_param=theta_param,pol_mod=pol_mod)
        print "Done."


    if t is None:
        t = (wavelengths-wavelengths.min())/(wavelengths.max()-wavelengths.min())
        if supp is None:
            supp = transpose(array(where(Px>0)))

    if neighbors_graph is None:
        print "Creating the displacement architecture..."
        shap = obs_psf.shape
        neighbors_graph,dist_neighbors,cent,coord_map,knn = full_displacement(shap,supp,t,pol_en=pol_en,cent=cent,theta_param=theta_param,pol_mod=pol_mod,coord_map=coord_map,knn=knn)
        print "Done."

    # Hyper parameters
    sigm = (1+(sum(abs(filters))**2+2)*prod(shap))**(-1)
    tau = None
    if (pwr_meth_en or spec_rad is not None) and not line_search:
        input = list()
        input.append(Px)
        input.append(shap)
        input.append(supp)
        input.append(neighbors_graph)
        input.append(dist_neighbors)
        input.append(spectrum)
        vect,spec_rad = pow_meth(pwr_mth_transpose,input,Px.shape,tol=0.5,ainit=None,nb_iter_max=30)
        print "Spectral radius: ",spec_rad
        tau = (spec_rad/2+1)**(-1)
        sigm = tau

    #print sigm,tau
    #
    rho = 0.9

    # Dual variables
    filters_rot = filters*0
    for i in range(0,filters.shape[-1]):
        filters_rot[:,:,i] = rot90(filters[:,:,i],2)
    Px_dual = Px*0
    WPx_dual = ones(Px.shape)

    space_sparse = zeros((shap[0],shap[1],filters.shape[-1],size(spectrum)))
    Wspace_sparse = ones((shap[0],shap[1],filters.shape[-1],size(spectrum)))

    spec_dct_sparse = zeros((shap[0],shap[1],size(spectrum)))
    Wspec_dct_sparse = ones((shap[0],shap[1],size(spectrum)))

    spec_smooth_sparse = zeros((shap[0],shap[1],size(spectrum)))
    Wspec_smooth_sparse = ones((shap[0],shap[1],size(spectrum)))


    # Spectral thresholds
    spec_wav_weights,spec_dct_weights = spectral_weights(shap,size(spectrum),fwhm,weights_spec[:,1],weights_spec[:,0],cent=None,k=2)


    rel_var = 100
    for j in range(0,nb_rw+1):
        i=0
        while i<nb_iter[j] and rel_var>tol:
            #ind_i = choice(supp.shape[0],nb_samp)
            #print ind_i
            ind_i = arange(0,supp.shape[0])
            print "iter ",i+1,"/",nb_iter[j]
            # Primal step
            grad,err = wavelengths_separation_gradient(obs_psf,Px,shap,supp,neighbors_graph,ind_i,dist_neighbors,spectrum)
            print "MSE: ",norm(err)**2
            print "Cost components: data attachment,", 0.5*norm(err)**2," dual components: ",sum(abs(space_sparse*ksig*weights_space*Wspace_sparse))+sum(abs(spec_dct_sparse*ksig*spec_dct_weights*Wspec_dct_sparse))+sum(abs(spec_smooth_sparse*ksig*spec_wav_weights*Wspec_smooth_sparse))+sum(abs(Px_dual*ksig*noise_map*WPx_dual))
            #print "Cost function: ", sum(abs(Px*ksig*noise_map*WPx_dual))
            temp1 = analysis_op_transpose(space_sparse,spec_dct_sparse,spec_smooth_sparse,filters_rot,info_file,shap,supp,neighbors_graph,ind_i,dist_neighbors,spectrum,nb_comp = 50)+Px_dual
            if line_search:
                tGrad = transport_plan_projections(grad+temp1,shap,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum=spectrum).sum(axis=2)
                tau = -sum(err*tGrad)/sum(tGrad**2)
                sigm = tau
            Px_temp = prox_pos(Px-tau*(grad+temp1),mass=mass,simplex_en=simplex_en)
            #Px_temp = Px-tau*(grad+temp1)


            # Dual step
            wavelets_coeff_temp,spec_coeff_dct_temp,spec_coeff_wav_temp = analysis_op(2*Px_temp-Px,filters,opt_wvlth,shap,supp,neighbors_graph,ind_i,dist_neighbors,spectrum,nb_comp = 50)
            P_thresh,stack_wav_thresh,stack_dct_thresh,stack_spec_wav_thresh = prox_sparsity(Px_dual+sigm*(Px_temp-Px),space_sparse+sigm*wavelets_coeff_temp,spec_dct_sparse+sigm*spec_coeff_dct_temp,spec_smooth_sparse+sigm*spec_coeff_wav_temp,ksig*noise_map*WPx_dual,ksig*weights_space*Wspace_sparse,ksig*spec_dct_weights*Wspec_dct_sparse,ksig*spec_wav_weights*Wspec_smooth_sparse,sigm,thresh_type=1)

            # Variables update
            Px = rho*Px_temp+(1-rho)*Px
            Px_dual = rho*P_thresh+(1-rho)*Px_dual
            space_sparse = rho*stack_wav_thresh+(1-rho)*space_sparse
            spec_dct_sparse = rho*stack_dct_thresh+(1-rho)*spec_dct_sparse
            spec_smooth_sparse = rho*stack_spec_wav_thresh+(1-rho)*spec_smooth_sparse
            i+=1
        # Reweighting

        if j<nb_rw:
            WPx_dual = (1+abs(Px)/(ksig*noise_map+eps))**(-1)
            wavelets_coeff_temp,spec_coeff_dct_temp,spec_coeff_wav_temp = analysis_op(Px,filters,opt_wvlth,shap,supp,neighbors_graph,ind_i,dist_neighbors,spectrum,nb_comp = 50)
            Wspace_sparse = (1+abs(wavelets_coeff_temp)/(ksig*weights_space+eps))**(-1)
            Wspec_dct_sparse = (1+abs(spec_coeff_dct_temp)/(ksig*spec_dct_weights+eps))**(-1)
            Wspec_smooth_sparse = (1+abs(spec_coeff_wav_temp)/(ksig*spec_wav_weights+eps))**(-1)
            print "Reweighting ",j+1,"/",nb_rw

    return transport_plan_projections(Px,shap,supp,neighbors_graph,dist_neighbors),Px,err,spec_rad


def wavelengths_separation_gfb(obs_psf,wavelengths,spectrum,supp_rad,noise_map,weights_spec,weights_space,filters,info_file,opt_wvlth,mu=0.5,Gamma=100,tol = 0.01,nb_iter=500,knn=None,nsig=3.,Px=None,supp=None,neighbors_graph=None,dist_neighbors=None,t=None,spec_rad=None,pwr_meth_en=False,grad_en=False,line_search=True,rand_frac = 0.2,simplex_en=False,mass=1,ksig=3,fwhm=3,nb_rw=1,eps=1e-15):
    from utils import diagonally_dominated_mat,im_gauss_nois_est,thresholding
    from numpy.linalg import norm
    from optim_utils import pos_proj_mat,prox_coeff_sum,pow_meth
    from simplex_projection import euclidean_proj_simplex
    from numpy import copy,multiply
    from numpy.random import choice

    nb_bands = size(spectrum)

    # Hyper-parameters
    gamma = 1
    lambd = 1
    w1 = 0.5 # Positivity or positive simplex constraint
    w2 = 1-w1 # Sparsity

    # Init
    shap = obs_psf.shape
    if Px is None:
        print "Initializing the transport plan..."
        Px = diagonally_dominated_mat(shap,sig=supp_rad,thresh_en=thresh_en,coord_map=coord_map,pol_en=pol_en,cent=cent,theta_param=theta_param,pol_mod=pol_mod)
        print "Done."

    space_sparse = zeros((shap[0],shap[1],filters.shape[-1],size(spectrum)))
    Wspace_sparse = ones((shap[0],shap[1],filters.shape[-1],size(spectrum)))

    spec_dct_sparse = zeros((shap[0],shap[1],size(spectrum)))
    Wspec_dct_sparse = ones((shap[0],shap[1],size(spectrum)))

    spec_smooth_sparse = zeros((shap[0],shap[1],size(spectrum)))
    Wspec_smooth_sparse = ones((shap[0],shap[1],size(spectrum)))

    # First auxiliary variables and related weights
    Pz1 = w1*Px
    filters_rot = filters*0
    for i in range(0,filters.shape[-1]):
        filters_rot[:,:,i] = rot90(filters[:,:,i],2)

    # Spectral thresholds
    spec_wav_weights,spec_dct_weights = spectral_weights(shap,size(spectrum),fwhm,weights_spec[:,1],weights_spec[:,0],cent=None,k=2)


    # Second auxiliary variables
    Pz2 = w2*Px
    WPz2 = ones(Px.shape)

    if t is None:
        t = (wavelengths-wavelengths.min())/(wavelengths.max()-wavelengths.min())
        if supp is None:
            supp = transpose(array(where(Px>0)))
    if neighbors_graph is None:
        print "Creating the displacement architecture..."
        shap = obs_psf.shape
        neighbors_graph,dist_neighbors,cent,coord_map,knn = full_displacement(shap,supp,t,pol_en=pol_en,cent=cent,theta_param=theta_param,pol_mod=pol_mod,coord_map=coord_map,knn=knn)
        print "Done."
    i=0
    rel_var = 100

    # Noise estimation


    if noise_map is None and grad_en is False:
        print "Noise map setting..."
        sig,filt = im_gauss_nois_est(obs_psf)
        Gamma *= sig
        noise_map = transport_plan_noise_map(spectrum,supp,neighbors_graph,dist_neighbors,shap)*sig
        print "Done."

    # Spectral radius estmation

    if (pwr_meth_en or spec_rad is not None) and not line_search and not mult_update:
        if spec_rad is None:
            input = list()
            input.append(Px)
            input.append(shap)
            input.append(supp)
            input.append(neighbors_graph)
            input.append(dist_neighbors)
            input.append(spectrum)
            vect,spec_rad = pow_meth(pwr_mth_transpose,input,Px.shape,tol=0.5,ainit=None,nb_iter_max=30)
            print "Spectral radius: ",spec_rad
        mu/=(spec_rad*(1+Gamma))
    else:
        mu/= size(obs_psf)*sum(spectrum**2)


    print "Main loop:"

    nb_samp = int(supp.shape[0]*rand_frac)
    while i<nb_iter and rel_var>tol:
        #ind_i = choice(supp.shape[0],nb_samp)
        #print ind_i
        ind_i = arange(0,supp.shape[0])
        print "iter ",i+1,"/",nb_iter

        mono_comp_est = transport_plan_projections(Px,shap,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:])
        cur_est = multiply(mono_comp_est,spectrum).sum(axis=2)
        wav_const_comp = analysis_op_transpose_2(space_sparse,spec_dct_sparse,spec_smooth_sparse,filters_rot,info_file,shap,spectrum,nb_comp = 50)

        err_1 = cur_est-obs_psf
        err_2 = mono_comp_est - wav_const_comp
        sum_err_1 = sum(err_1**2)
        sum_err_2 = Gamma*sum(err_2**2)
        print "Error: ", sum_err_1
        print "Quadratic part of the cost: ",sum_err_1+sum_err_2
        print "L1 norms:"," transport plan: ",sum(abs(Px*WPz2*noise_map*ksig))," Spatial smoothness: ",sum(abs(space_sparse*ksig*weights_space*Wspace_sparse))
        print "Spectral regularity: ",sum(abs(spec_dct_sparse*ksig*spec_dct_weights*Wspec_dct_sparse))+sum(abs(spec_smooth_sparse*ksig*spec_wav_weights*Wspec_smooth_sparse))
        print "Constraint transfer: ", (1 - norm(err_2)**2/max(norm(mono_comp_est)**2,norm(wav_const_comp)**2))*100,"%"
        Pxold = copy(Px)


        grad_comp_1 = transport_plan_projections_transpose(err_1,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum).sum(axis=2)+transport_plan_projections_transpose_2(Gamma*err_2,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],ones((nb_bands,))).sum(axis=2)
        grad_wavelets_coeff,grad_spec_coeff_dct,grad_spec_coeff_wav = analysis_op_2(-Gamma*err_2,filters,opt_wvlth,shap,spectrum,nb_comp = 50)

        if line_search:
            comp_tgrad_1 = -transport_plan_projections(grad_comp_1,shap,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:])
            tGrad_1 = multiply(comp_tgrad_1,spectrum).sum(axis=2)
            diff_tGrad = comp_tgrad_1-analysis_op_transpose_2(grad_wavelets_coeff,grad_spec_coeff_dct,grad_spec_coeff_wav,filters_rot,info_file,shap,spectrum,nb_comp = 50)
            mu = (sum(-err_1*tGrad_1)-Gamma*sum(err_2*diff_tGrad))/(sum(tGrad_1**2)+Gamma*sum(diff_tGrad**2))

        if grad_en:
            Px -= mu*grad_comp_1
            space_sparse -= mu*grad_wavelets_coeff
            spec_dct_sparse -= mu*grad_spec_coeff_dct
            spec_smooth_sparse -= mu*grad_spec_coeff_wav
        else:
            # Positivity/simplex constraint
            temp = 2*Px - Pz1 - mu*gamma*grad_comp_1
            Pz1 += lambd*prox_pos(temp,mass=mass,simplex_en=simplex_en)

            # Wavelet sparsity constraint
            temp1 = space_sparse - mu*gamma*grad_wavelets_coeff
            temp2 = spec_dct_sparse - mu*gamma*grad_spec_coeff_dct
            temp3 = spec_smooth_sparse - mu*gamma*grad_spec_coeff_wav

            temp1,temp2,temp3 = prox_sparsity_2(temp1,temp2,temp3,ksig*weights_space*Wspace_sparse,ksig*spec_dct_weights*Wspec_dct_sparse,ksig*spec_wav_weights*Wspec_smooth_sparse,mu*gamma,thresh_type=1)

            space_sparse += lambd*temp1
            spec_dct_sparse += lambd*temp2
            spec_smooth_sparse += lambd*temp3

            # Sparsity constraint
            temp = 2*Px - Pz2 - mu*gamma*grad_comp_1
            Pz2 += lambd*(thresholding(temp,WPz2*noise_map*ksig*mu*gamma/w2,thresh_type=1)-Px)

            # Main variable update
            Px = w1*Pz1+w2*Pz2

        rel_var = 100*sum((Px-Pxold)**2)/sum(Px**2)
        i+=1
    print "Done."

    return transport_plan_projections(Px,shap,supp,neighbors_graph,dist_neighbors),Px,err_1,spec_rad




def wavelengths_separation(obs_psf,wavelengths,spectrum,supp_rad,mu=0.5,tol = 0.01,nb_iter=500,thresh_en=False,coord_map=None,pol_en=False,cent=None,theta_param=1,pol_mod=False,knn=None,nsig=3.,Px=None,supp=None,neighbors_graph=None,dist_neighbors=None,t=None,noise_map=None,grad_en=True,spec_rad=None,pwr_meth_en=False,line_search=True,mult_update=True,rand_frac = 0.2,simplex_en=False,mass=1):

    from utils import diagonally_dominated_mat,im_gauss_nois_est,thresholding
    from numpy.linalg import norm
    from optim_utils import pos_proj_mat,prox_coeff_sum,pow_meth
    from simplex_projection import euclidean_proj_simplex
    from numpy import copy
    from numpy.random import choice


    if mult_update:
        grad_en = True
        spec_rad_en = False
        obs_psf_pos = pos_proj_mat(obs_psf)
        tObs = transport_plan_projections_transpose(obs_psf_pos,supp,neighbors_graph,dist_neighbors,spectrum).sum(axis=2)

    # Hyper-parameters
    gamma = 1
    lambd = 1
    w1 = 0.5 # Positivity or positive simplex constraint
    w2 = 1-w1 # Sparsity

    # Init
    shap = obs_psf.shape
    if Px is None:
        print "Initializing the transport plan..."
        Px = diagonally_dominated_mat(shap,sig=supp_rad,thresh_en=thresh_en,coord_map=coord_map,pol_en=pol_en,cent=cent,theta_param=theta_param,pol_mod=pol_mod)
        print "Done."
    Pz1 = w1*Px
    Pz2 = w2*Px

    if t is None:
        t = (wavelengths-wavelengths.min())/(wavelengths.max()-wavelengths.min())
    if supp is None:
        supp = transpose(array(where(Px>0)))
    if neighbors_graph is None:
        print "Creating the displacement architecture..."
        shap = obs_psf.shape
        neighbors_graph,dist_neighbors,cent,coord_map,knn = full_displacement(shap,supp,t,pol_en=pol_en,cent=cent,theta_param=theta_param,pol_mod=pol_mod,coord_map=coord_map,knn=knn)
        print "Done."
    i=0
    rel_var = 100

    # Noise estimation


    if noise_map is None and grad_en is False:
        print "Noise map setting..."
        sig,filt = im_gauss_nois_est(obs_psf)
        noise_map = transport_plan_noise_map(spectrum,supp,neighbors_graph,dist_neighbors,shap)*sig
        print "Done."

    # Spectral radius estmation

    if (pwr_meth_en or spec_rad is not None) and not line_search and not mult_update:
        if spec_rad is None:
            input = list()
            input.append(Px)
            input.append(shap)
            input.append(supp)
            input.append(neighbors_graph)
            input.append(dist_neighbors)
            input.append(spectrum)
            vect,spec_rad = pow_meth(pwr_mth_transpose,input,Px.shape,tol=0.5,ainit=None,nb_iter_max=30)
            print "Spectral radius: ",spec_rad
        mu/=spec_rad
    else:
        mu/= size(obs_psf)*sum(spectrum**2)


    print "Main loop:"

    nb_samp = int(supp.shape[0]*rand_frac)
    while i<nb_iter and rel_var>tol:
        #ind_i = choice(supp.shape[0],nb_samp)
        #print ind_i
        ind_i = arange(0,supp.shape[0])
        print "iter ",i+1,"/",nb_iter
        err = obs_psf - transport_plan_projections(Px,shap,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum=spectrum).sum(axis=2)
        print "Error: ",norm(err)**2
        print "Cost function: ",0.5*norm(err)**2 + sum(abs(Px*noise_map*nsig))
        Pxold = copy(Px)

        if mult_update:
            temp = transport_plan_projections_transpose(transport_plan_projections(Px,shap,supp[ind_i,:],\
            neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum=spectrum).sum(axis=2),supp,neighbors_graph,dist_neighbors,spectrum).sum(axis=2)
            ind = where(temp[supp[:,0],supp[:,1]]>0)
            Px[supp[ind_i,0],supp[ind_i,1]]*= tObs[supp[ind_i,0],supp[ind_i,1]]/temp[supp[ind_i,0],supp[ind_i,1]]
        else:
            grad = -transport_plan_projections_transpose(err,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum).sum(axis=2)
            if line_search:
                tGrad = transport_plan_projections(grad,shap,supp[ind_i,:],neighbors_graph[ind_i,:,:,:],dist_neighbors[ind_i,:,:],spectrum=spectrum).sum(axis=2)
                mu = -sum(err*tGrad)/sum(tGrad**2)
            if grad_en:
                Px -= mu*gamma*grad
                Px = pos_proj_mat(Px)
            else:
                # Positivity constraint
                temp = 2*Px - Pz1 - mu*gamma*grad
                if simplex_en:
                    Pz1 += lambd*(euclidean_proj_simplex(temp.reshape(((shap[0]*shap[1])**2,)),s=mass).reshape((shap[0]*shap[1],shap[0]*shap[1]))-Px)
                else:
                    Pz1 += lambd*(pos_proj_mat(temp)-Px)


                # Sparsity constraint
                temp = 2*Px - Pz2 - mu*gamma*grad
                Pz2 += lambd*(thresholding(temp,noise_map*nsig*mu*gamma/w2,thresh_type=1)-Px)

                Pxold = copy(Px)
                #Px = w1*Pz1+w2*Pz2+w3*Pz3
                Px = w1*Pz1+w2*Pz2
        rel_var = 100*norm(Px-Pxold)/norm(Px)
        i+=1
    print "Done."

    return transport_plan_projections(Px,shap,supp,neighbors_graph,dist_neighbors),Px,err,spec_rad

def bar_1d_2d(bar,shap,pol_en=True,scale_coeffr=1,scale_coefft=1,tol=0.001,cent=None,target_pos=None):

    im_out = zeros(shap)
    ind = argsort(bar[0,:])
    total = bar[0,:].sum()
    res = total
    ind = utils.argsort(bar[0,:])
    psf_interp = zeros(shap)
    l = -1
    while l>-(shap[0]*shap[1]) and res>total*tol:
        indP = ind[l]
        l-=1
        indPl = int(indP/(shap[0]))
        indPc = int(indP%(shap[0]))
        interp_pos = array([bar[1,ind[l]],bar[2,ind[l]]])
        mass = bar[0,ind[l]]
        if pol_en:
            iopt = cent[0]
            jopt = cent[1]

            if interp_pos[0]>0:
                iopt = (interp_pos[0]/scale_coeffr)*cos(scale_coeffr*interp_pos[1]/(scale_coefft*interp_pos[0]))+cent[0]
                iopt = min(max(0,iopt),shap[0]-1)
                jopt = (interp_pos[0]/scale_coeffr)*sin(scale_coeffr*interp_pos[1]/(scale_coefft*interp_pos[0]))+cent[1]
                jopt = min(max(0,jopt),shap[1]-1)

            #print np.int(iopt),np.int(jopt)
            iopt1 = min(max(0,np.int(iopt)),shap[0]-1)
            jopt1 = min(max(0,np.int(jopt)),shap[0]-1)
            if iopt1==iopt and jopt1==jopt:
                psf_interp[iopt1,jopt1]+=mass

            else:
                iopt2 = min(max(0,np.int(iopt1)+1),shap[0]-1)
                jopt2 = min(max(0,np.int(jopt1)+1),shap[0]-1)
                if iopt==iopt1:
                    d1 = abs(jopt-jopt1)**(-1)
                    d2 = abs(jopt-jopt2)**(-1)
                    d = d1+d2
                    psf_interp[iopt1,jopt1]+=d1*mass/d
                    psf_interp[iopt1,jopt2]+=d2*mass/d
                elif jopt==jopt1:
                    d1 = abs(iopt-iopt1)**(-1)
                    d2 = abs(iopt-iopt2)**(-1)
                    d = d1+d2
                    psf_interp[iopt1,jopt1]+=d1*mass/d
                    psf_interp[iopt2,jopt1]+=d2*mass/d

                else:
                    d1 = sqrt((iopt-iopt1)**2+(jopt-jopt1)**2)**(-1)
                    d2 = sqrt((iopt-iopt1)**2+(jopt-jopt2)**2)**(-1)
                    d3 = sqrt((iopt-iopt2)**2+(jopt-jopt1)**2)**(-1)
                    d4 = sqrt((iopt-iopt2)**2+(jopt-jopt2)**2)**(-1)
                    d = d1+d2+d3+d4
                    psf_interp[iopt1,jopt1]+=d1*mass/d
                    psf_interp[iopt1,jopt2]+=d2*mass/d
                    psf_interp[iopt2,jopt1]+=d3*mass/d
                    psf_interp[iopt2,jopt2]+=d4*mass/d
        else:
            ind_im = interp_pos+array([shap[1],shap[0]])/2-target_pos

            ind_im[0]= np.int(round(min(max(0,ind_im[0]),shap[1]-1)))
            ind_im[1]= np.int(round(min(max(0,ind_im[1]),shap[0]-1)))
            psf_interp[ind_im[1],ind_im[0]]+= mass

        res -= mass
    #print abs(l),"/",shap[0]*shap[1]
    return psf_interp

def bar_1d_2d_bis(bar_in,shap,cent,pol_en=True,log_param=None,tol=0.000000000000000000001,target_pos=None,tol_num=2.0**(-62)):
    from numpy import exp,argsort
    bar = copy(bar_in)

    total = bar[2,:].sum()
    res = total
    ind = argsort(bar[2,:])
    psf_interp = zeros(shap)
    l = -1
    while l>-(bar.shape[1]) and res>total*tol:

        interp_pos = array([bar[0,ind[l]],bar[1,ind[l]]])
        mass = bar[2,ind[l]]
        iopt = None
        jopt = None
        """if pol_en:
            iopt = cent[0]
            jopt = cent[1]

            if bar[1,ind[l]]>0:
                iopt = bar[1,ind[l]]*cos(bar[2,ind[l]]/bar[1,ind[l]])+cent[0]
                iopt = min(max(0,iopt),shap[0]-1)
                jopt = bar[1,ind[l]]*sin(bar[2,ind[l]]/bar[1,ind[l]])+cent[1]
                jopt = min(max(0,jopt),shap[1]-1)
        else:"""
        iopt = bar[0,ind[l]]
        jopt = bar[1,ind[l]]

        #print iopt,jopt
        iopt1 = min(max(0,np.int(iopt)),shap[0]-1)
        jopt1 = min(max(0,np.int(jopt)),shap[1]-1)
        if abs(iopt1-iopt)<tol_num and abs(jopt1-jopt)<tol_num:
            psf_interp[iopt1,jopt1]+=mass

        else:
            iopt2 = min(max(0,np.int(iopt1)+1),shap[0]-1)
            jopt2 = min(max(0,np.int(jopt1)+1),shap[1]-1)
            if abs(iopt-iopt1)<tol_num:
                d1 = abs(jopt-jopt1)**(-1)
                d2 = abs(jopt-jopt2)**(-1)
                d = d1+d2
                psf_interp[iopt1,jopt1]+=d1*mass/d
                psf_interp[iopt1,jopt2]+=d2*mass/d
            elif abs(jopt-jopt1)<tol_num:
                d1 = abs(iopt-iopt1)**(-1)
                d2 = abs(iopt-iopt2)**(-1)
                d = d1+d2
                psf_interp[iopt1,jopt1]+=d1*mass/d
                psf_interp[iopt2,jopt1]+=d2*mass/d

            else:
                d1 = sqrt((iopt-iopt1)**2+(jopt-jopt1)**2)**(-1)
                d2 = sqrt((iopt-iopt1)**2+(jopt-jopt2)**2)**(-1)
                d3 = sqrt((iopt-iopt2)**2+(jopt-jopt1)**2)**(-1)
                d4 = sqrt((iopt-iopt2)**2+(jopt-jopt2)**2)**(-1)
                d = d1+d2+d3+d4
                psf_interp[iopt1,jopt1]+=d1*mass/d
                psf_interp[iopt1,jopt2]+=d2*mass/d
                psf_interp[iopt2,jopt1]+=d3*mass/d
                psf_interp[iopt2,jopt2]+=d4*mass/d

        l-=1

        res -= mass

    return psf_interp




def displacement_interp_stack(P,t,shap,position_map_1,position_map_2,tol=0.001,pol_en=True,cent=None,scale_coefft=None,scale_coeffr=None):
    nb_im = len(t)
    psf_interp = zeros((shap[0],shap[1],nb_im))
    for i in range(0,nb_im):
        print "Image ",i+1,"/",nb_im
        if pol_en:
            psf_interp[:,:,i] = displacement_interp_pol_2(P,t[i],shap,position_map_1,position_map_2,cent,scale_coefft,scale_coeffr,tol=tol,var_speed=False)
        else:
            psf_interp[:,:,i] = displacement_interp(P,t[i],shap,position_map_1,position_map_2,tol=tol,var_speed=False)

    return psf_interp

def displacement_interp_stack_2(im_ref,im_proj,t,pos_map,final_pos,tol=0.001,pol_en=True,cent=None,scale_coefft=None,scale_coeffr=None):
    nb_im = len(t)
    shap = im_ref.shape
    psf_interp = zeros((shap[0],shap[1],nb_im))
    for i in range(0,nb_im):
        print "Image ",i+1,"/",nb_im
        if pol_en:
            psf_interp[:,:,i] = displacement_interp_pol_3(im_ref,im_proj,t[i],pos_map,final_pos,cent,scale_coefft,scale_coeffr,tol=0.001)
        else:
            psf_interp[:,:,i] = displacement_interp_2(P,t[i],shap,position_map_1,position_map_2,tol=tol)

    return psf_interp



def partial_transport_3(P,t,shap,tol=0,cent=None):
    psf_interp = zeros(shap)
    min_val = tol*P.max()
    i=0
    j=0
    if cent is None:
        cent = [shap[0]/2,shap[1]/2]
    for i in range(0,shap[0]*shap[1]):
        i1 = np.int(i/shap[1]) # Line index in position_map_1
        j1 = mod(i,shap[1])
        for j in range(0,shap[0]*shap[1]):
            if (P[i,j] >= min_val):
                i2 = np.int(j/shap[1]) # Line index in postion_map_2
                j2 = mod(j,shap[1])
                if (i1==i2 and j1==j2):
                    psf_interp[i1,j1] = psf_interp[i1,j1] + P[i,j]
                else :
                    iopt = round((1-t)*i1+t*i2)
                    jopt = round((1-t)*j1+t*j2)

                    if iopt < shap[0] and iopt>-1 and jopt < shap[1] and jopt>-1:
                        psf_interp[iopt,jopt] = psf_interp[iopt,jopt]+P[i,j]
    return psf_interp



def distance_matrices(weights,angles,shap,inf_val=1e64,cent=None):
    l = size(weights)
    dist_map = zeros((shap[0]*shap[1],shap[0]*shap[1],l))
    theta = sum(weights*angles)

    for i in range(0,l):
        mapr, maptheta = r_theta_rig_dist_map_3(shap,-theta+angles[i],inf_val=inf_val,cent=cent)
        dist_map[:,:,i] = mapr
    return dist_map


def rbf_interp(comp_cube,coeff,pos_data,target_pos):
    shap = comp_cube.shape
    interp_psf = zeros(shap[0:2])

    for i in range(0,shap[2]):
        rbf = Rbf(pos_data[:,0], pos_data[:,1], coeff[i,:], function='thin_plate')
        interp_psf += rbf(target_pos[0],target_pos[1])*comp_cube[:,:,i]

    return interp_psf


def rbf_3d(pos_data,values):
    rbf_list = list()
    for i in range(0,3):
        rbf_list.append(Rbf(pos_data[0,:], pos_data[1,:], pos_data[2,:], values[i,:], function='thin_plate'))
    return rbf_list

def rbf_3d_2(pos_data,flann_obj,pos_ref,data_ref,nb_neigh=20):
    shap = pos_data.shape
    out = pos_data*0
    result, dists = flann_obj.nn_index(transpose(pos_data), nb_neigh)
    for i in range(0,shap[1]):
        for j in range(0,3):
            rbf_interpij = Rbf(pos_ref[0,result[i,:]], pos_ref[1,result[i,:]], pos_ref[2,result[i,:]], data_ref[j,result[i,:]], function='thin_plate')
            out[j,i] = rbf_interpij(pos_data[0,i], pos_data[1,i], pos_data[2,i])

    return out


def apply_rbf_3d(pos,rbf_list):
    out = pos*0
    for i in range(0,3):
        out[i,:] = rbf_list[i](pos[0,:], pos[1,:], pos[2,:])
    return out



def rbf_stack(psf,pos_data,data_tree,dist,min_dist,nb_neighb,nb_comp=20):
    shap = psf.shape
    coeff,comp_cube,approx_cube = utils.cube_svd(psf,nb_comp=nb_comp)
    psf_interp = psf*0

    for i in range(0,shap[2]):
        print "psf ",i+1,"/",shap[2]
        ind0 = where(dist[i,:]>=min_dist[i])
        ind = data_tree[i,ind0[0][0:nb_neighb]]
        psf_interp[:,:,i] = rbf_interp(comp_cube,coeff[:,ind],pos_data[ind,:],pos_data[i,:])

    return psf_interp


def rbf_stack_2(data,pos_data,target_pos,nb_neigh,nb_comp=30,knn=None):

    from utils import cube_svd
    from pyflann import FLANN
    from numpy import zeros,transpose,diag
    import time
    if knn is None:
        knn = FLANN()
        params = knn.build_index(array(pos_data, dtype=float64))
    shap = data.shape
    nb_pts = target_pos.shape[0]

    # Computing the weights
    result, dists = knn.nn_index(target_pos, nb_neigh)
    coeff,comp_cube,approx_cube = cube_svd(data,nb_comp=min(nb_comp,nb_neigh))
    psf_interp = zeros((shap[0],shap[1],nb_pts))
    tmean = 0
    for i in range(0,nb_pts):
        t = time.time()
        psf_interp[:,:,i] = rbf_interp(comp_cube,coeff[:,result[i,:]],pos_data[result[i,:],:],target_pos[i,:])
        tmean+= time.time() - t

    print "Elapsed time: ",tmean/nb_pts
    return psf_interp



def rbf_stack_2_loc(data,pos_data,target_pos,nb_neigh,nb_comp=4,knn=None):

    from utils import cube_svd
    from pyflann import FLANN
    from numpy import zeros,transpose,diag

    if knn is None:
        knn = FLANN()
        params = knn.build_index(array(pos_data, dtype=float64))
    shap = data.shape
    nb_pts = target_pos.shape[0]
    psf_interp = zeros((shap[0],shap[1],nb_pts))
    # Computing the weights
    result, dists = knn.nn_index(target_pos, nb_neigh)
    for i in range(0,nb_pts):

        coeff,comp_cube,approx_cube = cube_svd(data[:,:,result[i,:]],nb_comp=min(nb_comp,nb_neigh))
        psf_interp[:,:,i] = rbf_interp(comp_cube,coeff,pos_data[result[i,:],:],target_pos[i,:])

    return psf_interp


def rbf_stack_2_feat_wise(list_data_feat,pos_data,target_pos,nb_neigh,nb_comp=30,knn=None):
    from numpy import zeros
    shap = list_data_feat[0].shape
    nb_points = target_pos.shape[0]
    psf_interp = zeros((shap[0],shap[1],nb_points))

    for i in range(0,len(list_data_feat)):
        psf_interp+=rbf_stack_2(list_data_feat[i],pos_data,target_pos,nb_neigh,nb_comp=nb_comp,knn=knn)

    return psf_interp

def rbf_stack_2m(data,pos_data,target_pos,nb_neigh,nb_comp=30):

    nb_real = len(nb_neigh)
    psf_interp = list()

    for i in range(0,nb_real):
        psf_interp.append(rbf_stack_2(data,pos_data,target_pos,nb_neigh[i],nb_comp=nb_comp))

    return psf_interp


def rbf_stack_2_loc_m(data,pos_data,target_pos,nb_neigh,nb_comp=30):

    nb_real = len(nb_neigh)
    psf_interp = list()

    for i in range(0,nb_real):
        psf_interp.append(rbf_stack_2_loc(data,pos_data,target_pos,nb_neigh[i],nb_comp=nb_comp))

    return psf_interp

def rbf_stack_2_feat_m(data,pos_data,target_pos,nb_neigh,nb_comp=30):

    nb_real = len(nb_neigh)
    psf_interp = list()

    for i in range(0,nb_real):
        psf_interp.append(rbf_stack_2_feat_wise(data,pos_data,target_pos,nb_neigh[i],nb_comp=nb_comp))

    return psf_interp


def rbf_stackm(psf,pos_data,data_tree,dist,dist_range,nb_neighb,nb_comp=20):
    nb_realizations = dist_range.shape[1]
    interp = list()

    for i in range(0,nb_realizations):
        print "realization ",i+1,"/",nb_realizations
        interp.append(rbf_stack(psf,pos_data,data_tree,dist,dist_range[:,i],nb_neighb,nb_comp=nb_comp))

    return interp


def rbf_stackmbis(psf,pos_data,data_tree,dist,dist_range,nb_neighb,nb_comp=20):
    nb_realizations = size(nb_neighb)
    interp = list()

    for i in range(0,nb_realizations):
        print "realization ",i+1,"/",nb_realizations
        interp.append(rbf_stack(psf,pos_data,data_tree,dist,dist_range,nb_neighb[i],nb_comp=nb_comp))

    return interp



def ri_profile(psf):
    prof = zeros((2,size(psf)))
    i,j = where(psf==psf.max())
    im = mean(i)
    jm = mean(j)
    shap = psf.shape
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            prof[0,i+j*shap[0]] = sqrt((im-i)**2+(jm-j)**2)
            prof[1,i+j*shap[0]] = psf[i,j]

    return prof



def im_to_point_cloud(im):

    from numpy import ones,zeros

    shap = im.shape

    point_cloud = None
    point_cloud = zeros((3,shap[0]*shap[1]))

    point_cloud[0,:] = (arange(0,shap[0]).reshape((shap[0],1)).dot(ones((1,shap[1])))).reshape((shap[0]*shap[1],))
    point_cloud[1,:] = ones((shap[0],1)).dot(arange(0,shap[1]).reshape((1,shap[1]))).reshape((shap[0]*shap[1],))
    point_cloud[2,:] = im.reshape((shap[0]*shap[1],))


    return point_cloud

def xlog(arr,a=None):

    from numpy import log as lognp,median
    t=1
    if a is None:
        a = (1-t)*median(arr)+ t*arr.min()
    arr_out = copy(arr)
    i,j = where(arr<a)
    arr_out[i,j] = a*lognp(arr[i,j])+a*(1-lognp(a))

    return arr_out


def xexp(arr,c=0.0000007):
    from numpy import exp as expnp,median
    t=1
    x = (1-t)*median(arr)+ t*arr.min()
    arr_out = copy(arr)
    i,j = where(arr<x)
    arr_out[i,j] = x*expnp(-c/x)*expnp(c/arr[i,j])

    return arr_out



def cube_to_point_cloud(cube):

    from numpy import zeros

    shap = cube.shape
    nb_pix = shap[0]*shap[1]
    nb_im = shap[2]

    point_cloud = zeros((3,shap[0]*shap[1],nb_im))

    for i in range(0,nb_im):
        point_cloud[:,:,i]= im_to_point_cloud(cube[:,:,i])

    return point_cloud


def isomap_interface(data,dim,nb_neigh_min=2,nb_neigh_max=29,step=2): # Data: each sample is in a row
    from sklearn import manifold
    from utils import distances_mat
    from copy import copy,deepcopy
    nb_samp = data.shape[1]
    dist_data = distances_mat(data)
    nb_step = floor((nb_neigh_max-nb_neigh_min+1)/step).astype(int)
    neigh = range(nb_neigh_min,nb_neigh_max+1,step)
    kernel_embedding_error = zeros((nb_step,))
    dist_embedding_error = zeros((nb_step,))
    over_fit = zeros((nb_step,))
    select_func = zeros((nb_step,))
    err_min = 1e15
    iso_out = None
    embedding_out = None


    for i in range(0,nb_step):
        print i+1,'/',nb_step
        iso = manifold.Isomap(neigh[i], dim).fit(transpose(data))
        Y = transpose(iso.transform(transpose(data)))
        dist_y = distances_mat(Y)
        k,l = where(iso.dist_matrix_>0)
        dist_embedding_error[i] = 100*abs((iso.dist_matrix_[k,l]**2 - dist_y[k,l])/iso.dist_matrix_[k,l]).mean()
        kernel_embedding_error[i] = iso.reconstruction_error()
        over_fit[i] = abs(iso.dist_matrix_**2 - dist_data).sum()/nb_samp

    return iso_out,embedding_out,kernel_embedding_error,dist_embedding_error,over_fit

def LLE_sliced_transport_bar(im_cube,weights,nb_points=None,cent=None,coord_map=None,log_param=1.5,zeros_inc=True,nb_neigh_emb=None,nb_comp=None,nb_neigh_inv_map=3,nb_samp=8,embedding_en=False,embedding_closest=True,log_en=True):
    from numpy import size,transpose,array,float64
    from sklearn import manifold
    from pyflann import FLANN
    from utils import knn_bar,circle_vect
    from optim_utils import sliced_transport_bar
    shap = im_cube.shape
    if cent is None:
        cent  = where(im_cube[:,:,0]==im_cube[:,:,0].max())
    if nb_neigh_emb is None:
        if embedding_closest:
            nb_neigh_emb = 15
        else:
            nb_neigh_emb = 30*shap[2]


    if nb_points is none:
        nb_points = shap[0]*shap[1]
    embedd_point_clouds = zeros((nb_comp,nb_points,shap[2]))
    embedd_point_clouds_unwrp = zeros((nb_comp,nb_points*shap[2]))
    clouds = zeros((3,nb_points,shap[2]))


    for i in range(0,shap[2]):
        clouds[:,:,i] = im_to_point_cloud(im_cube[:,:,i],rand_en=False,nb_samp = None,pol_en=False,cent=cent,coord_map=coord_map,log_param=log_param,min_val = 1e-31,zeros_inc=zeros_inc,log_advanced=False,log_en=log_en)


    Yhess_obj = None
    flann = None
    if embedding_en:
        point_cloud = zeros((3,nb_points*shap[2]))

        for i in range(0,shap[2]):
            point_cloud[:,i*nb_points:(i+1)*nb_points] = clouds[:,:,i]
        print "Calculating the embedding..."
        if embedding_closest:
            imax = where(weights==weights.max())
            in_cloud = None
            if size(imax[0]==1):
                in_cloud = clouds[:,:,imax[0][0]]
            else:
                in_cloud = zeros((3,nb_points*size(imax[0])))
                for k in range(0,size(imax[0])):
                    in_cloud[:,k*nb_points:(k+1)*nb_points] = clouds[:,:,imax[0][k]]
            Yhess_obj = manifold.LocallyLinearEmbedding(nb_neigh_emb, nb_comp,eigen_solver='dense',method='hessian').fit(transpose(in_cloud))
            #Yhess_obj = manifold.Isomap(nb_neigh_emb, nb_comp).fit(transpose(in_cloud))
        else:
            Yhess_obj = manifold.LocallyLinearEmbedding(nb_neigh_emb, nb_comp,eigen_solver='dense',method='hessian').fit(transpose(point_cloud))
            #Yhess_obj = manifold.Isomap(nb_neigh_emb, nb_comp).fit(transpose(point_cloud))

        for i in range(0,shap[2]):
            if embedding_closest:
                embedd_point_clouds[:,:,i] = transpose(Yhess_obj.transform(transpose(clouds[:,:,i])))
            else:
                embedd_point_clouds[:,:,i] = transpose(Yhess_obj.embedding_[i*nb_points:(i+1)*nb_points,:])
            embedd_point_clouds_unwrp[:,i*nb_points:(i+1)*nb_points] = embedd_point_clouds[:,:,i]

    rand_en=False
    basis = None
    if nb_comp==2 and embedding_en:
        basis = circle_vect(nb_samp)
    else:
        rand_en = True




    if embedding_en:
        print "Calculating the sliced barycenter..."
        init_bar = knn_bar(embedd_point_clouds,weights)

        emb_bar,basis = sliced_transport_bar(embedd_point_clouds,weights,nb_iter=1000,tol=0.0000001,alph=0.01,bar_init=init_bar,rand_en=rand_en,basis=basis)

        print "Inverting the embedding..."
        flann = FLANN()
        params = flann.build_index(array(Yhess_obj.embedding_, dtype=float64))
        bar = inverse_mapping(emb_bar,embedd_point_clouds_unwrp,point_cloud,flann,nb_neigh=nb_neigh_inv_map)

        print "Mapping the final point cloud to an image..."
        im_out = bar_1d_2d_bis(bar,shap[0:2],cent,pol_en=False,log_param=log_param,tol=0.001)

    else:
        init_bar = knn_bar(clouds,weights)
        bar,basis = sliced_transport_bar(clouds,weights,nb_iter=1000,tol=0.0000001,alph=0.01,bar_init=init_bar,rand_en=rand_en,basis=basis)
        print "Mapping the final point cloud to an image..."
        im_out = bar_1d_2d_bis(bar,shap[0:2],cent,pol_en=False,log_param=log_param,tol=0.001)

    print "Done..."

    return im_out,Yhess_obj,flann



def structured_sliced_transport_bar(im_cube,weights,nb_points=None,cent=None,coord_map=None,log_param=1.5,zeros_inc=True,nb_neigh_emb=None,nb_comp=None,nb_neigh_inv_map=3,nb_samp=8,embedding_en=False,embedding_closest=True,log_en=True,kmad=5,deg=2,win=9,win2=7):
    from numpy import zeros
    shap = im_cube.shape
    # Detecting and extracting structures
    label_im,poly_fit,nb_samples = ring_stack(im_cube,center=None,kmad=kmad,deg=deg,win=win,win2=win2)

    nb_zeros = len(nb_samples)
    indexes = [1,[2,3],[4,5],range(6,nb_zeros+1)]
    nb_points = zeros((4,))
    for i in range(0,4):
        if size(indexes[i])==1:
            nb_points[i] = nb_samples[indexes[i]]
        else:
            for k in range(0,size(indexes[i])):
                nb_points[i] += nb_samples[indexes[i][k]]

    list_structures = get_ring_stack(im_cube,map,indexes)

    bar_out = zeros((shap[0],shap[1],4))
    hess_obj = list()
    knn_obj = list()

    for i in range(0,4):
        bar_out[:,:,i],Yhess_obj,flann = LLE_sliced_transport_bar(list_structures[i],weights,nb_points=nb_points[i],cent=cent,coord_map=coord_map,log_param=log_param,zeros_inc=zeros_inc,nb_neigh_emb=nb_neigh_emb,nb_comp=nb_comp,nb_neigh_inv_map=nb_neigh_inv_map,nb_samp=nb_samp,embedding_en=embedding_en,embedding_closest=embedding_closest,log_en=log_en)
        hess_obj.append(Yhess_obj)
        knn_obj.append(flann)


    return bar_out,hess_obj,knn_obj


def arc_length(map,samp=None):
    shap = map.shape
    if samp is None:
        samp = max(shap)
    curv_abs = 0
    t = double(arange(0,samp+1))/samp
    step = sqrt((shap[0]-1)**2+(shap[1]-1)**2)/samp

    for i in range(0,samp):

        curv_abs+= abs(map[int(t[i]*(shap[0]-1)),int(t[i]*(shap[1]-1))]-map[int(t[i+1]*(shap[0]-1)),int(t[i+1]*(shap[1]-1))])*step
        #curv_abs+= map[int(t[i]*(shap[0]-1)),int(t[i]*(shap[1]-1))]*step
    return curv_abs



def embedding(mani_obj,im_stack,pol_en=True,cent=None,coord_map=None,log_param=1.5,log_advanced=False):
    from numpy import transpose
    shap =im_stack.shape

    out = zeros((mani_obj.embedding_.shape[1],shap[0]*shap[1],shap[2]))
    for i in range(0,shap[2]):
        point_cloud = im_to_point_cloud(im_stack[:,:,i],rand_en=False,pol_en=pol_en,cent=cent,coord_map=coord_map,log_param=log_param,zeros_inc=True,log_advanced=log_advanced)
        out[:,:,i] = transpose(mani_obj.transform(transpose(point_cloud_ref)))

    return out


def inverse_mapping(embed_coord,embed_ref,original_ref,flann_obj,nb_neigh=3):
    from numpy import zeros,diag
    shap1 = embed_coord.shape
    shap2 = original_ref.shape

    output = zeros((shap2[0],shap1[1]))
    result, dists = flann_obj.nn_index(transpose(embed_coord), nb_neigh)
    bar_coord = zeros((shap1[1],nb_neigh))

    for i in range(0,shap1[1]):
        bar_coord[i,:] = bar_coord2d(embed_ref[:,result[i]],embed_coord[:,i],cond=1./50).reshape((nb_neigh,))
        output[:,i] = diag(bar_coord[i,:]).dot(original_ref[:,result[i]]).sum(axis=1)

    return output,bar_coord,result


def inverse_mapping_2D(embed_coord,embed_ref,original_ref,flann_obj,nb_neigh=3):
    from numpy import zeros,diag
    shap1 = embed_coord.shape
    shap2 = original_ref.shape
    output = None
    bar_coord = None
    if len(shap1)<=1:
        output = zeros((shap2[0],shap2[1]))
        bar_coord = zeros((nb_neigh,))
    else:
        output = zeros((shap2[0],shap2[1],shap1[1]))
        bar_coord = zeros((shap1[1],nb_neigh))

    result, dists = flann_obj.nn_index(embed_coord, nb_neigh)



    if len(shap1)<=1:
        bar_coord = bar_coord2d(embed_ref[:,result[0,:]],embed_coord,cond=1./50).reshape((nb_neigh,))
        for j in range(0,nb_neigh):
            output+= bar_coord[j]*original_ref[:,:,result[0,j]]
    else:
        for i in range(0,shap1[1]):
            bar_coord[i,:] = bar_coord2d(embed_ref[:,result[i]],embed_coord[:,i],cond=1./50).reshape((nb_neigh,))
            for j in range(0,nb_neigh):
                output[:,:,i]+= bar_coord[i,j]*original_ref[:,:,result[i,j]]

    return output,bar_coord,result


def w_inverse_mapping_2D(embed_coord,embed_ref,original_ref,flann_obj,cent,coeff_poly,basis,nb_neigh=3,nb_iter=1000,support=True):
    from numpy import zeros,diag,argsort
    from optim_utils import sliced_transport_bar
    from numpy.polynomial.polynomial import polyval
    shap1 = embed_coord.shape

    shap2 = original_ref.shape
    output = None
    bar_coord = None
    if len(shap1)<=1:
        output = zeros((shap2[0],shap2[1]))
        bar_coord = zeros((nb_neigh,))
    else:
        output = zeros((shap2[0],shap2[1],shap1[1]))
        bar_coord = zeros((shap1[1],nb_neigh))

    result, dists = flann_obj.nn_index(transpose(embed_coord), nb_neigh)


    if len(shap1)<=1:
        bar_coord = bar_coord2d(embed_ref[:,result[0,:]],embed_coord,cond=1./50).reshape((nb_neigh,))
        point_cloud_w = copy(original_ref[:,:,result[0,:]])
        # Weighting
        for i in range(0,nb_neigh):
            rad_dist_bar = sqrt(sum((point_cloud_w[0:2,:,i]-array(cent).reshape((2,1)).dot(ones((1,shap2[1]))))**2,axis=0))
            if support:
                point_cloud_w[0:2,:,i] /= exp(polyval(rad_dist_bar,coeff_poly))
            else:
                point_cloud_w[2,:,i] *= exp(polyval(rad_dist_bar,coeff_poly))
        ind = argsort(bar_coord)
        init_bar = point_cloud_w[:,:,ind[0]]
        output,basis_out = sliced_transport_bar(point_cloud_w,bar_coord,nb_iter=nb_iter,tol=0.000000001,alph=0.01,bar_init=init_bar,rand_en=False,basis=basis,support=support)

    else:
        for i in range(0,shap1[1]):
            bar_coord[i,:] = bar_coord2d(embed_ref[:,result[i]],embed_coord[:,i],cond=1./50).reshape((nb_neigh,))

            point_cloud_w = copy(original_ref[:,:,result[0,:]])
            # Weighting
            for i in range(0,nb_neigh):
                rad_dist_bar = sqrt(sum((point_cloud_w[0:2,:,i]-array(cent).reshape((2,1)).dot(ones((1,shap2[1]))))**2,axis=0))
                if support:
                    point_cloud_w[0:2,:,i] /= exp(polyval(rad_dist_bar,coeff_poly))
                else:
                    point_cloud_w[2,:,i] *= exp(polyval(rad_dist_bar,coeff_poly))
            ind = argsort(bar_coord[i,:])
            init_bar = point_cloud_w[:,:,result[i,ind[0]]]
        output[:,:,i],basis_out = sliced_transport_bar(original_ref[:,:,result[i,:]],bar_coord,nb_iter=nb_iter,tol=0.0000001,alph=0.01,bar_init=init_bar,rand_en=False,basis=basis,support=support)

    return output,bar_coord,result


def w_inverse_mapping_hybrid_2D(embed_coord,embed_ref,original_ref_clouds,original_ref_im,flann_obj,cent,coeff_poly,basis,nb_neigh=3,nb_iter=1000):
    from numpy import zeros,diag,argsort
    from optim_utils import sliced_transport_bar,euc_support_barycenter
    from numpy.polynomial.polynomial import polyval
    shap1 = embed_coord.shape

    shap2 = original_ref.shape
    output = None
    bar_coord = None
    if len(shap1)<=1:
        output = zeros((shap2[0],shap2[1]))
        bar_coord = zeros((nb_neigh,))
    else:
        output = zeros((shap2[0],shap2[1],shap1[1]))
        bar_coord = zeros((shap1[1],nb_neigh))

    result, dists = flann_obj.nn_index(transpose(embed_coord), nb_neigh)


    if len(shap1)<=1:
        bar_coord = bar_coord2d(embed_ref[:,result[0,:]],embed_coord,cond=1./50).reshape((nb_neigh,))
        point_cloud_w = copy(original_ref[:,:,result[0,:]])
        # Weighting
        for i in range(0,nb_neigh):
            rad_dist_bar = sqrt(sum((point_cloud_w[0:2,:,i]-array(cent).reshape((2,1)).dot(ones((1,shap2[1]))))**2,axis=0))
            point_cloud_w[2,:,i] *= exp(polyval(rad_dist_bar,coeff_poly))
        ind = argsort(bar_coord)
        init_bar = point_cloud_w[:,:,ind[0]]
        output,basis_out = sliced_transport_bar(point_cloud_w,bar_coord,nb_iter=nb_iter,tol=0.000000001,alph=0.01,bar_init=init_bar,rand_en=False,basis=basis)

    else:
        for i in range(0,shap1[1]):
            bar_coord[i,:] = bar_coord2d(embed_ref[:,result[i]],embed_coord[:,i],cond=1./50).reshape((nb_neigh,))

            point_cloud_w = copy(original_ref[:,:,result[0,:]])
            # Weighting
            for i in range(0,nb_neigh):
                rad_dist_bar = sqrt(sum((point_cloud_ref[0:2,:,i]-array(cent).reshape((2,1)).dot(ones((1,shap2[1]))))**2,axis=0))
                point_cloud_w[2,:,i] *= exp(polyval(rad_dist_bar,coeff_poly))
            ind = argsort(bar_coord[i,:])
            init_bar = point_cloud_w[:,:,result[i,ind[0]]]
        output[:,:,i],basis_out = sliced_transport_bar(original_ref[:,:,result[i,:]],bar_coord,nb_iter=nb_iter,tol=0.0000001,alph=0.01,bar_init=init_bar,rand_en=False,basis=basis)

    return output,bar_coord,result




def psf_zeros_detector(im,center=None,kmad=5,deg=2,win=9,spline_ord=3,nb_rings = 7): # deg: degree of the polynomial interpolator
    from utils import polar,mad
    from numpy import less,zeros,float as floatnump,flipud,pi,array,argsort
    from numpy.polynomial.polynomial import polyfit
    from scipy.signal import argrelextrema
    from copy import deepcopy
    from scipy.signal import savgol_filter
    from scipy.interpolate import interp1d
    from scipy.signal import medfilt

    if win%2==0:
        win-=1

    if center is None:
        center = flipud(where(im==im.max()))
    pol_im = polar(im,center=center)

    nb_angles = pol_im.shape[0]
    angle_max = 2*pi*float(nb_angles-1)/nb_angles
    zeros_pos = list()
    nb_zeros_max = 0
    # First detection
    for i in range(0,nb_angles):
        rz = argrelextrema(pol_im[i,:],less)[0]
        if i==0:
            nb_zeros_max = size(rz)
            for l in range(0,size(rz)):
                zeros_pos.append({})
                zeros_pos[l][i*2*pi/nb_angles] = rz[l]
        else:
            for l in range(0,min(nb_zeros_max,size(rz))):
                zeros_pos[l][i*2*pi/nb_angles] = rz[l]
            if size(rz)>nb_zeros_max:
                for l in range(nb_zeros_max,size(rz)):
                    zeros_pos.append({})
                    zeros_pos[l][i*2*pi/nb_angles] = rz[l]
                nb_zeros_max = size(rz)



    # Remapping outliers
    #print "nb zeros: ",nb_zeros_max,len(zeros_pos)

    for l in range(0,nb_zeros_max):
        nb_samp = len(zeros_pos[l])
        angles = zeros_pos[l].keys()
        radius = zeros_pos[l].values()
        outliers = where(radius>kmad*mad(radius)+median(radius))
        out_detect=False
        if size(outliers[0])>0:
            if out_detect is False:
                out_detect = True
                zeros_pos.append({})
            zeros_pos_cp = deepcopy(zeros_pos)
            #print "nb zeros: ",len(zeros_pos)
            for i in range(0,size(outliers[0])):

                for k in range(l,nb_zeros_max+1):

                    if angles[outliers[0][i]] in zeros_pos[k].keys():
                        del zeros_pos[k][angles[outliers[0][i]]]
                    if k>0:
                        if angles[outliers[0][i]] in zeros_pos_cp[k-1].keys():
                            zeros_pos[k][angles[outliers[0][i]]] = zeros_pos_cp[k-1][angles[outliers[0][i]]]

    zeros_pos_clean = list()
    for i in range(0,len(zeros_pos)):
        if size(zeros_pos[i].keys())>0:
            zeros_pos_clean.append(zeros_pos[i])

    zeros_pos = zeros_pos_clean
    # Final mapping
    final_zeros_list = list()
    poly_out = list()
    width = list()


    # Preprocessings for interpolation

    for l in range(0,len(zeros_pos)):

        if 0 not in zeros_pos[l].keys():
            zeros_pos[l][0] = zeros_pos[l][array(zeros_pos[l].keys()).min()]

        zeros_pos[l][2*pi] = zeros_pos[l][0]

        nb_samp = len(zeros_pos[l])
        final_zeros_list.append(zeros((2,nb_samp)))
        dict_keys = zeros_pos[l].keys()
        ind = argsort(dict_keys)
        for k in range(0,nb_samp):
            final_zeros_list[l][1,k] = dict_keys[ind[k]]
            final_zeros_list[l][0,k] = zeros_pos[l][dict_keys[ind[k]]]

        #final_zeros_list[l][1,:] = array(zeros_pos[l].keys())
        #final_zeros_list[l][0,:] = array(zeros_pos[l].values())


        if nb_samp>win:
            #print final_zeros_list[l][0,:]
            if nb_samp>2*win:
                final_zeros_list[l][0,:] = medfilt(final_zeros_list[l][0,:],win)
            final_zeros_list[l][0,:] = savgol_filter(final_zeros_list[l][0,:],win,deg)

    # Enforcing radial monotonicity
    for k in range(0,nb_angles):
        key = i*2*pi/nb_angles
        val = list()
        for l in range(0,len(zeros_pos)):
            ind = where(final_zeros_list[l][1,:]==key)
            if len(ind[0])>0:
                val.append(final_zeros_list[l][0,ind[0]])

        ind_val = argsort(array(val))
        count = 0
        for l in range(0,len(zeros_pos)):
            ind = where(final_zeros_list[l][1,:]==key)
            if len(ind[0])>0:
                final_zeros_list[l][0,ind[0]] = val[count]
                count+=1

    for l in range(0,len(zeros_pos)):
        nb_samp = len(zeros_pos[l])
        if nb_samp>win:
            final_zeros_list[l][0,0] = (final_zeros_list[l][0,0]+final_zeros_list[l][0,-1])/2
            final_zeros_list[l][0,-1] = final_zeros_list[l][0,0]
            poly_out.append(interp1d(final_zeros_list[l][1,:],final_zeros_list[l][0,:],kind=spline_ord))
            if l< nb_rings:
                if l==0:
                    width.append(array(zeros_pos[l].values()).mean())
                else:
                    width.append(array(zeros_pos[l].values()).mean()-width[l-1])




    return final_zeros_list,poly_out,width



def ring_separation(im,center=None,kmad=5,deg=2,win=9,win2=1,nb_rings = 7):
    from utils import polar_coord
    from numpy import argsort,flipud,median,zeros,ones,sqrt
    from numpy.linalg import norm
    from scipy.signal import medfilt

    if center is None:
        center = flipud(where(im==im.max()))
    final_zeros_list,poly_out,widths = psf_zeros_detector(im,center=center,kmad=5,deg=deg,win=win,nb_rings=nb_rings)

    rtheta_pos = zeros((2,size(im)))
    cart_pos = zeros((2,size(im)))
    shap = im.shape
    for i in range(0,shap[1]):
        for j in range(0,shap[0]):
            rtheta_pos[:,j+i*shap[0]] = polar_coord([i,j],center)
            cart_pos[:,j+i*shap[0]] = [i,j]

    ind = argsort(rtheta_pos[0,:])

    label_im = zeros(shap)

    if nb_rings is None:
        nb_rings = len(poly_out)



    for k in range(0,size(im)):
        found = False
        l=0
        """r = zeros((nb_rings+1,))
        for l in range(0,nb_rings):
            r[l] = poly_out[l](rtheta_pos[1,ind[k]])

        r[nb_rings] = rtheta_pos[0,ind[k]]

        argsr = argsort(r)
        #print sum(abs(r[0:nb_rings]-r[argsr[0:nb_rings]]))

        if argsr[nb_rings]==nb_rings:
            label_im[cart_pos[1,ind[k]],cart_pos[0,ind[k]]] = nb_rings
        else:
            label_im[cart_pos[1,ind[k]],cart_pos[0,ind[k]]] = where(argsr==argsr[nb_rings]+1)[0]+1"""

        while found is False and l< nb_rings:


            if poly_out[l](rtheta_pos[1,k]) >= rtheta_pos[0,k]:

                if l==0:
                    label_im[int(rtheta_pos[0,k]*sin(rtheta_pos[1,k])+center[1]),int(rtheta_pos[0,k]*cos(rtheta_pos[1,k])+center[0])] = l+1
                    found = True
                else:
                    if poly_out[l-1](rtheta_pos[1,k]) <= rtheta_pos[0,k]:
                        label_im[int(rtheta_pos[0,k]*sin(rtheta_pos[1,k])+center[1]),int(rtheta_pos[0,k]*cos(rtheta_pos[1,k])+center[0])] = l+1
                        found = True

            if l==nb_rings-1:

                if poly_out[l](rtheta_pos[1,k]) <= rtheta_pos[0,k]:
                    label_im[int(rtheta_pos[0,k]*sin(rtheta_pos[1,k])+center[1]),int(rtheta_pos[0,k]*cos(rtheta_pos[1,k])+center[0])] = l+1
                    found = True

            l+=1



    i,j = where(label_im==0)
    label_im_ref = copy(label_im)

    if len(i)>0:
        for k in range(0,len(i)):
            increase=True
            rad = 1
            while increase:
                wind = label_im_ref[max(i[k]-rad,0):min(i[k]+rad+1,label_im.shape[0]-1),max(j[k]-rad,0):min(j[k]+rad+1,label_im.shape[1]-1)]
                a = median(wind)
                if a>0:
                    increase=False
                    label_im[i[k],j[k]] = a
                else:
                    rad+=2

    label_im = medfilt(label_im,win2)

    i,j = where(label_im==0)
    label_im_ref = copy(label_im)

    if len(i)>0:
        for k in range(0,len(i)):
            increase=True
            rad = 1
            while increase:
                wind = label_im_ref[max(i[k]-rad,0):min(i[k]+rad+1,label_im.shape[0]-1),max(j[k]-rad,0):min(j[k]+rad+1,label_im.shape[1]-1)]
                a = median(wind)
                if a>0:
                    increase=False
                    label_im[i[k],j[k]] = a
                else:
                    rad+=2



    heights = list()
    mean_rad = list()
    for l in range(1,int(label_im.max())+1):
        i,j = where(label_im==l)
        heights.append(im[i,j].max()-im[i,j].min())
        nb_points = len(i)
        mean_rad.append(sqrt(norm(i-center[0]*ones((nb_points,)))**2+norm(j-center[1]*ones((nb_points,)))**2)/sqrt(nb_points))


    return final_zeros_list,poly_out,label_im,rtheta_pos,widths,heights,mean_rad

def ring_equalizer(map,nb_pix,rad):
    from numpy import flipud,argsort#,median,zeros
    map_out = copy(map)
    shap = map.shape

    for i in range(0,size(nb_pix)):
        k,l = where(map_out==i+1)

        if size(k)>0:
            ind_lin = l+k*shap[1]
            nb_pts = size(k)

            if nb_pts>nb_pix[i]:
                ind = argsort(rad[ind_lin])
                for p in range(0,nb_pts-nb_pix[i]):
                    map_out[k[ind[-(p+1)]],l[ind[-(p+1)]]] = i+2
            if nb_pts<nb_pix[i]:
                count = 0
                ring_ind = i+2
                it,jt = where(map_out>i+1)

                while nb_pts+count<nb_pix[i]:

                    k1,l1 = where(map_out==ring_ind)
                    nb_pts_2 = size(k1)

                    if nb_pts_2>0:

                        ind_lin = l1+k1*shap[1]
                        ind = argsort(rad[ind_lin])

                        if nb_pts_2> nb_pix[i]-nb_pts-count:
                            for p in range(0,nb_pix[i]-nb_pts-count):
                                map_out[k1[ind[-(p+1)]],l1[ind[-(p+1)]]] = i+1
                            count = nb_pix[i]-nb_pts
                        else:
                            count += nb_pts_2
                            for p in range(0,nb_pts_2):
                                map_out[k1[ind[-(p+1)]],l1[ind[-(p+1)]]] = i+1



                        for p in range(0,int(min(nb_pts_2,nb_pix[i]-nb_pts-count))):
                            map_out[k1[ind[-(p+1)]],l1[ind[-(p+1)]]] = i+1
                    ring_ind+=1
                k,l = where(map_out==i+1)


    return map_out



def ring_stack(psf_stack,center=None,kmad=5,deg=2,win=9,win2=7,equalize_en=False,nb_rings = 7):

    from numpy import flipud,median,zeros,array

    if center is None:
        center = flipud(where(psf_stack[:,:,0]==psf_stack[:,:,0].max()))

    label_im = psf_stack*0
    nb_zeros = 1000000
    shap = psf_stack.shape
    pol_coord = None
    nb_samples_per_ring = zeros((shap[0]*shap[1],shap[2]))
    poly_fit = list()
    widths = list()
    heights = list()
    mean_rad = list()


    for i in range(0,shap[2]):
        print i+1," image/",shap[2]
        final_zeros_list,poly_out,label_im[:,:,i],pol_coord,widths_i,heights_i,mean_rad_i = ring_separation(psf_stack[:,:,i],center=center,kmad=kmad,deg=deg,win=win,win2=win2,nb_rings = nb_rings)
        mean_rad.append(mean_rad_i)
        poly_fit.append(poly_out)
        nb_zeros_i = len(poly_out)
        widths.append(widths_i)
        heights.append(heights_i)



        for j in range(0,nb_rings):
            x,y = where(label_im[:,:,i]==j+1)
            nb_samples_per_ring[j,i] = size(x)

    # Widths and heights equalizing
    nb_min = len(widths[0])

    for i in range(0,len(widths)):
        if nb_min>len(widths[i]):
            nb_min = len(widths[i])
        for j in range(0,i+1):
            if len(widths[j]) > nb_min:
                for k in range(nb_min,len(widths[j])):
                    widths[j].remove(widths[j][-1])
                    heights[j].remove(heights[j][-1])
                    mean_rad[j].remove(mean_rad[j][-1])


    #print nb_samples_per_ring[0:nb_zeros,:].sum(axis=1)


    final_nb_samples = mean(nb_samples_per_ring[0:nb_rings,:],axis=1)
    final_nb_samples = final_nb_samples.astype(int)

    # Equalizing zeros maps
    if equalize_en:
        for i in range(0,shap[2]):
            label_im[:,:,i] = ring_equalizer(label_im[:,:,i],final_nb_samples,pol_coord[0,:])



    return label_im,poly_fit,final_nb_samples,pol_coord,array(widths),array(heights),array(mean_rad)


def get_ring(im,map,indexes,theta_im,scaling_fact=1):
    from numpy import zeros,ones,pi,copy

    shap = im.shape
    ring_out = zeros(shap)
    pol_coord_ring = ones(shap)*2*pi
    nb_points = 0

    if size(indexes)==1:
        i,j = where(map==indexes)
        ring_out[i,j] = copy(im[i,j])
        pol_coord_ring[i,j] = copy(theta_im[i,j])

    else:
        for k in range(0,len(indexes)):
            i,j = where(map==indexes[k])
            ring_out[i,j] += copy(im[i,j])
            pol_coord_ring[i,j] = copy(theta_im[i,j])


    point_cloud = im_to_point_cloud(ring_out)


    return ring_out,pol_coord_ring,point_cloud


def get_ring_stack(im_stack,map,indexes,theta_im):
    from numpy import zeros,swapaxes
    shap = im_stack.shape
    list_map = list()
    list_pol_coord = list()
    list_point_clouds = list()
    nb_points = list()
    for i in range(0,len(indexes)):

        rings_i = zeros(shap)
        pol_coord_ring_i = ones(shap)*2*pi
        pt_cloud = list()
        for j in range(0,shap[2]):
            rings_i[:,:,j],pol_coord_ring_i[:,:,j],point_cloud_j = get_ring(im_stack[:,:,j],map[:,:,j],indexes[i],theta_im)
            pt_cloud.append(point_cloud_j)
            if j==0:
                nb_points.append(point_cloud_j.shape[1])

        list_point_clouds.append(swapaxes(swapaxes(array(pt_cloud),0,1),1,2))
        list_map.append(rings_i)
        list_pol_coord.append(pol_coord_ring_i)
    return list_map,list_pol_coord,list_point_clouds,array(nb_points)



def partionn_circle(nb_parts,offset,pol_coord):
    from numpy import arange,pi,zeros,pi
    i0,j0 = where(pol_coord<2*pi)
    nb_pts = size(i0)
    step = 2*pi/nb_parts
    z = arange(0,nb_parts+1)*step+offset
    z = z%(2*pi)
    labels = pol_coord*0
    for i in range(0,nb_parts-1):
        i1,j1 = where(pol_coord>=z[i])
        i2 = where(pol_coord[i1,j1]<z[(i+1)%nb_parts])
        labels[i1[i2[0]],j1[i2[0]]] = i+1

    i1,j1 = where(pol_coord<z[0])
    i2,j2 = where(pol_coord>=z[nb_parts-1])
    i3 = where(pol_coord[i2,j2]<2*pi)

    labels[i1,j1] = nb_parts
    labels[i2[i3[0]],j2[i3[0]]] = nb_parts

    return labels

def partionn_circle_stack(nb_parts,offset,pol_coord_stack):
    from numpy import zeros,pi,where,squeeze
    from optim_utils import optimal_assignement_1d
    nb_sets = pol_coord_stack.shape[2]

    labels = zeros(pol_coord_stack.shape)

    labels[:,:,0] = partionn_circle(nb_parts,offset,pol_coord_stack[:,:,0])
    i0,j0 = where(pol_coord_stack[:,:,0]<2*pi)

    for i in range(1,nb_sets):
        i1,j1 = where(pol_coord_stack[:,:,i]<2*pi)

        opt_ind,proj = optimal_assignement_1d(pol_coord_stack[i1,j1,i],pol_coord_stack[i0,j0,0])
        labels[i1,j1,i] = labels[i0[opt_ind],j0[opt_ind],0]

    return labels


def extract_point_cloud(ring_stack,pol_coord_stack,part_id,normalize_en=True):
    from numpy.linalg import svd
    from numpy import sqrt as n_sqrt
    i,j = where(pol_coord_stack[:,:,0]==part_id)
    nb_points = len(i)*ring_stack.shape[2]
    point_cloud = zeros((3,nb_points))
    point_cloud[0:2,0:len(i)] =  array([i,j])
    point_cloud[2,0:len(i)] = ring_stack[i,j,0]

    for k in range(0,ring_stack.shape[2]):
        i,j = where(pol_coord_stack[:,:,k]==part_id)
        point_cloud[0:2,k*len(i):(k+1)*len(i)] =  array([i,j])
        point_cloud[2,k*len(i):(k+1)*len(i)] = ring_stack[i,j,k]

    if normalize_en:
        u,s,v = linalg.svd(point_cloud.dot(transpose(point_cloud)),full_matrices=True)
        point_cloud = u.dot(diag(n_sqrt(s)**(-1))).dot(transpose(u)).dot(point_cloud)
        scale_mat = u.dot(diag(n_sqrt(s))).dot(transpose(u))
        return point_cloud,scale_mat
    else:
        return point_cloud


def interpolation_setting(data,center=None,kmad=5,deg=2,win=9,win2=3,nb_ring_parts=10,offset=None,nb_components=2,neigh_fact=1.5,ring_wise=True):

    from numpy import arange,pi
    from sklearn import manifold
    from pyflann import FLANN
    from numpy.polynomial.polynomial import polyfit

    if center is None:
        center = where(data[:,:,0]==data[:,:,0].max())
        if size(center>2):
            center = array([center[0][0],center[1][0]])
    shap = data.shape
    nb_data = shap[2]


    # -- Features setting -- #

    # Rings detection
    map,poly_fit,final_nb_samples,pol_coord,widths,heights,mean_rad = ring_stack(data,center=center,kmad=kmad,deg=deg,win=win,win2=win2)
    scale_poly = polyfit(mean_rad.mean(axis=0),log(widths.mean(axis=0)/heights.mean(axis=0)),2)


    # Rings extraction and scaling
    # Rings extraction
    indexes = list()



    if ring_wise:
        indexes = list(arange(1,size(final_nb_samples)+1))
    else:
        indexes = [1,list(arange(2,size(final_nb_samples)+1))]



    list_rings_learning,list_pol_coord,list_points_clouds,nb_points = get_ring_stack(data,map,indexes,pol_coord[1:].reshape(data.shape[0:2]))


    """
    # Computing embeddings
    embeddings1 = list()
    embeddings2 = list()
    knn1 = list()
    knn2 = list()
    scale_mats_1 = list()
    scale_mats_2 = list()
    for j in range(0,nb_struct):
        embeddings1.append([])
        embeddings2.append([])
        knn1.append([])
        knn2.append([])
        scale_mats_1.append([])
        scale_mats_2.append([])
        for i in range(0,nb_ring_parts):
            pt_cloud1,scale_mat1 = extract_point_cloud(list_rings_learning[j],section_numb0,i+1)
            scale_mats_1[j].append(scale_mat1)
            Yhess_obj = manifold.LocallyLinearEmbedding(int(neigh_fact*nb_data), nb_comp,eigen_solver='dense',method='hessian').fit(transpose(pt_clouds_list[i]))
    embed_list.append(Yhess_obj)
    flann = FLANN()
    params = flann.build_index(array(Yhess_obj.embedding_, dtype=float64))
    knn_list.append(flann)
    """

    return list_rings_learning,list_points_clouds,nb_points,widths,heights,mean_rad,scale_poly


def merger(list_rings,list_points_clouds,indices):
    list_rings_merge = list()
    list_points_clouds_merge = list()

    for i in range(0,len(indices)):
        if size(indices[i])==1:
            list_rings_merge.append(list_rings[indices[i]])
        else:
            list_rings_merge.append(list_rings[indices[i][0]])
            for j in range(1,size(indices[i])):
                list_rings_merge[i]+=list_rings[indices[i][j]]
        list_points_clouds_merge.append(cube_to_point_cloud(list_rings_merge[i]))

    return list_rings_merge,list_points_clouds_merge






def interpolation_setting_2(data,center=None,kmad=5,deg=2,win=9,win2=3,offset=None,nb_components=2,nb_neigh=None ,ring_wise=False,equalize_en=False):

    from numpy import arange,pi
    from sklearn import manifold
    from pyflann import FLANN
    from numpy.polynomial.polynomial import polyfit
    from utils import cube_to_mat


    if nb_neigh<= round(nb_components * (nb_components + 3) / 2):
        nb_neigh = int(round(nb_components * (nb_components + 3) / 2)+1)

    if center is None:
        center = where(data[:,:,0]==data[:,:,0].max())
        if size(center>2):
            center = array([center[0][0],center[1][0]])
    shap = data.shape
    nb_data = shap[2]


    # -- Features setting -- #

    # Rings detection
    map,poly_fit,final_nb_samples,pol_coord,widths,heights,mean_rad = ring_stack(data,center=center,kmad=kmad,deg=deg,win=win,win2=win2,equalize_en=False)

    scale_poly = polyfit(mean_rad.mean(axis=0),log(widths.mean(axis=0)/heights.mean(axis=0)),2)

    # Rings extraction
    indexes = list()

    if ring_wise:
        indexes = list(arange(1,size(final_nb_samples)+1))
    else:
        indexes = [1,list(arange(2,size(final_nb_samples)+1))]



    list_rings_learning,list_pol_coord,list_points_clouds,nb_points = get_ring_stack(data,map,indexes,pol_coord[1:].reshape(data.shape[0:2]))


    # Computing embeddings
    embed_list = list()
    knn_list = list()

    for i in range(0,len(list_rings_learning)):
        Yhess_obj = manifold.LocallyLinearEmbedding(nb_neigh[i],nb_components,eigen_solver='dense',method='hessian').fit(cube_to_mat(list_rings_learning[i]))
        embed_list.append(Yhess_obj)
        flann = FLANN()
        params = flann.build_index(array(Yhess_obj.embedding_, dtype=float64))
        knn_list.append(flann)

    return embed_list,knn_list,list_rings_learning,list_points_clouds,scale_poly



def field_mapping(embed_list,pos_field):

    from optim_utils import optim_poly_fit_2d

    nb_struct = len(embed_list)
    poly_modes = list()
    deg = list()

    for i in range(0,nb_struct):
        s,mean_err,w,deg_i = optim_poly_fit_2d(pos_field,transpose(embed_list[i]))
        poly_modes.append(s)
        deg.append(deg_i)
        print "Position fittig accuracy: ",mean_err,"%"

    return poly_modes,deg

def sliced_transport_time_assessing_interface(stack,nb_pts,param,nb_real,nb_iter=1000,tol=0.001,alph = 0.0,basis=None,rand=True,disc_err=0.1,smart_init_en = True,rad=15,ind=None):

    from utils import rand_diff_integ_pair,window_ind
    from numpy import zeros
    from optim_utils import sliced_transport_time_assessing_m

    indf=None
    indg=None
    shap = stack.shape
    if ind is None:
        ind = rand_diff_integ_pair(0,shap[2],nb_pts)
    else:
        nb_pts = ind.shape[0]
    if smart_init_en:
        indf = zeros((nb_pts,(2*rad+1)**2))
        indg = zeros((nb_pts,(2*rad+1)**2))

        for i in range(0,nb_pts):
            cent1 = where(stack[:,:,ind[i,0]]==stack[:,:,ind[i,0]].max())
            cent2 = where(stack[:,:,ind[i,1]]==stack[:,:,ind[i,1]].max())
            indf[i,:] = window_ind(cent1,shap[0:2],rad)
            indg[i,:] = window_ind(cent2,shap[0:2],rad)
        indf = indf.astype(int)
        indg = indg.astype(int)

    pt1 = cube_to_point_cloud(stack[:,:,ind[:,0]])
    pt2 = cube_to_point_cloud(stack[:,:,ind[:,1]])

    pt1[0:2,:]*= param
    pt2[0:2,:]*= param

    return [sliced_transport_time_assessing_m(pt1,pt2,nb_real,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand,disc_err=disc_err,indf=indf,indg=indg,smart_init_en = smart_init_en),ind]

def sliced_transport_time_assessing_interface_2(im1,im2,param,nb_real,nb_iter=20000,tol=0.001,alph = 0.0,basis=None,rand=True,disc_err=0.1,smart_init_en = True,ind=None,monte_carlo=100,rad=15):
    from optim_utils import sliced_transport_time_assessing_2
    pt1 = im_to_point_cloud(im1)
    pt2 = im_to_point_cloud(im2)
    pt1[0:2,:]*= param
    pt2[0:2,:]*= param

    return sliced_transport_time_assessing_2(pt1,pt2,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,nb_real=nb_real,rand=rand,disc_err=disc_err,smart_init_en = smart_init_en,monte_carlo=monte_carlo,gap=param,shap=im1.shape,rad=rad)


def sliced_displacement_interp_test(im1,im2,param,nb_steps=10,nb_iter=1000,nb_real=10,tol=0.001,basis=None,rand=False,rad=15,smart_init_en=True,output_control=True,theta_en=False,max_dist=None,cent=None,alpha=0.1,norm_en=False,pf=None):
    from optim_utils import sliced_transport,setting_polar,setting_cart_from_polar,cloud_size_constraint,dist_map_2,lasso_constraint,energy_warping_metric,cloud_size_constraint_simp
    from utils import window_ind
    from numpy import copy,linspace,ones,pi,float
    from numpy.linalg import norm

    pt1 = im_to_point_cloud(im1)
    pt2 = im_to_point_cloud(im2)
    if pf is None:
        if theta_en:
            pt1[0:2,:] = setting_polar(pt1[0:2,:],param,cent,max_dist)
            pt2[0:2,:] = copy(pt1[0:2,:])
        else:
            pt1[0:2,:]*=param
            pt2[0:2,:]*=param
        iopt = None
        indf = None
        indg = None
        if smart_init_en:
            cent1 = where(im1==im1.max())
            cent2 = where(im2==im2.max())
            indf = window_ind(cent1,im1.shape,rad)
            indg = window_ind(cent2,im2.shape,rad)
        #print indf,"------",indg
        pf,dist,sig,i_opt = sliced_transport(pt1,pt2,nb_real=nb_real,nb_iter=nb_iter,tol=tol,alph = alpha,basis=basis,rand=rand,output_control=output_control,iter_control=False,gap=param,shap = im1.shape,rad=rad,indf=indf,indg=indg,smart_init_en=smart_init_en)

        if theta_en:
            pf[0:2,:] = setting_cart_from_polar(pf[0:2,:],param,cent,max_dist)
            pt1[0:2,:] = setting_cart_from_polar(pt1[0:2,:],param,cent,max_dist)
        else:
            pf[0:2,:]/=param
            pt1[0:2,:]/=param

    im_interp = zeros((im1.shape[0],im1.shape[1],nb_steps))
    t = linspace(0,1,nb_steps)
    cent_mat = cent.reshape((2,1)).dot(ones((1,pf.shape[1])))
    dist_ref = norm(cent_mat-pf[0:2,:])
    siz_i = sum((norm((pt1[0:2,:]-cent_mat),axis=0)**2)*pt1[2,:])
    siz_f = sum((norm((pf[0:2,:]-cent_mat),axis=0)**2)*pf[2,:])
    cloud = copy(pt1)
    angle_min = pi/3
    energy_distrib = zeros((nb_steps,))

    for i in range(0,nb_steps):
        if i<nb_steps-1 and i>0:
            cloud = (1-(float(nb_steps-i))**(-1))*cloud+((1+float(i-1)/(nb_steps-i))/(nb_steps-1))*pf
            if norm_en:
                cloud[0:2,:] = cloud_size_constraint_simp(cloud,cent,(1-t[i])*siz_i+t[i]*siz_f)
            print "cloud_err: ",norm(cloud-(1-t[i])*pt1-t[i]*pf)
        elif i==nb_steps-1:
            cloud = pf
        intensities,inv_dist_weights,energy_distrib[i],knn = energy_warping_metric(cloud,nb_neigh=4)
        dist_mat = dist_map_2(cloud[0:2,:])
        k,l = where(dist_mat>0)
        print "Mean dist before correction: ",dist_mat[k,l].mean()

        """if norm_en:
            #cloud[0:2,:] = lasso_constraint(cloud[0:2,:],im1.shape,0.4,angle_min,nb_iter=400,mu=0.0005,tol=1.e-2)
            cloud[0:2,:] = lasso_constraint((1-t[i])*pt1[0:2,:]-t[i]*pf[0:2,:],im1.shape,0.2,angle_min,nb_iter=400,mu=0.0005,tol=1.e-2)
            #cloud = cloud_size_constraint(cloud,cent,(1-t[i])*siz_i+t[i]*siz_f,dist_ref)
            #cloud[0:2,:] = dist_ref*(cloud[0:2,:]-cent_mat)/norm(cloud[0:2,:]-cent_mat) + cent_mat"""
        moment = sum((norm((cloud[0:2,:]-cent_mat),axis=0)**2)*cloud[2,:])
        print "Step ",i+1,"/",nb_steps
        print "Moment: ",moment
        print "Energy: ",norm(cloud[2,:])**2
        im_interp[:,:,i] = bar_1d_2d_bis(cloud,im1.shape,[0,0],pol_en=False,log_param=None,tol=10.0**(-15))
        print "Image Energy: ",norm(im_interp[:,:,i])**2


    return im_interp,pf,energy_distrib

def sliced_displacement_interp_test_2(im1,im2,param,cent=None,nb_iter=1000,nb_real=10,tol=1.e-15):
    from optim_utils import sliced_transport_single_dir
    from numpy import zeros,array
    from numpy.linalg import norm

    pt1 = im_to_point_cloud(im1)
    pt2 = im_to_point_cloud(im2)
    shap = im1.shape

    if cent is None:
        cent =  array(where(im1==im1.max()))

    pt1[0:2,:]*=param
    pt2[0:2,:]*=param
    rmin = 0.00000000000000001

    displacement = sliced_transport_single_dir(pt1,pt2,nb_iter=nb_iter,tol=tol,nb_real=nb_real)

    nb_time_steps = 11
    advect_summary = advection_synthesis(displacement,nb_time_steps=nb_time_steps)
    im_interp = zeros((im1.shape[0],im1.shape[1],nb_time_steps))

    for i in range(0,nb_time_steps):
        advect_summary[0:2,:,i]/=param
        im_interp[:,:,i] = bar_1d_2d_bis(advect_summary[:,:,i],im1.shape,[0,0],pol_en=False,log_param=None,tol=10.0**(-15))
        print "Image Energy: ",norm(im_interp[:,:,i])**2

    return im_interp




def sliced_displacement_interp_test_3(im1,im2,param,step_max,nb_iter=1000,nb_real=10,tol=1.e-15,basis=None,rand=False,rad=15,smart_init_en=True,output_control=True,theta_en=False,max_dist=None,cent=None,alpha=0.0,norm_en=False,pf=None,nb_time_steps = 11):
    from optim_utils import sliced_transport,setting_polar,setting_cart_from_polar,cloud_size_constraint,dist_map_2,lasso_constraint
    from utils import window_ind
    from numpy import copy,linspace,ones,pi,float,array
    from numpy.linalg import norm



    pt1 = im_to_point_cloud(im1)
    pt2 = im_to_point_cloud(im2)
    cent1 = None
    if pf is None:
        if theta_en:
            pt1[0:2,:] = setting_polar(pt1[0:2,:],param,cent,max_dist)
            pt2[0:2,:] = copy(pt1[0:2,:])
        else:
            pt1[0:2,:]*=param
            pt2[0:2,:]*=param
        iopt = None
        indf = None
        indg = None
        if smart_init_en:
            cent1 = where(im1==im1.max())
            cent2 = where(im2==im2.max())
            indf = window_ind(cent1,im1.shape,rad)
            indg = window_ind(cent2,im2.shape,rad)
        #print indf,"------",indg
        pf,dist,sig,i_opt = sliced_transport(pt1,pt2,nb_real=nb_real,nb_iter=nb_iter,tol=tol,alph = alpha,basis=basis,rand=rand,output_control=output_control,iter_control=False,gap=param,shap = im1.shape,rad=rad,indf=indf,indg=indg,smart_init_en=smart_init_en)

        if theta_en:
            pf[0:2,:] = setting_cart_from_polar(pf[0:2,:],param,cent,max_dist)
            pt1[0:2,:] = setting_cart_from_polar(pt1[0:2,:],param,cent,max_dist)
        else:
            pf[0:2,:]/=param
            pt1[0:2,:]/=param

    if cent1 is None:
        cent1 = where(im1==im1.max())

    advect_list = size_constraint_advection(pt1,pf,array(cent1),gap=1,step_max=step_max,tol=1.e-13,nb_steps=50000)

    advect_summary = advection_synthesis(advect_list,nb_time_steps=nb_time_steps)
    im_interp = zeros((im1.shape[0],im1.shape[1],nb_time_steps))

    for i in range(0,nb_time_steps):
        im_interp[:,:,i] = bar_1d_2d_bis(advect_summary[:,:,i],im1.shape,[0,0],pol_en=False,log_param=None,tol=10.0**(-15))
        print "Image Energy: ",norm(im_interp[:,:,i])**2

    return im_interp,pf




def size_constraint_advection(pt1,pf,cent,gap=1,step_max=None,tol=1.e-15,nb_steps=5000,imax = 200000,energy_cons=False,sheep_constraint=True):
    from numpy.linalg import norm,svd
    from optim_utils import shape_tg_proj,energy_warping_tg_proj
    from numpy import transpose,ones
    if step_max is None:
        step_max = norm(pt1-pf)/nb_steps

    print "step max: ",step_max

    var = norm(pt1-pf)
    i=0
    advect_list = list()
    advect_list.append(pt1)
    nb_points = pt1.shape[1]
    while(var>tol and i<imax):
        if energy_cons:
            ascent_dir = energy_warping_tg_proj(advect_list[i],advect_list[i]-pf,nb_neigh=4)
        elif sheep_constraint:
            intensity_weigh_mat = ones((2,1)).dot(advect_list[i][2,:].reshape((1,nb_points)))
            #U,s,V = svd((intensity_weigh_mat*(advect_list[i][0:2,:]-pf[0:2,:])).dot(transpose(intensity_weigh_mat*(advect_list[i][0:2,:]-pf[0:2,:]))))
            U,s,V = svd((advect_list[i]-pf).dot(transpose(advect_list[i]-pf)))
            #print "advection energy: ",s
            #ascent_dir = copy(advect_list[i]-pf)
            ascent_dir = U[:,0].reshape((3,1)).dot(U[:,0].reshape((1,3))).dot(advect_list[i]-pf)
        else:
            ascent_dir = shape_tg_proj(advect_list[i],advect_list[i]-pf,cent,gap=gap)

        step = min(step_max,sum((advect_list[i]-pf)*ascent_dir)/norm(ascent_dir)**2)
        #step = sum((advect_list[i]-pf)*ascent_dir)/norm(ascent_dir)**2
        cloud_i = advect_list[i] - step*ascent_dir
        var = norm(step*ascent_dir)
        print "Advection vector norm: ",var," Relative error: ",100*norm(cloud_i-pf)/norm(pf)
        advect_list.append(cloud_i)
        i+=1
    advect_list.append(pf)
    return advect_list

def advection_analysis(advect_list):
    from numpy import zeros
    from numpy.linalg import norm
    dist = zeros((len(advect_list)-1,))
    dist[0] = norm(advect_list[0]-advect_list[1])
    for i in range(1,len(advect_list)-1):
        dist[i] = dist[i-1]+norm(advect_list[i]-advect_list[i+1])
    return dist

def advection_synthesis(advect_list,nb_time_steps=11):
    from numpy import linspace,sort
    shap = advect_list[0].shape
    cumulated_dist = list(advection_analysis(advect_list))

    output = zeros((shap[0],shap[1],nb_time_steps))
    t = linspace(0,1,nb_time_steps)
    output[:,:,0] = advect_list[0]
    output[:,:,-1] = advect_list[-1]

    for i in range(1,nb_time_steps-1):
        geod_t = t[i]*cumulated_dist[-1]
        cumulated_dist.append(geod_t)
        sort_dist = sort(cumulated_dist)
        j = where(sort_dist==geod_t)
        output[:,:,i] = ((sort_dist[j[0]+1]-geod_t)*advect_list[j[0]-1]+(geod_t-sort_dist[j[0]-1])*advect_list[j[0]])/(sort_dist[j[0]+1]-sort_dist[j[0]-1])
        cumulated_dist.remove(geod_t)

    return output

def sliced_wassertein_dist_interface(im1,im2,param,nb_iter=1000,nb_real=10,tol=0.001,basis=None,rand=False,im_output=False,matched_output=False,rad=15,smart_init_en=True,output_control=True,theta_en=False,max_dist=None,cent=None,alpha=1.0):
    from optim_utils import sliced_transport,opt_assignment,setting_polar,setting_cart_from_polar
    from utils import window_ind
    import time
    from numpy import copy

    pt1 = im_to_point_cloud(im1)
    pt2 = im_to_point_cloud(im2)

    if theta_en:
        pt1[0:2,:] = setting_polar(pt1[0:2,:],param,cent,max_dist)
        pt2[0:2,:] = copy(pt1[0:2,:])
    else:
        pt1[0:2,:]*=param
        pt2[0:2,:]*=param
    iopt = None
    indf = None
    indg = None
    if smart_init_en:
        cent1 = where(im1==im1.max())
        cent2 = where(im2==im2.max())
        indf = window_ind(cent1,im1.shape,rad)
        indg = window_ind(cent2,im2.shape,rad)
    #print indf,"------",indg
    pf,dist,sig,i_opt,iter= sliced_transport(pt1,pt2,nb_real=nb_real,nb_iter=nb_iter,tol=tol,alph = alpha,basis=basis,rand=rand,output_control=output_control,iter_control=True,gap=param,shap = im1.shape,rad=rad,indf=indf,indg=indg,smart_init_en=smart_init_en)



    if theta_en:
        pf[0:2,:] = setting_cart_from_polar(pf[0:2,:],param,cent,max_dist)
        pt1[0:2,:] = setting_cart_from_polar(pt1[0:2,:],param,cent,max_dist)
    else:
        pf[0:2,:]/=param
        pt1[0:2,:]/=param
    matched = [pt1,pf]

    if im_output:
        projim1 = bar_1d_2d_bis(pf,im1.shape,[0,0],pol_en=False,log_param=None,tol=10.0**(-15))
        iter_im = zeros((projim1.shape[0],projim1.shape[1],iter.shape[2]))
        for i in range(0,iter.shape[2]):
            if theta_en:
                iter_im[0:2,:,i] = setting_cart_from_polar(iter_im[0:2,:,i],param,cent,max_dist)
            iter_im[:,:,i] = bar_1d_2d_bis(iter[:,:,i],im1.shape,[0,0],pol_en=False,log_param=None,tol=10.0**(-15))

        return dist,sig,projim1,i_opt,iter_im
    elif matched_output:
        return dist,sig,matched
    else:
        return dist,sig




def sliced_wassertein_dist_interface_target(stack,pos_field,target_pos,param,nb_iter=500,tol=10.0**(-10),basis=None,rand=True,nb_neigh=30):
    from numpy import zeros
    from pyflann import FLANN
    knn = FLANN()
    params = knn.build_index(array(pos_field, dtype=float64))


    # Computing the weights
    result, dists = knn.nn_index(target_pos, nb_neigh)
    dist = zeros((nb_neigh,nb_neigh))

    for i in range(0,nb_neigh-1):
        print "Observation ",i+1,"/",nb_neigh
        for j in range(i+1,nb_neigh):
            if dist[i,j]==0:
                dist[i,j],sig = sliced_wassertein_dist_interface(stack[:,:,result[0,i]],stack[:,:,result[0,j]],param,nb_iter=nb_iter,tol=tol,basis=basis,rand=rand)
            dist[j,i] = dist[i,j]

    return dist



def transport_pair_wise_distances(stack,pos_field,target_field,nb_neigh=30,gap=None,nb_iter=500,tol=10.0**(-10),basis=None,rand=False,sliced_dist=None,nb_real=10):
    from optim_utils import max_gap
    from numpy import zeros,arange,copy,argsort
    from pyflann import FLANN

    knn = FLANN()
    params = knn.build_index(array(pos_field, dtype=float64))

    if gap is None:
        gap,inear,jnear,dist_mat = max_gap(stack,pos_field,tol=0.001,nb_neigh=nb_neigh,max_val =2**63,dist_mat=None)

    nb_points = stack.shape[2]
    nb_val = stack.shape[0]*stack.shape[1]
    if sliced_dist is None:
        sliced_dist = zeros((nb_points,nb_points))

    nb_targets = target_field.shape[0]
    neighborhood = zeros((nb_targets,nb_neigh))
    permutations = zeros((nb_points,nb_val,nb_points)) # convention: the lines M[i,:,j] contains a permutation sigma verifying fi approx fj[sigma]
    Id = arange(0,nb_points).astype(int)

    for i in range(0,nb_targets):
        print i+1,"th target/",nb_targets
        results,dist = knn.nn_index(target_field[i,:], nb_neigh)
        neighborhood[i,:] = results[0,:]
        for k in range(0,nb_neigh):
            print k+1," neighbor/",nb_neigh
            for l in range(k+1,nb_neigh):
                if sliced_dist[results[0,k],results[0,l]]==0:
                    print "PSFs indices: ",results[0,k],results[0,l]
                    sliced_dist[results[0,k],results[0,l]],sig = sliced_wassertein_dist_interface(stack[:,:,results[0,k]],stack[:,:,results[0,l]],gap,nb_iter=nb_iter,tol=tol,basis=basis,rand=rand,nb_real=nb_real)
                    sliced_dist[results[0,l],results[0,k]] = sliced_dist[results[0,k],results[0,l]]
                    permutations[results[0,k],:,results[0,l]] = copy(sig)
                    permutations[results[0,l],:,results[0,k]] = argsort(sig)

    return sliced_dist,neighborhood.astype(int),permutations.astype(int)

def local_embedding_bar_coordinates(dist_mat,neighborhood,pos_field,target_pos,local_dim=10,nb_neigh_inv_map=None):
    from numpy import zeros,ones,copy,transpose
    from pyflann import FLANN
    from utils import bar_coord2d,mds
    from optim_utils import bar_coord_pb
    import time
    knn = FLANN()

    nb_neigh = neighborhood.shape[1]
    nb_points = target_pos.shape[0]
    if nb_neigh_inv_map is None:
        nb_neigh_inv_map = nb_neigh
    else:
        nb_neigh_inv_map = min(nb_neigh,nb_neigh_inv_map)
    weights = zeros((nb_points,nb_neigh_inv_map))
    ones_line = ones((1,nb_neigh_inv_map))
    loc_eig_val = zeros((nb_points,local_dim))
    if nb_neigh_inv_map is None:
        nb_neigh_inv_map = local_dim+1

    acc = zeros((nb_points,))
    neighborhood_inv = zeros((nb_points,nb_neigh_inv_map))
    list_dist = list()
    list_dim = list()

    if local_dim>=nb_neigh_inv_map:
        local_dim = nb_neigh_inv_map-1

    tmean = 0


    for i in range(0,nb_points):
        t = time.time()
        # Calculating local embedded coordinates
        neigh_indices0 = ((neighborhood[i,0:nb_neigh_inv_map].reshape((nb_neigh_inv_map,1))).dot(ones_line)).astype(int)
        dist_i = copy(dist_mat[neigh_indices0,transpose(neigh_indices0)])
        #print "max dist check: ",dist_i.max()," Number non zeros: ",size(where(dist_mat[i,:]>0)[0])
        loc_coord_i,w = mds(dist_i,nb_components = local_dim) # Embedded coordiantes are in loc_coord_i lines
        list_dim.append(w)
        # Interpolating at the target position according to the target coodiantes
        loc_coord_interp = zeros((local_dim,))
        for p in range(0,local_dim):
            rbfp = Rbf(pos_field[neighborhood[i,range(0,nb_neigh_inv_map)],0], pos_field[neighborhood[i,range(0,nb_neigh_inv_map)],1], loc_coord_i[:,p], function='thin_plate')
            loc_coord_interp[p] = rbfp(target_pos[i,0],target_pos[i,1])
        # Calculating barycentric coordinates
        #params = knn.build_index(array(loc_coord_i, dtype=float64))

        #results,dist = knn.nn_index(loc_coord_interp, nb_neigh_inv_map)
        #list_dist.append(dist)
        neighborhood_inv[i,:] = copy(neighborhood[i,range(0,nb_neigh_inv_map)])
        #wi,acc[i] = bar_coord2d(transpose(loc_coord_i[results[0,:],:]),loc_coord_interp.reshape((local_dim,1)),cond=1./50,acc=True)
        wi,acc[i] = bar_coord_pb(transpose(loc_coord_i),loc_coord_interp.reshape((local_dim,1)),300)
        print "Percentage of accuracy: ",acc[i]
        weights[i,:] = wi.reshape((nb_neigh_inv_map,))
        tmean += time.time()-t

    print "Elapsed time", tmean/nb_points
    return weights,acc,neighborhood_inv.astype(int)#,list_dist,list_dim



def neighborhood_analysis(dist_mat,intrinsic_dim,target_pos,pos_field,max_size=40,tol=1,dim_max = 20,n=3):
    from numpy import ones,transpose,array,argsort,arange
    from utils import mds
    from pyflann import FLANN
    from numpy.polynomial.polynomial import polyfit,polyval


    knn = FLANN()
    params = knn.build_index(array(pos_field, dtype=float64))

    neighborhood = list()
    embeddings = list()
    eig_val_r = list()
    obs_dist = list()


    for i in range(0,target_pos.shape[0]):
        nb_neigh = intrinsic_dim+3
        eig_r = 0
        eig_val_ri = list()
        obs_dist_i = list()
        loc_coord_i = None
        results,dist = knn.nn_index(target_pos[i,:],1)
        for nb_neigh in range(intrinsic_dim+1,max_size):
            ones_line = ones((1,nb_neigh))
            ind_neigh = ((argsort(dist_mat[results[0],:])[0:nb_neigh].reshape((nb_neigh,1))).dot(ones_line)).astype(int)
            dist_loc = dist_mat[ind_neigh,transpose(ind_neigh)]
            loc_coord_i,w = mds(dist_loc,nb_components = dim_max)
            obs_dist_i.append(dist_loc.max())
            eig_val_ri.append(sum(w[2:]))
            nb_neigh+=1
        eig_val_r.append(array(eig_val_ri))
        obs_dist.append(array(obs_dist_i))

    r = array(obs_dist).reshape((size(array(obs_dist)),))
    eig_fit = array(eig_val_r).reshape((size(array(eig_val_r)),))
    a = sum(eig_fit*r**n)/sum(r**(2*n))

    fit_val = a*(r.max()*arange(0,max_size)/(max_size-1))**n

    return eig_val_r,obs_dist,fit_val,array(obs_dist).max()*arange(0,max_size)/(max_size-1),a

def neighborhood_analysis_2(pos_target,pos_ref,nb_neigh):
    from numpy import zeros,ones
    from numpy.linalg import norm
    from pyflann import FLANN

    knn = FLANN()
    params = knn.build_index(array(pos_ref, dtype=float64))
    results,dist = knn.nn_index(pos_target,nb_neigh)
    shap = pos_target.shape
    mean_dist = zeros((shap[0],))
    neigh_compactness = zeros((shap[0],))
    ones_vect = ones((nb_neigh,1))

    for i in range(0,shap[0]):
        bar = pos_ref[results[i,:],:].mean(axis=0)
        mean_dist[i] = mean(norm(ones_vect.dot(bar.reshape((1,shap[1])))-pos_ref[results[i,:],:],axis=1))

        neigh_compactness[i] = norm(bar-pos_target[i,:])/mean_dist[i]

    return mean_dist,neigh_compactness

def neighborhood_analysis_3(pos_target,pos_ref,nb_neigh):
    from numpy import zeros,ones,pi,arange,sort
    from numpy.linalg import norm
    from pyflann import FLANN
    from utils import polar_coord

    knn = FLANN()
    params = knn.build_index(array(pos_ref, dtype=float64))
    results,dist = knn.nn_index(pos_target,nb_neigh)
    shap = pos_target.shape
    mean_dist = zeros((shap[0],))
    neigh_isotropy = zeros((shap[0],))
    ones_vect = ones((nb_neigh,1))
    angle_vect = (2*pi)*arange(0,nb_neigh)/nb_neigh

    for i in range(0,shap[0]):
        bar = pos_ref[results[i,:],:].mean(axis=0)
        mean_dist[i] = mean(norm(ones_vect.dot(bar.reshape((1,shap[1])))-pos_ref[results[i,:],:],axis=1))
        ang = zeros((nb_neigh,))
        for j in range(0,nb_neigh):
            r,ang[j] = polar_coord(pos_ref[results[i,j],:],pos_target[i,:])

        neigh_isotropy[i] = norm(angle_vect-sort(ang))/nb_neigh

    return mean_dist,neigh_isotropy


def neighborhood_analysis_4(pos_target,pos_ref,nb_neigh,coherence_mat):
    from numpy import zeros,ones,pi,arange,sort,transpose
    from numpy.linalg import norm
    from pyflann import FLANN
    from utils import polar_coord

    knn = FLANN()
    params = knn.build_index(array(pos_ref, dtype=float64))
    results,dist = knn.nn_index(pos_target,nb_neigh)
    shap = pos_target.shape
    mean_dist = zeros((shap[0],))
    neigh_coherence = zeros((shap[0],))
    ones_vect = ones((nb_neigh,1))
    angle_vect = (2*pi)*arange(0,nb_neigh)/nb_neigh

    for i in range(0,shap[0]):
        bar = pos_ref[results[i,:],:].mean(axis=0)
        mean_dist[i] = mean(norm(ones_vect.dot(bar.reshape((1,shap[1])))-pos_ref[results[i,:],:],axis=1))
        ind = ones_vect.dot(results[i,:].reshape((1,nb_neigh)))
        ind = ind.astype(int)
        neigh_coherence[i] = coherence_mat[ind,transpose(ind)].max()

    return mean_dist,neigh_coherence



def param_dispersion(ell,dist_map):
    from numpy import zeros,argsort

    nb_neigh = ell.shape[0]

    disp_map = zeros((nb_neigh,nb_neigh-1))

    for i in range(0,nb_neigh):
        ind_i = argsort(dist_map[i,:])
        for j in range(1,nb_neigh):
            disp_map[i,j-1] = max(ell[ind_i[0:j],:].std(axis=0))

    return disp_map



def local_sliced_transport_bar_interface_2(stack,gap,dist_mat,neighborhood,pos_ref,target_ref,nb_iter=500,local_dim=10,nb_neigh_inv_map=11,output_path='../../../Data/Result_interp/',id=None,return_all=False,cent=None,theta_en=False):

    weights,acc,neighborhood_inv = local_embedding_bar_coordinates(dist_mat,neighborhood,pos_ref,target_ref,local_dim=local_dim,nb_neigh_inv_map=nb_neigh_inv_map)
    psf_interp_local_wass,sig = local_sliced_transport_bar_interface(stack,gap,neighborhood_inv,weights,nb_iter=nb_iter,tol=10.0**(-20),single_dir=False,output_path=output_path,save_file_en = True,id=id,cent=cent,theta_en=theta_en)
    return psf_interp_local_wass

def euclidian_interp(stack,neighborhood_inv,weights):
    from numpy import zeros,diag
    from utils import cube_to_mat,mat_to_cube,transpose

    shap = stack.shape
    interp = zeros((shap[0],shap[1],weights.shape[0]))
    for i in range(0,weights.shape[0]):
        interp[:,:,i] = (transpose(cube_to_mat(stack[:,:,neighborhood_inv[i,:]])).dot(weights[i,:].reshape((weights.shape[1],1)))).reshape((shap[0],shap[1]))
    return interp

def local_euclidean_bar(stack,dist_mat,neighborhood,pos_ref,target_ref,local_dim=10,nb_neigh_inv_map=11):

    weights,acc,neighborhood_inv = local_embedding_bar_coordinates(dist_mat,neighborhood,pos_ref,target_ref,local_dim=local_dim,nb_neigh_inv_map=nb_neigh_inv_map)
    psf_interp_local_euc = euclidian_interp(stack,neighborhood_inv,weights)

    return psf_interp_local_euc

def local_sliced_transport_bar_interface(stack,gap,neighborhood,weights,nb_iter=500,tol=10.0**(-20),single_dir=False,output_path='',save_file_en = False,rad=15,id='',cent=None,theta_en=False):
    from numpy import zeros,copy,squeeze
    from optim_utils import sliced_transport_bar,sliced_transport_ptbar,setting_polar_2,setting_cart_from_polar_2
    from astropy.io import fits
    from utils import compute_centroid
    import time
    pt_clouds = cube_to_point_cloud(stack)
    if theta_en:
        if cent is None:
            im_mean = stack.mean(axis=2)
            cent,w = compute_centroid(im_mean,sigw=100000000)
            cent = squeeze(cent)

        for i in range(0,pt_clouds.shape[2]):
            pt_clouds[0:2,:,i] = setting_polar_2(pt_clouds[0:2,:,i],gap,cent)
    else:
        pt_clouds[0:2,:,:]*=gap
    nb_points = weights.shape[0]
    shap = stack.shape[0:2]
    psf_interp = zeros((shap[0],shap[1],nb_points))
    sig = list()
    tmean = 0

    for i in range(0,nb_points):
        print "Target ",i+1,"/",nb_points
        t = time.time()
        w = copy(weights[i,:])
        if w.min()<0:
            ind = where(w<0)
            w[ind[0]] = 0
            w/=w.sum()
        ind = where(w==w.max())
        #bar,basis,sig_out = sliced_transport_bar(pt_clouds[:,:,neighborhood[i,:]],w,nb_iter=nb_iter,tol=tol,alph=0.01,bar_init=pt_clouds[:,:,neighborhood[i,ind[0][0]]],rand_en=True,basis=None,adapted_basis=False,pos_en=False,min_basis=False,support=False,assign_out=True,single_dir=single_dir)
        bar = sliced_transport_ptbar(pt_clouds[:,:,neighborhood[i,:]],w,nb_iter=nb_iter,tol=tol,alph=0.01,bar_init=pt_clouds[:,:,neighborhood[i,ind[0][0]]],rand_en=True,basis=None,adapted_basis=False,pos_en=False,min_basis=False,support=False,assign_out=True,single_dir=single_dir,shap=shap,gap=gap,rad=rad)
        if theta_en:
            bar[0:2,:] = setting_cart_from_polar_2(bar[0:2,:],gap,cent,max_dist)
        else:
            bar[0:2,:]/=gap
        #sig.append(sig_out)
        psf_interp[:,:,i] = bar_1d_2d_bis(bar,shap,[0,0],pol_en=False,log_param=None,tol=10.0**(-15))
        print "elapsed time: ",time.time()-t
        if save_file_en:
            fits.writeto(output_path+'psf_'+str(i)+id+'.fits',psf_interp[:,:,i])

    return psf_interp,sig

def get_data(output_path,shap,ids,nb_points):
    from numpy import zeros
    from astropy.io import fits
    output = list()
    nb_samp = len(ids)
    for i in range(0,nb_samp):
        nb_reals = min(10,len(ids[i]))
        im = zeros((shap[0],shap[1],nb_points,nb_reals))
        print "samp ",i+1,"/",nb_samp
        for j in range(0,nb_reals):
            print "real ",j+1,"/",nb_reals
            for k in range(0,nb_points):
                im[:,:,k,j] = fits.getdata(output_path+'psf_'+str(k)+ids[i][j]+'.fits')
        output.append(im)
    return output


def sliced_transport_bar_interface(stack,gap,weights,nb_iter=500,tol=10.0**(-10),pt_en=False,nb_real=1,rad=15,smart_init_en=True,theta_en=True,cent=None,max_dist=None):
    from numpy import zeros,copy
    from optim_utils import sliced_transport_bar,sliced_transport_ptbar,setting_polar_2,setting_cart_from_polar_2

    pt_clouds = cube_to_point_cloud(stack)
    if theta_en:
        for i in range(0,pt_clouds.shape[2]):
            pt_clouds[0:2,:,i] = setting_polar_2(pt_clouds[0:2,:,i],gap,cent)
    else:
        pt_clouds[0:2,:,:]*=gap
    shap = stack.shape[0:2]
    i = where(weights==weights.max())
    bar = None
    if pt_en:
        bar = sliced_transport_ptbar(pt_clouds,weights,nb_iter=nb_iter,tol=tol,alph=0.0,bar_init=pt_clouds[:,:,i[0][0]],rand_en=True,basis=None,adapted_basis=False,pos_en=False,min_basis=False,support=False,assign_out=True,nb_real=nb_real,shap=shap,gap=gap,rad=rad,smart_init_en=smart_init_en)
    else:
        bar,basis,sig_out = sliced_transport_bar(pt_clouds,weights,nb_iter=nb_iter,tol=tol,alph=0.0,bar_init=pt_clouds[:,:,i[0][0]],rand_en=True,basis=None,adapted_basis=False,pos_en=False,min_basis=False,support=False,assign_out=True,nb_real=nb_real)
    if theta_en:
        bar[0:2,:] = setting_cart_from_polar_2(bar[0:2,:],gap,cent,max_dist)
    else:
        bar[0:2,:]/=gap

    return bar_1d_2d_bis(bar,shap,[0,0],pol_en=False,log_param=None,tol=10.0**(-15))

def local_sliced_transport_bar_interface_m(stack,gap,dist_mat,neighborhood,nb_neigh_inv_map,pos_field,target_pos,output_path,local_dim=10,nb_iter=500,save_file_en = True):
    import time
    nb_real = len(nb_neigh_inv_map)

    output = list()
    tmean = time.time
    for i in range(0,nb_real):
        print i+1," th/",nb_real

        weights,acc,neighborhood_inv = local_embedding_bar_coordinates(dist_mat,neighborhood,pos_field,target_pos,local_dim=local_dim,nb_neigh_inv_map=nb_neigh_inv_map[i])
        output.append(local_sliced_transport_bar_interface(stack,gap,neighborhood_inv,weights,nb_iter=nb_iter,tol=10.0**(-10),single_dir=False,output_path=output_path[i],save_file_en = save_file_en)[0])

    return output


def sliced_transport_ptbar_interface(stack,weights,gap,nb_iter=500,tol=10.0**(-10),single_dir=False,output_path='',save_file_en = True,cent=None,pol=False,theta_w = 1):
    from numpy import zeros,copy,pi
    from optim_utils import sliced_transport_bar,sliced_transport_ptbar,setting_polar
    from astropy.io import fits
    from utils import polar_to_cart_cloud
    pt_clouds = cube_to_point_cloud(stack)
    shap = stack.shape[0:2]
    if pol:
        for i in range(0,stack.shape[2]):
            pt_clouds[0:2,:,i] = setting_polar(pt_clouds[0:2,:,i],gap,cent)
        pt_clouds[1,:,:]*=theta_w
    else:
        pt_clouds[0:2,:,:]*=gap

    bar = sliced_transport_ptbar(pt_clouds,weights,nb_iter=nb_iter,tol=tol,alph=0.01,rand_en=True,basis=None,adapted_basis=False,pos_en=False,min_basis=False,support=False,assign_out=True,single_dir=single_dir,gap=gap,shap=shap)

    bar[0:2,:]/=gap
    bar[1,:]/=theta_w
    if pol:
        bar[1,:]*=pi
        bar[0:2,:] = polar_to_cart_cloud(bar[0:2,:],cent)

    #sig.append(sig_out)

    psf_interp = bar_1d_2d_bis(bar,shap,[0,0],pol_en=False,log_param=None,tol=10.0**(-15))

    return psf_interp

def get_res(nb_psf,shap,dir):
    from astropy.io import fits
    from numpy import zeros

    output = zeros((shap[0],shap[1],nb_psf))
    for i in range(0,nb_psf):
        output[:,:,i] = fits.getdata(dir+'psf_'+str(i)+'.fits')

    return output

def get_res_m(nb_psf,shap,dir_list):
    nb_res = len(dir_list)
    output = list()
    for i in range(0,nb_res):
        output.append(get_res(nb_psf,shap,dir_list[i]))

    return output


def local_sliced_dist_interface(stack,pos_field,scale_poly,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False,nb_neigh=30,cent=None,weights=None):
    from optim_utils import local_sliced_dist_mat
    from numpy import exp,copy
    from numpy.polynomial.polynomial import polyval

    shap_clouds = stack.shape
    if weights is None:
        weights = zeros((shap_clouds[2],shap_clouds[1]))
        for i in range(0,shap_clouds[2]):
            rad_dist = sqrt(sum((stack[0:2,:,i]-array(cent).reshape((2,1)).dot(ones((1,shap_clouds[1]))))**2,axis=0))
            a = exp(polyval(rad_dist,scale_poly))
            weights[i,:] = a

    stackw = copy(stack)
    for i in range(0,shap_clouds[2]):
        stackw[0:2,:,i]/=weights[i,:]

    return local_sliced_dist_mat(stackw,pos_field,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand,nb_neigh=nb_neigh),weights

def local_sliced_dist_target_interface(stack,pos_field,target_pos,scale_poly,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False,nb_neigh=30,cent=None,weights=None):
    from optim_utils import local_sliced_dist_mat_target

    from numpy import exp,copy
    from numpy.polynomial.polynomial import polyval

    shap_clouds = stack.shape
    if weights is None:
        weights = zeros((shap_clouds[2],shap_clouds[1]))
        for i in range(0,shap_clouds[2]):
            rad_dist = sqrt(sum((stack[0:2,:,i]-array(cent).reshape((2,1)).dot(ones((1,shap_clouds[1]))))**2,axis=0))
            a = exp(polyval(rad_dist,scale_poly))
            weights[i,:] = a

    stackw = copy(stack)
    for i in range(0,shap_clouds[2]):
        stackw[0:2,:,i]/=weights[i,:]

    return local_sliced_dist_mat_target(stackw,pos_field,target_pos,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand,nb_neigh=nb_neigh)


def local_sliced_dist_supp_interface(stack,pos_field,scale_poly,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False,nb_neigh=30,cent=None,weights=None,max_val = 2**63,knn=None):
    from optim_utils import local_sliced_dist_mat_supp
    from numpy import exp,copy
    from numpy.polynomial.polynomial import polyval
    from pyflann import FLANN
    if knn is None:
        knn = FLANN()
        params = knn.build_index(array(pos_field, dtype=float64))
    shap_clouds = stack.shape
    if weights is None:
        weights = zeros((shap_clouds[2],shap_clouds[1]))
        for i in range(0,shap_clouds[2]):
            rad_dist = sqrt(sum((stack[0:2,:,i]-array(cent).reshape((2,1)).dot(ones((1,shap_clouds[1]))))**2,axis=0))
            a = exp(polyval(rad_dist,scale_poly))
            weights[i,:] = a

    stackw = copy(stack)
    for i in range(0,shap_clouds[2]):
        stackw[0:2,:,i]/=weights[i,:]

    return local_sliced_dist_mat_supp(stackw,pos_field,knn=knn,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand,nb_neigh=nb_neigh,max_val=max_val),weights



def local_hybrid_distance(im_stack,clouds_stack,pos_field,scale_poly,nb_iter=1000,tol=0.000000000001,alph = 0.1,basis=None,rand=False,nb_neigh=30,cent=None,weights=None,max_val = 2**63):
    from numpy import sqrt
    from optim_utils import local_euc_dist_mat_support
    from pyflann import FLANN

    knn = FLANN()
    params = knn.build_index(array(pos_field, dtype=float64))

    dist_mat_1,weights = local_sliced_dist_supp_interface(clouds_stack,pos_field,scale_poly,knn=knn,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand,nb_neigh=nb_neigh,cent=cent,weights=weights,max_val = max_val)
    dist_mat_2 = local_euc_dist_mat_support(im_stack,pos_field,knn=knn,nb_neigh=nb_neigh,max_val =max_val)

    return sqrt(dist_mat_1**2+dist_mat_2**2)



def local_sliced_dist_interface_m(stack_list,pos_field,scale_poly,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False,nb_neigh=30,cent=None,weights=None):

    dist_list = list()
    for i in range(0,len(stack_list)):
        print "Feature ",i+1
        dist_i,weights = local_sliced_dist_interface(stack_list[i],pos_field,scale_poly,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand,nb_neigh=nb_neigh,cent=cent,weights=weights)
        dist_list.append(dist_i)

    return dist_list


def local_sliced_hybrid_dist_m(im_stack_list,cloud_stack_list,pos_field,scale_poly,nb_iter=1000,tol=0.001,alph = 0.1,basis=None,rand=False,nb_neigh=30,cent=None,weights=None):

    dist_list = list()
    for i in range(0,len(im_stack_list)):
        print "Feature ",i+1
        dist_i = local_hybrid_distance(im_stack_list[i],cloud_stack_list[i],pos_field,scale_poly,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand,nb_neigh=nb_neigh,cent=cent,weights=weights)
        dist_list.append(dist_i)

    return dist_list




def LLE_interpolation(pos,shap,embed_list,knn_list,poly_modes,deg,list_rings,nb_neigh=None,nb_struct_max=None):

    from utils import eval_vect_poly
    from numpy import zeros,transpose

    psf = zeros(shap)
    nb_struct = len(embed_list)
    auto_nb_neigh = False
    if nb_neigh is None:
        auto_nb_neigh = True

    weights = list()
    struct_neigh = list()

    if nb_struct_max is None:
        nb_struct_max = nb_struct



    for i in range(0,nb_struct_max):

        if auto_nb_neigh:
            nb_neigh = embed_list[i].shape[0]+1
        embed_coord = eval_vect_poly(pos[0],pos[1],poly_modes[i],deg[i])

        struct_i,weights_i,struct_neigh_i = inverse_mapping_2D(embed_coord,embed_list[i],list_rings[i],knn_list[i],nb_neigh=nb_neigh)
        psf+= struct_i
        weights.append(weights_i.reshape((nb_neigh,)))
        struct_neigh.append(struct_neigh_i.reshape((nb_neigh,)))

    return psf,weights,struct_neigh

def LLE_rbf_interpolation(target_pos,pos_field,shap,embed_coord,psf_ref,knn_pos=None,knn_embed=None,nb_neigh=20):
    from numpy import transpose,zeros,array
    from pyflann import FLANN

    if knn_pos is None:
        knn_pos = FLANN()
        params1 = knn_pos.build_index(array(pos_field, dtype=float64))

    if knn_embed is None:
        knn_embed = FLANN()
        params2 = knn_embed.build_index(array(transpose(embed_coord), dtype=float64))

    nb_points = target_pos.shape[0]

    psf_interp = zeros((shap[0],shap[1],nb_points))
    dim = embed_coord.shape[0]
    weights = list()

    for i in range(0,nb_points):
        print "PSF",i+1,"/",nb_points
        result_pos, dists = knn_pos.nn_index(target_pos[i,:], nb_neigh)
        embed_coord_i = zeros((dim,))
        for j in range(0,dim):
            rbf = Rbf(pos_field[result_pos[0,:],0], pos_field[result_pos[0,:],1], embed_coord[j,result_pos[0,:]], function='thin_plate')
            embed_coord_i[j] = rbf(target_pos[i,0],target_pos[i,1])
        psf_interp[:,:,i],weights_i,struct_neigh_i = inverse_mapping_2D(embed_coord_i,emax_mbed_coord,psf_ref,knn_embed,nb_neigh=dim+1)
        weights.append(weights_i)

    return psf_interp,knn_embed,knn_pos,array(weights)


#w_inverse_mapping_2D(embed_coord,embed_ref,original_ref,flann_obj,basis,nb_neigh=3,nb_iter=1000)


def W_LLE_interpolation(pos,shap,embed_list,knn_list,poly_modes,coeff_poly,cent,deg,list_points_clouds,basis,nb_neigh=None,nb_iter = 1000,support=True,nb_struct_max=None):

    from utils import eval_vect_poly
    from numpy import zeros,transpose,sqrt
    from numpy.polynomial.polynomial import polyval

    psf = zeros(shap)
    nb_struct = len(embed_list)
    auto_nb_neigh = False
    if nb_neigh is None:
        auto_nb_neigh = True


    weights = list()
    struct_neigh = list()

    if nb_struct_max is None:
        nb_struct_max = nb_struct

    for i in range(0,nb_struct_max):

        embed_coord = eval_vect_poly(pos[0],pos[1],poly_modes[i],deg)

        if auto_nb_neigh:
            nb_neigh = embed_list[i].shape[0]+1

        bar_i,weights_i,struct_neigh_i = w_inverse_mapping_2D(embed_coord,embed_list[i],list_points_clouds[i],knn_list[i],cent,coeff_poly,basis,nb_neigh=nb_neigh,nb_iter=nb_iter,support=support)
        shap_clouds = list_points_clouds[i].shape
        rad_dist_bar = sqrt(sum((bar_i[0:2,:]-array(cent).reshape((2,1)).dot(ones((1,shap_clouds[1]))))**2,axis=0))
        if support:
            bar_i[0:2,:] += exp(polyval(rad_dist_bar,coeff_poly))
        else:
            bar_i[2,:] /= exp(polyval(rad_dist_bar,coeff_poly))

        struct_i = bar_1d_2d_bis(bar_i,shap,cent,pol_en=False,log_param=None,tol=10.0**(-15))
        psf+= struct_i
        weights.append(weights_i.reshape((nb_neigh,)))
        struct_neigh.append(struct_neigh_i.reshape((nb_neigh,)))

    return psf,weights,struct_neigh


def im_weighting(im,cent,scale_poly):
    from numpy import sqrt,exp
    from numpy.polynomial.polynomial import polyval
    shap = im.shape
    im_out = copy(im)
    for i in range(0,shap[0]):
        for j in range(1,shap[1]):
            im_out[i,j]/=exp(polyval(sqrt((i-cent[0])**2+(j-cent[1])**2),scale_poly))

    return im_out

def im_deweighting(im,cent,scale_poly):
    from numpy import sqrt,exp
    from numpy.polynomial.polynomial import polyval
    shap = im.shape
    im_out = copy(im)
    for i in range(0,shap[0]):
        for j in range(1,shap[1]):
            im_out[i,j]*=exp(polyval(sqrt((i-cent[0])**2+(j-cent[1])**2),scale_poly))

    return im_out


def im_weighting_stack(im_stack,cent,scale_poly):

    im_out = im_stack*0
    for i in range(0,im_out.shape[2]):
        im_out[:,:,i] = im_weighting(im_stack[:,:,i],cent,scale_poly)

    return im_out

def im_deweighting_stack(im_stack,cent,scale_poly):

    im_out = im_stack*0
    for i in range(0,im_out.shape[2]):
        im_out[:,:,i] = im_deweighting(im_stack[:,:,i],cent,scale_poly)

    return im_out

def LLE_interpolation_m(pos,shap,embed_list,knn_list,poly_modes,deg,list_rings,nb_neigh=None,nb_struct_max=4):

    from numpy import zeros

    nb_points = pos.shape[0]
    psf_interp = zeros((shap[0],shap[1],nb_points))

    for i in range(0,nb_points):
        psf_interp[:,:,i],weights,struct_neigh = LLE_interpolation(pos[i,:],shap,embed_list,knn_list,poly_modes,deg,list_rings,nb_neigh=nb_neigh,nb_struct_max = nb_struct_max)

    return psf_interp

def LLE_approx_interface(data_cube,dist_mat,nb_neigh,nb_comp):
    from optim_utils import approx_HLLE
    from utils import cube_to_mat,mat_to_cube
    from numpy import transpose

    shap = data_cube.shape

    embedding,loc_comp,data_approx = approx_HLLE(transpose(cube_to_mat(data_cube)),dist_mat,nb_neigh,nb_comp)



    return embedding,loc_comp,mat_to_cube(transpose(data_approx),shap[0],shap[1])

def pair_wise_distance_interface(data_cube,dist_mat,full_output=False,details_mat=None,nb_iter=100):
    from optim_utils import pair_wise_distances_constraint
    from utils import cube_to_mat,mat_to_cube
    from numpy import transpose


    shap = data_cube.shape
    data_mat = transpose(cube_to_mat(data_cube))

    details,U,details_mat = pair_wise_distances_constraint(data_mat,dist_mat,details_mat=details_mat,nb_iter=nb_iter)

    if full_output:
        return mat_to_cube(transpose(details),shap[0],shap[1]),U,details_mat
    else:
        return mat_to_cube(transpose(details),shap[0],shap[1])

def pair_wise_distance_lagrangian_interface(data_cube,dist_ref,dist_target,beta=10):
    from optim_utils import pair_wise_distances_constraint_lagrangian
    from utils import cube_to_mat,mat_to_cube
    from numpy import transpose
    shap = data_cube.shape
    data_mat = transpose(cube_to_mat(data_cube))


    data_tightened,const_mat = pair_wise_distances_constraint_lagrangian(data_mat,dist_ref,dist_target,beta=beta)

    return mat_to_cube(transpose(data_tightened),shap[0],shap[1]),const_mat


def local_reduction_interface(data_cube,nb_neigh,nb_comp,dist_mat):
    from optim_utils import local_reduction
    from utils import cube_to_mat,mat_to_cube
    from numpy import transpose


    shap = data_cube.shape

    return mat_to_cube(transpose(local_reduction(transpose(cube_to_mat(data_cube)),nb_neigh,nb_comp,dist_mat)),shap[0],shap[1])


def WLLE_interpolation_m(pos,shap,embed_list,knn_list,poly_modes,coeff_poly,cent,deg,list_points_clouds,basis,nb_neigh=None,nb_iter=1000,support=True,nb_struct_max=None):

    from numpy import zeros



    nb_points = pos.shape[0]
    psf_interp = zeros((shap[0],shap[1],nb_points))

    for i in range(0,nb_points):
        print i+1,'th PSF/',nb_points

        psf_interp[:,:,i],weights,struct_neigh = W_LLE_interpolation(pos[i,:],shap,embed_list,knn_list,poly_modes,coeff_poly,cent,deg,list_points_clouds,basis,nb_neigh=nb_neigh,nb_iter = 1000,support=support)

    return psf_interp


def LLE_interpolation_m2(data,ref_pos,target,pos,embed_list,knn_list,deg_mapping,list_rings,center=None,kmad=5,deg=2,win=9,win2=7,offset=None,nb_components=2,nb_neigh_embedding=None ,ring_wise=True,nb_neigh_inverse_mapping=5): # Interpolation of different data sets

    shap = data.shape[0:2]
    nb_sets = len(data)
    psf_interp = list()
    for i in range(0,nb_sets):
        embed_list,knn_list,list_rings_learning = interpolation_setting_2(data[i],center=center[i,:],kmad=kmad,deg=deg,win=win,win2=win2,offset=offset,nb_components=nb_components,nb_neigh=nb_neigh_embedding,ring_wise=ring_wise)
        poly_modes = field_mapping(embed_list,ref_pos[i],deg=deg_mapping)
        psf_interp.append(LLE_interpolation_m(target_pos[i],shap,embed_list,knn_list,poly_modes,deg,list_rings_learning,nb_neigh=nb_neigh_inverse_mapping))

    return psf_interp




def rings_barycenter(list_points_clouds,w,scaling_fact,shap,cent,rand_en=True,basis=None):
    from numpy import zeros
    from utils import knn_bar,lin_bar
    from optim_utils import sliced_transport_bar

    bar_init = zeros(shap)
    barycenters = zeros((shap[0],shap[1],len(list_points_clouds)-1))

    for i in range(1,len(list_points_clouds)):
        #init_bar = lin_bar(list_points_clouds[i],w)
        init_bar = list_points_clouds[i][:,:,0]
        bar,basis_out = sliced_transport_bar(list_points_clouds[i],w,nb_iter=1000,tol=0.0000001,alph=0.01,bar_init=init_bar,rand_en=rand_en,basis=basis[i])
        bar[2,:] *= scaling_fact[i]
        init_bar[2,:] *= scaling_fact[i]
        barycenters[:,:,i-1] = bar_1d_2d_bis(bar,shap,cent,pol_en=False,log_param=None,tol=0.001)
        bar_init += bar_1d_2d_bis(init_bar,shap,cent,pol_en=False,log_param=None,tol=0.001)

    return barycenters,bar_init


def weighted_rings_barycenter(arrays_points_clouds,w,coeff_poly,shap,cent,rand_en=True,basis=None):
    from numpy import zeros,ones,sqrt,exp
    from utils import knn_bar,lin_bar
    from optim_utils import sliced_transport_bar
    from numpy.polynomial.polynomial import polyval


    # Calculate the weighted point cloud
    shap_clouds = arrays_points_clouds.shape
    w_pt_clouds = copy(arrays_points_clouds)

    for i in range(0,shap_clouds[2]):
        rad_dist = sqrt(sum((arrays_points_clouds[0:2,:,i]-array(cent).reshape((2,1)).dot(ones((1,shap_clouds[1]))))**2,axis=0))
        w_pt_clouds[2,:,i]*= exp(polyval(rad_dist,coeff_poly))


    bar,basis_out = sliced_transport_bar(w_pt_clouds,w,nb_iter=200,tol=1e-20,alph=0.01,bar_init=copy(w_pt_clouds[:,:,0]),rand_en=rand_en,basis=basis[1])

    rad_dist_bar = sqrt(sum((bar[0:2,:]-array(cent).reshape((2,1)).dot(ones((1,shap_clouds[1]))))**2,axis=0))
    bar[2,:] /= exp(polyval(rad_dist_bar,coeff_poly))

    barycenters = bar_1d_2d_bis(bar,shap,cent,pol_en=False,log_param=None,tol=0.001)

    return barycenters

def weighted_rings_dist_matrix(list_points_clouds,coeff_poly,shap,cent,rand_en=False,basis=None,sphere_samp=8,nb_iter=1000,tol=0.001,alph = 0.0):
    from optim_utils import sliced_dist_mat
    from numpy import copy,zeros,ones,sqrt,exp
    from numpy.polynomial.polynomial import polyval
    from utils import sphere_vect

    if rand_en is False and basis is None:
        basis = sphere_vect(sphere_samp)


    dist_mats = list()
    for i in range(0,len(list_points_clouds)):
        w_pt_clouds = copy(list_points_clouds[i])
        shap_clouds = w_pt_clouds.shape
        for j in range(0,w_pt_clouds.shape[2]):
            rad_dist = sqrt(sum((w_pt_clouds[0:2,:,j]-array(cent).reshape((2,1)).dot(ones((1,shap_clouds[1]))))**2,axis=0))
            w_pt_clouds[2,:,j]*= exp(polyval(rad_dist,coeff_poly))
        print sum(abs(w_pt_clouds))
        dist_mats.append(sliced_dist_mat(w_pt_clouds,nb_iter=nb_iter,tol=tol,alph = alph,basis=basis,rand=rand_en))

    return dist_mats




def rings_barycenter_interface(main_lobes,list_points_clouds,pos_field,target_pos,coeff_scale_poly,nb_points,shap,cent,rand_en=True,knn=None,nb_neigh=10,p=2,sphere_nb_samp=10):
    from pyflann import FLANN
    from utils import sphere_vect_w
    if knn is None:
        knn = FLANN()
        params = knn.build_index(array(pos_field, dtype=float64))


    # Computing the weights
    result, dists = knn.nn_index(target_pos, nb_neigh)

    weights = dists**(-p)/sum(dists**(-p))

    list_points_clouds_reduced = list()

    # Setting the neighbors point clouds and weighed basis for sliced transport
    basis = list()
    for i in range(0,2):
        """stack = zeros((3,nb_points[i],nb_neigh))
        for j in range(0,nb_neigh):
            stack[:,:,j] = copy(list_points_clouds[i][:,result[j]*nb_points[i]:(result[j]+1)*nb_points[i]])"""
        list_points_clouds_reduced.append(list_points_clouds[i][:,:,result.reshape((nb_neigh,))])
        basis.append(sphere_vect_w(sphere_nb_samp,list_points_clouds[i]))

    barycenters = zeros((shap[0],shap[1],2))
    # Point cloud main lobe
    main_lobe_cloud = zeros((3,nb_points[0]))
    for i in range(0,nb_neigh):
        barycenters[:,:,0]+=weights[0,i]*main_lobes[:,:,result[0,i]]
        #main_lobe_cloud += weights[0,i]*list_points_clouds_reduced[0][:,:,i]

    #main_lobe_cloud[2,:] *= scaling_fact[0]

    #barycenters[:,:,0] = bar_1d_2d_bis(main_lobe_cloud,shap,cent,pol_en=False,log_param=None,tol=0.00000001)
    barycenters[:,:,1] = weighted_rings_barycenter(list_points_clouds_reduced[1],weights.reshape((nb_neigh,)),coeff_scale_poly,shap,cent,rand_en=rand_en,basis=basis)

    return barycenters,knn


def rings_barycenter_interface_m(main_lobes,list_points_clouds,pos_field,target_pos,coeff_scale_poly,nb_points,shap,cent,rand_en=True,knn=None,nb_neigh=10,p=2,sphere_nb_samp=10):

    from numpy import zeros
    nb_points = target_pos.shape[0]
    barycenters = zeros((shap[0],shap[1],nb_points))


    for i in range(0,nb_points):
        barycenters_i,knn = rings_barycenter_interface(main_lobes,list_points_clouds,pos_field,target_pos[i,:],coeff_scale_poly,nb_points,shap,cent,rand_en=True,knn=knn,nb_neigh=nb_neigh,p=p,sphere_nb_samp=sphere_nb_samp)
        barycenters[:,:,i] = barycenters_i.sum(axis=2)


    return barycenters



def rings_barycenter_interface_m2(main_lobes,list_points_clouds,pos_field,target_pos,coeff_scale_poly,nb_points,nb_neigh,shap,cent,rand_en=True,knn=None,p=2,sphere_nb_samp=10):

    barycenters = list()
    for i in range(0,len(nb_neigh)):
        barycenters.append(rings_barycenter_interface_m(main_lobes,list_points_clouds,pos_field,target_pos,coeff_scale_poly,nb_points,shap,cent,rand_en=rand_en,knn=knn,nb_neigh=nb_neigh[i],p=p,sphere_nb_samp=sphere_nb_samp))

    return barycenters



def lin_interp_m(im,pos_field,target_pos,nb_components=30,nb_neigh=10,coeff=None,comp_cube=None,p=2,knn=None):
    from utils import cube_svd
    from pyflann import FLANN
    from numpy import zeros,transpose,diag
    import time

    if knn is None:
        knn = FLANN()
        params = knn.build_index(array(pos_field, dtype=float64))


    # Computing the weights
    result, dists = knn.nn_index(target_pos, nb_neigh)

    weights = diag(sum(dists**(-p),axis=1)**(-1)).dot(dists**(-p))


    if coeff is None or comp_cube is None:
        coeff,comp_cube,approx_cube = cube_svd(im,nb_comp=nb_components,mean_sub=False)


    shap = im.shape
    nb_targets = target_pos.shape[0]
    barycenters = zeros((shap[0],shap[1],nb_targets))
    tmean = 0
    for i in range(0,nb_targets):
        print i+1, "th PSF/",nb_targets
        t = time.time()
        temp = comp_cube.reshape((shap[0]*shap[1],min(nb_components,shap[2]))).dot(coeff[:,result[i,:]].dot(weights[i,:].reshape((nb_neigh,1))))
        tmean+= time.time() - t
        barycenters[:,:,i] = temp.reshape((shap[0],shap[1]))
    print "Elapsed time: ",tmean/nb_targets
    return barycenters,coeff,comp_cube,knn



def lin_interp_m2(im,pos_field,target_pos,nb_components=30,nb_neigh=None,coeff=None,comp_cube=None):

    from numpy import zeros
    shap = im.shape
    nb_points = len(nb_neigh)
    barycenters = list()


    for i in range(0,nb_points):
        bar_i,coeff,comp_cube,knn = lin_interp_m(im,pos_field,target_pos,nb_components=nb_components,nb_neigh=nb_neigh[i],coeff=coeff,comp_cube=comp_cube)
        barycenters.append(bar_i)

    return barycenters


def extract_samp(pos_ind,nb_samp):
    from numpy.random import choice
    from numpy import arange,sort
    total = pos_ind.shape[0]
    samp = choice(arange(0,total),size=nb_samp,replace=False)
    density = nb_samp/((pos_ind[samp,0].max()-pos_ind[samp,0].min())*(pos_ind[samp,1].max()-pos_ind[samp,1].min()))
    return samp,density


def extract_samp_m(pos_ind,nb_samp,nb_real):
    from numpy import zeros
    samp = zeros((nb_samp,nb_real)).astype(int)
    id = list()
    density = 0
    for i in range(0,nb_real):
        samp[:,i],density_i = extract_samp(pos_ind,nb_samp)
        density += density_i
        id.append(str(nb_samp)+str(i))
    density /= nb_real

    return samp,density,id

def extract_samp_m2(pos_ind,nb_samp,nb_real):
    from numpy import zeros
    density = zeros((size(nb_real),))
    samp = list()
    id = list()
    for i in range(0,size(nb_samp)):
        samp_i,density[i],id_i = extract_samp_m(pos_ind,nb_samp[i],nb_real[i])
        samp.append(samp_i)
        id.append(id_i)

    return samp,density,id

def psfex_interf(samp,data,ref_pos,target_pos,PSF_SAMPLING,detect_thresh,sig,resol,var_deg=4,dir=True):
    from psfex_utils import psfextractor_2
    from numpy import array,ndarray
    shap = array(data.shape)
    print shap[0]
    shap*=PSF_SAMPLING
    output_est = None
    nb_samp = target_pos.shape[0]
    if type(samp) is ndarray:
        if size(shape(samp))==1:
            src,output_est,resi,hdu = psfextractor_2(PSF_SAMPLING,detect_thresh,data[:,:,samp],ref_pos[samp,:],sig,resol,var_deg=var_deg,dir=dir,target_pos=target_pos)
        else:
            output_est = zeros((shap[0],shap[1],target_pos.shape[0],samp.shape[1]))
            for i in range(0,samp.shape[1]):
                src,output_est[:,:,:,i],resi,hdu = psfextractor_2(PSF_SAMPLING,detect_thresh,data[:,:,samp[:,i]],ref_pos[samp[:,i],:],sig,resol,var_deg=var_deg,dir=dir,target_pos=target_pos)
    else: # samp has to be a list
        output_est = list()
        for i in range(0,len(samp)):
            output_est.append(psfex_interf(samp[i],data,ref_pos,target_pos,PSF_SAMPLING,detect_thresh,sig,resol,var_deg=4,dir=True))
    return output_est

def rbf_stack_2mbis(samp,data,pos_data,target_pos,nb_neigh,nb_comp=20,knn=None):
    from numpy import array,ndarray
    shap = data.shape
    output_est = None

    if type(samp) is ndarray:
        if size(shape(samp))==1:
            output_est = rbf_stack_2(data[:,:,samp],pos_data[samp,],target_pos,min(nb_neigh,size(samp)),nb_comp=nb_comp,knn=knn)
        else:
            output_est = zeros((shap[0],shap[1],target_pos.shape[0],samp.shape[1]))
            for i in range(0,samp.shape[1]):
                output_est[:,:,:,i] = rbf_stack_2(data[:,:,samp[:,i]],pos_data[samp[:,i],],target_pos,min(nb_neigh,samp.shape[0]),nb_comp=nb_comp,knn=knn)
    else: # samp has to be a list
        output_est = list()
        for i in range(0,len(samp)):
            print "Nb samples: ",samp[i].shape[0]
            output_est.append(rbf_stack_2mbis(samp[i],data,pos_data,target_pos,nb_neigh,nb_comp=nb_comp,knn=knn))

    return output_est

def lin_interp_m2bis(samp,im,pos_field,target_pos,nb_components=30,nb_neigh=10,coeff=None,comp_cube=None,p=2):
    from numpy import array,ndarray
    shap = im.shape
    output_est = None

    if type(samp) is ndarray:
        if size(shape(samp))==1:
            output_est,coeff,comp_cube,knn = lin_interp_m(im[:,:,samp],pos_field[samp,:],target_pos,nb_components=nb_components,nb_neigh=min(nb_neigh,size(samp)),p=p)
        else:
            output_est = zeros((shap[0],shap[1],target_pos.shape[0],samp.shape[1]))
            for i in range(0,samp.shape[1]):
                output_est[:,:,:,i],coeff,comp_cube,knn = lin_interp_m(im[:,:,samp[:,i]],pos_field[samp[:,i],:],target_pos,nb_components=nb_components,nb_neigh=min(nb_neigh,samp.shape[0]),p=p)
    else: # samp has to be a list
        output_est = list()
        for i in range(0,len(samp)):
            output_est.append(lin_interp_m2bis(samp[i],im,pos_field,target_pos,nb_components=nb_components,nb_neigh=nb_neigh,p=2))

    return output_est

def local_sliced_euc_bar_interface_m(stack,samp,dist_mat,pos_ref,target_ref,local_dim=10,nb_neigh_inv_map=11):
    from numpy import ndarray,ones,transpose
    from pyflann import FLANN
    output_est = None
    shap = stack.shape
    nb_samp = target_ref.shape[0]

    if type(samp) is ndarray:
        if size(shape(samp))==1:
            ind = samp.reshape((size(samp),1)).dot(ones((1,size(samp))))
            ind = ind.astype(int)
            knn = FLANN()
            params = knn.build_index(array(pos_ref[samp,:], dtype=float64))
            neighborhood, dists = knn.nn_index(target_ref, min(30,size(samp)))
            output_est = local_euclidean_bar(stack[:,:,samp],dist_mat[ind,transpose(ind)],neighborhood,pos_ref[samp,:],target_ref,local_dim=10,nb_neigh_inv_map=nb_neigh_inv_map)
        else:
            output_est = zeros((shap[0],shap[1],target_ref.shape[0],samp.shape[1]))
            for i in range(0,samp.shape[1]):
                ind = samp[:,i].reshape((samp.shape[0],1)).dot(ones((1,samp.shape[0])))
                ind = ind.astype(int)
                knn = FLANN()
                params = knn.build_index(array(pos_ref[samp[:,i],:], dtype=float64))
                neighborhood, dists = knn.nn_index(target_ref, min(30,samp.shape[0]))
                output_est[:,:,:,i] =local_euclidean_bar(stack[:,:,samp[:,i]],dist_mat[ind,transpose(ind)],neighborhood,pos_ref[samp[:,i],:],target_ref,local_dim=10,nb_neigh_inv_map=nb_neigh_inv_map)
                print "l1 norm: ",sum(output_est[:,:,:,i])
    else: # samp has to be a list
        output_est = list()
        for i in range(0,len(samp)):
            output_est.append(local_sliced_euc_bar_interface_m(stack,samp[i],dist_mat,pos_ref,target_ref,local_dim=10,nb_neigh_inv_map=nb_neigh_inv_map))

    return output_est

def local_sliced_euc_bar_interface_m_ext(stack,samp,pos_ref,target_ref,local_dim=10,nb_neigh_inv_map=11):
    from numpy import zeros,squeeze
    from optim_utils import dist_map

    shap = stack.shape
    nb_real = stack.shape[3]
    output = zeros((shap[0],shap[1],target_ref.shape[0],nb_real))
    for i in range(0,nb_real):
        dist_mat = dist_map(stack[:,:,:,i])
        output[:,:,:,i] = squeeze(local_sliced_euc_bar_interface_m(stack[:,:,:,i],samp,dist_mat,pos_ref,target_ref,local_dim=10,nb_neigh_inv_map=11))
    return output


def local_sliced_transport_bar_interface_2m(stack,samp,pos_ref,target_ref,nb_iter=500,local_dim=10,nb_neigh_inv_map=11,output_path='../../../Data/Result_interp/',id=None,list_dist_mat=None,cent=None,theta_en=False):
    from numpy import ndarray,ones,transpose,zeros
    from optim_utils import max_gap
    from pyflann import FLANN

    output_est = None
    shap = stack.shape
    nb_samp = target_ref.shape[0]
    #gap,inear,jnear,dist_mat_temp = max_gap(stack,pos_ref,tol=0.001,nb_neigh=10,max_val =2**63,dist_mat=None)
    dist_out = None
    if type(samp) is ndarray:
        if size(shape(samp))==1:
            knn = FLANN()
            params = knn.build_index(array(pos_ref[samp,:], dtype=float64))
            neighborhood, dists = knn.nn_index(target_ref, min(30,size(samp)))
            #ind = samp.reshape((size(samp),1)).dot(ones((1,size(samp))))
            #ind = ind.astype(int)
            dist_mat_in = None
            if list_dist_mat is not None:
                dist_out = list_dist_mat
                gap,inear,jnear,dist_mat_temp = max_gap(stack[:,:,samp],pos_ref[samp,:],tol=0.001,nb_neigh=min(10,size(samp)),max_val =2**63,dist_mat=None,pol_en=theta_en)
            else:
                dist_out,neighborhood,gap = dist_wasserstein_map(stack[:,:,samp],pos_ref[samp,:],target_ref,gap=None,max_val=2**63,nb_neigh=min(20,size(samp)),nb_real=50,tol=1.e-20,nb_iter=nb_iter,smart_init_en = True,rad=15,rand=True,cent=cent,theta_en=theta_en)
            output_est = local_sliced_transport_bar_interface_2(stack[:,:,samp],gap,dist_out,neighborhood,pos_ref[samp,:],target_ref,nb_iter=nb_iter,local_dim=local_dim,nb_neigh_inv_map=nb_neigh_inv_map,output_path=output_path,id=id,cent=cent,theta_en=theta_en)
        else:
            output_est = zeros((shap[0],shap[1],target_ref.shape[0],samp.shape[1]))
            if list_dist_mat is None:
                dist_out = zeros((samp.shape[0],samp.shape[0],samp.shape[1]))
            for i in range(0,samp.shape[1]):
                #ind = samp[:,i].reshape((samp.shape[0],1)).dot(ones((1,samp.shape[0])))
                #ind = ind.astype(int)
                dist_mat_in = None
                if list_dist_mat is not None:
                    dist_mat_in = list_dist_mat[:,:,i]
                    knn = FLANN()
                    params = knn.build_index(array(pos_ref[samp[:,i],:], dtype=float64))
                    neighborhood, dists = knn.nn_index(target_ref, min(30,samp.shape[0]))
                    gap,inear,jnear,dist_mat_temp = max_gap(stack[:,:,samp[:,i]],pos_ref[samp[:,i],:],tol=0.001,nb_neigh=min(10,samp.shape[0]),max_val =2**63,dist_mat=None,pol_en=theta_en)
                else:
                    dist_mat_in,neighborhood,gap = dist_wasserstein_map(stack[:,:,samp[:,i]],pos_ref[samp[:,i],:],target_ref,gap=None,max_val=2**63,nb_neigh=min(20,samp.shape[0]),nb_real=50,tol=1.e-20,nb_iter=nb_iter,smart_init_en = True,rad=15,rand=True,cent=cent,theta_en=theta_en)
                    dist_out[:,:,i] = dist_mat_in

                output_est[:,:,:,i] = local_sliced_transport_bar_interface_2(stack[:,:,samp[:,i]],gap,dist_mat_in,neighborhood,pos_ref[samp[:,i],],target_ref,nb_iter=nb_iter,local_dim=local_dim,nb_neigh_inv_map=nb_neigh_inv_map,output_path=output_path,id=id[i],cent=cent,theta_en=theta_en)
    else: # samp has to be a list
        output_est = list()
        dist_out = list()
        if list_dist_mat is not None:
            dist_out = list_dist_mat
            for i in range(0,len(samp)):
                output_est_i,dist_out_i = local_sliced_transport_bar_interface_2m(stack,samp[i],pos_ref,target_ref,nb_iter=nb_iter,local_dim=10,nb_neigh_inv_map=nb_neigh_inv_map,output_path=output_path,id=id[i],list_dist_mat=list_dist_mat[i],cent=cent,theta_en=theta_en)
                output_est.append(output_est_i)
        else:
            for i in range(0,len(samp)):
                output_est_i,dist_out_i = local_sliced_transport_bar_interface_2m(stack,samp[i],pos_ref,target_ref,nb_iter=nb_iter,local_dim=10,nb_neigh_inv_map=nb_neigh_inv_map,output_path=output_path,id=id[i],cent=cent,theta_en=theta_en)
                output_est.append(output_est_i)
                dist_out.append(dist_out_i)
    return output_est,dist_out


def dist_wasserstein_map(im_stack,ref_pos,target_pos,gap=None,max_val=1.e30,nb_neigh=15,nb_real=50,tol=1.e-20,nb_iter=100,smart_init_en = True,rad=15,rand=True,cent=None,theta_en=False):
    from numpy import ones,arange
    from optim_utils import sliced_transport,max_gap,setting_polar_2,setting_cart_from_polar_2
    from astropy.io import fits
    from utils import compute_centroid


    knn = FLANN()
    params = knn.build_index(ref_pos, dtype=float64)
    neighborhood, dists = knn.nn_index(target_pos, nb_neigh)

    if gap is None:
        gap,inear,jnear,dist_mat_temp = max_gap(im_stack,ref_pos,tol=0.001,nb_neigh=10,max_val =2**63,dist_mat=None,pol_en=theta_en)

    nb_psf = im_stack.shape[2]
    nb_target = target_pos.shape[0]
    dist_mat = ones((nb_psf,nb_psf))*max_val
    ind = arange(0,nb_psf).astype(int)
    dist_mat[ind,ind] *= 0
    pt_clouds = cube_to_point_cloud(im_stack)
    if theta_en:
        print "--- Polar coordinates are being used ---"
        if cent is None:
            im_mean = im_stack.mean(axis=2)
            cent,w = compute_centroid(im_mean,sigw=100000000)
            cent = squeeze(cent)

        for i in range(0,pt_clouds.shape[2]):
            pt_clouds[0:2,:,i] = setting_polar_2(pt_clouds[0:2,:,i],gap,cent)
    else:
        pt_clouds[0:2,:,:]*=gap
    shap = im_stack.shape[0:2]
    print "---------- Pairwise distances setting ---------"
    for i in range(0,nb_target):
        print "Target ",i+1,"/",nb_target
        for k in range(0,nb_neigh-1):
            for l in range(k+1,nb_neigh):
                if (dist_mat[neighborhood[i,k],neighborhood[i,l]] == max_val):
                    proj,dist_mat[neighborhood[i,k],neighborhood[i,l]],assignment,f_disc = sliced_transport(pt_clouds[:,:,neighborhood[i,k]],pt_clouds[:,:,neighborhood[i,l]],nb_iter=nb_iter,tol=tol,nb_real=nb_real,rand=rand,smart_init_en = smart_init_en,rad=rad,gap=gap,shap=shap)
                    dist_mat[neighborhood[i,k],neighborhood[i,l]] = dist_mat[neighborhood[i,l],neighborhood[i,k]]

    return dist_mat,neighborhood,gap

def gap_select(im_stack,pos,gap_fact_arr,sub_samp,nb_comp=2,nb_real=50,tol=1.e-20,nb_iter=100,smart_init_en = True,rad=15,rand=True,cent=None,theta_en=False):
    mean_pos = pos.mean(axis=0)
    from numpy import arange,ones,size,zeros,squeeze
    from numpy.random import choice
    from numpy.linalg import norm
    from optim_utils import max_gap,sliced_transport_proj
    from utils import cube_svd,compute_centroid


    shap = im_stack.shape[0:2]
    nb_im = im_stack.shape[2]

    ind = arange(0,nb_im)
    samp = choice(ind,size=sub_samp,replace=False)

    ind[array(samp).astype(int)] = -1

    ind_rmv = where(ind>=0)[0]

    pos_mean = pos.mean(axis=0)

    dist = norm(pos[ind_rmv,:]- ones((nb_im-sub_samp,1)).dot(pos_mean.reshape((1,2))))

    i = ind_rmv[where(dist==dist.min())[0][0]]
    ind_in = zeros((sub_samp+1,))
    ind_in[0:-1] = samp
    ind_in[-1] = i
    ind_in = ind_in.astype(int)

    gap_guess,inear,jnear,dist_mat_temp = max_gap(im_stack[:,:,ind_in],pos[ind_in,:],tol=0.001,pol_en=False)

    nb_test = size(gap_fact_arr)
    capture_measure = zeros((nb_test,))

    if cent is None:
        im_mean = im_stack.mean(axis=2)
        cent,w = compute_centroid(im_mean,sigw=100000000)
        cent = squeeze(cent)
    pt_clouds = cube_to_point_cloud(im_stack[:,:,ind_in])

    coeff,comp_cube,approx_cube,data_mean,centered_data = cube_svd(im_stack,nb_comp=nb_comp,mean_sub=True)
    print "Ref capture: ",norm(approx_cube-centered_data)**2/norm(centered_data)**2," nb comp: ",nb_comp

    for i in range(0,nb_test):
        if theta_en:
            for i in range(0,pt_clouds.shape[2]):
                pt_clouds[0:2,:,i] = setting_polar_2(pt_clouds[0:2,:,i],gap_guess*gap_fact_arr[i],cent)
        else:
            pt_clouds[0:2,:,:]*=gap_guess*gap_fact_arr[i]

        proj_tg = sliced_transport_proj(pt_clouds[:,:,-1],pt_clouds[:,:,0:-1],gap_guess*gap_fact_arr[i],nb_iter=nb_iter,tol=tol,alph = 0.0,basis=None,nb_real=200,rand=rand,disc_err=0.5,smart_init_en = True,output_control=True,shap=shap)
        coeff,comp_cube,approx_cube,data_mean,centered_data = cube_svd(proj_tg,nb_comp=nb_comp,mean_sub=True)

        capture_measure[i] = norm(approx_cube)**2/norm(proj_tg)**2

    return capture_measure,gap_guess
