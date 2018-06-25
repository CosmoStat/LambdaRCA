import subprocess
import os
import random
import cv2
from numpy import zeros,size,where,ones,copy,around,double,sinc,random,pi,arange,cos,sin,arccos,transpose,diag,sqrt,arange,floor,exp,array,mean,roots,float64,int,pi,median,rot90,argsort,tile,repeat,squeeze
from numpy.linalg import svd,norm,inv,eigh
import numpy.ma as npma
from numpy.random import randn,choice
import scipy.ndimage
import scipy
import scipy.signal as scisig
import scipy.ndimage.interpolation as ndii
import scipy.fftpack as scipy_fft
from pyflann import *
from multiprocessing import Process, Queue
from astropy.io import fits
import sys
sys.path.append('../utilities')
import gaussfitter
#import great3_util
#import optim_utils
#import isap
from scipy.spatial import ConvexHull
import scipy.fftpack as fftp
#from astropy.modeling import models#, fitting
import warnings
from scipy.ndimage.interpolation import zoom
import scipy.stats as scistats
import pywt
import copy as cp
from matplotlib import pyplot as plt,animation
import datetime,time

from scipy import interpolate

import scipy.linalg as sci_lin

def diagonally_dominated_mat(shap,sig=4.,thresh_en=True,coord_map=None,pol_en=False,pol_mod=False,theta_param=1,cent=None):
    from optim_utils import dist_map_2
    from numpy import sqrt as numsqrt
    coord_cloud = None
    if coord_map is None:
        coord_map = zeros((shap[0],shap[1],2))
        coord_map[:,:,0] = arange(0,shap[0]).reshape((shap[0],1)).dot(ones((1,shap[1])))
        coord_map[:,:,1] = ones((shap[0],1)).dot(arange(0,shap[1]).reshape((1,shap[1])))
        coord_cloud = zeros((2,shap[0]*shap[1]))
        coord_cloud[0,:] = coord_map[:,:,0].reshape((shap[0]*shap[1],))
        coord_cloud[1,:] = coord_map[:,:,1].reshape((shap[0]*shap[1],))
        if pol_en:
            if cent is None:
                cent = array([shap[0]/2,shap[1]/2])
            coord_cloud = polar_coord_cloud(coord_cloud,cent)
            coord_map[:,:,0] = cloud_out[0,:].reshape((shap[0],shap[1]))
            coord_map[:,:,1] = theta_param*cloud_out[1,:].reshape((shap[0],shap[1]))/(2*pi)
            if pol_mod:
                coord_map[:,:,1] *= coord_map[:,:,0]
                coord_cloud[1,:] *= coord_cloud[0,:]
    dist_map = sqrt(dist_map_2(coord_cloud))
    mat = exp(-dist_map**2/sig**2)
    if thresh_en:
        i,j = where(mat>exp(-1.))
        mat*=0
        mat[i,j] = 1.
    mat/=mat.sum()

    return mat

def diagonally_dominated_mat_stack(shap,nb_mat,sig=4.,thresh_en=True,coord_map=None,pol_en=False,\
    pol_mod=False,theta_param=1,cent=None):
    mat_ref = diagonally_dominated_mat(shap,sig=sig,thresh_en=thresh_en,coord_map=coord_map,\
    pol_en=pol_en,pol_mod=pol_mod,theta_param=theta_param,cent=cent)
    mat = zeros((shap[0]*shap[1],shap[0]*shap[1],nb_mat))
    for i in range(0,nb_mat):
        mat[:,:,i] = copy(mat_ref)
    return mat


def radial_support(mat,shap,cent=None,tol_deg=15.,coord_map=None):
    mat_out = copy(mat)
    coord_cloud = None
    if coord_map is None:
        coord_map = zeros((shap[0],shap[1],2))
        coord_map[:,:,0] = arange(0,shap[0]).reshape((shap[0],1)).dot(ones((1,shap[1])))
        coord_map[:,:,1] = ones((shap[0],1)).dot(arange(0,shap[1]).reshape((1,shap[1])))
        coord_cloud = zeros((2,shap[0]*shap[1]))
        coord_cloud[0,:] = coord_map[:,:,0].reshape((shap[0]*shap[1],))
        coord_cloud[1,:] = coord_map[:,:,1].reshape((shap[0]*shap[1],))
        if cent is None:
            cent = array([shap[0]/2,shap[1]/2])
        coord_cloud = polar_coord_cloud(coord_cloud,cent)

    for i in range(0,shap[0]*shap[1]):
        j = where(mat[i,:]>0)
        for k in range(0,len(j[0])):
            if coord_cloud[0,j[0][k]]<coord_cloud[0,i] or abs(coord_cloud[1,j[0][k]]-coord_cloud[1,i])>pi*tol_deg/180 :
                mat_out[i,j[0][k]] = 0
    return mat_out



def gauss_ker(shap,sigx,sigy):
    ker = np.zeros(shap)
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            ker[i,j] = np.exp(-(i-double(shap[0])/2)**2/(2*sigx**2))*np.exp(-(j-double(shap[1])/2)**2/(2*sigy**2))
    ker = ker/np.sqrt((ker**2).sum())
    return ker

def window_ind(ref_ind,shap,rad):

    rad = min(rad,max(shap[0]/2,shap[1]/2))

    ind_out = zeros(((2*rad+1)**2,))

    for i in range(0,2*rad+1):
        pos_refx = ref_ind[0]-rad+i
        pos_refy = ref_ind[1]-rad
        ind_lin = pos_refx*shap[1]+pos_refy
        ind_out[i*(2*rad+1):(i+1)*(2*rad+1)] = arange(int(ind_lin),int(ind_lin)+2*rad+1)

    return ind_out.astype(int)

def knearest(pos,pos_set,k):
    shap = pos_set.shape
    ref_pos = ones((shap[0],1)).dot(pos.reshape((1,2)))
    dist = ((ref_pos-pos_set)**2).sum(axis=1)
    ind = argsort(dist)
    return ind[0:k]

def saturate(im,perc=0.5): # perc = percentage of energy represented by the pixels saturated
    en = (im**2).sum()
    max_val = im.max()
    im_vect = im.reshape((size(im),))
    ind = argsort(im_vect)
    acc = 0
    i=-1
    while acc < perc*en:
        acc+=(im_vect[ind[i]])**2
        i-=1

    thresh = im_vect[ind[i]]
    im_out = copy(im)
    i,j = where(im_out>thresh)
    im_out[i,j] = thresh

    #im_out = im_out*max_val/thresh

    return im_out


def field_reshape(data_cube,field_shap):
    shap = data_cube.shape # prod(field_shap) = shap[2]
    data_field = zeros((field_shap[0],field_shap[1],shap[0]*shap[1]))
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            data_field[:,:,i+j*shap[0]] = data_cube[i,j,:].reshape(field_shap[0],field_shap[1])
    return data_field


def field_spectrum(data_cube,field_shap):
    data_field = field_reshape(data_cube,field_shap)
    dct_stack = dct_cube(data_field)**2
    a = dct_stack.sum(axis=2)
    spectrum = sqrt(dct_stack.sum(axis=2))
    return spectrum,dct_stack,data_field


def decim(im,d,av_en=1,fft=1):

    im_filt=copy(im)
    im_d = copy(im)
    if d>1:
        if av_en==1:
            siz = d+1-(d%2)
            mask = ones((siz,siz))/siz**2
            if fft==1:im_filt = scisig.fftconvolve(im, mask, mode='same')
            else:im_filt = scisig.convolve(im, mask, mode='same')
        n1 = int(floor(im.shape[0]/d))
        n2 = int(floor(im.shape[1]/d))
        im_d = zeros((n1,n2))
        i,j=0,0
        for i in range(0,n1):
            for j in range(0,n2):
                im_d[i,j] = im[i*d,j*d]
    if av_en==1:
        return (im_filt,im_d)
    else:
        return im_d


def space_dist_filt(r,a=1,exp=1):
    filt = zeros((2*r+1,))
    cumul = 0
    for i in range(0,r):
        filt[i] = -1.0/abs(double(r-i))**exp
        cumul += filt[i]
        filt[2*r-i] = filt[i]
    filt[r] = -a*cumul
    filt = filt/(sqrt(sum(filt**2)))
    return filt

def feat_dist_mat(feat_mat):
    shap = feat_mat.shape
    mat_out = zeros((shap[0],shap[0]-1))
    a = array(range(0,shap[0]))
    for i in range(0,shap[0]):
        ind_i = where(a!=i)

        for k in range(0,shap[0]-1):
            mat_out[i,k] = sqrt(sum((feat_mat[i,:]-feat_mat[ind_i[0][k],:])**2))
    return mat_out

def transpose_decim(im,decim_fact,av_en=0):
    shap = im.shape
    im_out = zeros((shap[0]*decim_fact,shap[1]*decim_fact))

    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            im_out[decim_fact*i,decim_fact*j]=im[i,j]

    if av_en==1:
        siz = decim_fact+1-(decim_fact%2)
        mask = ones((siz,siz))/siz**2
        im_out = scisig.fftconvolve(im, mask, mode='same')

    return im_out

def decim_arr(im_arr,d,av_en=1,fft=1):
    n3 = im_arr.shape[2]
    im_arr_d = copy(im_arr)
    im_arr_filt = copy(im_arr)
    if d>1:
        n1 = int(floor(im_arr.shape[0]/d))
        n2 = int(floor(im_arr.shape[1]/d))
        im_arr_d = zeros((n1,n2,n3))
        i=0
        for i in range(0,n3):
            im_filt = None
            im_d = None
            if av_en==1:
                im_filt,im_d = decim(im_arr[:,:,i],d,av_en=av_en,fft=fft)
                im_arr_filt[:,:,i] = im_filt
            else:
                im_d = decim(im_arr[:,:,i],d,av_en=av_en,fft=fft)
            im_arr_d[:,:,i] = im_d
    return im_arr_filt,im_arr_d

def rect_crop(im,n1,n2,sigw=0,dx=0,dy=0):
    nb_im = 1
    if len(im.shape)>2:nb_im = im.shape[2]

    #cent = [im.shape[0]/2-1,im.shape[1]/2-1]

    if nb_im==1:
        i,j = where(im==im.max())
        cent = array([i,j]).reshape((1,2))
        if sigw!=0: cent,Wc = compute_centroid(im,sigw)

        im_crop = im[around(cent[0,0]+dx-n1/2):around(cent[0,0]+dx-n1/2)+n1,around(cent[0,1]+dy-n2/2):around(cent[0,1]+dy-n2/2)+n2]
    else:
        im_crop = zeros((n1,n2,nb_im))
        k=0
        for k in range(0,nb_im):
            imk = im[:,:,k]
            i,j = where(imk==imk.max())
            cent = array([i,j])
            if sigw!=0: cent,Wc = compute_centroid(imk,sigw)
            im_crop[:,:,k] = imk[around(cent[0,0]+dx-n1/2):around(cent[0,0]+dx-n1/2)+n1,around(cent[0,1]+dy-n2/2):around(cent[0,1]+dy-n2/2)+n2]
    return im_crop

def stack_err(stack_ref,stack_est,shap,sigw=6):
    shap_ref = stack_est.shape
    error = zeros((shap_ref[4],))
    ref_crop = rect_crop(stack_ref,shap[0],shap[1],sigw=sigw)
    for i in range(0,shap_ref[4]):
        a=0
        for j in range(0,shap_ref[3]):
            est_crop = rect_crop(stack_est[:,:,:,j,i],shap[0],shap[1],sigw=sigw)
            print "field ",i+1,"/",shap_ref[4],"realization ",j+1,"/",shap_ref[3]
            for j in range(0,shap_ref[2]):
                a+= sum((est_crop[:,:,j]/sqrt(sum(ref_crop[:,:,j]**2)) - ref_crop[:,:,j]/sqrt(sum(ref_crop[:,:,j]**2)))**2)
        error[i] = a/(shap_ref[2]*shap_ref[3])
    return error

def rect_crop_c(im,n1,n2,cent,dx=0,dy=0):
    nb_im = 1
    if len(im.shape)>2:nb_im = im.shape[2]

    #cent = [im.shape[0]/2-1,im.shape[1]/2-1]
    im_crop = None
    if nb_im==1:
        im_crop = im[around(cent[0,0]+dx-n1/2):around(cent[0,0]+dx-n1/2)+n1,around(cent[0,1]+dy-n2/2):around(cent[0,1]+dy-n2/2)+n2]
    else:
        im_crop = zeros((n1,n2,nb_im))
        for k in range(0,nb_im):
            imk = im[:,:,k]
            im_crop[:,:,k] = imk[around(cent[k,0]+dx-n1/2):around(cent[k,0]+dx-n1/2)+n1,around(cent[k,1]+dy-n2/2):around(cent[k,1]+dy-n2/2)+n2]
    return im_crop

def gauss_gen(sigma_x,rho,siz):
    gal = zeros((siz[0],siz[1]))
    sigma_y = sigma_x*rho
    i=0
    j=0
    for i in range(0,siz[0]):
        for j in range(0,siz[1]):
            gal[i,j]=exp(-((i/(2*sigma_x))**2+(j/(2*sigma_y))**2))
    return gal

def gauss_gen_2(sigma_x,sigma_y,cent,theta,rad):

    gauss_map = zeros((2*rad+1,2*rad+1))
    i=0
    j=0
    for i in range(0,2*rad+1):
        for j in range(0,2*rad+1):
            gauss_map[i,j]=exp(-(((cos(theta)*(i-cent[0])+sin(theta)*(j-cent[1]))/(2*sigma_x))**2+((-sin(theta)*(i-cent[0])+cos(theta)*(j-cent[1]))/(2*sigma_y))**2))
    return gauss_map


def compute_centroid(im,sigw=None,nb_iter=4):
    if sigw is None:
        param=gaussfitter.gaussfit(im,returnfitimage=False)
        #print param
        sigw = (param[3]+param[4])/2
    sigw = float(sigw)
    n1 = im.shape[0]
    n2 = im.shape[1]
    rx = array(range(0,n1))
    ry = array(range(0,n2))
    Wc = ones((n1,n2))
    centroid = zeros((1,2))
    # Four iteration loop to compute the centroid
    i=0
    for i in range(0,nb_iter):

        xx = npma.outerproduct(rx-centroid[0,0],ones(n2))
        yy = npma.outerproduct(ones(n1),ry-centroid[0,1])
        W = npma.exp(-(xx**2+yy**2)/(2*sigw**2))
        centroid = zeros((1,2))
        # Estimate Centroid
        Wc = copy(W)
        if i == 0:Wc = ones((n1,n2))
        totx=0.0
        toty=0.0
        cx=0
        cy=0

        for cx in range(0,n1):
            centroid[0,0] += (im[cx,:]*Wc[cx,:]).sum()*(cx)
            totx += (im[cx,:]*Wc[cx,:]).sum()
        for cy in range(0,n2):
            centroid[0,1] += (im[:,cy]*Wc[:,cy]).sum()*(cy)
            toty += (im[:,cy]*Wc[:,cy]).sum()
        centroid = centroid*array([1/totx,1/toty])


    return (centroid,Wc)

def compute_centroid_2(im,Wc):
    n1 = im.shape[0]
    n2 = im.shape[1]
    rx = array(range(0,n1))
    ry = array(range(0,n2))
    centroid = zeros((2,))

    totx=0.0
    toty=0.0
    cx=0
    cy=0

    for cx in range(0,n1):
        centroid[0] += (im[cx,:]*Wc[cx,:]).sum()*(cx)
        totx += (im[cx,:]*Wc[cx,:]).sum()
    for cy in range(0,n2):
        centroid[1] += (im[:,cy]*Wc[:,cy]).sum()*(cy)
        toty += (im[:,cy]*Wc[:,cy]).sum()
    centroid = centroid*array([1/totx,1/toty])
    return centroid

def compute_centroid_arr(data_cube,sigw=None,nb_iter=4):
    nb_psf = data_cube.shape[2]
    centroid = zeros((nb_psf,2))
    for i in range(0,nb_psf):
        centroid_i,wc = compute_centroid(data_cube[:,:,i],sigw=sigw,nb_iter=nb_iter)
        centroid[i,0] = centroid_i[0,0]
        centroid[i,1] = centroid_i[0,1]
    return centroid

def mk_ellipticity(im,sigw,niter_cent=4,cent_return=True,get_quad=False):
    q = zeros((2,2))
    ell = zeros((2,))
    centroid,Wc = compute_centroid(im,sigw,niter_cent)
    n1 = im.shape[0]
    n2 = im.shape[1]
    rx = array(range(0,n1))
    xx = npma.outerproduct(rx-centroid[0,0],ones(n2))
    ry = array(range(0,n2))
    yy = npma.outerproduct(ones(n1),ry-centroid[0,1])
    q[0,0]=(im*(xx**2)*Wc).sum()
    q[1,1]=(im*(yy**2)*Wc).sum()
    q[0,1]=(im*xx*yy*Wc).sum()
    q[1,0]=(transpose(im)*xx*yy*Wc).sum()
    q = q/((im*Wc).sum())
    ell[0] = (q[0,0]-q[1,1])/(q[0,0]+q[1,1])
    ell[1] = (q[0,1]+q[1,0])/(q[0,0]+q[1,1])

    if cent_return==True:
        return ell,centroid
    elif get_quad:
        return ell,q
    else:
        return ell

def mk_ellipticity_2(im,Wc):
    q = zeros((2,2))
    ell = zeros((2,))
    centroid = compute_centroid_2(im,Wc)
    n1 = im.shape[0]
    n2 = im.shape[1]
    rx = array(range(0,n1))
    xx = npma.outerproduct(rx-centroid[0],ones(n2))
    ry = array(range(0,n2))
    yy = npma.outerproduct(ones(n1),ry-centroid[1])
    q[0,0]=(im*(xx**2)*Wc).sum()
    q[1,1]=(im*(yy**2)*Wc).sum()
    q[0,1]=(im*xx*yy*Wc).sum()
    q[1,0]=(transpose(im)*xx*yy*Wc).sum()
    q = q/((im*Wc).sum())
    ell[0] = (q[0,0]-q[1,1])/(q[0,0]+q[1,1])
    ell[1] = (q[0,1]+q[1,0])/(q[0,0]+q[1,1])
    fwhm=0
    if q[0,0]+q[1,1]>0:
        fwhm = sqrt(q[0,0]+q[1,1])
    return ell,centroid,fwhm

def mk_elliticity_2_arr(cube,Wc=None,get_quad=True):
    shap = cube.shape
    if Wc is None:
        Wc = ones((shap[0],shap[1]))

    ell_out = zeros((shap[2],2))
    fwhm = zeros((shap[2],))
    q = zeros((2,2,shap[2]))
    for i in range(0,shap[2]):
        ell_out[i,:],q[:,:,i] = mk_ellipticity(cube[:,:,i],10.0**(10),cent_return=False,get_quad=True)
        if q[0,0,i]+q[1,1,i]>=0:
            fwhm[i] = sqrt(q[0,0,i]+q[1,1,i])

    if get_quad:
        return ell_out,fwhm,q
    else:
        return ell_out,fwhm

def mk_ellipticity_error(ref_ell,ref_fwhm,cube,Wc=None):
    ell,fwhm = mk_elliticity_2_arr(cube,Wc=Wc)
    err = abs(ref_ell-ell)
    mean_err = sqrt(sum((ref_ell-ell)**2,axis=1)).mean()
    std_err = err.std(axis=0)
    return mean_err,std_err,ell,err

def mk_ellipticity_error_m(ref_ell,cube_m,Wc=None):
    shap = cube_m.shape
    mean_err = zeros((shap[3],2))
    std_err = zeros((shap[3],2))
    for i in range(0,shap[3]):
        mean_erri,std_erri,elli,erri = mk_ellipticity_error(ref_ell,cube_m[:,:,:,i],Wc=Wc)
        mean_err[i,:] = mean_erri
        std_err[i,:] = std_erri
    return mean_err,std_err

def moffat_fitting_2d(im,width=5,exp=1):
    shap = im.shape
    y, x = mgrid[:shap[0], :shap[1]]
    z = im

    # Fit the data using astropy.modeling
    amplitude = im.max()
    x_0,y_0 = where(im==amplitude)
    p_init = models.Moffat2D(amplitude=im.max(),x_0 = x_0,y_0 = y_0,gamma=width,alpha=exp)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)
    alpha = p.alpha.value
    gamma = p.gamma.value
    fwhm = None
    if alpha <= 0:
        fwhm = 0
    else:
        fwhm = 2*gamma*sqrt(2**(1.0/alpha)-1)
    fit_mod = zeros((shap[0],shap[1]))
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            fit_mod[i,j] = p(i,j)
    return p,fwhm,fit_mod

def mk_ellipticity_arr(im_arr,sigw,niter_cent=4):
    n3 = im_arr.shape[2]
    ell_arr = zeros((n3,2))
    theta = zeros((n3,))
    i=0
    for i in range(0,n3):
        ell = mk_ellipticity(im_arr[:,:,i],sigw,niter_cent=niter_cent)
        ell_arr[i,:]=ell
        r = sqrt(ell[0,0]**2+ell[0,1]**2)
        cos_om = ell[0,0]/r
        om = arccos(cos_om)
        theta[i] = om/2
    return ell_arr,theta

def psf_shape(im,width=5,exp=1,weighting=True):
    fit_mod = None
    fwhm = None
    ell = None
    #p,fwhm,fit_mod = moffat_fitting_2d(im,width=width,exp=exp)
    if weighting is False:
        shap = im.shape
        fit_mod = ones((shap[0],shap[1]))
        ell,centroid,fwhm = mk_ellipticity_2(im,fit_mod)
    else:
        p,fwhmk,fit_mod = moffat_fitting_2d(im,width=width,exp=exp)
        ell,centroid,fwhm = mk_ellipticity_2(im,fit_mod)

    return ell,fwhm,centroid

def psf_shape_arr(psf_cube,width=5,exp=1,weighting=True):
    shap = psf_cube.shape
    fwhm = zeros((shap[2],))
    ell = zeros((shap[2],2))
    centroid = zeros((shap[2],2))
    for i in range(0,shap[2]):
        #print i+1,"/",shap[2]
        ell_i,fwhm_i,centroid_i = psf_shape(psf_cube[:,:,i],width=width,exp=exp,weighting=weighting)
        ell[i,:] = ell_i
        centroid[i,:] = centroid_i
        fwhm[i] = fwhm_i

    return ell,centroid,fwhm

def psf_shape_eucl(im,sigw=10000000000):
    ell,q = mk_ellipticity(im,sigw,cent_return=False,get_quad=True)
    R = q[0,0]+q[1,1]
    return q,R

def psf_shape_eucl_arr(psf,sigw=10000000000):

    nb_psfs = psf.shape[2]
    q_out = zeros((2,2,nb_psfs))
    R = zeros((nb_psfs,))
    for i in range(0,nb_psfs):
        q_out[:,:,i],R[i] = psf_shape_eucl(psf[:,:,i],sigw=sigw)

    return q_out,R

def moments_err(res,R,sigw=10000000000):
    q_res,R_res = psf_shape_eucl_arr(res,sigw=sigw)
    rms00 = (q_res[0,0,:]/R).std()
    rms01 = (q_res[0,1,:]/R).std()
    rms10 = (q_res[1,0,:]/R).std()
    rms11 = (q_res[1,1,:]/R).std()

    return rms00,rms01,rms10,rms11

def moments_err_2(psf_est,psf_ref,sigw=10000000000):
    q,R = psf_shape_eucl_arr(psf_ref,sigw=sigw)
    nb_real = psf_est.shape[3]

    rms_out = zeros((nb_real,4))
    for i in range(0,nb_real):
        rms_out[i,0],rms_out[i,1],rms_out[i,2],rms_out[i,3] = moments_err(psf_est[:,:,:,i]-psf_ref,R,sigw=sigw)

    return rms_out,R




def psf_err_bias(psf_ref,psf_est_list):
    nb_real = len(psf_est_list)
    q_mean_1 = zeros((2,2,nb_real))
    q_mean_2 = zeros((2,2,nb_real))
    ell_out,fwhm,q1 = mk_elliticity_2_arr(psf_ref,Wc=None,get_quad=True)
    for i in range(0,nb_real):
        ell_out,fwhm,qi = mk_elliticity_2_arr(psf_est_list[i],Wc=None,get_quad=True)
        q = qi/(qi[0,0,:]+qi[1,1,:]) - q1/(q1[0,0,:]+q1[1,1,:])
        q_mean_1[:,:,i] = sqrt((q**2).mean(axis=2))
        mean_res = (psf_ref-psf_est_list[i]).mean(axis=2)
        ell,q_mean_2[:,:,i] = mk_ellipticity(mean_res,10.0**(10),niter_cent=4,cent_return=False,get_quad=True)


    return q_mean_1,q_mean_2


def psf_shape_list(psf_cube,width=5,exp=1,weighting=False):
    nb_r  = len(psf_cube)
    list_ell = list()
    list_fwhm = list()
    for i in range(0,nb_r):
        ell,centroid,fwhm = psf_shape_arr(psf_cube[i],width=width,exp=exp,weighting=weighting)
        list_ell.append(ell)
        list_fwhm.append(fwhm)

    return list_ell,list_fwhm

def im_shape(im,scale=None):
    if scale is None:
        scale = im.max()
    shap = im.shape
    cent = compute_centroid_2(im,ones(shap))
    data_matrix = zeros((2,shap[0]*shap[1]))
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            lin_ind = i+j*shap[0]
            data_matrix[:,lin_ind] = (array([i,j])-cent)*im[i,j]/scale + cent
    m = data_matrix.mean(axis=1)
    m = m.reshape((2,1))
    ones_v = ones((1,shap[0]*shap[1]))
    U, s, Vt = linalg.svd(data_matrix-m.dot(ones_v),full_matrices=True)

    return U,s,data_matrix

def cloud_shape(cloud_pos,weights,scale=None):

    if scale is None:
        scale = weights.max()
    nb_points = cloud_pos.shape[0]
    cent = zeros((2,))
    cent = cloud_centroid(cloud_pos,weights)

    data_matrix = zeros((2,nb_points))
    for i in range(0,nb_points):
        data_matrix[:,i] = (cloud_pos[i,:]-cent)*weights[i]/scale + cent
    m = data_matrix.mean(axis=1)
    m = m.reshape((2,1))
    ones_v = ones((1,nb_points))
    U, s, Vt = linalg.svd(data_matrix-m.dot(ones_v),full_matrices=True)

    return U,s,data_matrix


def cloud_centroid(cloud_pos,weights):
    nb_points = cloud_pos.shape[0]
    cent = zeros((2,))
    for i in range(0,nb_points):
        cent+=weights[i]*cloud_pos[i,:]
    cent/=sum(weights)
    return cent

def shape_error(ell_ref,fwhm_ref,data,width=5,exp=1,weighting=True):

    shap = ell_ref.shape
    shap2 = data.shape

    if len(shap2)==3:
        data = data.reshape(shap2[0],shap2[1],shap2[2],1)
        shap2 = data.shape
    ell_err = zeros((shap2[2]*shap2[3],2))
    fwhm_err = zeros((shap2[2]*shap2[3],))
    fwhm = zeros((shap2[2]*shap2[3],))


    for i in range(0,shap2[3]):
        ell,centroid,fwhm_i = psf_shape_arr(data[:,:,:,i],width=width,exp=exp,weighting=weighting)
        ell_err[i*shap2[2]:(i+1)*shap2[2],:] = ell-ell_ref
        fwhm_err[i*shap2[2]:(i+1)*shap2[2]] = fwhm_i - fwhm_ref
        fwhm[i*shap2[2]:(i+1)*shap2[2]] = fwhm_i

    mean_err_ell = abs(ell_err).mean(axis=0)
    std_err_ell = ell_err.std(axis=0)


    mean_err_fwhm = abs(fwhm_err).mean()
    std_err_fwhm = abs(fwhm_err).std()
    return mean_err_ell,std_err_ell,mean_err_fwhm,std_err_fwhm,ell_err,fwhm_err

def error_list(data_ref,data,width=5,exp=1,weighting=False):
    from numpy import zeros
    from numpy.linalg import norm
    nb_real = len(data)
    ell_ref,centroid_ref,fwhm_ref = psf_shape_arr(data_ref,width=width,exp=exp,weighting=weighting)
    mse = zeros((nb_real,))
    ell_err = zeros((nb_real,))
    fwhm_err = zeros((nb_real,))
    norm_data = norm(data_ref,axis=(0,1))
    norm_ell = norm(ell_ref,axis=1)
    for i in range(0,nb_real):
        for j in range(0,data[i].shape[3]):
            ell_j,centroid_j,fwhm_j = psf_shape_arr(data[i][:,:,:,j],width=width,exp=exp,weighting=weighting)
            mse[i]+=mean(norm(data_ref-data[i][:,:,:,j],axis=(0,1))/norm_data)
            ell_err[i]+=mean(norm(ell_ref-ell_j,axis=1)/norm_ell)
            fwhm_err[i]+=mean(abs(fwhm_ref-fwhm_j))
        mse[i]/=data[i].shape[3]
        ell_err[i]/=data[i].shape[3]
        fwhm_err[i]/=data[i].shape[3]

    return mse,fwhm_err,ell_err

def shape_error_m2(ell_ref,fwhm_ref,data_list,width=5,exp=1,weighting=False):

    nb_real = len(data_list)
    mean_err_ell = zeros((nb_real,2))
    std_err_ell = zeros((nb_real,2))
    mean_err_fwhm = zeros((nb_real,))
    std_err_fwhm = zeros((nb_real,))
    err_ell = list()
    err_fwhm = list()

    for i in range(0,nb_real):
        mean_err_ell[i,:],std_err_ell[i,:],mean_err_fwhm[i],std_err_fwhm[i],ell_err_i,fwhm_err_i = shape_error(ell_ref,fwhm_ref,data_list[i],width=width,exp=exp,weighting=weighting)
        err_ell.append(ell_err_i)
        err_fwhm.append(fwhm_err_i)

    return mean_err_ell,std_err_ell,mean_err_fwhm,std_err_fwhm,err_ell,err_fwhm

def mean_squared_error(data_ref,est_list):
    n = len(est_list)
    err = zeros((n,))
    err_raw = list()
    shap = data_ref.shape
    norms = norm(data_ref.reshape((shap[2],shap[0]*shap[1])),axis=1)
    for i in range(0,n):
        err_raw.append(mean(((data_ref-est_list[i])/norms)**2,axis=(0,1)))
        err[i] = mean(err_raw[i])
    return err,err_raw

def shape_error_m(ell_ref,fwhm_ref,data,width=5,exp=1,weighting=True):
    nb_fields = data.shape[3]
    nb_samp = data.shape[2]
    mean_err_ell = zeros((nb_fields,2))
    std_err_ell = zeros((nb_fields,2))
    mean_err_fwhm = zeros((nb_fields,))
    std_err_fwhm = zeros((nb_fields,))
    err_ell = zeros((nb_samp,2,nb_fields))
    err_fwhm = zeros((nb_samp,nb_fields))


    for i in range(0,nb_fields):
        print "field ",i+1,"/",nb_fields
        mean_err_ell_i,std_err_ell_i,mean_err_fwhm_i,std_err_fwhm_i,ell_err_i,fwhm_err_i = shape_error(ell_ref,fwhm_ref,data[:,:,:,i],width=5,exp=1,weighting=weighting)
        mean_err_ell[i,:] = mean_err_ell_i
        std_err_ell[i,:] = std_err_ell_i
        mean_err_fwhm[i] = mean_err_fwhm_i
        std_err_fwhm[i] = std_err_fwhm_i
        err_ell[:,:,i] = ell_err_i
        err_fwhm[:,i] = fwhm_err_i

    return mean_err_ell,std_err_ell,mean_err_fwhm,std_err_fwhm,err_ell,err_fwhm

def shape_error_list(data_ref,ell_ref,fwhm_ref,data,width=5,exp=1,weighting=False):

    nb_fields = len(ell_ref)
    mean_err_fwhm = zeros((nb_fields,))
    std_err_fwhm = zeros((nb_fields,))
    err_ell = list()
    err_fwhm = list()
    mean_err_ell = zeros((nb_fields,))
    std_err_ell = zeros((nb_fields,))
    mean_lsq = zeros((nb_fields,))


    for i in range(0,nb_fields):
        mean_err_ell_i,std_err_ell_i,mean_err_fwhm_i,std_err_fwhm_i,ell_err_i,fwhm_err_i = shape_error(ell_ref[i],fwhm_ref[i],data[i],width=width,exp=exp,weighting=False)
        mean_err_fwhm[i] = mean_err_fwhm_i
        std_err_fwhm[i] = std_err_fwhm_i
        err_ell.append(ell_err_i)
        err_fwhm.append(fwhm_err_i)
        mean_err_ell[i] = mean(sqrt((ell_err_i**2).sum(axis=1)))
        std_err_ell[i] = vect_error_disp(ell_err_i)
        mean_lsq[i] = ((data_ref[i] - data[i])**2).sum()/data_ref[i].shape[2]

    return mean_err_fwhm,std_err_fwhm,err_ell,err_fwhm,mean_err_ell,std_err_ell,mean_lsq

def vect_error_disp(err_mat): # Each line is an error vector
    shap = err_mat.shape
    mean_err = err_mat.mean(axis=0)
    mean_err = mean_err.reshape((2,1))
    mat_cent = transpose(err_mat) - mean_err.dot(ones((1,shap[0])))
    svecl, s, svecr = linalg.svd(mat_cent,full_matrices=False)
    nuc_norm = s.sum()
    return nuc_norm

def vect_error_disp_stack(err_mat):
    shap = err_mat.shape
    nuc_norms = zeros((shap[2],))
    for i in range(0,shap[2]):
        nuc_norms[i] = vect_error_disp(err_mat[:,:,i])

    return nuc_norms

def proj_mat_2(siz,offset=0):
    n1,n2= siz[0],siz[1]
    mat_proj = zeros((6,n1*n2))
    mat_proj_temp,normv=proj_mat(siz,offset)
    i=0
    for i in range(0,6):mat_proj[i,:] = (mat_proj_temp[:,:,i]).reshape(1,n1*n2)
    return mat_proj

def proj_mat(siz,offset=0):
    n1,n2= siz[0],siz[1]
    mat_proj = zeros((n1,n2,6))
    normv = zeros((6,))
    rx = array(range(0,n1))
    mat_proj[:,:,0] = npma.outerproduct(rx+offset,ones(n2))
    normv[0] = sqrt((mat_proj[:,:,0]**2).sum())
    ry = array(range(0,n2))
    mat_proj[:,:,1] = npma.outerproduct(ones(n1),ry+offset)
    normv[1] = sqrt((mat_proj[:,:,1]**2).sum())
    mat_proj[:,:,2] = ones((n1,n2))
    normv[2] = sqrt((mat_proj[:,:,2]**2).sum())
    mat_proj[:,:,3] = mat_proj[:,:,0]**2 + mat_proj[:,:,1]**2
    normv[3] = sqrt((mat_proj[:,:,3]**2).sum())
    mat_proj[:,:,4] = mat_proj[:,:,0]**2 - mat_proj[:,:,1]**2
    normv[4] = sqrt((mat_proj[:,:,4]**2).sum())
    mat_proj[:,:,5] = mat_proj[:,:,0]*mat_proj[:,:,1]
    normv[5] = sqrt((mat_proj[:,:,5]**2).sum())
    return mat_proj,normv


def shape_energy(shape_comp,coeff,principal_comp):
    nb_comp = coeff.shape[0]
    comp_shape_en = zeros((nb_comp,))
    data_shape_en = zeros((nb_comp,))

    for i in range(0,nb_comp):
        for j in range(0,shape_comp.shape[2]):
            comp_shape_en[i] += ((principal_comp[:,:,i]*shape_comp[:,:,j]).sum())**2
        data_shape_en[i] = comp_shape_en[i]*(coeff[i,:].std())**2

    return comp_shape_en,data_shape_en

#def shape_analysis(coeff,cube_comp,):

def mk_ellipticity_atoms(im,offset=0,sigw=0):
    if sigw !=0 :
        centroid,Wc = compute_centroid(im,sigw)
        im = im*Wc
    mat_proj,normv = proj_mat(im.shape,offset)
    U = zeros((6,1))
    i = 0
    for i in range(0,6):U[i,0] = (im*mat_proj[:,:,i]).sum()

    e1 = U[4,0]*U[2,0]-U[0,0]**2+U[1,0]**2
    e2 = U[3,0]*U[2,0]-U[0,0]**2-U[1,0]**2
    e3 = 2*(U[5,0]*U[2,0]-U[0,0]*U[1,0])
    ell = zeros((1,2))
    ell[0,0] = e1/e2
    ell[0,1] = e3/e2
    centroid = zeros((1,2))
    centroid[0,0] = (im*mat_proj[:,:,0]).sum()/(im*mat_proj[:,:,2]).sum()
    centroid[0,1] = (im*mat_proj[:,:,1]).sum()/(im*mat_proj[:,:,2]).sum()
    return centroid,U,ell

def mk_ellipticity_atoms_basic(U):
    e1 = U[4]*U[2]-U[0]**2+U[1]**2
    e2 = U[3]*U[2]-U[0]**2-U[1]**2
    e3 = 2*(U[5]*U[2]-U[0]*U[1])
    ell = zeros((1,2))
    ell[0,0] = e1/e2
    ell[0,1] = e3/e2
    return ell

def ellipticity_basics_grad(U):
    den = (U[3]*U[2]-U[0]**2-U[1]**2)**2
    grad_comp_1 = zeros((5,1))
    grad_comp_2 = zeros((5,1))
    grad_comp_1[0] = -2*U[0]*((U[3]-U[4])*U[2]-2*U[1]**2)/den
    grad_comp_1[1] = 2*U[1]*((U[3]+U[4])*U[2]-2*U[0]**2)/den
    grad_comp_1[2] = ((U[3]-U[4])*U[0]**2-(U[3]+U[4])*U[1]**2)/den
    grad_comp_1[3] = -U[2]*(U[4]*U[2]-U[0]**2+U[1]**2)/den
    grad_comp_1[4] = U[2]*(U[3]*U[2]-U[0]**2-U[1]**2)/den

    grad_comp_2[0] = (-2*U[1]*U[2]*U[3]+4*U[0]*U[2]*U[5]-2*U[1]*U[0]**2+2*U[1]**3)/den
    grad_comp_2[1] = (-2*U[0]*U[2]*U[3]+4*U[1]*U[2]*U[5]-2*U[0]*U[1]**2+2*U[0]**3)/den
    grad_comp_2[2] = (-2*U[5]*(U[0]**2+U[1]**2) + 2*U[0]*U[1]*U[3])/den
    grad_comp_2[3] = -2*U[2]*(U[5]*U[2]-U[0]*U[1])/den
    grad_comp_2[4] = 2*U[2]*(U[3]*U[2]-U[0]**2-U[1]**2)/den

    return grad_comp_1,grad_comp_2

def ellipticity_grad(im):
    mat_proj,normv = proj_mat(im.shape,0)

    i = 0
    for i in range(0,6):U[i] = (im*mat_proj[:,:,i]).sum()
    centroid,U,ell = mk_ellipticity_atoms(im)
    c = U[3]*U[2]-U[0]**2-U[1]**2
    grad1 = U[2]*mat_proj[:,:,4] - ell[0,0]*U[2]*mat_proj[:,:,3] -2*(1-ell[0,0])*U[0]*mat_proj[:,:,0] + 2*(1+ell[0,0])*U[1]*mat_proj[:,:,1]+(U[4]-ell[0,0]*U[3])*mat_proj[:,:,2]
    grad1 = grad1/c

    grad2 = 2*U[2]*mat_proj[:,:,5] + (2*U[5]-ell[0,1]*U[3])*mat_proj[:,:,2] + 2*(ell[0,1]*U[0]-U[1])*mat_proj[:,:,0] + 2*(ell[0,1]*U[1]-U[0])*mat_proj[:,:,1] - ell[0,1]*U[2]*mat_proj[:,:,3]
    grad2 = grad2/c
    return grad1,grad2

def ellipticity_gradt(im,u,t):
    mat_proj,normv = proj_mat(im.shape,0)
    Uu = zeros((6,))
    Ug = zeros((6,))
    i = 0
    for i in range(0,6):
        Uu[i] = (im*mat_proj[:,:,i]).sum()
        Ug[i] = (u*mat_proj[:,:,i]).sum()

    centroid,Ut,ellt = mk_ellipticity_atoms(im+t*u)
    c = Ut[3]*Ut[2]-Ut[0]**2-Ut[1]**2
    a1 = Uu[4]*Ug[2]+Ug[4]*Uu[2]-2*(Uu[0]*Ug[0]-Uu[1]*Ug[1])
    a2 = 2*(Uu[4]*Uu[2]-Uu[0]**2+Uu[1]**2)
    d1 = Uu[3]*Ug[2]+Ug[3]*Uu[2]-2*(Uu[0]*Ug[0]+Uu[1]*Ug[1])
    d2 = 2*(Uu[3]*Uu[2]-Uu[0]**2-Uu[1]**2)

    b1 = 2*(Uu[5]*Ug[2]+Ug[5]*Uu[2]-Uu[0]*Ug[1]-Ug[0]*Uu[1])
    b2 = 4*(Uu[5]*Uu[2]-Uu[0]*Uu[1])

    der1 = (a1+a2*t)/c - ellt[0,0]*(d1+d2*t)/c
    der2 = (b1+b2*t)/c - ellt[0,1]*(d1+d2*t)/c

    return der1,der2,ellt

def field_comp_wise_ell_grad(stack,comp,coeff):
    nb_im = stack.shape[2]
    nb_comp = comp.shape[2]

    ell_map = zeros((nb_im,nb_comp,2))
    grad_map = zeros((nb_im,nb_comp,2))

    for i in range(0,nb_im):
        for j in range(0,nb_comp):
            grad_map[i,j,0],grad_map[i,j,1],ell_map[i,j,:] = ellipticity_gradt(stack[:,:,i],comp[:,:,j],0)

    return ell_map,grad_map

def field_comp_wise_ell_taylor_approx(coeff,ell_map,grad_map,pos_field):#,pos_ref=None):
    knn = FLANN()

    params = knn.build_index(array(pos_field, dtype=float64))
    result, dists = knn.nn_index(pos_field,2)
    shap = ell_map.shape
    ell_est = zeros((pos_field.shape[0],2))

    for i in range(0,pos_field.shape[0]):
        ell_est[i,0] = ell_map[result[i,-1],0,0]+ sum((coeff[:,i]-coeff[:,result[i,-1]])*grad_map[result[i,-1],:,0])
        ell_est[i,1] = ell_map[result[i,-1],0,1]+ sum((coeff[:,i]-coeff[:,result[i,-1]])*grad_map[result[i,-1],:,1])

    err_out = 100*norm(ell_est-ell_map[:,0,:],axis=1)/norm(ell_map[:,0,:],axis=1)

    return ell_est,err_out


def ellt_zeros(im,u):
    mat_proj,normv = proj_mat(im.shape,0)
    Uu = zeros((6,))
    Ug = zeros((6,))
    i = 0
    for i in range(0,6):
        Uu[i] = (im*mat_proj[:,:,i]).sum()
        Ug[i] = (u*mat_proj[:,:,i]).sum()
    a1 = Ug[4]*Ug[2]-Ug[0]**2+Ug[1]**2
    a2 = Uu[4]*Ug[2]+Uu[2]*Ug[4]-2*Ug[0]*Uu[0]+2*Ug[1]*Uu[1]
    a3 = Uu[4]*Uu[2]-Uu[0]**2+Uu[1]**2

    b1 = Ug[3]*Ug[2]-Ug[0]**2-Ug[1]**2
    b2 = Uu[3]*Ug[2]+Uu[2]*Ug[3]-2*Ug[0]*Uu[0]-2*Ug[1]*Uu[1]
    b3 = Uu[3]*Uu[2]-Uu[0]**2-Uu[1]**2

    d1 = 2*(Ug[5]*Ug[2]-Ug[0]*Ug[1])
    d2 = 2*(Uu[5]*Ug[2]+Uu[2]*Ug[5]-Uu[0]*Ug[1]-Uu[1]*Ug[0])
    d3 = 2*(Uu[5]*Uu[2]-Uu[0]*Uu[1])

    A = zeros((3,))
    A[0] = a3
    A[1] = a2
    A[2] = a1

    B = zeros((3,))
    B[0] = b3
    B[1] = b2
    B[2] = b1

    A1 = zeros((3,))
    A1[0] = a3*b2 - a2*b3
    A1[1] = 2*(b1*a3-b3*a1)
    A1[2] = a2*b1-a1*b2

    B1 = zeros((3,))
    B1[0] = d3*b2 - d2*b3
    B1[1] = 2*(b1*d3-b3*d1)
    B1[2] = d2*b1-d1*b2

    root_ell1=roots(A)
    root_ell2=roots(B)
    root_ell3=roots(A1)
    root_ell4=roots(B1)

    return root_ell1,root_ell2,root_ell3,root_ell4


def check_grad_ellipticity(im,h_max=0.01,nb_points=100):
    from numpy.random import randn
    shap = im.shape
    t = randn(shap[0],shap[1])
    t = t/sqrt((t**2).sum())
    a = array(range(1,nb_points+1))*sqrt((im**2).sum())*h_max/nb_points
    i=0
    ell_err = zeros((2,nb_points))
    grad_proj = zeros((2,nb_points))
    centroid,U,ell_ref = mk_ellipticity_atoms(im)
    grad1,grad2 = ellipticity_grad(im)
    for i in range(0,nb_points):
        im_i = im + a[i]*t
        centroid,U,elli = mk_ellipticity_atoms(im_i)
        ell_err[0,i] = abs(elli[0,0]-ell_ref[0,0])
        ell_err[1,i] = abs(elli[0,1]-ell_ref[0,1])
        grad_proj[0,i] = abs((grad1*(im_i-im)).sum())
        grad_proj[1,i] = abs((grad2*(im_i-im)).sum())

    return ell_err,grad_proj

def nul_ell_proj(im,mean_sub=None):
    mat_proj,normv = proj_mat(im.shape,0)
    shap = im.shape
    mat_proj1 = zeros((shap[0],shap[1],2))
    mat_proj2 = zeros((shap[0],shap[1],3))
    mat_proj3 = zeros((shap[0],shap[1],4))
    if mean_sub is not None:
        mat_proj1[:,:,0] = mat_proj[:,:,2]
    else:
        mat_proj1[:,:,0] = mat_proj[:,:,4]
    mat_proj1[:,:,1] = mat_proj[:,:,0]-mat_proj[:,:,1]

    mat_proj2[:,:,0] = mat_proj[:,:,5]
    mat_proj2[:,:,1] = mat_proj[:,:,0]
    mat_proj2[:,:,2] = mat_proj[:,:,1]

    mat_proj3[:,:,0] = mat_proj[:,:,5]
    mat_proj3[:,:,1] = mat_proj[:,:,0]
    mat_proj3[:,:,2] = mat_proj[:,:,1]
    mat_proj3[:,:,3] = mat_proj[:,:,4]

    proj1 = im-proj_cube(im,mat_proj1,ortho=1)
    proj2 = im-proj_cube(im,mat_proj2,ortho=1)
    proj3 = im-proj_cube(im,mat_proj3,ortho=1)

    centroid,U,ell1 = mk_ellipticity_atoms(proj1)
    centroid,U,ell2 = mk_ellipticity_atoms(proj2)
    centroid,U,ell3 = mk_ellipticity_atoms(proj3)

    return proj1,proj2,proj3,ell1,ell2,ell3




def mk_ellipticity_atoms_basic_arr(U):
    shap = U.shape
    ell_arr = zeros((shap[0],2))
    i = 0
    for i in range(0,shap[0]):
        ell_i = mk_ellipticity_atoms_basic(U[i,:])
        ell_arr[i,0] = ell_i[0,0]
        ell_arr[i,1] = ell_i[0,1]
    return ell_arr

def mat_mm_ell(im,offset=0,sigw=0,path='../../'):
    import scipy.signal as scisig
    if sigw !=0 :
        centroid,Wc = compute_centroid(im,sigw)
        im = im*Wc
    mat_proj,normv = proj_mat(im.shape,offset)
    mat_ell = zeros((6,im.size))
    i=0
    Li = copy(im)
    for i in range(0,6):
        Ui = mat_proj[:,:,i]
        #Li = scisig.correlate(im,Ui,mode='same')
        Li = correl_c(im,Ui,path=path)
        mat_ell[i,:] = Li.reshape(1,im.shape[0]*im.shape[1])
    return mat_ell

def corr_coeff(im1,im2):
    corr = scisig.correlate(im1,im2,mode='same').max()/(sqrt((im1**2).sum())*sqrt((im2**2).sum()))
    return corr

def stack_corr_coeff(cube_1,cube_2):
    nb_im = cube_1.shape[2]
    corr_out = zeros((nb_im,))
    for i in range(0,nb_im):
        corr_out[i] = corr_coeff(cube_1[:,:,i],cube_2[:,:,i])
    return corr_out

def stack_corr_coeff_m(cube_1m,cube_2):
    nb_eval = cube_1m.shape[3]
    mean_corr = zeros((nb_eval,))
    std_corr = zeros((nb_eval,))
    for i in range(0,nb_eval):
        corr = stack_corr_coeff(cube_1m[:,:,:,i],cube_2)
        mean_corr[i] = corr.mean()
        std_corr[i] = corr.std()

    return mean_corr,std_corr

def transpose_mat_ell(mat_in,shap,offset=0,sigw=0,path=''):
    import scipy.signal as scisig
    mat_proj,normv = proj_mat(shap,offset)
    im = zeros((shap[0],shap[1]))
    i=0
    imi = copy(im)
    Ui = copy(im)
    for i in range(0,6):
        mat_proj_i = mat_proj[:,:,i]
        Ui = rot180(mat_proj_i)
        #Li = scisig.correlate(im,Ui,mode='same')
        imi = mat_in[i,:].reshape(shap[0],shap[1])
        Li = correl_c(imi,Ui,path=path)
        im = im + Li
    return im

def cube_mean_norm(X):
    shap = X.shape
    out = zeros((shap[2],))
    norm_out = zeros((shap[0],shap[2]))
    bias_out = zeros((shap[2],))
    for i in range(0,shap[2]):
        norm_out[:,i] = sqrt((X[:,:,i]**2).sum(axis=1))
        out[i] = mean(norm_out[:,i])
        bias_out[i] = sum(sqrt((X[:,:,i].mean(axis=0))**2))

    return out,norm_out,bias_out




def cube_mm_ell(im,offset=0,sigw=0,path=''):
    import scipy.signal as scisig
    if sigw !=0 :
        centroid,Wc = compute_centroid(im,sigw)
        im = im*Wc
    mat_proj,normv = proj_mat(im.shape,offset)
    cube_ell = zeros((im.shape[0],im.shape[1],6))
    i=0
    for i in range(0,6):
        Ui = mat_proj[:,:,i]
        #Li = scisig.correlate(im,Ui,mode='same')
        cube_ell[:,:,i] = correl_c(im,Ui,path=path)
    return cube_ell

def mat_mm_ell_fft(im,offset=0,sigw=0):
    if sigw !=0 :
        centroid,Wc = compute_centroid(im,sigw)
        im = im*Wc
    mat_proj,normv = proj_mat(im.shape,offset)
    mat_ell = zeros((6,im.size))
    siz=im.shape
    padd_width = siz[0]*0
    im_padd = padding(im,padd_width)
    i=0
    for i in range(0,6):
        Ui = mat_proj[:,:,i]
        Ui_padd = padding(Ui,padd_width)
        Li_padd = scisig.fftconvolve(im_padd,Ui_padd,mode='same')
        Li = Li_padd[padd_width:siz[0]+padd_width,padd_width:siz[1]+padd_width]
        mat_ell[i,:] = Li.reshape(1,im.shape[0]*im.shape[1])
    return mat_ell

def padding(im,padding_width,val=0):

    siz = im.shape
    im_out = ones((siz[0]+2*padding_width,siz[1]+2*padding_width))*val
    im_out[padding_width:siz[0]+padding_width,padding_width:siz[1]+padding_width]=im
    return im_out

def fista_ell(x,h,mu,mat_init,mat_cor,thresh,thresh_type,im_size,gradx_ker,grady_ker,lambda_grad,nb_iter=200): # The code assumes columns vectors
    U = copy(mat_init)
    n1 = int(x.shape[0])
    n2 = int(h.shape[0])
    vect = ones((n2,))
    mse = zeros((nb_iter,1))
    mat,normv = proj_mat(im_size)
    invmat_cor = inv(mat_cor)
    w,v = eigh(invmat_cor)
    V = zeros((n1,n2))
    V_old = copy(V)
    #lip = h.sum()*abs(w).max()+((abs(gradx_ker).sum())**2+(abs(grady_ker).sum())**2)*lambda_grad
    lip = h.sum()*abs(w).max()
    #print lip,(h**2).sum(),(abs(gradx_ker).sum())**2,(abs(grady_ker).sum())**2,lambda_grad*max((normv[3]/normv)**2)
    t=1
    told=t
    i = 0
    s = zeros((1,6))

    tikhx_mat = zeros((6,im_size[0]*im_size[1]))
    tikhy_mat = zeros((6,im_size[0]*im_size[1]))
    for i in range(0,nb_iter):
        res = x - U.dot(h)
        mse[i] = res.sum()**2

        #svecl, s, svecr = linalg.svd(Yi,full_matrices=False)
        #s_thresh = thresholding(s,thresh,thresh_type) # Low rank reg
        #V = svecl.dot(diag(s_thresh).dot(svecr))
        k = 0
        for k in range(0,6):
            compvk = U[k,:]
            compk = compvk.reshape(im_size[0],im_size[1])
            gradx,grady = grad2d(compk,gradx_ker,grady_ker)
            tgradx,tgrady = transp_grad2d(gradx,grady,gradx_ker,grady_ker)
            tikhx_mat[k,:] = (tgradx.reshape(1,im_size[0]*im_size[1]))
            tikhy_mat[k,:] = (tgrady.reshape(1,im_size[0]*im_size[1]))

        #V = U + (mu/lip)*(npma.outer(res,h) - lambda_grad*(tikhx_mat+tikhy_mat)) # Tikhonov reg
        V = U + (mu/lip)*(npma.outer(invmat_cor.dot(res),vect))
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        U = V_old + lamb*(V-V_old)
        V_old = copy(V)
    return mse,s,U


def fista_ell_2(x,h,mu,mat_init,mat_cor,thresh,thresh_type,im_size,gradx_ker,grady_ker,lambda_grad,nb_iter=200): # The code assumes columns vectors; the optimization variable is incov*M

    n1 = int(x.shape[0])
    n2 = int(h.shape[0])
    vect = ones((n2,))
    mse = zeros((nb_iter,1))
    mat,normv = proj_mat(im_size)
    invmat_cor = inv(mat_cor)
    x = invmat_cor.dot(x)
    U = invmat_cor.dot(mat_init)
    w,v = eigh(invmat_cor)
    w2,v = eigh(mat_cor)
    V = zeros((n1,n2))
    V_old = copy(V)
    lip = 2*(h.sum()+((abs(gradx_ker).sum())**2+(abs(grady_ker).sum())**2)*lambda_grad*(abs(w2).max())**2)
    #lip = h.sum()
    #print lip,(h**2).sum(),(abs(gradx_ker).sum())**2,(abs(grady_ker).sum())**2,lambda_grad*max((normv[3]/normv)**2)
    t=1
    told=t
    i = 0
    s = zeros((1,6))

    tikhx_mat = zeros((6,im_size[0]*im_size[1]))
    tikhy_mat = zeros((6,im_size[0]*im_size[1]))
    for i in range(0,nb_iter):
        res = x - U.dot(h)
        mse[i] = res.sum()**2

        #svecl, s, svecr = linalg.svd(Yi,full_matrices=False)
        #s_thresh = thresholding(s,thresh,thresh_type) # Low rank reg
        #V = svecl.dot(diag(s_thresh).dot(svecr))
        k = 0
        for k in range(0,6):
            compvk = U[k,:]
            compk = compvk.reshape(im_size[0],im_size[1])
            gradx,grady = grad2d(compk,gradx_ker,grady_ker)
            tgradx,tgrady = transp_grad2d(gradx,grady,gradx_ker,grady_ker)
            tikhx_mat[k,:] = (tgradx.reshape(1,im_size[0]*im_size[1]))
            tikhy_mat[k,:] = (tgrady.reshape(1,im_size[0]*im_size[1]))
        tikhx_mat = mat_cor.dot(mat_cor.dot(tikhx_mat))
        tikhy_mat = mat_cor.dot(mat_cor.dot(tikhy_mat))
        V = U + (mu/lip)*(npma.outer(res,vect) - lambda_grad*(tikhx_mat+tikhy_mat)) # Tikhonov reg
        #V = U + (mu/lip)*(npma.outer(res,vect))
        t = (1+sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        U = V_old + lamb*(V-V_old)
        V_old = copy(V)
    U = mat_cor.dot(U)
    return mse,s,U

def const_mat_gen(x,h,mat_cor,mat_dict):

    invmat_cor = LA.inv(mat_cor)
    siz = mat_dict.shape
    i,j = 0,0
    Mhs = zeros((siz[2],siz[2]))
    Mxs = zeros((siz[2],))
    for i in range(0,siz[2]):
        Mi = mat_dict[:,:,i]
        ai = invmat_cor.dot(Mi.dot(h))
        Mxs[i] = transpose(x).dot(ai)
        for j in range(0,siz[2]):
            Mj = mat_dict[:,:,j]
            Mhs[i,j] = transpose(h).dot(transpose(Mj).dot(ai))
    return Mhs,Mxs

def const_mat_gen_2(x,h,mat_cor,mat_dict,coeff_dict):

    invmat_cor = LA.inv(mat_cor)
    siz = coeff_dict.shape
    i,j = 0,0
    Mhs = zeros((siz[1],siz[1]))
    Mxs = zeros((siz[1],))
    for i in range(0,siz[1]):
        Mi = mat_dict[:,:,i]
        ai = invmat_cor.dot(Mi.dot(h))
        li = array(coeff_dict[i,:])
        Mxs = Mxs+transpose(x).dot(ai)*transpose(li)
        for j in range(0,siz[1]):
            Mj = mat_dict[:,:,j]
            lj = array(coeff_dict[j,:])
            Mhs[i,j] = transpose(h).dot(transpose(Mj).dot(ai))*(li.dot(transpose(lj)))
    return Mhs,Mxs

def fista_ell_3(x,h,mu,mat_cor,mat_dict,pca_eig_val,tol,nb_iter=200,mean_comp=None,xinit=None): # The code assumes columns vectors; synthesis scheme; a dictionary should be provided

    mat_dict_in  =None
    if mean_comp is not None:
        mean_mat = mat_dict[:,:,0]
        xmean = mean_mat.dot(h)
        x =x-xmean
        mat_dict_in = mat_dict[:,:,1:]
    else: mat_dict_in = mat_dict
    Mhs,Mxs = const_mat_gen(x,h,mat_cor,mat_dict_in)
    siz = mat_dict_in.shape
    if xinit is None:xinit = zeros((siz[2],))
    proj_thresh = sqrt(pca_eig_val)*tol
    mse,U = fista_bas(Mxs,xinit,Mhs,mu,proj_cons=1,proj_thresh=proj_thresh,nb_iter=nb_iter)
    output = zeros((siz[0],siz[1]))
    i = 0
    for i in range(0,siz[2]):
        output = output + U[i]*mat_dict_in[:,:,i]
    if mean_comp is not None:output = output+mean_mat
    return U,mse,output

def fista_ell_4(x,h,mu,mat_cor,mat_dict,thresholds,max_1st_comp,mean_1st_comp,nb_iter=200,mean_comp=None,xinit=None):

    mat_dict_in  =None
    if mean_comp is not None:
        mean_mat = mat_dict[:,:,0]
        xmean = mean_mat.dot(h)
        x =x-xmean
        mat_dict_in = mat_dict[:,:,1:]
    else: mat_dict_in = mat_dict
    Mhs,Mxs = const_mat_gen(x,h,mat_cor,mat_dict_in)
    siz = mat_dict_in.shape
    if xinit is None:xinit = zeros((siz[2],))
    w1=0.8
    w2=1-w1
    w,v = LA.eigh(Mhs)
    lip = (abs(w).max())
    lamb = 1.5
    z1 = copy(xinit)
    z2 = copy(xinit)
    U = copy(xinit)
    i=0
    for i in range(0,nb_iter):
        z1_temp = 2*U-z1-(Mhs.dot(U)-Mxs)/lip
        z1_temp[siz[2]-1]=z1_temp[siz[2]-1]-mean_1st_comp
        z1_temp = thresholding(z1_temp,thresholds/w1,thresh_type=1)
        z1_temp[siz[2]-1]=z1_temp[siz[2]-1]+mean_1st_comp
        z1 = z1 + lamb*(z1_temp-U)
        z2_temp = 2*U-z2-(Mhs.dot(U)-Mxs)/lip
        z2_temp[siz[2]-1]=z2_temp[siz[2]-1]-thresholding(z2_temp[siz[2]-1],max_1st_comp,thresh_type=1)
        z2 = z2 + lamb*(z2_temp-U)
        U = w1*z1+w2*z2
    output = zeros((siz[0],siz[1]))
    i = 0
    for i in range(0,siz[2]):
        output = output + U[i]*mat_dict_in[:,:,i]
    if mean_comp is not None:output = output+mean_mat
    return U,output

def fista_ell_5(x,h,mu,mat_cor,mat_dict,coeff_dict,thresh,nb_iter=200,nb_reweighting=1,mean_comp=None,xinit=None): # The code assumes columns vectors; synthesis scheme; a dictionary should be provided

    mat_dict_in  =None
    if mean_comp is not None:
        mean_mat = mat_dict[:,:,0]
        xmean = mean_mat.dot(h)
        x =x-xmean
        mat_dict_in = mat_dict[:,:,1:]
    else: mat_dict_in = mat_dict
    Mhs,Mxs = const_mat_gen_2(x,h,mat_cor,mat_dict_in,coeff_dict)
    siz = mat_dict_in.shape
    siz_2 = coeff_dict.shape
    if xinit is None:xinit = zeros((siz_2[0],))
    weights = ones((siz_2[0],))
    k=0
    for k in range(0,nb_reweighting+1):
        mse,alpha = fista_bas(Mxs,xinit,Mhs,mu,nb_iter=nb_iter,thresh_cons=1,thresh=thresh*weights)
        weights = ones((siz_2[0],))/(abs(alpha)/thresh + 1)
    U = coeff_dict.dot(alpha)
    output = zeros((siz[0],siz[1]))
    i = 0
    for i in range(0,siz[2]):
        output = output + U[i]*mat_dict_in[:,:,i]
    if mean_comp is not None:output = output+mean_mat
    return U,alpha,mse,output


def fista_ell_6(x,h,mu,mat_cor,mat_dict,coeff_dict,cells_means,nb_cells):

    siz1 = cells_means.shape
    nb_cells_tot = siz1[0]
    res = zeros((nb_cells_tot,1))
    scal = zeros((nb_cells_tot,1))
    mean_mat = mat_dict[:,:,0]
    xmean = mean_mat.dot(h)
    x =x-xmean
    mat_dict_in = mat_dict[:,:,1:]
    Mhs0,Mxs0 = const_mat_gen(x,h,mat_cor,mat_dict_in)
    i=0
    for i in range(0,nb_cells_tot):
        x0 = Mhs0.dot(transpose(cells_means[i,:]))
        a = transpose(Mxs0).dot(x0)
        b = transpose(x0).dot(x0)
        #opt = a/b
        opt=1
        res[i] = transpose(Mxs0 - opt*x0).dot(Mxs0 - opt*x0)
        scal[i]=opt
    ind = argsort(squeeze(res))
    siz2 = coeff_dict.shape
    siz3 = mat_dict_in.shape
    mean_mat_2 = mean_mat*0

    res_opt = zeros((nb_cells,1))
    opt_candidate = zeros((siz2[1],nb_cells)) # siz2[1] = sparsity level
    j = 0
    for i in range(0,nb_cells):
        for j in range(0,siz3[2]):
            mean_mat_2=mean_mat_2+ mat_dict_in[:,:,j]*scal[ind[i]]*cells_means[ind[i],j]
        xtemp = x - mean_mat_2.dot(h)
        Mhs,Mxs = const_mat_gen_2(xtemp,h,mat_cor,mat_dict_in,squeeze(coeff_dict[:,:,ind[i]]))
        Mxs = squeeze(transpose(Mxs))
        #M0 = transpose(Mhs).dot(Mhs)
        w,v = LA.eigh(M0)
        mu = 1
        mse,U = fista_bas(y,x_init,M,mu,nb_iter=100)
        #opt_candidate[:,i] = LA.inv(M0).dot(transpose(Mhs)).dot(Mxs)
        opt_candidate[:,i] = U
        #res_opt[i] = transpose(Mxs - Mhs.dot(opt_candidate[:,i])).dot(Mxs - Mhs.dot(opt_candidate[:,i]))
        res_opt[i] = mse[900]
    print res_opt
    ind2 = argsort(squeeze(res_opt))
    alpha = transpose(cells_means[ind[ind2[0]],j])*scal[ind[ind2[0]]]
    for i in range(0,siz2[1]):
        alpha = alpha + coeff_dict[:,:,ind[ind2[0]]].dot(opt_candidate[:,ind2[0]])
    output = zeros((siz3[0],siz3[1]))
    i = 0
    for i in range(0,siz3[2]):
        output = output + alpha[i]*mat_dict_in[:,:,i]
    output = output+mean_mat
    return alpha,alpha,output


def fista_ell_7(x,h,mu,mat_cor,mat_dict,cells_means,nb_cells,iter_en=0,nb_iter=50):
    siz1 = cells_means.shape
    siz2 = mat_dict.shape
    nb_cells_tot = siz1[0]
    #nb_cells=nb_cells_tot ## Warnign, debugg
    res = zeros((nb_cells_tot,1))
    scal = zeros((nb_cells_tot,1))
    mean_mat = zeros((6,siz1[1]/6,nb_cells_tot))
    mat_dict_mat = zeros((6,siz1[1]/6,siz2[1],nb_cells))
    i=0
    for i in range(0,nb_cells_tot):
        mean_mat_vect = cells_means[i,:]
        mean_mati = mean_mat_vect.reshape(6,siz1[1]/6)
        xmean = mean_mati.dot(h)
        res[i] = transpose(x-xmean).dot(x-xmean)
        mean_mat[:,:,i]=mean_mati
    ind = argsort(squeeze(res))

    res_opt = zeros((nb_cells,1))
    opt_candidate = zeros((siz2[1],nb_cells)) # siz2[1] = sparsity level
    xinit = zeros((siz2[1],))

    for i in range(0,nb_cells):
        mean_mati = array(mean_mat[:,:,ind[i]])
        xtemp = x - mean_mati.dot(h)
        mat_dicti = zeros((6,siz1[1]/6,siz2[1]))
        j = 0
        for j in range(0,siz2[1]):
            mat_dict_vectij = mat_dict[:,j,ind[i]]
            mat_dicti[:,:,j] = mat_dict_vectij.reshape(6,siz1[1]/6)
        mat_dict_mat[:,:,:,i]=mat_dicti
        Mhs,Mxs = const_mat_gen(xtemp,h,mat_cor,mat_dicti)
        Mxs = squeeze(transpose(Mxs))
        #M0 = transpose(Mhs).dot(Mhs)
        w,v = LA.eigh(Mhs)
        print 'eigen_val: ', w
        U = zeros((siz2[1],1))
        if iter_en==1:
            mu = 1
            mse,U = fista_bas(Mxs,xinit,Mhs,mu,nb_iter=nb_iter)
            #
            opt_candidate[:,i] = U
            #res_opt[i] = transpose(Mxs - Mhs.dot(opt_candidate[:,i])).dot(Mxs - Mhs.dot(opt_candidate[:,i]))
            res_opt[i] = mse[nb_iter-1]
        else:
            opt_candidate[:,i] = LA.inv(Mhs).dot(Mxs)
            res_opt[i] = transpose(Mxs - Mhs.dot(opt_candidate[:,i])).dot(Mxs - Mhs.dot(opt_candidate[:,i]))

#print res_opt
    ind2 = argsort(squeeze(res_opt))
    output = mean_mat[:,:,ind[ind2[0]]]

    for i in range(0,siz2[1]):
        output = output + opt_candidate[i,ind2[0]]*mat_dict_mat[:,:,i,ind2[0]]
    return output,ind,ind2,res_opt,opt_candidate



def fista_bas(y,x_init,M,mu,proj_cons=0,proj_thresh=None,nb_iter=100,thresh_cons=0,thresh=None): # Minimizes : J(X) = X^T*M*X - y^T*X + C, M being symetric positive
    import numpy.ma as npma
    from numpy import *
    import numpy.linalg as LA
    siz = M.shape
    """a=1
        M = M + a*eye(siz[0])"""
    n1 = int(x_init.shape[0])
    mse = zeros((nb_iter,1))
    U = copy(x_init)
    V = zeros((n1,))
    V_old = copy(V)
    w,v = LA.eigh(M)
    lip = (abs(w).max())

    #lip = h.sum()
    #print lip,(h**2).sum(),(abs(gradx_ker).sum())**2,(abs(grady_ker).sum())**2,lambda_grad*max((normv[3]/normv)**2)
    t=1
    told=t
    i = 0

    for i in range(0,nb_iter):

        res = y - M.dot(U)
        mse[i] = res.sum()**2
        V = U + (mu/lip)*res
        #U = U + (mu/lip)*(transpose(M)).dot(res)
        k=0
        if proj_cons==1:
            for k in range(0,n1):
                if abs(V[k])>proj_thresh[k]:
                    V[k] = (V[k]/abs(V[k]))*proj_thresh[k]
        if thresh_cons==1:
            thresh_type=1
            V = thresholding(V,thresh,thresh_type)
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        U = V_old + lamb*(V-V_old)
        V_old = copy(V)
    return mse,U


def sig_sparsity_analysis(x,nb_points):
    pts = 100*double(array(range(1,nb_points+1)))/nb_points
    y = x.reshape(size(x))
    ind_sort = argsort(y)

    energy = sum(x**2)
    en_perc = zeros((nb_points,))
    for i in range(0,nb_points):
        en_perc[i] = 100*sum(y[ind_sort[-int(pts[i]*size(x)/100):]]**2)/energy

    return en_perc,pts

def sig_sparsity_analysis_m(cube,nb_points):
    out  = zeros((cube.shape[2],nb_points))
    pts=None
    for i in range(0,cube.shape[2]):
        en_perc_i,pts =sig_sparsity_analysis(cube[:,:,i],nb_points)
        out[i,:] = en_perc_i

    return out,pts



def thresholding(x,thresh,thresh_type): # x is a 1D or 2D array, thresh is an array of the same size, thresh_type is 1 or 0, 1 for soft thresholding, 0 for hard thresholding
    xthresh = copy(x)
    n = x.shape

    if len(n)>0:
        n1 = n[0]
    else:
        n1=1
    n2=1
    if len(n)==2:n2 =n[1]
    i,j = 0,0
    if len(n)==2:
        for i in range(0,n1):
            for j in range(0,n2):
                if abs(xthresh[i,j])<thresh[i,j]:xthresh[i,j]=0
                else:
                    if xthresh[i,j]!=0:xthresh[i,j]=(abs(xthresh[i,j])/xthresh[i,j])*(abs(xthresh[i,j])-thresh_type*thresh[i,j])

    elif len(n)==1:
        for i in range(0,n1):
            if abs(xthresh[i])<thresh[i]:xthresh[i]=0
            else:
                if xthresh[i]!=0:xthresh[i]=(abs(xthresh[i])/xthresh[i])*(abs(xthresh[i])-thresh_type*thresh[i])
    elif len(n)==0:
        if abs(xthresh)<thresh:xthresh=0
        else:
            if xthresh!=0:xthresh=(abs(xthresh)/xthresh)*(abs(xthresh)-thresh_type*thresh)

    return xthresh

def thresholding_perc(x,perc,thresh_type):
    shap = x.shape
    thres_map = None
    thresh =None
    thresh_map = ones(shap)
    xthresh = None
    if len(shap)<3:
        y = x.reshape(size(x))
        ind_sort = argsort(abs(y))
        thresh = abs(y)[ind_sort[-int(perc*size(x))]]
        thresh_map = ones(shap)*thresh
        xthresh = thresholding(x,thresh_map,thresh_type)
    elif len(shap)==3:
        for i in range(shap[2]):
            y = x[:,:,i].reshape(size(x[:,:,i]))
            ind_sort = argsort(abs(y))
            thresh = abs(y)[ind_sort[-int(perc*size(x[:,:,i]))]]
            thresh_map[:,:,i] *= thresh
        xthresh = thresholding_3D(x,thresh_map,thresh_type)

    return xthresh,thresh_map

def threshold_perc(x,perc=0.99):
    E = sum(x**2)
    i = -1
    res = 0
    v = x.reshape((size(x),))
    while(res<perc*E):
        res+=v[i]**2
        i-=1
    return v[i+1]


def kthresholding(x,k):
    k = int(k)
    if k<1:
        print "Warning: wrong k value for k-thresholding"
        k = 1
    if k>len(x):
        return x
    else:
        xout = copy(x)*0
        ind = argsort(abs(x))
        xout[ind[-k:]] = x[ind[-k:]]
        return xout

def kthresholding_im(x,k,thresh_type):
    if k<1:
        print "Warning: wrong k value for k-thresholding"
        k = 1
    shap = x.shape
    xv = copy(x.reshape((shap[0]*shap[1],)))
    ind = argsort(abs(xv))
    thresh_map = ones((shap[0],shap[1]))*xv[ind[-k-1]]
#print xv[ind[-k-1]], ind[-k:]
    xthresh = thresholding(x,thresh_map,thresh_type)

    return xthresh

def khighest(x,k):
    shap = x.shape
    xv = copy(x.reshape((shap[0]*shap[1],)))
    ind = argsort(abs(xv))
    thresh = xv[ind[-k-2]]
    i,j = where(abs(x)>=thresh)
    map = x*0
    map[i,j] = 1

    return map


def lineskthresholding(mat,k):
    mat_out = copy(mat)
    shap = mat.shape
    for j in range(0,shap[0]):
        mat_out[j,:] = kthresholding(mat[j,:],k)
    return mat_out


def l_inf_ball_proj(x,thresh,thresh_type,cent=None):
    xthresh=None
    if cent is not None:
        xthresh = x - thresholding(x-cent,thresh,thresh_type)
    else:
        xthresh = x - thresholding(x,thresh,thresh_type)
    return xthresh

def thresholding_3D(x,thresh,thresh_type):
    from numpy import copy
    shap = x.shape
    nb_plan = shap[2]
    k=0
    xthresh = copy(x)
    for k in range(0,nb_plan):
        xthresh[:,:,k] = thresholding(copy(x[:,:,k]),thresh[:,:,k],thresh_type)

    return xthresh

def orth_comple(A,tol=99.999): # This function computes an orthogonal basis of the orthogonal complement of A columns space
    shap = A.shape
    U, s1, Vt = linalg.svd(A,full_matrices=False)
    rank = 1
    i=1
    acc = 100*sum(s1[:i]**2)/sum(s1[:]**2)
    while acc<tol:
        i+=1
        acc = 100*sum(s1[:i]**2)/sum(s1[:]**2)
    rank = i
    A_ortho = U[:,:rank]
    B = eye((shap[0])) - A_ortho.dot(transpose(A_ortho))
    U,s2, Vt = linalg.svd(B,full_matrices=False)
    comple_ortho = U[:,:shap[0]-rank]
    return comple_ortho,s2,rank

def kernel_ext(mat,tol = 0.01): # The kernel of the matrix mat is defined as the vector space spanned by the eigenvectors corresponding to 1% of the sum of the squared singular values
    from numpy.linalg import svd
    U, s, Vt = svd(mat,full_matrices=True)
    e = (s**2).sum()
    eker = 0
    count = 0
    while eker<e*tol:
        count+=1
        eker+=s[-count]**2
    count -=1
    #ker = Vt[-count:,:]
    ker = copy(Vt)

    return ker


def kernel_test(mat,vect_test,tol=0.01):
    ker = kernel_ext(mat,tol = tol)
    proj_coeff = ker.dot(vect_test)

    loss = 100*(proj_coeff**2).sum()/(vect_test**2).sum()
    return loss


def kernel_mat_test(mat,mat_test,tol=0.01):
    ker = kernel_ext(mat,tol = tol)
    loss = 100*((mat_test - transpose(ker).dot(ker.dot(mat_test)))**2).sum()/((mat_test**2).sum())

    return loss

def kernel_mat_test_unit(mat,mat_test,tol=0.01):
    ker = kernel_ext(mat,tol = tol)

    shap = ker.shape
    nb_vect = shap[0]
    #print "Null space size: ",nb_vect
    loss = (mat_test**2).sum()
    select_ind = -1
    u= None
    uout=None
    for i in range(0,nb_vect):
        u = copy(ker[i,:])
        u = u.reshape((1,shap[1]))
        err = ((mat_test - transpose(u).dot(u.dot(mat_test)))**2).sum()
        if err <loss:
            select_ind = i
            loss = err
            uout = copy(u)
    return loss,uout,ker,select_ind

def kernel_mat_stack_test_unit(mat_stack,mat_test,tol=0):
    shap = mat_stack.shape
    nb_mat = shap[2]
    loss = (mat_test**2).sum()
    #print "========== Ref Loss =============",loss
    select_ind = None
    vect_out = None
    ker_out = None
    select_ind_2 = None
    for i in range(0,nb_mat):
        lossi,ui,keri,indi = kernel_mat_test_unit(mat_stack[:,:,i],mat_test,tol=tol)
        #print "========== ith =============",lossi
        if lossi<loss:
            loss = lossi
            vect_out = copy(ui)
            select_ind = i
            select_ind_2 = indi
            ker_out = keri
    return vect_out,select_ind,loss,ker_out,select_ind_2

def kernel_test_stack(mat,vect_test,tol=0.01):
    nb_mat = mat.shape[2]
    loss = zeros((nb_mat,))
    for i in range(0,nb_mat):
        loss[i] = kernel_test(mat[:,:,i],vect_test,tol=tol)

    return loss

def kernel_testmat_stack(mat,mat_test,tol=0.01):
    nb_mat = mat.shape[2]
    loss = zeros((nb_mat,))
    for i in range(0,nb_mat):
        loss[i] = kernel_mat_test(mat[:,:,i],mat_test,tol=tol)

    return loss






def log_sampling(val_min,val_max,nb_samp):
    from  numpy import log,double,array,exp
    lval_min = log(val_min)
    lval_max = log(val_max)
    a = double(array(range(0,nb_samp)))/(nb_samp-1)
    a = a*(lval_max-lval_min) + lval_min
    a = exp(a)
    return a

def lin_sampling(val_min,val_max,nb_samp):
    a = double(array(range(0,nb_samp)))/(nb_samp-1)
    a = a*(val_max-val_min) +val_min
    return a

def l_inf_ball_proj_3D(x,thresh,thresh_type):
    xthresh = x - thresholding_3D(x,thresh,thresh_type)
    return xthresh

def low_rank_approx(mat_ell,thresh,thresh_type,mu,nb_iter=100):
    import numpy.ma as npma
    from numpy import *
    U = mat_ell
    n1 = mat_ell.shape[0]
    n2 = mat_ell.shape[1]
    mse = zeros((nb_iter,1))
    V = zeros((n1,n2))
    V_old = V
    t=1
    told=t
    i = 0
    s = zeros((1,6))
    for i in range(0,nb_iter-1):
        res = mat_ell - U
        mse[i] = res.sum()**2
        Yi = U + mu*res
        svecl, s, svecr = linalg.svd(Yi,full_matrices=False)
        s_thresh = thresholding(array(s),thresh,thresh_type) # Low rank reg
        V = svecl.dot(diag(s_thresh).dot(svecr))
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        U = V_old + lamb*(V-V_old)
        V_old = V
    return mse,s_thresh,U


def gal_psnr(gal,patch_size):
    from numpy import *
    n1 = gal.shape[0]
    n2 = gal.shape[1]
    noise_samp = zeros((2*patch_size,2*patch_size))
    noise_samp[0:patch_size,0:patch_size] = gal[0:patch_size,0:patch_size]
    noise_samp[0:patch_size,patch_size:2*patch_size] = gal[0:patch_size,n2-patch_size:n2]
    noise_samp[patch_size:2*patch_size,0:patch_size] = gal[n1-patch_size:n1,0:patch_size]
    noise_samp[patch_size:2*patch_size,patch_size:2*patch_size] = gal[n1-patch_size:n1,n2-patch_size:n2]
    noise_std = nanstd(noise_samp)
    psnr = gal.max()/noise_std
    return psnr,noise_samp

def gal_selector(gal_cube,nb_gal,patch_size,min_psnr,p=1):
    from numpy import *
    n1 = gal_cube.shape[0]
    n2 = gal_cube.shape[1]
    n3 = gal_cube.shape[2]
    index = ones((nb_gal,))*-1
    psnr_select = ones((nb_gal,))*-1
    i = 0
    if nb_gal<n3/2:
        for i in range(0,nb_gal):
            a = False
            count = 0
            while (a == False and count < n3*p ):
                ind = random.randint(0, n3)
                psnr,noise_stamp = gal_psnr(gal_cube[:,:,ind],patch_size)
                if psnr>min_psnr and abs(index-ind).min()>0 :
                    a = True
                    index[i]=ind
                    psnr_select[i] = psnr
                count+=1
    else:
        print 'SNR argument ignored, too much images asked'
        index = array(range(0,nb_gal))
    return index

def grad2d(im,maskx,masky):
    import scipy.signal as scisig
    from numpy import *
    gradx = scisig.fftconvolve(im, maskx, mode='same')
    grady = scisig.fftconvolve(im, masky, mode='same')
    return gradx,grady

def transp_grad2d(imx,imy,maskx,masky):
    import scipy.signal as scisig
    from numpy import *
    tgradx = scisig.fftconvolve(imx, rot90(rot90(maskx)), mode='same')
    tgrady = scisig.fftconvolve(imy, rot90(rot90(masky)), mode='same')
    return tgradx,tgrady

def factorielle(x):
    if x < 2:
        return 1
    else:
        return x * factorielle(x-1)

def anscomb(x):
    from numpy import *
    x_stab = 2*sqrt(x+(3/8))
    return x_stab

def inv_anscomb_nv(x):
    from numpy import *
    xinv = (x/2)**2 - (3/8)
    return xinv


def inv_anscomb(x,n=0):
    from numpy import *
    i=0
    xinv = x*0
    if n==0:
        xinv = (x**2)/4 + (double(x)**-1)*sqrt(3/32) - 11*(double(x)**-2) + (double(x)**-3)*sqrt(75/128) - 1/8
    else:
        for i in range(0,n):
            xinv = xinv + 2*sqrt(i+3/8)*(x**i)*exp(-x)/factorielle(i)
    return xinv

def compact_spike1d(rad,w,deg):
    from numpy import *
    spike = zeros((2*rad+1,))
    i=0
    for i in range(0,2*rad+1):spike[i]=exp(-abs((double(i-rad)/w))**(double(1)/deg))*sinc(double(i-rad)/rad)
    spike = spike/sqrt((spike**2).sum())
    return spike


def spike_transform(x,rad,nb_deg,nb_width,deg_min=1,deg_max=2,width_min=5,width_max=20):
    import scipy.signal as scisig
    from numpy import *
    l = size(x)
    step_deg = double(deg_max-deg_min)/nb_deg
    step_width = double(width_max-width_min)/nb_width
    spike_coeff = zeros((l,nb_deg*nb_width))
    i,j = 0,0
    spec_rad_approx = 0
    for i in range(0,nb_deg):
        for j in range(0,nb_width):
            spike_ij = compact_spike1d(rad,double(rad)/(width_min+j*step_width),deg_min+i*step_deg)
            spec_rad_approx = spec_rad_approx + (spike_ij.sum())**2
            spike_coeff[:,j+i*nb_width]= scisig.fftconvolve(x, spike_ij, mode='same')

    return spike_coeff,spec_rad_approx


def spike_transform_transp(spike_coeff,rad,nb_deg,nb_width,deg_min=1,deg_max=2,width_min=5,width_max=20):
    import scipy.signal as scisig
    from numpy import *
    step_deg = double(deg_max-deg_min)/nb_deg
    step_width = double(width_max-width_min)/nb_width
    l = spike_coeff.shape[0]
    spike = zeros((l,))
    for i in range(0,nb_deg):
        for j in range(0,nb_width):
            spike_ij = compact_spike1d(rad,double(rad)/(width_min+j*step_width),deg_min+i*step_deg)
            spike = spike+scisig.fftconvolve(spike_coeff[:,j+i*nb_width], spike_ij, mode='same')
    return spike


def noise_sim_spike(sig,len,nb_mont,rad,nb_deg,nb_width,deg_min=1,deg_max=2,width_min=5,width_max=20):
    from numpy import *
    from numpy.matlib import *

    noise_vec = zeros((len,nb_deg*nb_width))
    sample = ndarray(shape=(len,nb_deg*nb_width,nb_mont), dtype=float)
    i,j,k,z=0,0,0,0
    for k in range(0,nb_mont):
        noise = sig*randn((len,1))
        sample[:,:,k]=spike_transform(noise,rad,nb_deg,nb_width,deg_min,deg_max,width_min,width_max)
    for z in range(0,len):
        for i in range(0,nb_deg):
            for j in range(0,nb_width): noise_vec[z,j+i*nb_width]=std(sample[z,j+i*nb_width,:])

    return noise_vec



def spike_analysis(x,rad,nb_deg,nb_width,mu,nb_iter,nsigma,sigma,thresh_type,nb_mont,pos_cons=True,deg_min=1,deg_max=2,width_min=5,width_max=20):
    U,spec_rad = spike_transform(x,rad,nb_deg,nb_width,deg_min,deg_max,width_min,width_max)
    Urec = spike_transform_transp(U,rad,nb_deg,nb_width,deg_min,deg_max,width_min,width_max)
    a = (x*Urec).sum()/(Urec**2).sum()
    U = a*U
    #U = U*0
    from numpy import *
    n1 = size(x)
    mse = zeros((nb_iter,))
    V = zeros((n1,nb_deg*nb_width))

    V_old = V
    t=1
    told=t
    i = 0
    Urec = x*0
    #noise_std = noise_sim_spike(sigma,n1,nb_mont,rad,nb_deg,nb_width,deg_min,deg_max,width_min,width_max)
    #noise_std = mu*noise_std/spec_rad
    #print noise_std
    #thresh = nsigma*noise_std
    for i in range(0,nb_iter):
        print i
        Urec = spike_transform_transp(U,rad,nb_deg,nb_width,deg_min,deg_max,width_min,width_max)
        res = x - Urec
        mse[i] = res.sum()**2
        res_coeff,spec_rad = spike_transform(res,rad,nb_deg,nb_width,deg_min,deg_max,width_min,width_max)
        grad = mu*res_coeff/spec_rad
        Yi = U + grad
        thresh = nsigma*mu*ones((n1,nb_deg*nb_width))/spec_rad
        """ki,kj = 0,0
            for ki in range(0,nb_deg):
            for kj in range(0,nb_width):
            sigma = 1.4826*mad(grad[:,kj+ki*nb_width])
            print sigma
            thresh[:,kj+ki*nb_width] = nsigma*sigma*thresh[:,kj+ki*nb_width]"""
        V = thresholding(Yi,thresh,thresh_type)
        if pos_cons==True:
            ind = where(V<0)
            V[ind]=0
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        U = V_old + lamb*(V-V_old)
        V_old = copy(V)
    Urec = spike_transform_transp(U,rad,nb_deg,nb_width,deg_min,deg_max,width_min,width_max)

    return Urec,mse

def mad(x):
    from numpy import *
    return median(abs(x-median(x)))

def cartesian_product(arrays):
    import numpy as numpy
    broadcastable = numpy.ix_(*arrays)
    broadcasted = numpy.broadcast_arrays(*broadcastable)
    rows, cols = reduce(numpy.multiply, broadcasted[0].shape), len(broadcasted)
    out = numpy.empty(rows * cols, dtype=broadcasted[0].dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T

def get_noise(im,nb_iter=5,k=3):
    sig = 1.4826*mad(im)
    for i in range(0,nb_iter):
        im_thresh = im*(abs(im)>k*sig)
        sig = 1.4826*mad(im-im_thresh)
    return sig

def get_noise_arr(arr):

    shap = arr.shape
    ind = list()
    for i in shap[2:]:
        ind.append(arange(0,i))
    coord = cartesian_product(ind)
    noise_map = ones(arr.shape)
    s = slice(None) # equivalent to ':
    for i in range(0,coord.shape[0]):
        sig = get_noise(arr[(s,s)+tuple(coord[i,:])])
        noise_map[(s,s)+tuple(coord[i,:])]*=sig

    return noise_map

def PCA(data,nb_comp,mean_sub=None): # Rows represent obsrvations and columns represent the variables
    import numpy.ma as npma
    from numpy import *
    import scipy.linalg as LA
    siz = data.shape
    mean_data = zeros((1,siz[1]))
    if mean_sub is not None:
        k=0
        for k in range(0,siz[0]):
            mean_data = mean_data + data[k,:]
        mean_data = mean_data/siz[0]
    data_cp = data - npma.outerproduct(ones((siz[0],1)),mean_data)
    cov_mat = transpose(data_cp).dot(data_cp)/siz[0]
    #print 'PCA start'
    w,v = LA.eigh(cov_mat,eigvals=(siz[1]-nb_comp,siz[1]-1))
    coeff = data_cp.dot(v)
    #print 'PCA done'
    return w,v,coeff,mean_data

def SVD_interf(data,nb_comp,mean_sub=None,coeff_comp=None): # Rows represent obsrvations and columns represent the variables
    import numpy.ma as npma
    siz = data.shape
    mean_data = zeros((1,siz[1]))
    data_cp = copy(data)
    if mean_sub is not None:
        k=0
        for k in range(0,siz[0]):
            mean_data = mean_data + data[k,:]
        mean_data = mean_data/siz[0]
        data_cp = data - npma.outerproduct(ones((siz[0],1)),mean_data)
    U, s, Vt = linalg.svd(data_cp,full_matrices=False)
    eig_vect = transpose(Vt[0:nb_comp,:])
    w = s**2
    coeff=None
    if coeff_comp is not None:
        coeff = data.dot(eig_vect)
    return w,eig_vect,coeff,mean_data

def cube_svd(cube,nb_comp=None,ind=None,mean_sub=False):
    shap = cube.shape
    if nb_comp is None:
        nb_comp = min(shap[0]*shap[1],shap[2])
    mat = cube.reshape((shap[0]*shap[1],shap[2]))
    data_mean = None
    centered_data = None
    if ind is None:
        ind = range(0,shap[2])
    if mean_sub:
        data_mean = cube.mean(axis=2)
        mat -= data_mean.reshape((shap[0]*shap[1],1)).dot(ones((1,shap[2])))
        centered_data = copy(cube)
        for i in range(0,shap[2]):
            centered_data[:,:,i]-=data_mean
    U, s, Vt = svd(mat[:,ind],full_matrices=False)
    shap_u = U.shape
    coeff = transpose(U[:,0:nb_comp]).dot(mat)
    approx = U[:,0:nb_comp].dot(coeff)
    comp_cube = U[:,0:nb_comp].reshape((shap[0],shap[1],min(nb_comp,shap[2])))
    approx_cube =  approx.reshape((shap[0],shap[1],shap[2]))
    if mean_sub:
        return coeff,comp_cube,approx_cube,data_mean,centered_data
    else:
        return coeff,comp_cube,approx_cube

def cube_svd_m(cube,nb_comp=None):
    shap = cube.shape
    est = copy(cube)*0
    for i in range(0,shap[4]):
        for j in range(0,shap[3]):
            print "==============================================",j,"/",shap[3],"======================================================="
            coeff,comp_cube,approx_cube = cube_svd(cube[:,:,:,j,i],nb_comp=nb_comp)
            est[:,:,:,j,i] = approx_cube
    return est

"""def psf_interp(psf,data_pos,method="spline",nb_comp=20):
    coeff,comp_cube,approx_cube = cube_svd(psf,nb_comp=nb_comp)
    interpolator  = list()
    x =
    for i in range(0,nb_comp):"""





def shape_dict_coeff(im,dict,mean_sub=None): # We assume that the dictionary columns represent the atoms
    from numpy import *
    n = im.size
    data_mat_shape = mat_mm_ell(im)
    data_mat_vec = data_mat_shape.reshape(1,6*n)
    if mean_sub is not None: data_mat_vec = data_mat_vec-transpose(dict[:,0])
    coeff_temp = data_mat_vec.dot(dict)
    coeff = coeff_temp[0,1:]
    return coeff

def err_ell_trunc(im,dict,nb_comp,mean_sub=None):
    from numpy import *
    coeff = shape_dict_coeff(im,dict,mean_sub=mean_sub)

    n = im.size
    l = coeff.size
    for k in range(0,l-nb_comp):
        coeff[k]=0
    mat_shape_approx_vec = None
    if mean_sub is not None:
        mat_shape_approx_vec = coeff.dot(transpose(dict[:,1:]))
        mat_shape_approx_vec = mat_shape_approx_vec+transpose(dict[:,0])
    else:
        mat_shape_approx_vec = coeff.dot(transpose(dict))
    mat_shape_approx = mat_shape_approx_vec.reshape(6,n)
    centroid,U,ell= mk_ellipticity_atoms(im)
    ell_approx = mk_ellipticity_atoms_basic(mat_shape_approx[:,(n-1)/2])
    err = 100*(ell-ell_approx)/ell
    return ell,ell_approx,err

def laplace_gal_PCA_coeff_fitting(coeff):
    from numpy import *
    import scipy.stats as scisig_stat
    siz = coeff.shape
    scale_vect = zeros((siz[1],1))
    i=0
    for i in range(0,siz[1]-1):
        loc,scale = scisig_stat.laplace.fit(coeff[:,i])
        print scale
        scale_vect[i]=scale
    hist,edges = histogram(coeff[:,siz[1]-1], bins=siz[0]/100)
    i = where(hist==max(hist))
    j = i[0][0]
    cent = (edges[j]+edges[j+1])/2
    siz_2 = 2*hist[0:j].sum()
    coeff_sym = zeros((siz_2,1))
    k=0
    for k in range(0,siz_2/2):
        if coeff[k,siz[1]-1] <= cent:
            coeff_sym[2*k]= coeff[k,siz[1]-1]
            coeff_sym[2*k+1]= -coeff[k,siz[1]-1]
    loc,scale = scisig_stat.laplace.fit(coeff_sym,loc=cent,scale=1)
    scale_vect[siz[1]-1]=scale
    max_1st_comp = max(coeff[:,siz[1]-1])
    mean_1st_comp=cent
    return scale_vect,max_1st_comp,mean_1st_comp

def sarah_poisson_filt(signal,nb_iter,thresh,thresh_type,pos_cons=True,mu=0.9):
    from numpy import *
    # Anscombe transform
    sig_stab = 2*sqrt(signal+(3/8))
    # Offset removing
    off = median(sig_stab)
    sig_stab = sig_stab - off
    # Transformed signal filtering
    i=0
    mse = zeros((nb_iter,))
    filt_sig = sig_stab
    # Signal restoration
    for i in range(0,nb_iter):
        res = sig_stab - filt_sig
        mse[i] = (res**2).sum()
        filt_sig = filt_sig + mu*res
        # Sparsity constraint
        filt_sig = thresholding(filt_sig,thresh,thresh_type)
        if pos_cons==True:
            ind = where(filt_sig<0)
            filt_sig[ind]=0
    sig_stab = filt_sig+off
    signal_filt = (sig_stab/2)**2 - 3/8
    return signal_filt,mse

def sarah_poisson_filt_2(signal,nb_iter,pos_cons=True,mu=0.9,opt=None):
    from numpy import *
    # Anscombe transform
    sig_stab = 2*sqrt(signal+(3/8))
    # Offset removing
    off = median(sig_stab)
    sig_stab = sig_stab - off
    # Transformed signal filtering
    i=0
    mse = zeros((nb_iter,))
    filt_sig = sig_stab
    # Signal restoration
    print(pos_cons)
    print(opt)
    print(mu)
    for i in range(0,nb_iter):
        res = sig_stab - filt_sig
        mse[i] = (res**2).sum()
        filt_sig = filt_sig + mu*res
        # Sparsity constraint
        filt_sig = isap.mr1d_filter(filt_sig, opt)
        if pos_cons==True:
            ind = where(filt_sig<0)
            filt_sig[ind]=0
    sig_stab = filt_sig+off
    signal_filt = (sig_stab/2)**2 - 3/8
    return signal_filt,mse


def sarah_poisson_filt_3(signal,nb_iter,nsigma,thresh_type,pos_cons=True,mu=0.9,opt=None):
    from numpy import *
    # Anscombe transform
    sig_stab = 2*sqrt(signal+(3/8))
    # Offset removing
    off = median(sig_stab)
    sig_stab = sig_stab - off
    # Transformed signal filtering
    i=0
    mse = zeros((nb_iter,))
    filt_sig = sig_stab
    # Signal restoration
    print(pos_cons)
    print(opt)
    print(mu)
    info_filename = "temp_info.mr"
    for i in range(0,nb_iter):
        res = sig_stab - filt_sig
        mse[i] = (res**2).sum()
        filt_sig = filt_sig + mu*res
        # Sparsity constraint
        filt_sig = isap.mr1d_filter_beta(filt_sig, info_filename,nsigma,thresh_type,opt=None)
        if pos_cons==True:
            ind = where(filt_sig<0)
            filt_sig[ind]=0
    sig_stab = filt_sig+off
    signal_filt = (sig_stab/2)**2 - 3/8
    return signal_filt,mse



def convol_check(im1,im2):
    from numpy import *
    n1 = im1.shape[0]
    n2 = im1.shape[1]
    out = zeros((n1,n2))
    for i in range(0,n1):
        for j in range(0,n2):
            bi = array([-i,-(n1-1)/2]).max()
            bj = array([-j,-(n2-1)/2]).max()
            ej = array([n2-j-1,(n2-1)/2]).min()
            ei = array([n1-i-1,(n1-1)/2]).min()
            for k in range(bi,ei+1):
                for l in range(bj,ej+1):
                    out[i,j] = out[i,j]+im1[i+k,j+l]*im2[-k+(n1-1)/2,-l+(n2-1)/2]
    return out


def correl_check(im1,im2):
    from numpy import *
    n1 = im1.shape[0]
    n2 = im1.shape[1]
    out = zeros((n1,n2))
    for i in range(0,n1):
        for j in range(0,n2):
            bi = array([-i,-(n1-1)/2]).max()
            bj = array([-j,-(n2-1)/2]).max()
            ej = array([n2-j-1,(n2-1)/2]).min()
            ei = array([n1-i-1,(n1-1)/2]).min()
            for k in range(bi,ei+1):
                for l in range(bj,ej+1):
                    out[i,j] = out[i,j]+im1[i+k,j+l]*im2[k+(n1-1)/2,l+(n2-1)/2]
    return out

def correl_c(im1,im2,path=''):

    name1 = path+rand_file_name('.fits')
    name2 = path+rand_file_name('.fits')
    name3 = path+rand_file_name('.fits')

    fits.writeto(name1,im1)
    fits.writeto(name2,im2)
    exe = path+'CPP/sprite/bin/pxi_correl'
    subprocess.call([exe, name1, name2, name3])
    im3 = fits.getdata(name3)
    os.remove(name1)
    os.remove(name2)
    os.remove(name3)
    return im3


def rot180(im1):

    n1 = im1.shape[0]
    n2 = im1.shape[1]
    out = zeros((n1,n2))
    for i in range(0,n1):
        for j in range(0,n2):
            out[i,j] = im1[n1-1-i,n2-1-j]

    return out

def rot180_stack(cube):
    shap = cube.shape
    cube_out = copy(cube)
    for i in range(0,shap[2]):
        cube_out[:,:,i] = rot180(cube[:,:,i])

    return cube_out

def rot90_stack(cube):
    shap = cube.shape
    cube_out = copy(cube)
    for i in range(0,shap[2]):
        cube_out[:,:,i] = rot90(cube[:,:,i],2)

    return cube_out


def dot_check(im1,im2):
    from numpy import *
    n1 = im1.shape[0]
    n2 = im1.shape[1]
    a = 0
    for i in range(0,n1):
        for j in range(0,n2):
            a = a+im1[i,j]*im2[i,j]
    return a

"""def dot_conv_check(im1,im2,im3):"""


def correl_comp(coeff_data_pca,w): # The data are assume to be centered, the variance sorted in ascending order and columns represent the variables
    from numpy import *
    import numpy.ma as npma

    siz = coeff_data_pca.shape
    mat = ones(siz[0])
    wmat = npma.outerproduct(mat,w)
    coeff_data_pca_norm = coeff_data_pca/sqrt(wmat)
    corr_mat = (transpose(coeff_data_pca_norm[:,0:siz[1]-2])).dot(coeff_data_pca_norm[:,0:siz[1]-2])/siz[0] # The higher component is excluded
    return corr_mat


def cumul(vect):
    from numpy import *
    l = vect.size
    vect_cumul = zeros(l)
    for i in range(0,l):
        for k in range(0,i+1):
            vect_cumul[i] = vect_cumul[i] + vect[k]
    return vect_cumul

def sym_histo(x,bin_w):
    from numpy import *
    hist,edges = histogram(x, bins=bin_w)
    max_val_ind = where(hist==max(hist))
    ind = max_val_ind[0][0]
    max_val = double(edges[ind]+edges[ind+1])/2 # We assume it's positive
    small_val_ind = where(x<=max_val)
    l = array(small_val_ind).size
    new_set = zeros(2*l)
    L = x.size
    j=0
    for i in range(0,L):
        if x[i]<=max_val:
            new_set[j]=x[i]
            new_set[j+1]=x[i]+2*(max_val-x[i])
            j=j+2

    hist_sim,edges_new_set = histogram(new_set, bins=bin_w)
    return hist_sim,edges_new_set,new_set,max_val

def rand_file_name(ext):
    current_time = datetime.datetime.now().time()
    return 'file'+str(time.clock())+ext


def dictionary_learning(data,data_init,sparsity,nb_iter,dictionary_size,dict_file,coeff_file):
    opt = '-v -S '+str(sparsity)+' -I '+str(nb_iter)
    opt0 = '-v -S '+str(sparsity)
    subprocess.call(['dl_dl1d', opt, data_init, data, dict_file])
    subprocess.call(['dl_omp', opt0, dict_file, data, coeff_file])
    return 0

def tree_approx_test(data,tree,dictionary,tol,cells_means,list_split,rel_en=True): # Data are arranged in lines
    from numpy import *
    a=0
    err_tot = 0
    tree_siz = shape(tree)
    new_list_split = copy(list_split)
    i=0
    for i in range(0,tree_siz[0]):
        if list_split[i][0]==1:
            siz = size(tree[i])
            datai = data[array(tree[i]),:] - ones((siz,1))*cells_means[i,:]
            dict_i = copy(dictionary[:,:,i])
            err_i,err_pos = proj_err(datai,dict_i)
            thresh = tol*(datai**2).sum()
            if rel_en is not True:
                thresh = tol*siz
            if err_i <= thresh:
                new_list_split[i][0]=0
            else:
                a=1
    return a,new_list_split

def tree_approx_test_2(data,tree,dictionary,tol_pix,tol_pos,cells_means,list_split): # Data are arranged in lines
    from numpy import *
    a=0
    err_tot = 0
    tree_siz = shape(tree)
    new_list_split = copy(list_split)
    i=0
    for i in range(0,tree_siz[0]):
        if list_split[i][0]==1:
            siz = size(tree[i])
            datai = data[array(tree[i]),:] - ones((siz,1))*cells_means[i,:]
            dict_i = copy(dictionary[:,:,i])
            err_i_pix,err_i_pos = proj_err(datai,dict_i)
            print err_i_pix,tol_pix*(datai[:,:-2]**2).sum(),sqrt(err_i_pos),tol_pos
            if err_i_pix <= tol_pix*(datai[:,:-2]**2).sum() and sqrt(err_i_pos) <= tol_pos:
                new_list_split[i][0]=0
            else:
                a=1
    return a,new_list_split


def tree_PCA(data,tree,sparsity_lev,list_split,old_dictionary=None,old_cells_means=None):
    from numpy import *
    import copy
    tree_size = shape(tree)
    data_size = data.shape
    dictionary = zeros((data_size[1],sparsity_lev,tree_size[0]))
    cells_means = zeros((tree_size[0],data_size[1]))
    coeff_tree = copy.deepcopy(tree)
    i=0
    for i in range(0,tree_size[0]):
        #print tree[i]
        if list_split[i][0]==1:
            datai = data[array(tree[i]),:]
            wi,vi,coeffi,mean_datai = PCA(datai,sparsity_lev+1,mean_sub=1)
            dictionary[:,:,i]=vi[:,1:]
            siz = shape(datai)
            cells_means[i,:]= mean_datai
            l = size(tree[i])
            k=0
            for k in range(0,l):
                coeff_tree[i][k] = coeffi[k,0]
        else:
            dictionary[:,:,i] = old_dictionary[:,:,list_split[i][1]]
            cells_means[i,:] = old_cells_means[list_split[i][1],:]
    return dictionary,cells_means,coeff_tree

def tree_SVD(data,tree,sparsity_lev,list_split,old_dictionary=None,old_cells_means=None,coeff_comp=None):
    from numpy import *
    import copy
    tree_size = shape(tree)
    data_size = data.shape
    dictionary = zeros((data_size[1],sparsity_lev,tree_size[0]))
    cells_means = zeros((tree_size[0],data_size[1]))
    coeff_tree = copy.deepcopy(tree)
    i=0
    for i in range(0,tree_size[0]):
        #print tree[i]
        if list_split[i][0]==1:
            datai = data[array(tree[i]),:]
            wi,vi,coeffi,mean_datai = SVD_interf(datai,sparsity_lev+1,mean_sub=1,coeff_comp=coeff_comp)
            dictionary[:,:,i]=vi[:,0:-1]
            siz = shape(datai)
            cells_means[i,:]= mean_datai
            l = size(tree[i])
            k=0
            if coeff_comp is not None :
                for k in range(0,l):
                    coeff_tree[i][k] = coeffi[k,0]
        else:
            dictionary[:,:,i] = old_dictionary[:,:,list_split[i][1]]
            cells_means[i,:] = old_cells_means[list_split[i][1],:]
    return dictionary,cells_means,coeff_tree

def tree_PCA_pos(data,tree,sparsity_lev,list_split,old_dictionary=None,old_cells_means=None):
    from numpy import *
    import copy
    tree_size = shape(tree)
    data_size = data.shape
    dictionary = zeros((data_size[1],sparsity_lev,tree_size[0]))
    cells_means = zeros((tree_size[0],data_size[1]))
    i=0
    for i in range(0,tree_size[0]):
        #print tree[i]
        if list_split[i][0]==1:
            datai = data[array(tree[i]),:]
            wi,vi,coeffi,mean_datai = PCA(datai,sparsity_lev,mean_sub=1)
            dictionary[:,:,i]=vi
            siz = shape(datai)
            cells_means[i,:]= mean_datai
        else:
            dictionary[:,:,i] = old_dictionary[:,:,list_split[i][1]]
            cells_means[i,:] = old_cells_means[list_split[i][1],:]
    return dictionary,cells_means

def tree_svd_pos(data,tree,sparsity_lev,list_split,old_dictionary=None,old_cells_means=None,coeff_comp=None):
    from numpy import *
    import copy
    tree_size = shape(tree)
    data_size = data.shape
    dictionary = zeros((data_size[1],sparsity_lev,tree_size[0]))
    cells_means = zeros((tree_size[0],data_size[1]))
    i=0
    for i in range(0,tree_size[0]):
        #print tree[i]
        if list_split[i][0]==1:
            datai = data[array(tree[i]),:]
            wi,vi,coeffi,mean_datai = SVD_interf(datai,sparsity_lev,mean_sub=1,coeff_comp=coeff_comp)
            dictionary[:,: vi.shape[1],i]=vi
            siz = shape(datai)
            cells_means[i,:]= mean_datai
        else:
            dictionary[:,:,i] = old_dictionary[:,:,list_split[i][1]]
            cells_means[i,:] = old_cells_means[list_split[i][1],:]
    return dictionary,cells_means


def tree_split(tree,coeff_tree,list_split,sparsity_level):
    from numpy import *
    new_tree  =list([])
    new_list_split = list([])
    tree_size = shape(tree)
    i=0

    for i in range(0,tree_size[0]):
        l = size(tree[i])
        if list_split[i][0]==1 and l > sparsity_level :
            pos_ind = []
            neg_ind = []
            for k in range(0,l):
                if coeff_tree[i][k] >=0:
                    pos_ind.append(tree[i][k])
                if coeff_tree[i][k] <=0:
                    neg_ind.append(tree[i][k])
            new_tree.append(neg_ind)
            new_tree.append(pos_ind)
            new_list_split.append([1,-1])
            new_list_split.append([1,-1])
        else:
            new_tree.append(tree[i])
            new_list_split.append([0,i]) # we save the index in the old tree
    return new_tree,new_list_split


def tree_split_pos(tree,list_split,sparsity_level,data):
    from numpy import *
    new_tree  =list([])
    new_list_split = list([])
    tree_size = shape(tree)
    i=0
    shape_dat = shape(data)
    for i in range(0,tree_size[0]):
        l = size(tree[i])
        if list_split[i][0]==1 and l > sparsity_level :
            pos_ind = []
            neg_ind = []
            coord_i = data[tree[i],shape_dat[1]-2:shape_dat[1]]
            stdx = std(coord_i[:,0])
            stdy = std(coord_i[:,1])
            id=0
            if stdy > stdx:
                id=1
            mean_id = mean(coord_i[:,id])
            for k in range(0,l):
                if data[tree[i][k],shape_dat[1]-2+id] >=mean_id:
                    pos_ind.append(tree[i][k])
                if data[tree[i][k],shape_dat[1]-2+id]<=mean_id:
                    neg_ind.append(tree[i][k])
            new_tree.append(neg_ind)
            new_tree.append(pos_ind)
            new_list_split.append([1,-1])
            new_list_split.append([1,-1])
        else:
            new_tree.append(tree[i])
            new_list_split.append([0,i]) # we save the index in the old tree
    return new_tree,new_list_split




def proj_err(data,dictionary): # Data in lines, dictionary vector in columns
    from numpy import *
    coeff = data.dot(dictionary)
    approx = coeff.dot(transpose(dictionary))
    err = ((data-approx)**2).sum(axis=0)
    err_pix = sum(err[:-2])
    err_pos = sum(err)-err_pix
    return err_pix,err_pos

def geometric_dictionary_learning(data,sparsity_lev,tol): # Tol = pourcentage in terms of quadratic error
    from numpy import *
    #tol_abs = tol*(data**2).sum()
    split_en=1
    tree_init = list([])
    siz_data = data.shape
    tree_init.append(range(0,siz_data[0]))
    tree = tree_init
    list_split = []
    list_split.append([1,-1])
    dictionary,cells_means,coeff_tree=tree_PCA(data,tree_init,sparsity_lev,list_split)
    nb_sc=0
    while (split_en == 1):
        print 'tree splitting'
        tree,list_split=tree_split(tree,coeff_tree,list_split,sparsity_lev)
        print 'iteratice PCA'
        dictionary,cells_means,coeff_tree=tree_PCA(data,tree,sparsity_lev,list_split,dictionary,cells_means)
        print 'projection error'
        split_en,list_split = tree_approx_test(data,tree,dictionary,tol,cells_means,list_split)
        nb_sc=nb_sc+1
        print 'scale ',nb_sc
        print 'number of cells:', shape(tree)

    return tree,dictionary,cells_means,nb_sc

def geometric_dictionary_learning_field(data,sparsity_lev,tol): # Tol = pourcentage in terms of quadratic error
    from numpy import *
    #tol_abs = tol*(data**2).sum()
    split_en=1
    tree_init = list([])
    siz_data = data.shape
    tree_init.append(range(0,siz_data[0]))
    tree = tree_init
    list_split = []
    list_split.append([1,-1])
    dictionary,cells_means=tree_PCA_pos(data,tree_init,sparsity_lev,list_split)
    nb_sc=0
    while (split_en == 1):
        print 'tree splitting'
        tree,list_split=tree_split_pos(tree,list_split,sparsity_lev,data)
        print tree[0]
        print 'iteratice PCA'
        dictionary,cells_means=tree_PCA_pos(data,tree,sparsity_lev,list_split,dictionary,cells_means)
        print 'projection error'
        split_en,list_split = tree_approx_test(data,tree,dictionary,tol,cells_means,list_split)
        nb_sc=nb_sc+1
        print 'scale ',nb_sc
        print 'number of cells:', shape(tree)

    return tree,dictionary,cells_means,nb_sc

def geometric_dictionary_learning_field_2(data,sparsity_lev,tol1=0.00001,tol2=0.1/3600): # Tol1 = pourcentage in terms of quadratic error on the pixels intensities : tol2 error on the position in deg (euclid pixel size)
    from numpy import *
    #tol_abs = tol*(data**2).sum()
    split_en=1
    tree_init = list([])
    siz_data = data.shape
    tree_init.append(range(0,siz_data[0]))
    tree = tree_init
    list_split = []
    list_split.append([1,-1])
    #dictionary,cells_means=tree_PCA_pos(data,tree_init,sparsity_lev,list_split)
    dictionary,cells_means=tree_svd_pos(data,tree_init,sparsity_lev,list_split)
    nb_sc=0
    while (split_en == 1):
        print 'tree splitting'
        tree,list_split=tree_split_pos(tree,list_split,sparsity_lev,data)
        #print tree
        print 'iteratice PCA'
        dictionary,cells_means=tree_svd_pos(data,tree,sparsity_lev,list_split,dictionary,cells_means)
        print 'projection error'
        split_en,list_split = tree_approx_test_2(data,tree,dictionary,tol1,tol2,cells_means,list_split)
        nb_sc=nb_sc+1
        print 'scale ',nb_sc
        print 'number of cells:', shape(tree)

    return tree,dictionary,cells_means,nb_sc


def geometric_dictionary_denoising(galaxy,dictionary,cells_means):
    from numpy import *
    siz = galaxy.shape
    x = galaxy.reshape(1,siz[0]*siz[1])
    dist = zeros((cells_means.shape[0],1))
    for i in range(0,cells_means.shape[0]):
        dist[i] = (x-cells_means[i,:]).dot(transpose(x-cells_means[i,:]))
    ind = argsort(squeeze(dist))
    i0=0
    coeffx = (x-cells_means[ind[i0],:]).dot(dictionary[:,:,ind[i0]])
    xapprox = coeffx.dot(transpose(dictionary[:,:,ind[i0]])) + cells_means[ind[i0],:]
    galaxy_denoised = xapprox.reshape(siz[0],siz[1])
    return galaxy_denoised



def affine_surf_eval(O,n1,n2,x,y): # {O, n1, n2} is an affine plan in R^n; this function computes a vector of this plan so that his last compnents are equal to x and y respectively
    from numpy import *
    import numpy.linalg as npl
    u = zeros((2,1))
    M = zeros((2,2))
    n = O.shape[0]
    M[:,0] = n1[n-2:n]
    M[:,1] = n2[n-2:n]
    u[0]=x-O[n-2]
    u[1]=y-O[n-1]
    sol = npl.lstsq(M,u)
    coeff = sol[0]
    output = O+coeff[0]*n1+coeff[1]*n2
    return output



def psf_gauss(psf,fwhmx):
    from numpy import *
    ell = mk_ellipticity(psf,1e10,niter_cent=1)
    r = sqrt(ell[0,0]**2+ell[0,1]**2)
    cos_om = ell[0,0]/r
    om = arccos(cos_om)
    theta = om/2
    fwhmy = sqrt((1-r)/(1+r))*fwhmx
    sigx = fwhmx/(2*sqrt(2*log(2)))
    sigy = fwhmy/(2*sqrt(2*log(2)))
    cent,wc = compute_centroid(psf,1e10,nb_iter=1)
    rad = (psf.shape[0]-1)/2
    ell1 = (sigx**2 - sigy**2)/(sigx**2 + sigy**2)*cos(2*theta)
    ell2 = (sigx**2 - sigy**2)/(sigx**2 + sigy**2)*sin(2*theta)
    gauss_map = gauss_gen_2(sigx,sigy,cent,theta,rad)
    gauss_map = gauss_map*psf.max()
    ell_fit = mk_ellipticity(gauss_map,1e10,niter_cent=1)
    ell_error = ell - ell_fit
    return gauss_map,ell,ell_error,theta


def psf_gauss_field(psf_field,fwhmx):
    from numpy import *
    n1 = psf_field.shape[0]
    n2 = psf_field.shape[1]
    nb_psf = psf_field.shape[2]
    ell_theta = zeros((nb_psf,3))
    gauss_field = zeros((n1,n2,nb_psf))
    i=0
    for i in range(0,nb_psf):
        gauss_map,ell,ell_error,theta = psf_gauss(psf_field[:,:,i],fwhmx)
        gauss_field[:,:,i]=gauss_map
        ell_theta[i,0] = ell[0,0]
        ell_theta[i,1] = ell[0,1]
        ell_theta[i,2] = theta
    return gauss_field,ell_theta


def rand_diff_integ(a,b,n): # This function returns n different integers sorted in ascending order
    from numpy import *
    out = array(range(0,n))
    i=0
    for i in range(0,n):
        bool=0
        if (i >0):
            while (bool==0):
                ind = random.randint(a, b)
                temp = abs(out[0:i]-ind)
                if (temp.min()>0):
                    out[i]=ind
                    bool=1
        else:
            ind = random.randint(a, b)
            out[i]=ind
    out = sort(out)
    return out.astype(int)

def rand_diff_integ_pair(a,b,n):
    from numpy import zeros,ones
    ind_out = zeros((n,2))
    ind_out = ind_out.astype(int)
    ind_arr = arange(a,b)

    ind0 = rand_diff_integ(0,b-a,n)

    ind_out[:,0] = ind_arr[ind0]
    ind_arr[ind0] = -1
    ind_arr2 = where(ind_arr>=0)[0]
    ind1 = rand_diff_integ(0,b-a-n,n)
    ind_out[:,1] = ind_arr2[ind1]

    return ind_out

def plot_circles(target,points,nb_samp=100,p=2):
    from pylab import figure, show, plot

    angles = 2*pi*arange(0,nb_samp)/(nb_samp-1)
    figure()
    r = norm(target.reshape((2,1)).dot(ones((1,points.shape[1])))-points,axis=0)
        #for i in range(0,points.shape[1]):

    plot(r.max()*cos(angles)+target[0],r.max()*sin(angles)+target[1],'r--')
    plot(r.min()*cos(angles)+target[0],r.min()*sin(angles)+target[1],'r--')
    plot(median(r)*cos(angles)+target[0],median(r)*sin(angles)+target[1],'r--')

    return None


def rand_in(shap,slice_x,slice_y,mean=0,sig=1):
    rand_out = zeros((shap[0],shap[1]))

    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            rand_out[i,j] = sig*random.randn(1)+mean

    return rand_out

def cube_sampling(cube,n): # Extracts randomly n slice from the input data cube

    n1 = cube.shape[2]
    n2 = cube.shape[0]
    n3 = cube.shape[1]

    ind_rm = rand_diff_integ(0,n1-1,n1-n)
    a = ones((n1,))
    a[ind_rm] = 0
    i = where(a==1)
    ind_kpt = i[0]
    cube_out = cube[:,:,ind_kpt]

    return cube_out,ind_rm,ind_kpt

def geo_data_sampling(cube,frac_obs,frac_valid,pos_data):
    from numpy import int
    shap = cube.shape
    nb_data  = int(round(shap[2]*frac_obs))
    cube_1,ind_rm,ind_kpt = cube_sampling(cube,shap[2]-nb_data)
    pos_1 = pos_data[ind_kpt,:]

    nb_valid = int(round(frac_valid*nb_data))
    print nb_valid,nb_data,cube_1.shape[2]
    cube_obs,ind_valid,ind_kpt = cube_sampling(cube_1,nb_valid)
    cube_valid = cube_1[:,:,ind_valid]
    pos_obs = pos_1[ind_kpt,:]
    pos_valid = pos_1[ind_valid,:]

    return cube_obs,pos_obs,cube_valid,pos_valid

def geo_data_samplingm(cube,frac_obs,frac_valid,pos_data):
    nb_r = len(frac_obs)
    list_cube_obs = list()
    list_pos_obs = list()
    list_cube_valid = list()
    list_pos_valid = list()

    for i in range(0,nb_r):
        cube_obs,pos_obs,cube_valid,pos_valid  = geo_data_sampling(cube,frac_obs[i],frac_valid,pos_data)
        list_cube_obs.append(cube_obs)
        list_pos_obs.append(pos_obs)
        list_cube_valid.append(cube_valid)
        list_pos_valid.append(pos_valid)

    return list_cube_obs,list_pos_obs,list_cube_valid,list_pos_valid

def cube_sampling_interior(cube,n,pos):
    from numpy import *
    n1 = cube.shape[2]
    n2 = cube.shape[0]
    n3 = cube.shape[1]
    hull = ConvexHull(pos)
    hull_ind = hull.vertices
    ind_ref = range(0,n1)
    c = in1d(ind_ref,hull_ind)
    ind_interior = where(c==False)
    ind_rm = rand_diff_integ(0,len(ind_interior[0])-1,n)
    ind_rm = ind_interior[0][ind_rm]
    ind_kpt = array(range(0,n1-n))
    cube_out = zeros((n2,n3,n1-n))
    count=0
    count2=0
    i =0
    #while (count < n1-n):
    for i in range(0,n1):
        if (i != ind_rm[count2]):
            cube_out[:,:,count] = cube[:,:,i]
            ind_kpt[count]=i
            count = count+1
        if (i==ind_rm[count2] and count2<n-1):
            count2=count2+1
    return cube_out,ind_rm,ind_kpt,hull_ind


def ind_cv(mat_ind,list_ind): # Remap the integers in mat_ind into [| 0,size(list_ind) |]
    l = size(list_ind)
    mat_ind_cv = copy(mat_ind)
    for k in range(0,l):
        i,j = where(mat_ind==list_ind[k])
        mat_ind_cv[i,j] = k
    return mat_ind_cv

def aa_shaping(cube):
    from numpy import *
    n1 = cube.shape[0]
    n2 = cube.shape[1]
    nb_data = cube.shape[2]
    mat = zeros((nb_data,n1*n2))
    i = 0
    for i in range(0,nb_data):
        cube_i = cube[:,:,i]
        mat[i,:] = cube_i.reshape(1,n1*n2)
    return mat

def vid(cube,filename=None):
    fig = plt.figure()
    shap = cube.shape
    back = plt.imshow(zeros(shap[0:2]),cmap=plt.get_cmap('gist_stern'))

    def init():
        back.set_data(zeros(shap[0:2]))
        return back

    def animate(i):
        back.set_data(cube[:,:,i])
        return back

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=cube.shape[2], blit=False,interval=1)

    if filename is None:
        filename = rand_file_name('.mp4')

    anim.save(filename, fps=30)

    return anim

def vect(u):
    v = copy(u).reshape((size(u),1))
    return v

def gram_schmidt(U): # This function orthonormalizes the set of columns of U
    from numpy import *
    V = copy(U)
    V = V*0
    nb_vect = U.shape[1]
    i = 0
    V[:,0]=U[:,0]/sqrt((U[:,0]**2).sum())
    weight_mat = zeros((nb_vect,nb_vect))
    weight_mat[0,0] = 1.0/sqrt((U[:,0]**2).sum())
    for i in range(1,nb_vect):
        V[:,i]=U[:,i]
        j=0
        for j in range(0,i):
            V[:,i]=V[:,i]-(transpose(V[:,j]).dot(U[:,i]))*V[:,j]
        V[:,i]=V[:,i]/sqrt((V[:,i]**2).sum())
    return V

def random_orth_basis(n):
    U = random.randn(n,n)
    V = gram_schmidt(U)
    return V

def random_orth_basis_cat(n,nb_real=3):
    from numpy import zeros
    basis_out = zeros((n,n*nb_real))
    for i in range(0,nb_real):
        basis_out[:,i*n:(i+1)*n] = random_orth_basis(n)

    return basis_out

def adapted_positive_basis(F,centering_en=False):
    shap = F.shape
    nz = list()
    nb_nz = 0
    for i in range(0,shap[2]):
        thresh = threshold_perc(F[-1,:,i],perc=0.99)
        id = where((F[0,:,i]>thresh))
        nb_nz+=size(id)
        nz.append(id[0])
    data = zeros((shap[0],nb_nz))
    l=0
    for i in range(0,shap[2]):
        data[:,l:l+size(nz[i])] = F[:,nz[i],i]
        l+=size(nz[i])
    if centering_en:
        data -= data.mean(axis=1).reshape((shap[0],1)).dot(ones((1,nb_nz)))
    basis,s,v = svd(data,full_matrices=True)
    s=s/sqrt(sum(s**2))
    #s=s/s[0]
    basis = basis.dot(diag(s))

    #print "Basis: ",basis,"; weighting: ",s
    return basis,s

def sphere_vect(n):

    theta = pi*arange(1.0,n+1)/(2*n)
    phi = pi*arange(1.0,n+1)/(2*n)
    mat_vect = zeros((3,n**2))
    vect1 = cos(theta).reshape((n,1))
    vect2 = sin(theta).reshape((n,1))
    vect3 = cos(phi).reshape((n,1))
    vect4 = sin(phi).reshape((n,1))
    ones_vect = ones((n,1))
    mat_vect[0,:] = (vect3.dot(transpose(vect1))).reshape((n**2,))
    mat_vect[1,:] = (vect3.dot(transpose(vect2))).reshape((n**2,))
    mat_vect[2,:] = (vect4.dot(transpose(ones_vect))).reshape((n**2,))

    return mat_vect

def sphere_vect_w(n,data,centering_en=True):
    mat_vect = sphere_vect(n)
    basis,s = adapted_positive_basis(data,centering_en=centering_en)
    weights = zeros((n**2,))
    for i in range(0,n**2):
        a = transpose(basis).dot(mat_vect[:,i].reshape((3,1)))
        weights[i] = norm(diag(s).dot(a),2)

    weights = weights/norm(weights,2)
    w_mat_vect = mat_vect.dot(diag(weights))
    return mat_vect


def circle_vect(n):
    from numpy import cos,sin
    theta = 2*pi*arange(0,n)/n
    mat_vect = zeros((2,n))
    mat_vect[0,:] = cos(theta)
    mat_vect[1,:] = sin(theta)

    return mat_vect


"""def random_rot(M,a=1):
    theta = 2*pi*numpy.random.rand(1)*a
    phi = pi*numpy.random.rand(1)*a
    M = zeros(())"""

def gram_schmidt_cube(V):
    shap = V.shape
    dict = zeros((shap[0]*shap[1],shap[2]))
    k=0
    for k in range(0,shap[2]):
        dict[:,k] = (V[:,:,k]).reshape((shap[0]*shap[1],))
    dict_ortho = gram_schmidt(dict)
    Vorth = copy(V)
    for k in range(0,shap[2]):
        Vorth[:,:,k] = (dict_ortho[:,k]).reshape((shap[0],shap[1]))
    return Vorth

def proj_vect(v,mat,ortho_en=True): # Compute the orthogonal projection of v onto mat column space
    mat_ortho = copy(mat)
    if ortho_en:
        mat_ortho = gram_schmidt(mat)
    u = mat_ortho.dot(transpose(mat_ortho).dot(v))
    if ortho_en:
        return u,mat_ortho
    else:
        return u

def proj_cube(im,cube,ortho=None):
    cube_proj = copy(cube)
    if ortho is not None:
        cube_proj = gram_schmidt_cube(cube)

    shap = cube.shape
    im_proj = im*0
    k = 0
    if size(shap)>1:
        for k in range(0,shap[2]):
            im_proj = im_proj+((im*cube_proj[:,:,k]).sum())*cube_proj[:,:,k]
    else:
        im_proj = im_proj+((im*cube_proj).sum())*cube_proj
    return im_proj

def shape_projector(im,cent_shape,shape_ortho_basis,mu,niter=50):
    from numpy import *
    mse = zeros((niter,))
    n1 = im.shape[0]
    n2 = im.shape[1]
    dim_shape = n1*n2*6
    U = zeros((dim_shape,))
    V = zeros((dim_shape,))
    V_old = copy(V)
    i=0
    t = 1
    told = t
    shap = im.shape
    im_shape_mat = mat_mm_ell(im)
    delta  = im_shape_mat-cent_shape
    delta_vec = delta.reshape((dim_shape,))
    proj_offset = delta_vec - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(delta_vec))
    for i in range(0,niter):
        temp_proj = U - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(U))
        mat_temp_proj = temp_proj.reshape(6,n1*n2)
        bck_proj_temp = transpose_mat_ell(mat_temp_proj,shap)
        mat_bck_proj_temp = mat_mm_ell(bck_proj_temp)
        mat_bck_proj_temp_vec = mat_bck_proj_temp.reshape((dim_shape,))
        grad = -proj_offset+(mat_bck_proj_temp_vec - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(mat_bck_proj_temp_vec)))
        mse[i] = (grad**2).sum()
        print mse[i]
        V = U - mu*grad
        t = (1+math.sqrt(4*told**2+1))/2
        lamb = 1+(told-1)/t
        told = t
        U = V_old + lamb*(V-V_old)
        V_old = copy(V)
    temp_proj = U - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(U))
    mat_temp_proj = temp_proj.reshape(6,n1*n2)
    bck_proj_temp = transpose_mat_ell(mat_temp_proj,shap)
    proj_im = im - bck_proj_temp

    mat_proj_im = mat_mm_ell(proj_im)
    delta  = mat_proj_im-cent_shape
    delta_vec = delta.reshape((dim_shape,))
    proj_offset = delta_vec - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(delta_vec))
    mat_proj_offset = proj_offset.reshape(6,n1,n2)
    return mse,proj_im,proj_offset,delta


def shape_projector_opt_step(im,cent_shape,back_cent_shape,shape_ortho_basis,mu,niter=50):
    from numpy import *
    mse = zeros((niter,))
    n1 = im.shape[0]
    n2 = im.shape[1]
    dim_shape = n1*n2*6
    U = zeros((dim_shape,))
    V = zeros((dim_shape,))
    V_old = copy(V)
    i=0
    t = 1
    told = t
    shap = im.shape
    im_shape_mat = mat_mm_ell(im)
    delta  = im_shape_mat-cent_shape
    im_delta = im - back_cent_shape
    delta_vec = delta.reshape((dim_shape,))
    proj_offset = delta_vec - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(delta_vec))
    step = -mu
    for i in range(0,niter):
        temp_proj = U - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(U))
        mat_temp_proj = temp_proj.reshape(6,n1*n2)
        bck_proj_temp = transpose_mat_ell(mat_temp_proj,shap)
        mat_bck_proj_temp = mat_mm_ell(bck_proj_temp)
        mat_bck_proj_temp_vec = mat_bck_proj_temp.reshape((dim_shape,))
        grad = -proj_offset+(mat_bck_proj_temp_vec - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(mat_bck_proj_temp_vec)))

        temp_proj_2 = grad - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(grad))
        mat_temp_proj_2 = temp_proj_2.reshape(6,n1*n2)
        bck_proj_temp_2 = transpose_mat_ell(mat_temp_proj_2,shap)
        a1 = ((im_delta - bck_proj_temp)**2).sum()
        a2 = ((im_delta - bck_proj_temp)*bck_proj_temp_2).sum()
        print 'step_size = ',step
        step = a1/a2
        mse[i] = (grad**2).sum()
        print 'err = ',mse[i]
        U = U+step*grad
    temp_proj = U - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(U))
    mat_temp_proj = temp_proj.reshape(6,n1*n2)
    bck_proj_temp = transpose_mat_ell(mat_temp_proj,shap)
    proj_im = im - bck_proj_temp

    mat_proj_im = mat_mm_ell(proj_im)
    delta  = mat_proj_im-cent_shape
    delta_vec = delta.reshape((dim_shape,))
    proj_offset = delta_vec - shape_ortho_basis.dot(transpose(shape_ortho_basis).dot(delta_vec))
    mat_proj_offset = proj_offset.reshape(6,n1,n2)
    return mse,proj_im,proj_offset,delta


def lanczos(U,n=10,n2=None):
    if n2 is None:
        n2 = n
    siz = size(U)
    H = None
    if (siz == 2):
        U_in = copy(U)
        if len(U.shape)==1:
            U_in = zeros((1,2))
            U_in[0,0]=U[0]
            U_in[0,1]=U[1]
        H = zeros((2*n+1,2*n2+1))
        if (U_in[0,0] == 0) and (U_in[0,1] == 0):
            H[n,n2] = 1
        else:
            i=0
            j=0
            for i in range(0,2*n+1):
                for j in range(0,2*n2+1):
                    H[i,j] = sinc(U_in[0,0]-(i-n))*sinc((U_in[0,0]-(i-n))/n)*sinc(U_in[0,1]-(j-n))*sinc((U_in[0,1]-(j-n))/n)

    else :
        H = zeros((2*n+1,))
        for i in range(0,2*n):
            H[i] = sinc(pi*(U-(i-n)))*sinc(pi*(U-(i-n))/n)
    return H

def shift(im,U,n=5):
    shap = im.shape
    M = None
    if n is None:
        M = lanczos(U,n=int((shap[0]-1)/2),n2=int((shap[1]-1)/2))
    else:
        M = lanczos(U,n=n,n2=n)
    im_out = scisig.fftconvolve(im,M,mode='same')

    return im_out

def shift_stack(im_stack,U,n=None):
    im_out = copy(im_stack)
    nb_psf = im_stack.shape[2]
    for i in range(0,nb_psf):
        im_out[:,:,i] = shift(im_stack[:,:,i],U[i,:],n=n)
    return im_out

def stack_norm(im_stack,norm=2):
    im_out = copy(im_stack)
    nb_psf = im_stack.shape[2]
    for i in range(0,nb_psf):
        if norm==2:
            im_out[:,:,i] = im_out[:,:,i]/sqrt((im_out[:,:,i]**2).sum())
        elif norm==1:
            im_out[:,:,i] = im_out[:,:,i]/(im_out[:,:,i].sum())
    return im_out

def stack_regist(im_stack,n=None):
    U = int_grid_shift(im_stack)
    im_reg = shift_stack(im_stack,-U,n=n)
    return im_reg

def corr_shift_est(im1,im2,upfact=1):


    im_1_in = copy(im1)
    im_2_in = copy(im2)
    if upfact>1:
        im_1_in = zoom(im1, upfact,order=3, mode='constant', cval=0.0)
        im_2_in = zoom(im2, upfact,order=3, mode='constant', cval=0.0)
    shap = im_1_in.shape
    di = shap[0]%2
    dj = shap[1]%2
    pos_ref = [shap[0]/2 +di-1,shap[1]/2 +dj-1]
    cross_corr = scisig.correlate2d(im_1_in, im_2_in, mode='same', boundary='fill', fillvalue=0)
    i,j = where(cross_corr==cross_corr.max())

    shifts = double([pos_ref[0]-i[0],pos_ref[1]-j[0]])/upfact

    return cross_corr,shifts

def corr_shift_est_stack(im_stack,upfact=1):
    shap = im_stack.shape
    shifts_est = zeros((shap[2],2))
    for i in range(0,shap[2]):
        cross_corr,shifts = corr_shift_est(im_stack[:,:,0],im_stack[:,:,i],upfact=1)
        shifts_est[i,:]=array(shifts)
    return shifts_est


def shift_transpose(im,U,n=10):
    shap = im.shape
    M = lanczos(U,n=10)
    im_out = scisig.fftconvolve(im,rot90(M,2),mode='same')
    return im_out

def lanczos_rot(img,theta,lanczos_rad=10,marg=0):
    print 'theta = ',theta
    img_rot = 0*padding(img,marg)
    shap = img.shape
    diag_l = sqrt((shap[0]+2*marg)**2+(shap[1]+2*marg)**2)
    padding_band = round((diag_l - min(shap[0],shap[1]))/2+lanczos_rad)
    img_padd = padding(img,padding_band)
    i=0
    j=0
    Mrot = array([[cos(theta),sin(theta)],[-sin(theta),cos(theta)]])

    O1 = zeros((2,1))
    O1[0,0]=(shap[0]-1)/2 + marg
    O1[1,0]=(shap[1]-1)/2 + marg
    O2 = zeros((2,1))
    O2[0,0]=(shap[0]-1)/2+padding_band
    O2[1,0]=(shap[1]-1)/2+padding_band

    Y = zeros((2,1))
    for i in range(0,shap[0]+2*marg):
        for j in range(0,shap[1]+2*marg):
            Y[0,0] = i
            Y[1,0] = j
            X1 = Mrot.dot(Y-O1)
            X2 = X1+O2
            X2_int = floor(X2)
            delta = X2-X2_int

            mask_lanc = lanczos(delta)
            l=0
            m=0
            valij=0
            for l in range(0,2*lanczos_rad):
                for m in range(0,2*lanczos_rad):
                    valij = valij+mask_lanc[l,m]*img_padd[int(X2_int[0,0])-lanczos_rad+l,int(X2_int[1,0])-lanczos_rad+m]
            img_rot[i,j]=valij
    return img_rot

def transpose_lanczos_rot(img,theta,lanczos_rad=10):
    im_out = 0*img
    shap = img.shape
    i=0
    j=0
    O1 = zeros((2,1))
    O1[0,0]=(shap[0]-1)/2
    O1[1,0]=(shap[1]-1)/2
    Mrot1 = array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    Mrot2 = array([[cos(theta),sin(theta)],[-sin(theta),cos(theta)]])
    diag_lanc_rad = sqrt(2)*(2*lanczos_rad+1)/2
    Y = zeros((2,1))
    Y1 = zeros((2,1))
    Lc = ones((2,1))*lanczos_rad
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            Y[0,0] = i
            Y[1,0] = j
            X1  = Mrot1.dot(Y-O1)+O1
            if (floor(X1[0])<shap[0]) and (floor(X1[0])>-1) and (floor(X1[1])<shap[1]) and (floor(X1[1])>-1):
                i1=0
                j1=0
                valij = 0
                for i1 in range(max(0,floor(X1[0]-diag_lanc_rad)),min(shap[0]-1,floor(X1[0]+diag_lanc_rad))):
                    for j1 in range(max(0,floor(X1[1]-diag_lanc_rad)),min(shap[1]-1,floor(X1[1]+diag_lanc_rad))):
                        Y1[0,0]=i1
                        Y1[1,0]=j1
                        Y2 = Mrot2.dot(Y1-O1)+O1
                        Y2_int = floor(Y2)
                        delta = Y2-Y2_int
                        mask_lanc = lanczos(delta)
                        pos_mask = Y-Y2_int+Lc
                        if (pos_mask[0,0]<2*lanczos_rad+1) and (pos_mask[0,0]>-1) and (pos_mask[1,0]<2*lanczos_rad+1) and (pos_mask[1,0]>-1):
                            valij = valij + img[i1,j1]*mask_lanc[pos_mask[0,0],pos_mask[1,0]]
                im_out[i,j]=valij
    return im_out




def rotate_positions_field(map,theta):
    shap = map.shape
    O = zeros((2,1))
    Y = zeros((2,1))
    O[0,0]=map[(shap[0]-1)/2,(shap[1]-1)/2,0]
    O[1,0]=map[(shap[0]-1)/2,(shap[1]-1)/2,1]
    rot_mat = array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    rot_map = 0*map
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            Y[0,0] = map[i,j,0]
            Y[1,0] = map[i,j,1]
            Yrot = rot_mat.dot(Y-O)+O
            rot_map[i,j,0] = Yrot[0,0]
            rot_map[i,j,1] = Yrot[1,0]

    return rot_map

def angle_est(data):

    theta = 0
    if len(data.shape)==2:
        param,fit=gaussfitter.gaussfit(data,returnfitimage=True)
        theta = param[6]*pi/180
        if param[5]>param[4]:
            theta = -theta
        else:
            theta = pi/2-theta
    elif len(data.shape)==3:
        theta = zeros((data.shape[2],))
        i=0
        for i in range(0,data.shape[2]):
            param,fit=gaussfitter.gaussfit(data[:,:,i],returnfitimage=True)
            if param[5]>param[4]:
                theta[i] = -param[6]*pi/180
            else:
                theta[i] = pi/2-param[6]*pi/180

    return theta

def rot_registration(data,theta):
    data_rot,cent = opencv_rot_interf(data,theta)
    #lanczos_rot(data,-theta)
    return data_rot


def rot_registration_cube(data,theta,ib=None,ie=None):
    shap = data.shape
    k=0
    data_rot = zeros((shap[0],shap[1],ie-ib+1))

    if ib is None:
        ib = 0
    if ie is None:
        ie = shap[2]-1
    for k in range(ib,ie+1):
        print k+1-ib,'/',ie-ib+1
        data_rot_k=rot_registration(data[:,:,k],theta[k-ib])
        data_rot[:,:,k-ib]=data_rot_k
    return data_rot

def mp_rand_rot(im_cube,nb_proc=5):
    shap = im_cube.shape
    nb_gal = shap[2]
    cube_rot = copy(im_cube)
    theta = double(rand_diff_integ(1,shap[2]*10,shap[2]))*(360.0/(shap[2]*10))
    print theta.max()
    slice = int(nb_gal/nb_proc)

    if __name__ == '__main__':
        q = Queue()
        jobs = []
        for i in range(0,nb_proc):
            ib = i*slice
            ie = ib+slice-1
            p = Process(target=mp_rot_cube, args=(gal_test,theta[ib,ie+1],ib,ie,q))
            p.start()
            jobs.append(p)
        for i in range(0,nb_proc):
            a = q.get()
            cube_rot[:,:,a[2]:a[3]+1] = a[0]

    return cube_rot,theta




def mp_rot_cube(data,theta,ib,ie,q):
    data_rot = rot_registration_cube(data,theta,ib=ib,ie=ie)
    q.put([data_rot,theta,ib,ie])
    print 'Data ',ib,' --> ',ie;' processed'

def mp_rot_registration_cube(data,ib,ie,q):
    theta = angle_est(data[:,:,ib:ie+1])
    data_rot = rot_registration_cube(data,-theta*180/pi,ib=ib,ie=ie)
    q.put([data_rot,theta,ib,ie])
    print 'Data ',ib,' --> ',ie;' processed'



def cube_to_mat(cube):
    shap = cube.shape
    mat = zeros((shap[2],shap[0]*shap[1]))
    k=0
    for k in range(0,shap[2]):
        mat[k,:] = cube[:,:,k].reshape(1,shap[0]*shap[1])
    return mat


def mat_to_cube(mat,n1,n2):
    shap = mat.shape
    cube = zeros((n1,n2,shap[0]))
    k=0
    for k in range(0,shap[0]):
        cube[:,:,k] = mat[k,:].reshape(n1,n2)
    return cube

def cube_to_map(cube):
    shap = cube.shape
    map = zeros((shap[0],shap[1]*shap[2]))
    for k in range(0,shap[2]):
        map[:,k*shap[1]:(k+1)*shap[1]] = cube[:,:,k]
    return map

def cube_to_map2(cube,n1,n2):
    shap = cube.shape
    map = zeros((shap[0]*n1,shap[1]*(shap[2]/n1)))
    for i in range(0,n1):
        for j in range(0,n2):
            map[i*shap[0]:(i+1)*shap[0],j*shap[1]:(j+1)*shap[1]] = cube[:,:,i+j*n1]
    return map


def map_to_cube(map,n1,n2):
    shap = map.shape
    ax1 = shap[0]/n1
    ax2 = shap[1]/n2
    nb_slice = ax1*ax2
    cube = zeros((n1,n2,nb_slice))
    for i in range(0,ax1):
        for j in range(0,ax2):
            cube[:,:,i+j*ax1] = map[n1*i:n1*(i+1),n2*j:n2*(j+1)]
    return cube


def cube_gram_mat(cube):
    shap = cube.shape
    mat = zeros((shap[2],shap[2]))
    for i in range(0,shap[2]):
        for j in range(i+1,shap[2]):
            mat[i,j] = sum(cube[:,:,i]*cube[:,:,j])
    mat = mat + transpose(mat)
    for i in range(0,shap[2]):
        mat[i,j] = sum(cube[:,:,i]**2)
    return mat

def surf_interface(im,hess_thresh,max_iter = 1000):

    des=None
    a1=False
    a2=False
    a3=False
    im_in = zeros((im.shape[0],im.shape[1],1), uint8)
    im_in[:,:,0] = 255*(im-im.min())/(im.max()-im.min())
    thresh_up=hess_thresh
    thresh_down=hess_thresh
    des_out = zeros((1,128))

    while (a1==False):
        surf = cv2.SURF(thresh_up)
        kp, des_temp = surf.detectAndCompute(im_in,None)
        if (len(kp) ==1):
            a3=True
            a1=True
            des = des_temp
        else:
            if(len(kp)==0):
                a1 = True
            else:
                thresh_up = thresh_up*2
    count = 0
    while (a2==False and count<max_iter):
        surf = cv2.SURF(thresh_down)
        kp, des_temp = surf.detectAndCompute(im_in,None)
        if (len(kp) ==1):
            a3=True
            a2=True
            des = des_temp
        else:
            if(len(kp)>0):
                a2 = True
            else:
                thresh_down = thresh_down/2
    #count +=1
    if count==max_iter:
        print 'max reached'
    a3 = True
    count = 0
    while (a3==False and count<max_iter):
        thresh =  (thresh_up+thresh_down)/2
        surf = cv2.SURF(thresh)
        kp, des_temp = surf.detectAndCompute(im_in,None)
        if (len(kp) ==1):
            a3=True
            #print des_temp
            des_out = des_temp[0]
        else:
            if(len(kp)==0):
                thresh_up = thresh
            else:
                thresh_down = thresh
        count+=1
        if count==max_iter:
            des_out = des_temp[0,:]


    return des_out

def surf_interface_2(im,hess_thresh,max_iter = 1000):
    des=None
    a=False
    des_out = zeros((1,128))
    count=0
    im_in = zeros((im.shape[0],im.shape[1],1), uint8)
    im_in[:,:,0] = 255*(im-im.min())/(im.max()-im.min())
    while (a==False and count<max_iter):
        surf = cv2.SURF(hess_thresh)
        kp, des_temp = surf.detectAndCompute(im_in,None)
        if(len(kp)>0):
            a = True
            des_out = des_temp[0,:]
        else:
            hess_thresh = hess_thresh/2
        count+=1
    return des_out


def surf_data_cube(data,hess_thresh):
    shap = data.shape
    k=0
    des = zeros((shap[2],128))
    good_im = list()
    bad_im = list()
    for k in range(0,shap[2]):
        imk = data[:,:,k]
        desk = surf_interface_2(imk,hess_thresh)
        des[k,:] = desk
        #print 'Gal ',k
        if (desk**2).sum()==0:
            print 'Bad im ',desk.max(),' ',desk.min()
            bad_im.append(k)
        else:
            good_im.append(k)

    return des,asarray(bad_im),asarray(good_im)

def kmeans_interface(data,nb_cluster,good_im_ind=None,nb_iter=10,eps=1.0,nb_attempts=5):
    data_in = float32(copy(data))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, nb_iter, eps)
    if good_im_ind is None:
        good_im_ind = array(range(0,data.shape[0]))
    label = zeros((data_in.shape[0]))
    ret,label,center=cv2.kmeans(data_in[good_im_ind,:], nb_cluster, label,criteria, nb_attempts, cv2.KMEANS_PP_CENTERS)
    label_ind = list()
    i=0
    for i in range(0,nb_cluster):
        ind = where(label==i)
        label_ind.append(good_im_ind[ind[0]])

    return ret,label_ind,label,center



def kmeans_preproc(im,opt=None,filters=None):
    coeff,filters = isap.mr_trans_2(im,filters=None,opt=None)
    shap = im.shape
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
    shap_coeff = coeff.shape
    stack_feat = zeros((shap[0],shap[1],shap_coeff[2]+2))
    stack_feat[:,:,0] = copy(im)
    for i in range(1,shap_coeff[2]+2):
        if i<shap_coeff[2]-1:
            stack_feat[:,:,i] = (copy(coeff[:,:,i])-mean(coeff[:,:,i])*ones(shap))/coeff[:,:,i].std()
        else:
            stack_feat[:,:,i] = (copy(pos_map[:,:,shap_coeff[2]-i])-mean(pos_map[:,:,shap_coeff[2]-i])*ones(shap))/pos_map[:,:,shap_coeff[2]-i].std()

    mat_out = transpose(cube_to_mat(stack_feat))
    i = where(mat_out[:,0]>0)

    #mat_out = mat_out[i[0],:]
    return mat_out,filters,i[0]



def clusters_analysis(label_ind,im):
    from numpy import int
    nb_clusters = len(label_ind)
    weights = zeros((nb_clusters,))
    shap = im.shape
    pos_cluster = zeros((nb_clusters,2))
    im_clusters = zeros((shap[0],shap[1],nb_clusters))
    ind_coord = list()
    for i in range(0,nb_clusters):
        nb_pts = len(label_ind[i])
        ind_coord_i = zeros((nb_pts,2))
        for k in range(0,nb_pts):
            il = int(label_ind[i][k]/shap[1])
            ic = label_ind[i][k]%shap[1]
            weights[i]+=im[il,ic]
            ind_coord_i[k,:] = array([il,ic])
            pos_cluster[i,:] += array([il,ic])
            im_clusters[il,ic,i] = im[il,ic]
        pos_cluster[i,:] /= nb_pts
        ind_coord.append(ind_coord_i)

    return im_clusters,weights,pos_cluster,ind_coord


def clusters_matching(label_ind1,label_ind2,P,tol=0.001):

    shap = P.shape
    err = tol*P.sum()
    # Setting threshold
    P_cpy = copy(P).reshape((shap[0]*shap[1],))
    ind = argsort(P_cpy)
    cm = 0
    i = 0
    while cm<err:
        cm += P_cpy[ind[i]]
        P_cpy[ind[i]] = 0
        i+=1

    P_thresh = P_cpy.reshape((shap[0],shap[1]))
    i,j = where(P_thresh>0)
    nb_prod = len(i)

    weights = zeros((nb_prod,))
    cart_prods = list()
    for k in range(0,nb_prod):
        l1 = len(label_ind2[i[k]])
        l2 = len(label_ind1[j[k]])
        ind = ones((l1*l2,2))
        for m in range(0,l1):
            ind[m*l2:(m+1)*l2,0]*= label_ind2[i[k]][m]
            ind[m*l2:(m+1)*l2,1] = copy(label_ind1[j[k]])
        ind = ind.astype(int)
        cart_prods.append(ind)
        weights[k] = P[i[k],j[k]]

    return cart_prods,weights

def mds_clustering(sim_matrix,nb_clusters,tol=0.99): # tol = percentage of the energy kept
    U, s, Vt = linalg.svd(sim_matrix,full_matrices=False)
    t = sum(s)
    nb_comp = 1
    while sum(s[0:nb_comp])<tol*t:
        nb_comp+=1
    print nb_comp
    w = diag(sqrt(s[0:nb_comp]))

    embedding = transpose(w.dot(Vt[0:nb_comp,:]))
    ret,label_ind,label,center = kmeans_interface(embedding,nb_clusters,nb_iter=10,eps=1.0,nb_attempts=5)
    return embedding,label_ind,center,ret

def centering_mat(M):
    Mout = copy(M)
    shap = M.shape
    ones_vect = ones((shap[0],1))
    Mout = Mout - ones_vect.dot(transpose(ones_vect).dot(M))/M.shape[0]
    return Mout

def gram_convert(M):
    M1 = centering_mat(M)
    Mout = -transpose(centering_mat(transpose(M1)))/2
    return Mout

def gal_kmeans(gal_cube,nb_cluster,hess_thresh=250):
    des,bad_im,good_im = surf_data_cube(gal_cube,hess_thresh)
    ret,label_ind,label,center = kmeans_interface(des,good_im,nb_cluster)
    if size(bad_im)>0:
        label_ind.append(bad_im)
    return ret,label_ind,center

def knn_bar(data,weights): # data is a 3D cube; the third dimension corresponds to the observables; each slice is a point cloud and the first dimension corresponds to the points coordinates
    from pyflann import FLANN
    from numpy import argsort,ones,array,zeros,where,diag,size,float64
    shap = data.shape
    ind = argsort(weights)
    flann = FLANN()
    weights_cp = ones((shap[0],1)).dot(weights.reshape((1,shap[2])))
    data_cp = array(data, dtype=float64)
    params = flann.build_index(data[:,:,ind[-1]])
    bar_out = zeros((shap[0],shap[1]))

    for i in range(0,shap[2]-1):
        result_temp, dists_temp = flann.nn_index(data[:,:,ind[i]], 1)
        for j in range(0,shap[0]):
            nn = where(result_temp==j)
            if size(nn[0])>0:
                bar_out[j,:]+= data[nn[0],:,ind[i]].sum(axis=0)*weights[ind[i]]/size(nn[0])
                weights_cp[j,ind[i]] = 0

    bar_out+=diag(weights_cp.sum(axis=1)).dot(data[:,:,ind[-1]])

    return bar_out

def lin_bar(data,weights):
    shap = data.shape
    bar_out = zeros((shap[0],shap[1]))

    for i in range(0,shap[2]):
        bar_out+=weights[i]*data[:,:,i]

    return bar_out


def knn_interf(data,nb_neigh,return_index=False):
    from numpy import array,float64
    flann = FLANN()
    data_cp = array(data, dtype=float64)
    params = flann.build_index(data_cp)
    result_temp, dists_temp = flann.nn_index(data_cp, nb_neigh+1)
    dists = dists_temp[:,1:nb_neigh+1]
    result = result_temp[:,1:nb_neigh+1]
    if return_index:
        return result,dists,flann
    else:
        return result,dists

def knn_splitting(data_pos,weights):
    nb_data = data_pos.shape[0]
    results,dists = knn_interf(data_pos,nb_data-1)
    i,j = where(dists==dists.max())
    weights_max = weights.sum()/2
    count = 0
    ind = 0
    while count <weights_max:
        count+=weights[results[i[0],ind]]
        ind+=1

    ind1 = zeros((ind,))
    ind1[0] = i[0]
    ind1[1:] = results[i[0],0:ind-1]
    ind2 = results[i[0],ind-1:]
    ind1 = ind1.astype(int)
    ind2 = ind2.astype(int)
    return ind1,ind2

def knn_splitting_eq(data_pos,weights):
    nb_data = data_pos.shape[0]
    results,dists = knn_interf(data_pos,nb_data-1)
    i,j = where(dists==dists.max())
    weights_max = weights.sum()/2
    count = 0
    ind = 0
    while count <weights_max:
        count+=weights[results[i[0],ind]]
        ind+=1

    ind1 = zeros((ind,))
    ind1[0] = i[0]
    ind1[1:] = results[i[0],0:ind-1]
    ind2 = results[i[0],ind-1:]
    ind1 = ind1.astype(int)
    ind2 = ind2.astype(int)
    return ind1,ind2

def knn_splitting_preproc(im):
    i,j = where(abs(im)>0)
    ind_lin = where(abs(im).reshape((size(im),))>0)
    nb_data = len(i)
    weights = im[i,j]
    pos = ones((nb_data,2))
    pos[:,0] = i
    pos[:,1] = j
    return pos,weights,ind_lin[0]


def knn_pos(ref_pos,data_pos,k):
    nb_data = data_pos.shape[0]
    ones_vect = ones((nb_data,1))
    sq_dist = ((data_pos - ones_vect.dot(transpose(vect(ref_pos))))**2).sum(axis=1)
    ind = argsort(sq_dist)
    return ind[0:k]

def mutli_scale_splitting(data_pos,weights,nb_split,ind_lin):
    nb_data = data_pos.shape[0]
    ind_init = arange(0,nb_data).astype(int)
    tree_loc = list()
    tree_loc.append(copy(ind_init))
    tree = None
    for i in range(0,nb_split):
        nb_cells = len(tree_loc)
        tree_cp = cp.deepcopy(tree_loc)
        tree = list()
        tree_loc = list()
        for k in range(0,nb_cells):
            ind1,ind2 = knn_splitting(data_pos[tree_cp[k],:],weights[tree_cp[k]])
            tree.append(ind_lin[tree_cp[k][ind1]])
            tree.append(ind_lin[tree_cp[k][ind2]])
            tree_loc.append(tree_cp[k][ind1])
            tree_loc.append(tree_cp[k][ind2])


    return tree


def vector_field_corr(mat,pos_field,nb_bin=20):
    shap = mat.shape
    nb_neigh = shap[0]-1
    result,dists = knn_interf(pos_field,nb_neigh)
    step = max(dists[:,nb_neigh-1])/nb_bin
    correl = zeros((nb_bin+1,))
    correl[0] = 1
    for i in range(0,nb_bin):
        count = 0

        for j in range(shap[0]):
            ind = where(dists[j,:]>i*step)
            if size(ind)>0:
                ind = ind[0][0]
                dist = dists[j,ind]

                while dist<step*(i+1) and dist > i*step and ind<nb_neigh-1:
                    count+=1
                    #print i
                    correl[i+1]+=(mat[j,:]*mat[result[j,ind],:]).sum()/(sqrt((mat[j,:]**2).sum())*sqrt((mat[result[j,ind],:]**2).sum()))
                    ind+=1
                    dist = dists[j,ind]
        if count>0:

            correl[i+1]/=count

    return correl

def diffusion_graph(data,nb_neigh,min_val=1,max_val=100000): # Samples are stored in the lines

    flann = FLANN()
    data_cp = array(data, dtype=float64)
    result_temp, dists_temp = flann.nn(data_cp, data_cp, nb_neigh+1)
    dists = dists_temp[:,1:nb_neigh+1]
    result = result_temp[:,1:nb_neigh+1]

    dists_weighted = copy(dists)
    k,l=0,0
    shap = data.shape
    for k in range(0,shap[0]):
        for l in range(0,nb_neigh):
            dists_weighted[k,l] = exp(-(dists[k,l]**2)/(dists[k,round(nb_neigh/2)]*dists[result[k,l],round(nb_neigh/2)]))
    shap = dists.shape
    ones_mat = ones((shap[0],shap[1]))
    dists_weighted = ((max_val-min_val)*(dists_weighted-dists_weighted.min()*ones_mat)/(dists_weighted.max()-dists_weighted.min()) + min_val*ones_mat)
    dists_weighted.astype(int)
    print '------------ ',dists_weighted.min(),' --------------'
    print '------------ ',dists.min(),' --------------'
    result_list = list()
    dist_list = list()
    dist_weighted_list = list()
    for k in range(0,shap[0]):
        result_list.append([])
        dist_list.append([])
        dist_weighted_list.append([])

    for k in range(0,shap[0]):
        for l in range(0,nb_neigh):
            if (result[k,l] in result_list[k])== False:
                result_list[k].append(result[k,l])
                dist_list[k].append(dists[k,l])
                dist_weighted_list[k].append(dists_weighted[k,l])
            if (k in result_list[result[k,l]])== False:
                result_list[result[k,l]].append(k)
                dist_list[result[k,l]].append(dists[k,l])
                dist_weighted_list[result[k,l]].append(dists_weighted[k,l])


    return result_list,dist_list,dist_weighted_list

def manifold_partionning_graph(data,nb_neigh_frac,max_val=100000):
    nb_points = data.shape[0]
    geod_dist_mat,next_mat,nb_neigh,min_mat,dist_mat = geodesic_distances(data,nb_neigh=2,prim_en=True)
    i,j = where(geod_dist_mat>0)
    min_dist = geod_dist_mat[i,j].min()
    max_dist = geod_dist_mat[i,j].max()
    ones_mat = ones((nb_points,nb_points))
    geod_dist_mat_int = max_val*geod_dist_mat/max_dist
    geod_dist_mat_int.astype(int)
    result_list = list()
    dist_list = list()
    for k in range(0,nb_points):
        result_list.append([])
        dist_list.append([])
    for k in range(0,nb_points):
        ind = argsort(geod_dist_mat_int[k,:])
        for l in range(0,int(nb_neigh_frac*nb_points)):
            if (ind[l+1] in result_list[k])== False:
                result_list[k].append(ind[l+1])
                dist_list[k].append(geod_dist_mat_int[k,ind[l+1]])
            if (k in result_list[ind[l+1]])== False:
                result_list[ind[l+1]].append(k)
                dist_list[ind[l+1]].append(geod_dist_mat_int[k,ind[l+1]])
    return result_list,dist_list,geod_dist_mat,next_mat,dist_mat

def manifold_partionning(data,nb_parts,nb_neigh_frac=1.0/3):
    edges,dists,geod_dist_mat,next_mat,dist_mat = manifold_partionning_graph(data,nb_neigh_frac)
    part_ind = metis_wrp(edges,dists,nb_parts,opt="-ptype=rb",del_file=True)
    return part_ind,geod_dist_mat,next_mat,dist_mat

def compute_nb_edges(edges): # All vertices have the same number of edges; the edges indices start at 0.
    shap0 = len(edges)

    k,l=0,0
    nb_edges=0
    for k in range(0,shap0):
        shap1 = len(edges[k])
        for  l in range(0,shap1):
            if (edges[k][l] > k):
                nb_edges+=1
            if  (edges[k][l] < k):
                if (k in edges[edges[k][l]])== False:
                    print "Warning: unproprer vertex"
    return nb_edges

def write_graph(edges,weights,graph_filename): # The weights should be integer
    shap0 = len(edges)
    nb_edges = compute_nb_edges(edges)
    f = open(graph_filename, 'w')
    # Header writing
    f.write(str(shap0)+' '+str(nb_edges)+' '+str(0)+str(0)+str(1)+'\n')
    k,l = 0,0
    for k in range(0,shap0):
        shap1 = len(edges[k])
        for l in range(0,shap1-1):
            f.write(str(int(edges[k][l])+1)+' '+str(int(weights[k][l]))+' ')
        f.write(str(int(edges[k][shap1-1]+1))+' '+str(int(weights[k][shap1-1]))+'\n')
    f.close
    return None

def read_graph_part(part_filename,nb_edges):
    labels = zeros((nb_edges,))
    k=0
    f = open(part_filename,'r')
    for k in range(0,nb_edges):
        labels[k] = int(f.readline())
    return labels

def randfilename(ext='.txt'):
    filename = str(random.randint(0, 1000000))+ext
    return filename

def metis_wrp(edges,weights,nb_parts,opt="-ptype=rb",del_file=True):
    filename = randfilename()
    write_graph(edges,weights,filename)
    subprocess.call(["gpmetis",opt,filename,str(nb_parts)])
    part_filename = filename+".part."+str(nb_parts)
    shap0 = len(edges)
    labels = read_graph_part(part_filename,shap0)
    part_ind = list([])
    l=0
    for l in range(0,nb_parts):
        ind  = where(labels==l)
        part_ind.append(ind[0])

    if del_file==True:
        os.remove(filename)
        os.remove(part_filename)
    return part_ind

def diffusion_partionning(data,nb_parts,nb_neigh=15):
    edges,dists,dists_weighted = diffusion_graph(data,nb_neigh)
    part_ind = metis_wrp(edges,dists_weighted,nb_parts,opt="-ptype=rb",del_file=True)
    return part_ind

def tree_split_2(data,tree,sparsity_level,list_split,nb_neigh=15):
    from numpy import *
    new_tree  =list([])
    new_list_split = list([])
    tree_size = shape(tree)
    i=0
    shape_dat = shape(data)
    for i in range(0,tree_size[0]):
        l = size(tree[i])

        if list_split[i][0]==1 and l >= 2*sparsity_level+2 :
            data_i = data[tree[i],:]
            nb_neigh_in = min(nb_neigh,int(l/2))
            part_ind = diffusion_partionning(data_i,2,nb_neigh=nb_neigh_in)
            tree_ind_1 = list()
            tree_ind_2 = list()
            l1 = size(part_ind[0])
            l2 = size(part_ind[1])
            if (l1<=sparsity_level) or (l2<=sparsity_level):
                new_tree.append(tree[i])
                new_list_split.append([0,i])
            else:
                for i1 in range(0,l1):
                    tree_ind_1.append(tree[i][part_ind[0][i1]])
                for i2 in range(0,l2):
                    tree_ind_2.append(tree[i][part_ind[1][i2]])

                new_tree.append(tree_ind_1)
                new_tree.append(tree_ind_2)
                new_list_split.append([1,-1])
                new_list_split.append([1,-1])
        else:
            new_tree.append(tree[i])
            new_list_split.append([0,i]) # we save the index in the old tree
    return new_tree,new_list_split

def tree_split_shap(data,tree,sparsity_level,list_split,nb_neigh=15):
    from numpy import *
    new_tree  =list([])
    new_list_split = list([])
    tree_size = shape(tree)
    i=0
    shape_dat = shape(data)
    for i in range(0,tree_size[0]):
        l = size(tree[i])

        if list_split[i][0]==1 and l >= 2*sparsity_level+2 :
            data_i = data[tree[i],:]
            u = array(range(0,6))
            shap = data_i.shape
            lx = int(shap[1]/6)
            id = int((lx-1)/2)
            data_shap_i = data_i[:,id+lx*u]
            nb_neigh_in = min(nb_neigh,int(l/2))
            part_ind = diffusion_partionning(data_shap_i,2,nb_neigh=nb_neigh_in)
            tree_ind_1 = list()
            tree_ind_2 = list()
            l1 = size(part_ind[0])
            l2 = size(part_ind[1])
            if (l1<=sparsity_level) or (l2<=sparsity_level):
                new_tree.append(tree[i])
                new_list_split.append([0,i])
            else:
                for i1 in range(0,l1):
                    tree_ind_1.append(tree[i][part_ind[0][i1]])
                for i2 in range(0,l2):
                    tree_ind_2.append(tree[i][part_ind[1][i2]])

                new_tree.append(tree_ind_1)
                new_tree.append(tree_ind_2)
                new_list_split.append([1,-1])
                new_list_split.append([1,-1])
        else:
            new_tree.append(tree[i])
            new_list_split.append([0,i]) # we save the index in the old tree
    return new_tree,new_list_split


def tree_split_shap_2(data,tree,sparsity_level,list_split,nb_neigh=15):
    from numpy import *
    new_tree  =list([])
    new_list_split = list([])
    tree_size = shape(tree)
    i=0
    shape_dat = shape(data)
    for i in range(0,tree_size[0]):
        l = size(tree[i])

        if list_split[i][0]==1 and l >= 2*sparsity_level+2 :
            data_i = data[tree[i],:]
            u = array(range(0,6))
            shap = data_i.shape
            lx = int(shap[1]/6)
            id = int((lx-1)/2)
            data_shap_i = data_i[:,id+lx*u]
            data_shap_i_2 = mk_ellipticity_atoms_basic_arr(data_shap_i)
            nb_neigh_in = min(nb_neigh,int(l/2))
            part_ind = diffusion_partionning(data_shap_i_2,2,nb_neigh=nb_neigh_in)
            tree_ind_1 = list()
            tree_ind_2 = list()
            l1 = size(part_ind[0])
            l2 = size(part_ind[1])
            if (l1<=sparsity_level) or (l2<=sparsity_level):
                new_tree.append(tree[i])
                new_list_split.append([0,i])
            else:
                for i1 in range(0,l1):
                    tree_ind_1.append(tree[i][part_ind[0][i1]])
                for i2 in range(0,l2):
                    tree_ind_2.append(tree[i][part_ind[1][i2]])

                new_tree.append(tree_ind_1)
                new_tree.append(tree_ind_2)
                new_list_split.append([1,-1])
                new_list_split.append([1,-1])
        else:
            new_tree.append(tree[i])
            new_list_split.append([0,i]) # we save the index in the old tree
    return new_tree,new_list_split


def geometric_dictionary_learning_2(data,sparsity_lev,tol,nb_neigh=15,rel_en=True): # Tol = pourcentage in terms of quadratic error
    from numpy import *
    #tol_abs = tol*(data**2).sum()
    split_en=1
    tree_init = list([])
    siz_data = data.shape
    tree_init.append(range(0,siz_data[0]))
    tree = tree_init
    list_split = []
    list_split.append([1,-1])
    dictionary,cells_means,coeff_tree=tree_SVD(data,tree_init,sparsity_lev,list_split)
    nb_sc=0
    while (split_en == 1):
        print 'tree splitting'
        tree,list_split=tree_split_2(data,tree,sparsity_lev,list_split,nb_neigh=nb_neigh)
        print 'iteratice SVD'
        dictionary,cells_means,coeff_tree=tree_SVD(data,tree,sparsity_lev,list_split,old_dictionary=dictionary,old_cells_means=cells_means,coeff_comp=None)
        print 'projection error'
        split_en,list_split = tree_approx_test(data,tree,dictionary,tol,cells_means,list_split,rel_en=rel_en)
        nb_sc=nb_sc+1
        print 'scale ',nb_sc
        print 'number of cells:', shape(tree)

    data_est = data*0
    nb_cells = len(tree)
    for i in range(0,nb_cells):
        nb_memb = len(tree[i])
        proji = dictionary[:,:,i].dot(transpose(dictionary[:,:,i]))
        centi = copy(cells_means[i,:]).reshape((siz_data[1],1))
        for j in range(0,nb_memb):
            dij = copy(data[tree[i][j],:]).reshape((siz_data[1],1))
            estij = proji.dot(dij-centi) + centi
            data_est[tree[i][j]] = estij.reshape((siz_data[1],))

    return tree,dictionary,cells_means,nb_sc,data_est

def geometric_dictionary_learning_cube_2(data,sparsity_lev,tol,nb_neigh=15,rel_en=True):
    shap = data.shape
    data_in = cube_to_mat(data)
    tree,dictionary,cells_means,nb_sc,data_est = geometric_dictionary_learning_2(data_in,sparsity_lev,tol,nb_neigh=nb_neigh,rel_en=rel_en)
    data_est_out = mat_to_cube(data_est,shap[0],shap[1])
    cells_means_out = mat_to_cube(cells_means,shap[0],shap[1])
    dictionnary_out = list()
    nb_cells = len(tree)
    for i in range(0,nb_cells):
        dictionnary_out.append(mat_to_cube(transpose(dictionary[:,:,i]),shap[0],shap[1]))
    return tree,dictionnary_out,cells_means_out,nb_sc,data_est_out

def geometric_dictionary_learning_cube_2_m(data,sparsity_lev,tol,nb_neigh=15,rel_en=True):
    shap = data.shape
    est = copy(data)*0
    for i in range(0,shap[3]):
        print "==============================================",i,"/",shap[3],"======================================================="
        tree,dictionnary_out,cells_means_out,nb_sc,data_est_out = geometric_dictionary_learning_cube_2(data[:,:,:,i],sparsity_lev,tol[i],nb_neigh=nb_neigh,rel_en=rel_en)
        est[:,:,:,i] = data_est_out
    return est


def shap_geometric_dictionary_learning_2(data,sparsity_lev,tol_ell=0.1,tol_init=0.0001,nb_neigh=15,iter_max=5):
    count=0
    stop=False
    tree=None
    dictionary=None
    cells_means=None
    nb_sc=None
    tol=tol_init
    while (stop==False) and (count<1):
        tree,dictionary,cells_means,nb_sc = geometric_dictionary_learning_2(data,sparsity_lev,tol,nb_neigh=nb_neigh)
        mell1_err,mell2_err,medell1_err,medell2_err,stdell1_err,stdell2_err = shap_dictionary_accuracy(data,dictionary,cells_means,tree)
        l = len(tree)
        k=0
        acc_reached = True
        while (acc_reached==True) and (k<l):
            if (medell1_err[k]>tol_ell) or (medell2_err[k]>tol_ell):
                acc_reached=False
                tol = tol/2
                print 'max(mell1_err),min(mell1_err),max(mell2_err),min(mell2_err) : ',max(mell1_err),min(mell1_err),max(mell2_err),min(mell2_err)
                print 'medell1_err,medell2_err: ',medell1_err,medell2_err
                print 'reducing square error tolerance'

            k+=1
        count+=1
        if (acc_reached==True):
            stop=True
    return tree,dictionary,cells_means,nb_sc,tol,mell1_err,mell2_err,medell1_err,medell2_err,stdell1_err,stdell2_err



def shap_tg_space_accuracy(shap_data,tg_mat,cell_mean):
    shap = shap_data.shape
    ell1_err = zeros((shap[0],))
    ell2_err = zeros((shap[0],))
    i=0
    shap_data_approx = ((shap_data-ones((shap[0],1))*cell_mean).dot(tg_mat)).dot(transpose(tg_mat))+ones((shap[0],1))*cell_mean
    for i in range(0,shap[0]):
        shap_mati = shap_data[i,:]
        shap_mati = shap_mati.reshape((6,shap[1]/6))
        shap_mat_approxi = shap_data_approx[i,:]
        shap_mat_approxi = shap_mat_approxi.reshape((6,shap[1]/6))
        li = shap[1]/6
        Ui = shap_mati[:,(li-1)/2]
        U_approxi = shap_mat_approxi[:,(li-1)/2]
        elli = mk_ellipticity_atoms_basic(Ui)
        ell_approxi = mk_ellipticity_atoms_basic(U_approxi)
        ell1_err[i] = 100*abs(elli[0,0]-ell_approxi[0,0])/abs(elli[0,0])
        ell2_err[i] = 100*abs(elli[0,1]-ell_approxi[0,1])/abs(elli[0,1])

    mell_err1 = mean(ell1_err)
    mell_err2 = mean(ell2_err)
    medell_err1 = median(ell1_err)
    medell_err2 = median(ell2_err)
    stdell_err1 = std(ell1_err)
    stdell_err2 = std(ell2_err)
    return mell_err1,mell_err2,medell_err1,medell_err2,stdell_err1,stdell_err2

def shap_dictionary_accuracy(shap_data,dictionary,cells_means,tree):
    l = len(tree)
    k = 0
    mell1_err = zeros((l,))
    mell2_err = zeros((l,))
    medell1_err = zeros((l,))
    medell2_err = zeros((l,))
    stdell1_err = zeros((l,))
    stdell2_err = zeros((l,))
    for k in range(0,l):
        shap_data_k = shap_data[tree[k],:]
        tg_mat = dictionary[:,:,k]
        cell_mean = cells_means[k,:]
        mell1_errk,mell2_errk,medell1_errk,medell2_errk,stdell1_errk,stdell2_errk = shap_tg_space_accuracy(shap_data_k,tg_mat,cell_mean)
        mell1_err[k] = mell1_errk
        mell2_err[k] = mell2_errk
        medell1_err[k] = medell1_errk
        medell2_err[k] = medell2_errk
        stdell1_err[k] = stdell1_errk
        stdell2_err[k] = stdell2_errk
    return mell1_err,mell2_err,medell1_err,medell2_err,stdell1_err,stdell2_err

def gal_shape_dic_learning(mat,sparsity_lev,tol,nb_neigh=15):
    from numpy import *
    tree,dictionary,cells_means,nb_sc = geometric_dictionary_learning_2(mat,sparsity_lev,tol,nb_neigh=nb_neigh)
    return tree,dictionary,cells_means,nb_sc

def gnd_tp_data_saving(pos_map,weights,pos_map_file,weights_file): # Data are saved in lines order
    f1 = open(pos_map_file,'w')
    f2 = open(weights_file,'w')
    shap = pos_map.shape
    i,j=0,0
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            f1.write("%f %f\n" % (pos_map[i,j,0],pos_map[i,j,1]))
            f2.write("%f\n" % (weights[i,j]))
    f1.close
    f2.close
    return 0

def read_opt_map(mapping_file,len_flow):
    f = open(mapping_file,'r')
    k = 0
    mapping = zeros((len_flow,3))
    data = fromfile(mapping_file,sep=" ")
    for k in range(0,len_flow):
        mapping[k,0] = data[k*3]   # Origin
        mapping[k,1] = data[k*3+1] # Destination
        mapping[k,2] = data[k*3+2] # Mass moved
    return mapping

def gnd_tp_solver_wrp(distrib_1,distrib_2,pos_map_1,pos_map_2,exe_path="../../CPP/NOT/build/gnd_tp_solve",del_file=True):

    pos_map_file_1 = randfilename()
    pos_map_file_2 = randfilename()
    weights_file_1 = randfilename()
    weights_file_2 = randfilename()
    mapping_file = randfilename()
    print pos_map_file_1,pos_map_file_2,weights_file_1,weights_file_2,mapping_file
    gnd_tp_data_saving(pos_map_1,distrib_1,pos_map_file_1,weights_file_1)
    gnd_tp_data_saving(pos_map_2,distrib_2,pos_map_file_2,weights_file_2)
    shap1 = distrib_1.shape
    shap2 = distrib_2.shape
    subprocess.call([exe_path,pos_map_file_1,weights_file_1,pos_map_file_2,weights_file_2,str(shap1[0]*shap1[1]),str(shap2[0]*shap2[1]),mapping_file])
    len_flow = shap1[0]*shap1[1] + shap2[0]*shap2[1] - 1
    mapping = read_opt_map(mapping_file,len_flow) # The 1st and 2nd columns contains linear indexes with the convention floor(ind/line_length) = line index; mod(ind,line_length) = column index

    if del_file==True:
        os.remove(pos_map_file_1)
        os.remove(pos_map_file_2)
        os.remove(weights_file_1)
        os.remove(weights_file_2)
        os.remove(mapping_file)
    return mapping


def space_var_conv_mat_feat(target_siz,samples_coord,lanczos_rad=4): # This function returns a matrix which performs a lanczos interpolation from a regular grid to a set arbitrary position; the non-zeros entries of the matrix are also provided.
    nb_data = samples_coord.shape[0]
    k=0
    nb_pix = target_siz[0]*target_siz[1]
    mat = zeros((nb_data,nb_pix))
    supp_cols = list({})
    supp_lines = list({})

    for k in range(0,nb_pix):
        supp_cols.append(list({}))
    for k in range(0,nb_data):
        supp_lines.append(list({}))
    lip = 0
    for k in range(0,nb_data):
        pos  = floor(samples_coord[k,:])
        delta = samples_coord[k,:] - pos
        kernel = lanczos(transpose(-delta),n=lanczos_rad)
        norm_ker = sqrt((kernel**2).sum())
        if norm_ker*nb_data > lip:
            lip = norm_ker*nb_data
        m,l=0,0
        for m in range(0,2*lanczos_rad+1):
            for l in range(0,2*lanczos_rad+1):
                id_line = int(pos[0])*target_siz[1]+int(pos[1])+l-lanczos_rad+(m-lanczos_rad)*target_siz[1]
                if (id_line>=0) and (id_line<nb_pix):
                    mat[k,id_line] = kernel[m,l]
                    supp_lines[k].append(id_line)

                    supp_cols[id_line].append(k)
    return mat,supp_cols,supp_lines,lip

def space_var_conv_mat(mat,X,supp_lines):

    n=0
    shap = mat.shape
    Y = zeros((shap[0],))
    for n in range(0,shap[0]):
        if supp_lines[n]:
            ind = array(supp_lines[n])
            Y[n] = mat[n,ind].dot(X[ind])

    return Y

def space_var_conv_transpose_mat(mat,Y,supp_cols):
    n=0
    shap = mat.shape
    X = zeros((shap[1],))
    for n in range(0,shap[1]):
        if supp_cols[n]:
            ind = array(supp_cols[n])
            X[n] = transpose(mat[ind,n]).dot(Y[ind])
    return X

def get_euclid_psf(nb_psf,shap=None,wvlgth=725,data_path='../../../Data/psf_euclid/',central=True,id=1): # wavlgt = wavelength in nanometers

    path = ''
    nb_psf_in = nb_psf
    wvlgth_min = 475
    wvlgth_max = 970
    if (wvlgth > wvlgth_max) or (wvlgth < wvlgth_min):
        print 'The specfied wavelength should be taken within [475,970]. Wavelength setr to 725nm'
        wvlgth = 725
    wvl_id = int(99*(wvlgth - wvlgth_min)/(wvlgth_max - wvlgth_min)+1)

    crop_en=True
    if central==True:
        path = data_path+'central4CCDs/'
        if nb_psf+id>599:
            print 'Warning: only 599 central PSF available'
            nb_psf_in = 599-id
        if shap is None:
            psf_test = fits.getdata(path+'PSF_MC_T0072.ZMX__Npup_4096x4096_lbda_0.800um_central4ccds.cat_0001.fits',wvl_id)
            shap = psf_test.shape
            crop_en=False
    if central==False:
        path = data_path+'corner36pos/'
        if nb_psf>36:
            print 'Warning: only 599 central PSF available'
            nb_psf_in = 36-id
        if shap==None:
            psf_test = fits.getdata(path+'PSF_MC_T0072.ZMX__Npup_4096x4096_lbda_0.800um_bords_guillard.cat_0001.fits',wvl_id)
            shap = psf_test.shape
            crop_en=False

    field_pos_vect = zeros((nb_psf_in,2))
    psf_cube = zeros((shap[0],shap[1],nb_psf_in))
    psf=None
    psf_file=None
    k=0
    psf_ind = None
    if central==True:
        psf_ind = rand_diff_integ(0,599,nb_psf_in)
    else:
        psf_ind = rand_diff_integ(0,35,nb_psf_in)

    for k in range(0,nb_psf_in):
        psf_id = psf_ind[k]+id
        if (central==True):
            if (psf_id<10):
                psf_file = path+'PSF_MC_T0072.ZMX__Npup_4096x4096_lbda_0.800um_central4ccds.cat_000'+str(psf_id)+'.fits'
            if (psf_id<100) and (psf_id>9):
                psf_file = path+'PSF_MC_T0072.ZMX__Npup_4096x4096_lbda_0.800um_central4ccds.cat_00'+str(psf_id)+'.fits'
            if (psf_id>99):
                psf_file = path+'PSF_MC_T0072.ZMX__Npup_4096x4096_lbda_0.800um_central4ccds.cat_0'+str(psf_id)+'.fits'
        if (central==False):
            if (psf_id<10):
                psf_file = path+'PSF_MC_T0072.ZMX__Npup_4096x4096_lbda_0.800um_bords_guillard.cat_000'+str(psf_id)+'.fits'
            if (psf_id>9):
                psf_file = path+'PSF_MC_T0072.ZMX__Npup_4096x4096_lbda_0.800um_bords_guillard.cat_00'+str(psf_id)+'.fits'
        print psf_file
        psf = fits.getdata(psf_file,wvl_id)
        field_pos_vect[k,0] = fits.getval(psf_file,'XFIELD',wvl_id)
        field_pos_vect[k,1] = fits.getval(psf_file,'YFIELD',wvl_id)
        if crop_en:
            psf_cube[:,:,k] = rect_crop(psf,shap[0],shap[1],sigw=12)
        else:
            psf_cube[:,:,k] = psf

    return psf_cube,field_pos_vect

def arcsec2pix_euclid(a,res_fact):

    b = a*12.0/(0.101*res_fact)

    return b

def argsort2D(arr):
    shap = arr.shape
    arr_in = reshape(arr,shap[0]*shap[1])
    ind = argsort(arr_in)

    return ind

def randwalk(rad=60,res_fact=6,nb_steps=100,amp=0.1):
    ker = zeros((2*rad+1,2*rad+1))
    ker[rad,rad]=1
    pos_init = array([rad,rad])
    for i in range(0,nb_steps):
        pos_init = pos_init + amp*(12.0/res_fact)*randn(2)
        print pos_init
        ind = array([int(pos_init[0]),int(pos_init[1])])
        ind[0] = max(min(2*rad,ind[0]),0)
        ind[1] = max(min(2*rad,ind[1]),0)
        print ind
        ker[ind[0],ind[1]]=1
    return ker

def randwalk_pol(rad=20,nb_steps=500,ampr=0.6,amp_deg = 15):
    ker = zeros((2*rad+1,2*rad+1))
    ker[rad,rad]=1
    pos_init = array([rad,rad])
    cent = copy(pos_init)
    r = 0
    theta = 0
    for i in range(0,nb_steps):
        r += ampr*randn(1)
        theta += abs(2*pi*amp_deg*randn(1)/180)
        pos_init = array([r*cos(theta)+rad,r*sin(theta)+rad])
        #print pos_init
        ind = array([int(pos_init[0]),int(pos_init[1])])
        ind[0] = max(min(2*rad,ind[0]),0)
        ind[1] = max(min(2*rad,ind[1]),0)
        #print ind
        ker[ind[0],ind[1]]+=1
    return ker


def get_euclid_psf_2(nb_psf,shap=None,wvlgth=730,data_path='../../../Data/psf_euclid_2/',rand_en=True,centering_en=False): # wavlgt = wavelength in nanometers

    psf_names = os.listdir(data_path)
    htest = fits.open(data_path+psf_names[1])
    psf_test = htest[1].data
    nb_wav = len(htest)-1

    wvlgth_min = htest[1].header['WLGTH0']*1000
    wvlgth_max = htest[nb_wav].header['WLGTH0']*1000
    wvl_mean = (wvlgth_min+wvlgth_max)/2
    if (wvlgth > wvlgth_max) or (wvlgth < wvlgth_min):
        print 'The specfied wavelength should be taken within [',wvlgth_min,',',wvlgth_max,']. Wavelength set to ',wvl_mean
        wvlgth = wvl_mean
    wvl_id = int((nb_wav-1)*(wvlgth - wvlgth_min)/(wvlgth_max - wvlgth_min)+1)
    nb_max = size(psf_names)
    if nb_psf > nb_max:
        print 'Too many PSF requested. Returning the max.'
        nb_psf = nb_max
    psf_ind = range(1,nb_psf+1)
    if rand_en and nb_psf<nb_max:
        psf_ind = rand_diff_integ(1,nb_max-1,nb_psf)
    print "PSF ind: ",psf_ind

    crop_en = 0
    if shap is None:
        shap = psf_test.shape
    else:
        crop_en=1
    htest.close()
    psf_cube = zeros((shap[0],shap[1],nb_psf))
    field_pos_vect = zeros((nb_psf,2))
    outliers = list()
    for i in range(0,nb_psf):
        print psf_ind[i]
        print data_path+psf_names[psf_ind[i]]
        hdu = fits.open(data_path+psf_names[psf_ind[i]])
        wvl_id_loc = wvl_id
        if len(hdu)-1 < wvl_id_loc:
            print 'Warning outlier wavelenght'
            outliers.append(psf_ind[i])
            wvl_id_loc = 1
        if crop_en==0:
            psf_cube[:,:,i] = hdu[wvl_id_loc].data
        else:
            psf_cube[:,:,i] = rect_crop(hdu[wvl_id_loc].data,shap[0],shap[1],sigw=1000)
        field_pos_vect[i,0] = hdu[wvl_id_loc].header['XFIELD']
        field_pos_vect[i,1] = hdu[wvl_id_loc].header['YFIELD']
        hdu.close()
    if centering_en:
        psf_cube = stack_regist(psf_cube,n=5)
    return psf_cube,field_pos_vect,psf_names,outliers


def get_euclid_psf_wvl(shap=None,data_path='../../../Data/psf_euclid/central4CCDs/',ind=0,centering_en=False,norm_en=True): # wavlgt = wavelength in nanometers

    psf_names = os.listdir(data_path)
    htest = fits.open(data_path+psf_names[ind+1])
    l = len(htest)-1
    shaptemp = htest[1].data.shape
    psf_test = zeros((shaptemp[0],shaptemp[1],l))
    wvl = zeros((l,))
    for i in range(0,l):
        psf_test[:,:,i] = copy(htest[i+1].data)
        wvl[i] = htest[i+1].header['WLGTH0']*1000
    if shap is not None:
        psf_test = rect_crop(psf_test,shap[0],shap[1],sigw=1000)
    if centering_en:
        psf_test = stack_regist(psf_test,n=5)
    if norm_en:
        psf_test = stack_norm(psf_test,norm=1)
    return psf_test,wvl

def get_euclid_psf_wvl_field(shap=None,data_path='../../../Data/psf_euclid/central4CCDs/',nb_psfs=200,nb_max = 599,centering_en=False,norm_en=True): # wavlgt = wavelength in nanometers

    psf_names = os.listdir(data_path)
    htest = fits.open(data_path+psf_names[1])
    nb_bands = len(htest)-1
    wvl = zeros((nb_bands,))
    for i in range(0,nb_bands):
        wvl[i] = htest[i+1].header['WLGTH0']*1000
    shaptemp = htest[1].data.shape

    id_ref = arange(1,nb_max)
    samp = choice(id_ref,size=nb_psfs,replace=False)
    psfs = list()
    field_pos_vect = zeros((nb_psfs,2))

    for i in range(0,nb_psfs):
        print i+1,"/",nb_psfs
        htest = fits.open(data_path+psf_names[id_ref[i]])
        if len(htest)-1==nb_bands:
            psf_test = zeros((shaptemp[0],shaptemp[1],nb_bands))
            for j in range(0,nb_bands):
                psf_test[:,:,j] = copy(htest[j+1].data)
            if shap is not None:
                psf_test = rect_crop(psf_test,shap[0],shap[1],sigw=1000)
            if centering_en:
                psf_test = stack_regist(psf_test,n=5)
            if norm_en:
                psf_test = stack_norm(psf_test,norm=1)
            psfs.append(psf_test)
            field_pos_vect[i,0] = htest[1].header['XFIELD']
            field_pos_vect[i,1] = htest[1].header['YFIELD']

    return psfs,wvl,field_pos_vect



def get_euclid_mix(nb_cent=600,nb_corn=36,nb_loc=400,shap=[512,512],wvlgth=730,rand_en=True,centering_en=False):
    nb_psf = nb_cent + nb_corn + nb_loc
    psf_cube = zeros((shap[0],shap[1],nb_psf))
    field_pos_vect = zeros((nb_psf,2))
    central_psf = '../../../Data/psf_euclid/central4CCDs/'
    corner_psf = '../../../Data/psf_euclid/corner36pos/'
    loc_psf = '../../../Data/psf_euclid_2/'

    psf_cube_1,field_pos_vect_1,psf_names,outliers_1 = get_euclid_psf_2(nb_cent,shap=shap,wvlgth=wvlgth,data_path=central_psf,rand_en=rand_en,centering_en=centering_en)
    psf_cube_2,field_pos_vect_2,psf_names,outliers_2 = get_euclid_psf_2(nb_corn,shap=shap,wvlgth=wvlgth,data_path=corner_psf,rand_en=rand_en,centering_en=centering_en)
    psf_cube_3,field_pos_vect_3,psf_names,outliers_3 = get_euclid_psf_2(nb_loc,shap=shap,wvlgth=wvlgth,data_path=loc_psf,rand_en=rand_en,centering_en=centering_en)
    psf_cube[:,:,0:nb_cent] = psf_cube_1
    psf_cube[:,:,nb_cent:nb_cent+nb_corn] = psf_cube_2
    psf_cube[:,:,nb_cent+nb_corn:nb_psf] = psf_cube_3
    nb_outliers = len(outliers_1)+len(outliers_2)+len(outliers_3)
    outliers = zeros((nb_outliers,))
    for i in range(0,nb_outliers):
        if i <len(outliers_1):
            outliers[i] = outliers_1[i]
        else:
            if i<len(outliers_1)+len(outliers_2):
                outliers[i] = outliers_1[i-len(outliers_1)]
            else:
                outliers[i] = outliers_1[i-(len(outliers_1)+len(outliers_2))]


    field_pos_vect[0:nb_cent,:] = field_pos_vect_1
    field_pos_vect[nb_cent:nb_cent+nb_corn,:] = field_pos_vect_2
    field_pos_vect[nb_cent+nb_corn:nb_psf,:] = field_pos_vect_3

    return psf_cube,field_pos_vect,outliers



def compact_sub_samp_extract(psf_cube,field_pos_vect,nb_psf):
    ind = distance_graph_sorting(field_pos_vect,nb_max=nb_psf)
    cube_out = psf_cube[:,:,ind]
    return cube_out,field_pos_vect[ind,:]


def psf_rot_regist(psf,sig,diagn=True):
    ell,cent = mk_ellipticity(psf,sig,niter_cent=4,cent_return=True)
    r = sqrt(ell[0,0]**2+ell[0,1]**2)
    cos_om = ell[0,0]/r
    om = arccos(cos_om)
    theta = om/2
    M = cv2.getRotationMatrix2D((cent[0,0],cent[0,1]),-theta*180/pi,1)
    psf_rot = cv2.warpAffine(psf,M,(psf.shape[1],psf.shape[0]))
    if diagn:
        ell_rot = mk_ellipticity(psf_rot,sig,niter_cent=4)
        r = sqrt(ell_rot[0,0]**2+ell_rot[0,1]**2)
        cos_om = ell_rot[0,0]/r
        om = arccos(cos_om)
        theta_rot = om/2
        return psf_rot,theta,ell,theta_rot
    else:
        return psf_rot,theta,ell

def opencv_rot_interf(im,theta,shap=None,cent=None,sig=None): # Theta in degrees
    if shap is None:
        shap = im.shape
    if (sig is not None) and (cent is None):
        cent,Wc = compute_centroid(im,sig)
    else:
        cent = array(((shap[0]-1)/2,(shap[1]-1)/2))
        cent = cent.reshape((1,2))
    M = cv2.getRotationMatrix2D((cent[0,0],cent[0,1]),theta,1)
    im_rot = cv2.warpAffine(im,M,(shap[1],shap[0]))
    return im_rot,cent




def logpolar(input,cent):
    from numpy import mgrid
    # This takes a numpy array and returns it in Log-Polar coordinates.

    coordinates = mgrid[0:max(input.shape[:])*2,0:360] # We create a cartesian array which will be used to compute log-polar coordinates.
    log_r = 10**(coordinates[0,:]/(input.shape[0]*2.)*log10(input.shape[1])) # This contains a normalized logarithmic gradient
    angle = 2.*pi*(coordinates[1,:]/360.) # This is a linear gradient going from 0 to 2*Pi

    # Using scipy's map_coordinates(), we map the input array on the log-polar coordinate. Do not forget to center the coordinates!
    lpinput = scipy.ndimage.interpolation.map_coordinates(input,(log_r*cos(angle)+cent[0,0],log_r*sin(angle)+cent[0,1]),order=3,mode='constant')

    # Returning log-normal...
    return lpinput



def polar(image,center=None,angles=None,radii=None): # No scale invariance
    from numpy import pi,sqrt,linspace,zeros,ones,copy,arange,sin as sinnp,cos as cosnp,empty_like,array
    """Return log-polar transformed image and log base."""
    shap = image.shape
    if center is None:
        center = where(image==image.max())
    if angles is None:
        angles = int(pi*min(shap[0],shap[1]))
    if radii is None:
        radii = round(sqrt(shap[0]**2+shap[1]**2)/2)

    theta = zeros((angles,radii), dtype=float64)
    theta.T[:] = -linspace(0, 2*pi, angles, endpoint=False)
    b = sinnp(theta)
    x = copy(image)
    y = copy(image)
    i,j=0,0

    #d = hypot(shape[0]-center[0], shape[1]-center[1])
    #log_base = 10.0 ** (math.log10(d) / (radii))
    radius = copy(theta)
    radius[:] = array(arange(radii,dtype=float64))

    x = radius*sinnp(theta)  + ones((angles,radii))*center[0]
    y = radius*cosnp(theta)  + ones((angles,radii))*center[1]
    output = empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output





def auto_correl_map(im,winds):
    shap = im.shape
    i,j,=0,0
    map = im*0
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            i1 = max(i-(winds-1)/2,0)
            i2 = min(i+(winds-1)/2,shap[0])
            j1 = max(j-(winds-1)/2,0)
            j2 = min(j+(winds-1)/2,shap[1])
            wind = im[i1:i2,j1:j2]
            wind_fft = abs(scipy_fft.fft2(wind))
            map[i,j] = sqrt((abs(wind_fft)**4).sum())

    map = map/map.max()
    return map

def struct_res_support(cube,thresh=15,wind=10):
    shap = cube.shape
    k=0
    supp = cube*0
    for k in range(0,shap[2]):
        map = auto_correl_map(cube[:,:,k],wind)
        md = mad(map)
        supp[:,:,k] = (map>thresh*md)
    return supp

def sum_prod_gauss(means1,means2,sigs1,sigs2): # Calculates the mean and the product of the sum of products of gaussian distributions
    m = sum(means1*means2)
    sig = sqrt(sum((means2**2)*(sigs1**2)+(means1**2)*(sigs2**2)+(sigs2**2)*(sigs1**2)))
    return m,sig

def autocorrel_detection(x,p_val=0.99,k=3,sig=None,detect_rad=3):

    if sig is None:
        sig = im_gauss_nois_est_cube(x)
    shap = array(x.shape)
    if shap[0]%2==0:
        shap[0]-=1
    if shap[1]%2==0:
        shap[1]-=1
    autocorr = scipy_fft.ifft2(abs(scipy_fft.fft2(x[0:shap[0],0:shap[1]]))**2)
    interv_1 = scistats.chi2.interval(p_val,shap[0]*shap[1],scale=sig)
    means = zeros((shap[0],shap[1]))
    sigs = ones((shap[0],shap[1]))*sig
    mcross,sig_cross = sum_prod_gauss(means,means,sigs,sigs)
    interv_2 = [mcross-k*sig_cross,mcross+k*sig_cross]
    ic = shap[0]/2
    jc = shap[1]/2
    detect = False
    if autocorr[ic,jc]>interv_1[1] or autocorr[ic,jc]<interv_1[0]:
        detect = True
    if detect is False:
        nb_tests = 0
        while detect is False and nb_tests < (2*detect_rad+1)**2:
            i1 = nb_tests%(2*detect_rad+1)-rad
            i2 = nb_tests/(2*detect_rad+1)-rad
            if i1**2+i2**2 >0:
                if autocorr[i1+ic,i2+jc]>interv_2[1] or autocorr[i1+ic,i2+jc]<interv_2[0]:
                    detect = True
            nb_ests+=1
    return detect,autocorr

def autocorrel_detection_stack(x,p_val=0.99,k=3,sig=None,detect_rad=4):
    shap = x.shape
    nb_im = shap[2]
    count = 0
    detect_flag = list()
    for i in range(0,nb_im):
        detect,autocorr = autocorrel_detection(x[:,:,i],p_val=p_val,k=k,sig=sig,detect_rad=detect_rad)
        detect_flag.append(detect)
        if detect is True:
            count+=1

    score = 100*double(count)/nb_im
    return detect_flag,score

def mp_struct_res_support_slice(cube,nb,ne,q,thresh=15,wind=10):
    slice = cube[:,:,nb:ne+1]
    supp_k = struct_res_support(slice,thresh=thresh,wind=wind)
    q.put([supp_k,nb,ne])

def mp_struct_res_support(cube,nb_proc,thresh=15,wind=10):
    supp = cube*0
    nb_proc = 5
    supp = 0*cube
    nb_gal = cube.shape[2]
    i = 0
    slice = int(nb_gal/nb_proc)
    #if __name__ == '__main__':
    q = Queue()
    jobs = []
    for i in range(0,nb_proc):
        ib = i*slice
        ie = ib+slice-1
        p = Process(target=mp_struct_res_support_slice, args=(cube,ib,ie,q,thresh,wind))
        p.start()
        jobs.append(p)

    for i in range(0,nb_proc):
        a = q.get()
        supp[:,:,a[1]:a[2]+1] = a[0]

    return supp


def wien_filt(im):
    snr_map = snr_map_gauss(im)
    w = snr_map/(snr_map+1)
    print w.min()
    print w.max()
    imdft = scipy_fft.fft2(im)
    imfilt = scipy_fft.ifft2(imdft*w)
    res_imag = imag(imfilt)
    leak = sqrt((real(imfilt).sum())**2/(res_imag.sum())**2)
    return real(imfilt),leak


def snr_map_gauss(im):
    shap = im.shape
    onesmat = ones((shap[0],shap[1]))
    sig = 1.4826*mad(im)
    imdft = scipy_fft.fft2(im)
    snr_map = sqrt((abs(imdft)**2/sig**2-onesmat)*double(abs(imdft)**2/sig**2>=onesmat))
    return snr_map

def shap_mat_learning_data_pipe(gal,nb_clusters,stamp_rad,filename):
    # Denoising
    cube_den,Om,res = optim_utils.spca_interf(gal)
    # Clustering
    ret,label_ind,center  = gal_kmeans(cube_den,nb_clusters)
    # Savings and outputs
    gal_clusters = list()
    shap_clusters = list()
    k=0
    for k in range(0,nb_clusters):
        cube_k,theta = mp_rand_rot(cube_den[:,:,label_ind[k]])
        gal_clusters.append(cube_k)
        fits.writeto('../data/'+filename+str(k)+'gal.fits',cube_to_mat(cube_k))
        matk = great3_util.data_mat_shapes(cube_k,stamp_rad)
        shap_clusters.append(matk)
        fits.writeto('../data/'+filename+str(k)+'shap_mat.fits',matk)
    return gal_clusters,shap_clusters


def wvl_norm(siz,opt=['-t2','-n1']):
    dirac = zeros(siz)
    dirac[(siz[0]-1)/2,(siz[1]-1)/2]=1
    Result,mr_file = isap.mr_trans(dirac,opt=opt)
    shap = Result.shape
    norm_vect = zeros((shap[2],))
    k=0
    for k in range(0,shap[2]):
        norm_vect[k] = sqrt((Result[:,:,k]**2).sum())
    os.remove(mr_file)
    return Result,norm_vect

def correlated_noise_est(u,nb_iter=5,k=5,tol=1.):

    sig = 1.4826*mad(u)
    i=0
    var = 100
    ones_vec = ones(u.shape)
    while i < nb_iter and var>100*tol:
        sig_old = sig
        sig = 1.4826*mad(u-thresholding(u,sig*ones_vec,thresh_type=0))
        var = 100*abs(sig-sig_old)/sig_old
        i+=1

    return sig


def im_gauss_nois_est(im,opt=['-t2','-n2'],filters=None):
    from isap import mr_trans_2
    Result,filters = mr_trans_2(im,filters=filters,opt=opt)
    siz = im.shape
    norm_wav = norm(filters[:,:,0])
    sigma = 1.4826*mad(Result[:,:,0])/norm_wav

    return sigma,filters

def im_gauss_nois_est_cube(cube,opt=None,filters=None,return_map=False):
    shap = cube.shape
    sig = zeros((shap[2],))
    map = None
    if return_map:
        map =ones(shap)

    for i in range(0,shap[2]):
        sig_i,filters = im_gauss_nois_est(cube[:,:,i],opt=opt,filters=filters)
        sig[i] = sig_i
        if return_map:
            map[:,:,i] *= sig[i]
    if return_map:
        return sig,filters,return_map
    else:
        return sig,filters

def im_thresh_map(im,k,opt=None):
    sig,filters = im_gauss_nois_est(im,opt)
    siz = im.shape
    Result,norm_vect = wvl_norm(siz,opt=opt)
    shap = Result.shape
    thresh_map = ones((shap[0],shap[1],shap[2]-1))
    i=0
    for i in range(0,shap[2]-1):
        thresh_map[:,:,i] = thresh_map[:,:,i]*k*sig*norm_vect[i]

    return thresh_map

def res_sig_map(res,opt=None):

    wvl_res,mr_file = isap.mr_trans(res,opt=opt)
    shap = wvl_res.shape
    sig_map = ones((shap[0],shap[1],shap[2]-1))
    k=0
    for k in range(0,shap[2]-1):
        sigk = 1.4826*mad(wvl_res[:,:,k])
        sig_map[:,:,k] = sig_map[:,:,k]*sigk
    os.remove(mr_file)
    return sig_map

def sig_map(res):
    shap = res.shape
    map = None
    if len(shap)==2:
        map = 1.4826*ones((shap[0],shap[1]))*mad(res)
    elif len(shap)==3:
        map = ones((shap[0],shap[1],shap[2]))
        for i in range(0,shap[2]):
            map[:,:,i] *= im_gauss_nois_est(res[:,:,i])
    return map

def acc_sig_map(shap_im,ker_stack,sig_est,flux_est,flux_ref,upfact,w,sig_data=None):
    shap = ker_stack.shape
    nb_im = shap[2]
    if sig_data is None:
        sig_data = ones((nb_im,))
    var_stack = ones((shap_im[0],shap_im[1],nb_im))
    map2 = zeros((shap_im[0]*upfact,shap_im[1]*upfact))
    ker_stack_in = copy(ker_stack)**2
    for l in range(0,shap[2]):
        var_stack[:,:,l]*=sig_data[l]**2
        #print "Noise estimation parameters: sigma data",sig_data[l]**2," sigma est: ",sig_est[l]**2," shift ker norm ",sum(ker_stack_in[:,:,l]**2)
        map2 += ((w[l]*flux_est[l]/(sig_est[l]*flux_ref))**2)*scisig.convolve(transpose_decim(var_stack[:,:,l],upfact),ker_stack_in[:,:,l],mode='same')
    map =  sqrt(map2)
    return map

def acc_sig_maps(shap_im,ker_stack,sig_est,flux_est,flux_ref,upfact,w,sig_data=None):
    shap = w.shape
    map_out = zeros((shap_im[0]*upfact,shap_im[1]*upfact,shap[0]))
    for i in range(0,shap[0]):
        map_out[:,:,i] = acc_sig_map(shap_im,ker_stack,sig_est,flux_est,flux_ref,upfact,w[i,:],sig_data=sig_data)
        #print "Noise estimation, weights norm: ",sum(w[i,:]**2)

    return map_out


def stack_convolve(cube_im,cube_psf):
    shap = cube_im.shape
    cube_out = copy(cube_im)
    for i in range(0,shap[2]):
        cube_out[:,:,i] = scisig.convolve(cube_im[:,:,i],cube_psf[:,:,i],mode='same')
    return cube_out

def stack_fft_deconv(cube_im,cube_psf,nb_iter=100):
    shap = cube_im.shape
    cube_out = copy(cube_im)
    for i in range(0,shap[2]):
        #cube_out[:,:,i] =  scipy_fft.ifft2(scipy_fft.fft2(cube_im[:,:,i])/scipy_fft.fft2(cube_psf[:,:,i]))
        print "=========",i,"/",shap[2],"============"
        cube_out[:,:,i] = optim_utils.deconvol(cube_im[:,:,i],cube_psf[:,:,i],mu=1,nb_iter=nb_iter)
    return cube_out

def stack_fft_deconv_m(cube_im,cube_psf_m):
    shap = cube_psf_m.shape
    shap2 = cube_im.shape
    cube_out = zeros((shap2[0],shap2[1],shap2[2],shap[3]))
    for i in range(0,shap[3]):
        print "=========",i,"/",shap[3],"============"
        cube_out[:,:,:,i] = stack_fft_deconv(cube_im,cube_psf_m[:,:,:,i])
    return cube_out

def min_pt(im1,im2):
    im_min = copy(im1)
    shap = im1.shape
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            im_min[i,j] = min(im1[i,j],im2[i,j])
    return im_min


def max_pt(im1,im2):
    im_max = copy(im1)
    shap = im1.shape
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            im_max[i,j] = max(im1[i,j],im2[i,j])
    return im_max


def hyp_point(u,b): # Returns a point of the hyperplane defined by <u,x>=b; this routine assumes that u is a 2D array
    ind1,ind2 = where(u>0)
    n = size(ind1)
    shap = u.shape
    x = zeros(shap)
    x[ind1,ind2] = (u[ind1,ind2])**(-1)
    x = x*b/n
    return x

def hyp_2_point(u1,c1,u2,c2): # Returns a point in the intersection (supposed to exist) of the hyperplanes <u1,x>=c1, <u2,x>=c2
    n1 = (u2**2).sum()
    a1 = (u1*u2).sum()/n1
    v0 = u1 - a1*u2
    b1 = sqrt((v0**2).sum())
    v = v0/b1
    a = c2/n1
    b = (c1-a1*c2)/b1
    x = a*u2+b*v
    return x

def proj_hyp(u,b,y): # Projects the point y on the hyperplane defined by <u,x>=b
    n = u/sqrt((u**2).sum())
    o = hyp_point(u,b)
    x = y - ((y-o)*n).sum()*n
    return x

def proj_2_hyp(u1,b1,u2,b2,y): # Projects y on the intersection (supposed to exist) of the hyperplanes <u1,x>=b1, <u2,x>=b2
    o = hyp_2_point(u1,b1,u2,b2)
    shap = u1.shape
    dict = zeros((shap[0],shap[1],2))
    dict[:,:,0]=u1
    dict[:,:,1]=u2
    x = y-proj_cube(y-o,dict,ortho=1)
    return x

def rand_lin_comb(u1,u2,u3,nb_vect):
    from numpy.random import rand
    shap = u1.shape
    com_stack = zeros((shap[0],shap[1],nb_vect))
    coeff = zeros((nb_vect,3))
    for i in range(0,nb_vect):
        a1 = rand(1)
        a2 = rand(1)*(1-a1)
        a3 = 1-a1-a2
        coeff[i,0] = a1
        coeff[i,1] = a2
        coeff[i,2] = a3
        com_stack[:,:,i] = a1*u1+a2*u2+a3*u3
        com_stack[:,:,i] = com_stack[:,:,i]/sqrt(((com_stack[:,:,i])**2).sum())
    return com_stack,coeff

def dir_check(dir_stack,ref_ell,cur_point):
    siz = dir_stack.shape
    ell_steepness = zeros((siz[2],2))
    dir_cand = zeros((siz[2],))
    i = 0
    for i in range(0,siz[2]):
        grad1i,grad2i,ellti = ellipticity_gradt(cur_point,dir_stack[:,:,i],0)
        ell_steepness[i,0] = grad1i
        ell_steepness[i,1] = grad2i
        if (ell_steepness[i,0]*ref_ell[0,0]<=0) and (ell_steepness[i,1]*ref_ell[0,1]<=0):
            dir_cand[i]=1
    return dir_cand,ell_steepness

def polar_coord(coord,cent): # Angle in radians
    from numpy import sqrt,arccos,abs,pi
    r = sqrt((coord[0]-cent[0])**2+(coord[1]-cent[1])**2)
    alpha=None
    if r==0:
        alpha=0
    else:
        alpha_ref = arccos(abs(coord[0]-cent[0])/r)
        alpha = alpha_ref
        if (coord[0]-cent[0]<0) and (coord[1]-cent[1]>=0):
            alpha = pi-alpha
        elif (coord[0]-cent[0]<0) and (coord[1]-cent[1]<0):
            alpha = pi+alpha
        elif (coord[0]-cent[0]>=0) and (coord[1]-cent[1]<0):
            alpha = 2*pi-alpha

    return r,alpha

def polar_coord_cloud(coord,cent):
    from numpy import zeros
    shap = coord.shape
    out = zeros((2,shap[1]))

    for i in range(0,shap[1]):
        out[0,i],out[1,i] = polar_coord(coord[:,i],cent)

    return out

def polar_to_cart(coord,cent):
    return coord[0]*cos(coord[1])+cent[0],coord[0]*sin(coord[1])+cent[1]

def polar_to_cart_cloud(coord,cent):

    from numpy import zeros
    shap = coord.shape
    out = zeros((2,shap[1]))

    for i in range(0,shap[1]):
        out[0,i],out[1,i] = polar_to_cart(coord[:,i],cent)

    return out

def pol_coord_map_check(siz,cent=None):
    if cent is None:
        cent = [siz[0]/2,siz[1]/2]
    map_r = zeros((siz[0],siz[1]))
    map_theta = zeros((siz[0],siz[1]))
    i,j=0,0

    for i in range(0,siz[0]):
        for j in range(0,siz[1]):
            r,theta = polar_coord([j,i],cent)
            map_r[i,j] = r
            map_theta[i,j] = theta

    return map_r,map_theta



def scaling(x,n,ainit=0,afinal=1): # returns y = ((2x-1)^(1/n)+1)/2; n should be odd
    return (afinal-ainit)*(sign(2*x-1)*abs(2*x-1)**(1.0/n)+1)/2 + ainit


def scaling_2(x,n,ainit=0,afinal=1): # returns y = (afinal-ainit)*(x)^n+ainit
    return (afinal-ainit)*(x)**n + ainit


def add_test(a):
    return sum(a)

def call_fun(fun,arg):
    return fun(arg)

def stack_transpose(cube):
    shap = cube.shape
    trans_stack = zeros((shap[1],shap[0],shap[2]))
    for i in range(0,shap[2]):
        trans_stack[:,:,i] = transpose(cube[:,:,i])

    return trans_stack

def rot_dissim(im_1,im2,theta): # Theta in degrees
    im2rot = opencv_rot_interf(im2,theta)
    d1 = abs(im_1-im2rot).max()
    return d1

def int_grid_shift(psf_stack): #
    shap = psf_stack.shape
    U = zeros((shap[2],2))

    for i in range(0,shap[2]):
        #param=gaussfitter.gaussfit(psf_stack[:,:,i],returnfitimage=False)
        #(centroid,Wc) = compute_centroid(psf_stack[:,:,i],(param[3]+param[4])/2)
        (centroid,Wc) = compute_centroid(psf_stack[:,:,i],1e20,nb_iter=1)
        U[i,0] = centroid[0,0]-double(shap[0])/2
        U[i,1] = centroid[0,1]-double(shap[1])/2

    return U

def shift_est(psf_stack): #
    shap = psf_stack.shape
    U = zeros((shap[2],2))
    param=gaussfitter.gaussfit(psf_stack[:,:,0],returnfitimage=False)
    #(centroid_ref,Wc) = compute_centroid(psf_stack[:,:,0],(param[3]+param[4])/2)
    centroid_out = zeros((shap[2],2))
    for i in range(0,shap[2]):
        param=gaussfitter.gaussfit(psf_stack[:,:,i],returnfitimage=False)
        #(centroid,Wc) = compute_centroid(psf_stack[:,:,i],(param[3]+param[4])/2)
        (centroid,Wc) = compute_centroid(psf_stack[:,:,i],(param[3]+param[4])/2)
        U[i,0] = centroid[0,0]-double(shap[0])/2
        U[i,1] = centroid[0,1]-double(shap[1])/2
        centroid_out[i,0]  = centroid[0,0]
        centroid_out[i,1]  = centroid[0,1]
    return U,centroid_out

def shift_est_2_anch(psf_stack,anchor,res=1,width=5,exp=1):
    shap = psf_stack.shape
    shifts = zeros((shap[2],2))
    for i in range(0,shap[2]):
        p,fwhm,fit_mod = moffat_fitting_2d(psf_stack[:,:,i],width=5,exp=1)
        centroid = compute_centroid_2(psf_stack[:,:,i],fit_mod)
        shifts[i,0] = anchor[0] - centroid[0]*res
        shifts[i,1] = anchor[1] - centroid[1]*res
    return shifts

def wvl_centroid(im,opt,nsig):
    wvl,mr_file = isap.mr_trans(im,opt=opt)
    shap = wvl.shape
    sig_map = ones((shap[0],shap[1],shap[2]-1))
    k=0
    for k in range(0,shap[2]-1):
        sigk = 1.4826*mad(wvl[:,:,k])
        sig_map[:,:,k] = sig_map[:,:,k]*sigk
    thresh_type = 0
    wvl_clean = thresholding(wvl,nsig*sig_map,thresh_type)
    wvl_cent = zeros((1,2))
    for k in range(0,shap[2]-1):
        cent,w = compute_centroid(im,10**3,nb_iter=1)
        wvl_cent = wvl_cent+cent
    wvl_cent = wvl_cent/(shap[2]-1)
    os.remove(mr_file)
    return wvl_cent

def wvl_shift_est(psf_stack,opt,nsig):
    shap = psf_stack.shape
    U = zeros((shap[2],2))
    centroid_ref = wvl_centroid(psf_stack[:,:,0],opt,nsig)
    for i in range(1,shap[2]):
        centroid = wvl_centroid(psf_stack[:,:,i],opt,nsig)
        U[i,0] = centroid[0,0]-centroid_ref[0,0]
        U[i,1] = centroid[0,1]-centroid_ref[0,1]

    return U

def thresh_centroid(im,sig_thresh=None,sigw=None,opt=None):
    thresh_type = 0
    if sig_thresh is None:
        sig_thresh = im_gauss_nois_est(im,opt=opt)
    thresh_map = ones(im.shape)*min(4*sig_thresh,abs(im).max()-sig_thresh)
    im_thresh = thresholding(im,thresh_map,thresh_type=thresh_type)
    if sigw is None:
        param=gaussfitter.gaussfit(im,returnfitimage=False)
        sigw = (param[3]+param[4])/2
    cent,w = compute_centroid(im_thresh,sigw,nb_iter=4)
    return cent

def thresh_shift_est(im_stack,sig_thresh=None,opt=None):
    shap = im_stack.shape
    sigwv = zeros((shap[2],))
    for i in range(0,shap[2]):
        param=gaussfitter.gaussfit(im_stack[:,:,i],returnfitimage=False)
        sigwv[i] = (param[3]+param[4])/2
    sigw_ref = median(sigwv)
    centroids = zeros((shap[2],2))
    for i in range(0,shap[2]):
        if sig_thresh is not None:
            centroids[i,:] = thresh_centroid(im_stack[:,:,i],sig_thresh=sig_thresh[i],sigw=sigw_ref,opt=opt)
        else:
            centroids[i,:] = thresh_centroid(im_stack[:,:,i],sigw=sigw_ref,opt=opt)
    U = zeros((shap[2],2))
    for i in range(1,shap[2]):
        U[i,0] = centroids[i,0]-centroids[0,0]
        U[i,1] = centroids[i,1]-centroids[0,1]
    return U


def flux_estimate(im,cent=None,rad=4): # Default value for the flux tunned for Euclid PSF at Euclid resolution
    flux = 0
    if cent is None:
        cent = array(where(im==im.max())).reshape((1,2))
    shap = im.shape
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            if sqrt((i-cent[0,0])**2+(j-cent[0,1])**2)<=rad:
                flux = flux+im[i,j]
    return flux

def flux_estimate_stack(stack,cent=None,rad=4):
    shap = stack.shape
    flux = zeros((shap[2],))
    for i in range(0,shap[2]):
        if cent is not None:
            flux[i] = flux_estimate(stack[:,:,i],cent=cent[i,:],rad=rad)
        else:
            flux[i] = flux_estimate(stack[:,:,i],rad=rad)
    return flux

def random_shifts(im_stack,amax=0.5,n=10): # random shifts in [-amax,amax]*pix
    from numpy.random import rand
    from scipy.signal import fftconvolve
    output_stack = copy(im_stack)
    shap = im_stack.shape
    shifts = zeros((shap[2],2))
    #print " ----------- /!\ Warning /!\: a single image is used in the shift simulation----------- "
    for i in range(0,shap[2]):
        u = zeros([1,2])

        u[0,0] = amax*(rand(1)-0.5)
        u[0,1] = amax*(rand(1)-0.5)
        shifts[i,:] = u.reshape((2,))
        output_stack[:,:,i] = fftconvolve(im_stack[:,:,i],lanczos(u,n=n),mode='same') # here
    return output_stack,shifts


def randshift_decim_flux(im_stack,down_fact,amax=0.5,n=10,var=0,nb_fields=1): # Var= Luminosity variability: 0=> same flux,1=> might tend to zero
    from numpy.random import random
    from numpy import squeeze
    shap = im_stack.shape
    output_stack = zeros((int(shap[0]/down_fact),int(shap[1]/down_fact),shap[2],nb_fields))
    P = zeros((shap[2],nb_fields))
    shifts = zeros((shap[2],2,nb_fields))
    flux = zeros((shap[2],nb_fields))

    for k in range(0,nb_fields):
        shift_stack,shifts_k = random_shifts(im_stack,amax=amax,n=n)
        temp,output_stack_k = decim_arr(shift_stack,down_fact,av_en=0,fft=1)
        P_k = zeros((shap[2],))
        i=0
        flux_k = zeros((shap[2],))
        for i in range(0,shap[2]):
            flux[i,k] = 1-var*random(1)
            P_k[i] = (output_stack_k[:,:,i]**2).sum()/(shap[0]*shap[1])
            output_stack_k[:,:,i] = flux[i,k]*output_stack_k[:,:,i]
            shifts_k[i,:] = shifts_k[i,:] - shifts_k[0,:]
        output_stack[:,:,:,k] = copy(output_stack_k)
        P[:,k] = copy(P_k)
        shifts[:,:,k] = shifts_k

    return output_stack,P,squeeze(shifts),squeeze(flux)

def multiframe_sr_data_gen(im,down_fact,snr,nb_obs,noise_var=0,flux_var=0,amax=0.5):
    from numpy import log10
    shap = im.shape
    im_stack = repeat(im.reshape((shap[0],shap[1],1)),nb_obs,axis=2)
    output_stack,sig,dec_stack,shifts,sig,flux = sr_data_gen(im_stack,20*log10(snr),down_fact,amax=amax,n=10,noise_var=noise_var,flux_var=flux_var,nb_comp=None,nb_fields=1)

    return output_stack,sig


def noise_cube(shap,snrdb_max,pow,var=0,nb_fields=1): # Var= snr variability; 0=same snr, 1=> there might twice more noise in an image compared to another
    from numpy.random import random
    cube = zeros(shap)
    sig = zeros((shap[2],nb_fields))
    for k in range(0,nb_fields):
        for i in range(0,shap[2]):
            sig[i,k] = (1+var*random(1))*sqrt(pow[i,k])*10**(-snrdb_max/20)
            cube[:,:,i,k] = sig[i,k]*randn(shap[0],shap[1])
    return cube,sig

def sr_data_gen(im_stack,snrdB,down_fact,amax=0.5,n=10,noise_var=0,flux_var=0,nb_comp=None,nb_fields=1):
    from optim_utils import trunc_svd_cube
    im_stack_in = copy(im_stack)
    if nb_comp is not None:
        im_stack_in = trunc_svd_cube(im_stack,nb_comp)
    dec_stack,P,shifts,flux = randshift_decim_flux(im_stack_in,down_fact,amax=amax,n=n,var=flux_var,nb_fields=nb_fields)
    shap = dec_stack.shape
    nb_snr = size(snrdB)
    output_stack = None
    sig = None
    if nb_snr==1:
        noise,sig = noise_cube(shap,snrdB,P,var=noise_var,nb_fields=nb_fields)
        output_stack = dec_stack+noise
    else:
        sig = zeros((shap[2],nb_fields,nb_snr))
        output_stack = zeros((shap[0],shap[1],shap[2],nb_fields,nb_snr))

        for i in range(0,nb_snr):
            noise_i,sig_i = noise_cube(shap,snrdB[i],P,var=noise_var,nb_fields=nb_fields)
            output_stack[:,:,:,:,i] = dec_stack+noise_i
            sig[:,:,i] = sig_i
    return output_stack,sig,dec_stack,shifts,flux

def polychrom_star_sim(poly_chrom_psf,spectrum,snr,noise_en=True,sampling=1,av_en=1):

    shap = poly_chrom_psf.shape
    output = zeros((shap[0]/sampling,shap[1]/sampling))

    if sampling>1:
        temp,psf_d = decim_arr(poly_chrom_psf,sampling,av_en=av_en,fft=1)

    for i in range(0,shap[2]):
        output+= spectrum[i]*psf_d[:,:,i]/psf_d[:,:,i].sum()

    sig = None
    if noise_en:
        sig = sqrt(((norm(output)**2)/(shap[0]*shap[1]*snr*sampling**2)))
        output += sig*randn(shap[0]/sampling,shap[1]/sampling)
    return output,sig,psf_d

def polychrom_star_field_sim(poly_chrom_psf_cube_list,snrdB,sampling=1,sampling_0=6,\
    flat_sed_en=True,spars_sed=False,nb_spikes=10,frac=0.8,amax=0.5,n=10,noise_var=0,flux_var=0,nb_comp=None,nb_fields=1):

    nb_bands = poly_chrom_psf_cube_list[0].shape[2]
    nb_psfs = len(poly_chrom_psf_cube_list)

    # Spectrum
    if flat_sed_en:
        spectrums = ones((nb_bands,nb_psfs))
    else:
        spectrums = abs(randn(nb_bands,nb_psfs))
        if spars_sed:
            for i in range(0,nb_psfs):
                spectrums[:,i] = sparsify_sed(spectrums[:,i],nb_spikes,frac)

    spectrums = diag((spectrums.sum(axis=0))**(-1)).dot(spectrums)

    # HR polychromatic PSFs
    shap = poly_chrom_psf_cube_list[0].shape
    stack_hr = zeros((shap[0]/sampling_0,shap[1]/sampling_0,nb_psfs))
    ref_hr = zeros((shap[0]/sampling_0,shap[1]/sampling_0,nb_bands,nb_psfs))
    for i in range(0,nb_psfs):
        stack_hr[:,:,i],sig,ref_hr[:,:,:,i] = polychrom_star_sim(poly_chrom_psf_cube_list[i],spectrums[:,i],0,noise_en=False,sampling=sampling_0,av_en=1)

    # Simulated observations
    output_stack,sig,dec_stack,shifts,flux = sr_data_gen(stack_hr,snrdB,sampling,amax=amax,n=n,noise_var=noise_var,flux_var=flux_var,nb_comp=nb_comp,nb_fields=nb_fields)


    return output_stack,sig,ref_hr,shifts,flux,spectrums


def sparsify_sed(spectrum,nb_spike,frac):
    spectrum_out = copy(spectrum)
    t = sum(spectrum)
    id_ref = arange(0,size(spectrum))
    samp = choice(id_ref,size=nb_spike,replace=False)
    id_ref[samp]=-1
    samp_comp = where(id_ref>=0)[0]
    spectrum_out[samp]*= t*frac/sum(spectrum_out[samp])
    spectrum_out[samp_comp]*= t*(1-frac)/sum(spectrum_out[samp_comp])
    return spectrum_out


def stack_satur(stack,perc,satur_interv=[0.2,0.8]): # This routine assumed a single field data stack is provided
    from numpy import int
    from numpy.random import rand
    shap = stack.shape
    nb_satur = int(shap[2]*perc)
    ind_satur = rand_diff_integ(0,shap[2]-1,nb_satur)
    stack_out = copy(stack)
    for i in range(0,nb_satur):
        print "im ",i+1,"/",nb_satur
        perc = (satur_interv[1]-satur_interv[0])*rand(1)+satur_interv[0]
        stack_out[:,:,ind_satur[i],0] = saturate(stack[:,:,ind_satur[i],0],perc=perc)
    return stack_out,ind_satur

def sr_data_gen_2(im_stack,snrdB,saturation_perc,down_fact,satur_interv=[0.2,0.8],amax=0.5,n=10,noise_var=0,flux_var=0,nb_comp=None,nb_fields=1):
    im_stack_in = copy(im_stack)
    if nb_comp is not None:
        im_stack_in = optim_utils.trunc_svd_cube(im_stack,nb_comp)
    dec_stack,P,shifts,flux = randshift_decim_flux(im_stack_in,down_fact,amax=amax,n=n,var=flux_var,nb_fields=nb_fields)
    shap = dec_stack.shape
    nb_satur = size(saturation_perc)
    output_stack = None
    sig = None
    ind_sat = None
    if nb_satur==1:
        noise,sig = noise_cube(shap,snrdB,P,var=noise_var,nb_fields=nb_fields)
        sat,ind_sat = stack_satur(dec_stack,saturation_perc,satur_interv=satur_interv)
        output_stack = sat+noise
    else:
        output_stack = zeros((shap[0],shap[1],shap[2],nb_fields,nb_satur))
        noise,sig = noise_cube(shap,snrdB,P,var=noise_var,nb_fields=nb_fields)
        ind_sat = list()
        for i in range(0,nb_satur):
            sat_i,ind_sat_i = stack_satur(dec_stack,saturation_perc[i],satur_interv=satur_interv)
            ind_sat.append(ind_sat_i)
            output_stack[:,:,:,:,i] = sat_i+noise

    return output_stack,sig,dec_stack,shifts,sig,flux,ind_sat



def cumul_energ_perc(s):
    E = (s**2).sum()
    out = s*0
    for i in range(0,size(s)):
        out[i] = sum(s[0:i]**2)/E
    return out

def rot_invar_correl(im1,im2):
    from numpy import conj
    pol1 = polar(im1)
    pol2 = polar(im2)
    corr = scipy_fft.ifft2(scipy_fft.fft2(pol1)*conj(scipy_fft.fft2(pol2)))
    n1 = sqrt((pol1**2).sum())
    n2 = sqrt((pol2**2).sum())
    corr_coeff = abs(corr).max()/(n1*n2)
    return corr_coeff

def shift_ker_stack(shifts,upfact,lanc_rad=4):
    from numpy import rot90
    shap = shifts.shape
    shift_ker_stack = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[0]))
    shift_ker_stack_adj = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[0]))

    for i in range(0,shap[0]):

        uin = shifts[i,:].reshape((1,2))*upfact
        shift_ker_stack[:,:,i] = lanczos(uin,n=lanc_rad)
        shift_ker_stack_adj[:,:,i] = rot90(shift_ker_stack[:,:,i],2)

    return shift_ker_stack,shift_ker_stack_adj


def laguerre_poly(x,m,q):
    # Abramovitz recurrence relation
    L0_x = 1
    L1m_x = (m+1)-x
    out = L0_x
    if q ==1:
        out = L1m_x
    elif q>1:
        out = zeros((q+1,))
        out[0] = L0_x
        out[1] = L1m_x
        for i in range(2,q+1):
            out[i] = (((2*(i-1)+m+1)-x)*out[i-1] - (i-1+m)*out[i-2])/i

    return out

def laguerre_exp_vect(siz,sig,p,q,cent=None):
    if cent==None:
        cent = array(siz)/2
    real_comp = zeros(siz)
    imag_comp = zeros(siz)
    for i in range(0,siz[0]):
        for j in range(0,siz[1]):
            r,theta = polar_coord([i,j],cent)
            a = laguerre_poly(r**2/sig**2,p-q,q)
            if q<=1:
                ain = a
            else:
                ain = a[q]
            real_comp[i,j] = ((-1)**q)/(sqrt(pi)*sig**2)*sqrt(double(factorielle(q))/factorielle(p))*((r/sig)**(p-q))*cos((p-q)*theta)*exp(-r**2/(2*sig**2))*ain
            imag_comp[i,j] = ((-1)**q)/(sqrt(pi)*sig**2)*sqrt(double(factorielle(q))/factorielle(p))*((r/sig)**(p-q))*sin((p-q)*theta)*exp(-r**2/(2*sig**2))*ain

    return real_comp,imag_comp


def laguerre_exp_dict(siz,sig_min,sig_max,nb_sc,p_max,cent=None):
    nb_el = p_max*(p_max+1)/2
    real_dict = zeros((siz[0],siz[1],nb_el,nb_sc))
    imag_dict = zeros((siz[0],siz[1],nb_el,nb_sc))
    sig = 0
    for k in range(0,nb_sc):
        sig = sig+sig_min+(sig_max-sig_min)*k/nb_sc
        for p in range(0,p_max):
            for q in range(0,p):
                nb_el_p = p*(p+1)/2
                r,im = laguerre_exp_vect(siz,sig,p,q,cent=cent)
                real_dict[:,:,nb_el_p+q,k] = r
                imag_dict[:,:,nb_el_p+q,k] = im
    dict = real_dict+1j*imag_dict
    return dict


def abs_val_reverting(w):
    max_val = abs(w).max()
    min_val = abs(w).min()
    ones_arr = ones(w.shape)
    w_out = sign(w)*((max_val+min_val)*ones_arr-abs(w))
    w_out = w_out/abs(w_out).max()
    return w_out

def vect_recond(w,cond_fact):
    ind = where(w**2<cond_fact*(w**2).max())
    ind2 = where(w**2>=cond_fact*(w**2).max())
    min_val = abs(w[ind2]).min()
    l = size(ind)
    w[ind] = min_val*ones((l,))*sign(w[ind])
    return w



"""def fourier_interpolation(im,D):

    shap = im.shape
    im_fft = scipy_fft.fft2(im)
    fft_padd = zeros((shap[0]*D,shap[1]*D))"""

def fft_stack(stack):
    out = copy(stack)
    shap = stack.shape
    for i in range(0,shap[2]):
        out[:,:,i] = scipy_fft.fft2(stack[:,:,i])
    return out

def shift_phase_map(dx,dy,shap):
    map = zeros(shap)
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            map[i,j] = exp(2*pi*1j*(double(dx*i)/shap[0]+double(dy*j)/shap[1]))
    return map

def sing_freq_mat_lr_fourier(freq1,freq2,dx,dy,lr_shap,down_fact): # The input frequencies are integers indexes in the range corresponding to the lr images size
    s = zeros((down_fact**2))
    f1 = exp(2*pi*1j/(lr_shap[0]*down_fact))
    f2 = exp(2*pi*1j/(lr_shap[1]*down_fact))
    for i in range(0,D):
        for j in range(0,D):
            s[i*D+j] = (f1**((freq1+i*lr_shap[0])*dx))*(f2**((freq2+j*lr_shap[1])*dy))
    S = diag(s)
    return S


def sing_mat_stack_lr_fourier(freq1,freq2,shift_stack,lr_shap,down_fact):
    shap = shift_stack.shape
    sing_freq_mat_stack = zeros((down_fact**2,down_fact**2,shap[0]))
    for k in range(0,shap[0]):
        sing_freq_mat_stack[:,:,k] = sing_freq_mat_lr_fourier(freq1,freq2,shift_stack[k,0],shift_stack[k,1],lr_shap,down_fact)
    return sing_freq_mat_stack

def fourier_dec_shift_transp_sing_freq(sing_freq_mat_stack,single_freq_fourier_stack,weights_stack):
    shap = sing_freq_mat_stack.shape
    nb_im = shap[2]
    D = sqrt(shap[0])
    ones_vect = zeros((D**2,1))
    g = zeros((D**2,1))
    v = zeros((D**2,nb_im))
    for k in range(0,nb_im):
        u = transpose(sing_freq_mat_stack[:,:,k]).dot(ones_vect)*single_freq_fourier_stack[k]*weights_stack[k]
        v[:,k] = u.reshape((D**2,))
        g+=copy(u)
    return g,v

def fourier_dec_shift_transp_all_freqs(shifts_stack,lr_shap,down_fact,tfourier_stack,weights_stack):
    shap = tfourier_stack.shape
    sing_freq_mat_stack = None
    G = zeros((down_fact**2,shap[0]*shap[1]))
    for freq1 in range(0,shap[0]):
        for freq2 in range(0,shap[1]):
            sing_freq_mat_stack_temp = sing_mat_stack_lr_fourier(freq1,freq2,shifts_stack,lr_shap,down_fact)
            if freq1+freq2==0:
                sing_freq_mat_stack = sing_freq_mat_stack_temp
        G[:,freq1*shap[1]+freq2] = (fourier_dec_shift_transp_sing_freq(sing_freq_mat_stack,tfourier_stack[freq1,freq2,:],weights_stack)).reshape((down_fact**2,))
    return G,sing_freq_mat_stack


def fourier_shifts_cost(G,freq0_mat_stack,weights_stack):
    shap = freq0_mat_stack.shape
    ones_vect = ones((1,shap[0]))
    kern_mat = zeros((shap[0],shap[1]))
    for k in range(0,shap[2]):
        u = ones_vect.dot(freq0_mat_stack[:,:,k])*weights_stack[0]
        kern_mat += transpose(u)*dot(u)
    w = LA.inv(kern_mat).dot(G)
    cost = real((w*G).sum())
    return w,kern_mat,cost

def convex_hull_init(X,nb_pts,proj_sphere_en=1): # Data are stored in the lines of X
    shap = X.shape
    ind = array(range(0,shap[0]))
    ret,label_ind,label,center = kmeans_interface(X,ind,nb_pts,nb_iter=10,eps=1.0,nb_attempts=5)
    centroid = X.sum(axis=0)/shap[0]
    centroid = centroid.reshape((1,shap[1]))
    if proj_sphere_en==1:
        ones_vect = ones((shap[0],1))
        centroidn = ones_vect.dot(centroid)
        dist = ((X-centroidn)**2).sum(axis=1)
        r = sqrt(dist.max())
        center = optim_utils.proj_sphere_mat(center,centroidn,r*ones_vect)
    return center,centroid,label_ind,ret

def convex_hull_init_2(X,nb_pts):
    shap = X.shape
    centroid = X.sum(axis=0)/shap[0]
    centroid = centroid.reshape((1,shap[1]))
    ones_vect = ones((shap[0],1))
    centroidn = ones_vect.dot(centroid)
    Xcent = X-centroidn
    U, s, Vt = linalg.svd(Xcent,full_matrices=False)
    coeff = Xcent.dot(transpose(Vt))
    S = zeros((shap[1],nb_pts))
    ind_list = list()
    for k in range(0,nb_pts/2):
        ind = argsort(coeff[k,:])
        ind_max = nb_pts/2-1
        while(ind[ind_max] in ind_list):
            ind_max -= 1
        ind_list.append(ind[ind_max])
        S[:,2*k] = X[ind[ind_max],:]
        ind_min = 0
        while(ind[ind_min] in ind_list):
            ind_min += 1
        ind_list.append(ind[ind_min])
        S[:,2*k+1] = X[ind[ind_min],:]
    return S,centroid

def convex_hull_init_3(X,nb_pts,nb_frac=0.1):
    shap = X.shape
    part_ind = diffusion_partionning(X,nb_pts,nb_neigh=int(nb_frac*(shap[0])))
    S = zeros((shap[1],nb_pts))
    for i in range(0,nb_pts):
        part = X[part_ind[i],:]
        S[:,i] = part.sum(axis=0)/size(part_ind[i])
    return S

def rand_oriented_vect(V,amp=None,w=None): # Directive vectors are stored in column

    shap = V.shape
    if w is None:
        w = ones((shap[1],1))
    if  amp is None:
        amp=1
    a = randn(shap[1],1)*(w.reshape((shap[1],1)))
    u = V.dot(a)
    u = amp*u/sqrt(((u)**2).sum())
    return u

def tg_rand_disp(o,x,sig,V,weights=None): # V is a set of column vector basis
    n = (o-x)/sqrt(((o-x)**2).sum())
    tgV = optim_utils.proj_dir(V,n)
    disp = rand_oriented_vect(tgV,sig,w=weights)
    return x+disp

def tg_rand_disp_set(o,X,sig,V,weights=None): # X and V are sets of column vectors and sig is a vector
    shap = X.shape
    Xdisp = copy(X)
    for k in range(0,shap[1]):
        x = X[:,k].reshape((shap[0],1))
        uk = tg_rand_disp(o,x,sig[k],V,weights=weights)
        Xdisp[:,k] = uk.reshape((shap[0],))
    return Xdisp


def norm2(u):
    normu = sqrt((u**2).sum())
    return normu

def norm1(u):
    normu = abs(u).sum()
    return normu

def mat_to_gnetworkx(mat):
    import networkx as nx
    FG=nx.Graph()

    for i in range(0,mat.shape[0]):
        for j in range(i+1,mat.shape[0]):
            FG.add_edge(i,j,weight=mat[i,j])

    return FG

def gnetworkx_to_mat(FG,inf_val=10**32):
    L = FG.number_of_nodes()
    mat = zeros((L,L))

    for i in FG.edges(data='weight'):
        if i[2]==0 and i[0]!=i[1]:
            mat[i[0],i[1]] = inf_val
            #print 'hey!!!'
        else:
            mat[i[0],i[1]] = i[2]
    mat+=transpose(mat)
    i,j = where(mat==0)
    k = where(i-j!=0)
    mat[i[k],j[k]] = inf_val
    return mat

#def minimum_sppanning_tree_interf(dist_mat):
#    nx.minimum_spanning_tree(mat_to_gnetworkx(dist_mat),weight='weight')



def floyd_alg(dist_mat):
    shap = dist_mat.shape
    nb_nodes = shap[0]*(shap[0]-1)/2
    max_dist = dist_mat.max()
    path = list()
    geod_dist_mat = copy(dist_mat)
    a = array(range(0,shap[1]))
    a = a.reshape((1,shap[1]))
    ones_vect = ones((shap[0],1))
    next_mat = ones_vect.dot(a)
    for i in range(0,nb_nodes):
        path.append(list())
    for k in range(0,shap[0]):
        for i in range(0,shap[0]):
            for j in range(0,shap[1]):
                if geod_dist_mat[i,k]+geod_dist_mat[k,j]<geod_dist_mat[i,j]:
                    geod_dist_mat[i,j] = geod_dist_mat[i,k]+geod_dist_mat[k,j]
                    next_mat[i,j] = next_mat[i,k]
    return geod_dist_mat,next_mat

def dijkstra_interf(dist_mat):
    import networkx as nx
    G = mat_to_gnetworkx(dist_mat)
    length=nx.all_pairs_dijkstra_path_length(G)
    geod_dist_mat = copy(dist_mat)*0
    shap = dist_mat.shape
    for i in range(0,shap[0]):
        for j in range(i+1,shap[0]):
            geod_dist_mat[i,j] = length[i][j]

    geod_dist_mat+=transpose(geod_dist_mat)
    return geod_dist_mat


def prim_alg(dist_mat):
    inf_val = 10**(32)
    shap = dist_mat.shape
    edge = array(range(0,shap[0]))
    min_mat = inf_val*ones((shap[0],shap[0]))
    for k in range(0,shap[0]):
        min_mat[k,k]=0
    a = edge.sum()
    used_edge = zeros((shap[0],))
    used_edge[0] = 1
    while a>0:
        id1 = where(used_edge==1)
        id2 = where(edge!=0)

        mat = dist_mat[transpose(id1),id2]
        i,j = where(mat==mat.min())
        ind1 = i[0]
        ind2 = j[0]


        min_mat[id1[0][ind1],id2[0][ind2]] = mat.min()
        min_mat[id2[0][ind2],id1[0][ind1]] = mat.min()
        edge[id2[0][ind2]] = 0
        used_edge[id2[0][ind2]] = 1
        a = edge.sum()
    return min_mat


def geodesic_distances(S,nb_neigh=2,prim_en=False,inf_val=10**32):
    import networkx as nx
    #inf_val = 10**(16)
    shap = S.shape
    dist_mat = inf_val*ones((shap[0],shap[0]))
    for k in range(0,shap[0]):
        dist_mat[k,k]=0
    dmax = inf_val
    flann = None

    while (dmax-inf_val)>=0:
        neigh,dists,flann = knn_interf(S,nb_neigh,return_index=True)
        for i in range(0,shap[0]):
            for k in range(0,nb_neigh):
                dist_mat[i,neigh[i,k]] = dists[i,k]
                dist_mat[neigh[i,k],i] = dists[i,k]

        geod_dist_mat = dijkstra_interf(dist_mat)
        dmax = geod_dist_mat.max()
        nb_neigh+=1
        print "number neigbhor: ",nb_neigh
        print "max geodesic dist: ",geod_dist_mat.max()
    if prim_en:
        #min_mat = prim_alg(dist_mat)
        min_mat = gnetworkx_to_mat(nx.minimum_spanning_tree(mat_to_gnetworkx(dist_mat),weight='weight'),inf_val=inf_val)
        print min_mat.max()
        geod_dist_mat = dijkstra_interf(min_mat)
        dmax = geod_dist_mat.max()
        print "dmax: ",dmax
        return geod_dist_mat.astype(float),nb_neigh,min_mat,dist_mat,flann
    else:
        return geod_dist_mat.astype(float),nb_neigh,dist_mat,flann

def mds(dist_matrix,nb_components=3):


    shap = dist_matrix.shape
    nb_components = min(shap[0],nb_components)
    gram_mat = gram_convert(dist_matrix**2)

    U,s,V = svd(gram_mat,full_matrices=False)

    #w,V = sci_lin.eigh(gram_mat,eigvals=(shap[0]-nb_components,shap[0]-1))

    embedding = U[:,0:nb_components].dot(diag(sqrt(s[0:nb_components])))
    return embedding,s[0:nb_components]

def new_point_embedding(new_coord_mat,geod_dist_mat,flann,w,embedding):
    result_temp, dists_temp = flann.nn_index(transpose(new_coord_mat), 2)
    ones_vect1 = ones((new_coord_mat.shape[1],1))
    ones_vect2 = ones((1,embedding.shape[1]))
    temp1 = ones_vect1.dot(ones_vect2).dot((geod_dist_mat**2)/embedding.shape[1])
    temp2 = (dists_temp[:,1]**2).reshape((new_coord_mat.shape[1],1)).dot(ones_vect2)+geod_dist_mat[result_temp[:,1],:]**2
    temp3 = transpose(embedding).dot(diag(w**(-1))/(2*embedding.shape[1]))
    new_embedding = transpose((temp1-temp2).dot(temp3))

    return new_embedding


def geodesic_distances_interf(im_stack,nb_neigh=2,prim_en=True):
    S = cube_to_mat(im_stack)
    geod_dist_mat,next_mat,nb_neigh,min_mat,dist_mat,flann = geodesic_distances(S,nb_neigh=nb_neigh,prim_en=True)
    return geod_dist_mat,next_mat,nb_neigh,min_mat,dist_mat

def get_path(next_mat,id1,id2):
    path = list()
    while id1 != id2:
        id1 = next_mat[id1,id2]
        path.append(int(id1))
    return path

def cube_vect_prod(cube,coeff_mat): # Coeff are in columns
    shap = cube.shape
    S = cube_to_mat(cube)
    out = transpose(transpose(S).dot(coeff_mat))
    out_cube = mat_to_cube(out,shap[0],shap[1])
    return out_cube

#def fourier_shifts_grad(G,freq0_mat_stack,weights_stack):

def pywt_ksig_noise(sig_map,opt='coif5',nb_scale=None,k=4):
    shap = sig_map.shape
    if nb_scale is None:
        nb_scale = int(floor(log2(min(shap))))
    var_map = sig_map**2
    wav = pywt.Wavelet(opt)
    dec_lo = wav.dec_lo
    dec_high = wav.dec_hi
    rec_lo = wav.rec_lo
    rec_high = wav.rec_hi
    dec_lo2 = copy(dec_lo)
    dec_high2 = copy(dec_high)
    rec_lo2 = copy(rec_lo)
    rec_high2 = copy(rec_high)

    for i in range(0,len(rec_lo)):
        rec_lo2[i] = rec_lo[i]**2
        rec_high2[i] = rec_high[i]**2

    for i in range(0,len(dec_lo)):
        dec_lo2[i] = dec_lo[i]**2
        dec_high2[i] = dec_high[i]**2


    filter_bank = [dec_lo2,dec_high2,rec_lo2,rec_high2]
    myWavelet = pywt.Wavelet(name="myWavelet", filter_bank=filter_bank)

    var_map_wv = pywt.wavedec2(var_map,myWavelet,mode='zpd',level=nb_scale)
    ksig_map = list()
    ksig_map.append(k*sqrt(var_map_wv[0]))
    for i in range(0,nb_scale):
        ksig_map.append([k*sqrt(var_map_wv[i+1][0]),k*sqrt(var_map_wv[i+1][1]),k*sqrt(var_map_wv[i+1][1])])

    return ksig_map

def pywt_ksig_noise_2(sig_map,opt='coif5',nb_scale=None,k=1,nb_montecarlo=1000):
    shap = sig_map.shape
    if nb_scale is None:
        nb_scale = numpy.int(floor(log2(min(shap))))
    zeros_mat = zeros((shap[0],shap[1]))
    ref_trans = pywt.wavedec2(zeros_mat,opt,mode='zpd',level=nb_scale)
    M2 = list()
    M2.append(copy(ref_trans[0]))
    Mean =list()
    Mean.append(copy(ref_trans[0]))
    for i in range(0,nb_scale):
        M2.append([copy(ref_trans[i+1][0]),copy(ref_trans[i+1][1]),copy(ref_trans[i+1][2])])
        Mean.append([copy(ref_trans[i+1][0]),copy(ref_trans[i+1][1]),copy(ref_trans[i+1][2])])



    for j in range(0,nb_montecarlo):

        randx = sig_map*numpy.random.randn(shap[0],shap[1])
        x = pywt.wavedec2(randx,opt,mode='zpd',level=nb_scale)

        trans_en = sum(x[0]**2)
        for i in range(0,nb_scale):
            trans_en+=sum(x[i+1][0])**2
            trans_en+=sum(x[i+1][1])**2
            trans_en+=sum(x[i+1][1])**2
        Delta = list()
        Delta.append(x[0]-Mean[0])

        for i in range(0,nb_scale):
            Delta.append([x[i+1][0]-Mean[i+1][0],x[i+1][1]-Mean[i+1][1],x[i+1][2]-Mean[i+1][2]])
        Mean[0]+= Delta[0]/(j+1)
        for i in range(0,nb_scale):
            Mean[i+1][0]+=Delta[i+1][0]/(j+1)
            Mean[i+1][1]+=Delta[i+1][1]/(j+1)
            Mean[i+1][2]+=Delta[i+1][2]/(j+1)
        M2[0]+=Delta[0]*(x[0]-Mean[0])
        for i in range(0,nb_scale):
            M2[i+1][0]+=Delta[i+1][0]*(x[i+1][0]-Mean[i+1][0])
            M2[i+1][1]+=Delta[i+1][1]*(x[i+1][1]-Mean[i+1][1])
            M2[i+1][2]+=Delta[i+1][2]*(x[i+1][2]-Mean[i+1][2])


    M2[0]/= nb_montecarlo-1
    for i in range(0,nb_scale):
        M2[i+1][0]/= nb_montecarlo-1
        M2[i+1][1]/= nb_montecarlo-1
        M2[i+1][2]/= nb_montecarlo-1

    M = list()
    M.append(k*sqrt(M2[0]))
    for i in range(0,nb_scale):
        M.append([k*sqrt(M2[i+1][0]),k*sqrt(M2[i+1][1]),k*sqrt(M2[i+1][2])])

    return Mean,M

def pywt_stack(im,opt='coif5',nb_scale=None):
    shap = im.shape

    if nb_scale is None:
        nb_scale = int(floor(log2(min([shap[0],shap[1]]))))
    coeff_wt = list()
    for i in range(0,shap[2]):
        coeff_wt.append(pywt.wavedec2(im[:,:,i],opt,mode='zpd',level=nb_scale))

    return coeff_wt

def pywt_ksig_noise_2_stack(sig_map,opt='coif5',nb_scale=None,k=1,nb_montecarlo=1000):
    M = list()
    shap=sig_map.shape

    for i in range(0,shap[2]):
        Mean_i,M_i = pywt_ksig_noise_2(sig_map[:,:,i],opt=opt,nb_scale=nb_scale,k=k,nb_montecarlo=nb_montecarlo)
        M.append(copy(M_i))


    return M

def pywt_thresh(wav_coeff,ksig_map,coarse_thresh=False,thresh_type=1):
    nb_scales = len(wav_coeff)
    #print len(wav_coeff),len(ksig_map)
    wav_coeff_thresh = list()
    if coarse_thresh:
        wav_coeff_thresh.append(thresholding(wav_coeff[0],ksig_map[0],thresh_type=thresh_type))
    else:
        wav_coeff_thresh.append(wav_coeff[0])
    for i in range(0,nb_scales-1):
        wav_coeff_thresh.append([thresholding(wav_coeff[i+1][0],ksig_map[i+1][0],thresh_type=thresh_type),thresholding(wav_coeff[i+1][1],ksig_map[i+1][1],thresh_type=thresh_type),thresholding(wav_coeff[i+1][2],ksig_map[i+1][2],thresh_type=thresh_type)])

    return wav_coeff_thresh

def pywt_filter(im,sig_map=None,ksig_map=None,opt='coif5',nb_scale=None,k=1,coarse_thresh=False,thresh_type=1):
    shap = im.shape
    if nb_scale is None:
        nb_scale = int(floor(log2(min(shap))))

    if ksig_map is None:
        mean_map,ksig_map = pywt_ksig_noise_2(sig_map,opt=opt,nb_scale=nb_scale,k=k)
    wav_coeff = pywt.wavedec2(im,opt,mode='zpd',level=nb_scale)
    wav_coeff_thresh = pywt_thresh(wav_coeff,ksig_map,coarse_thresh=coarse_thresh,thresh_type=thresh_type)

    im_filt = pywt.waverec2(wav_coeff_thresh, opt)

    return im_filt

def pywt_filter_stack(im,sig_map=None,ksig_map=None,opt='coif5',nb_scale=None,k=1,coarse_thresh=False,thresh_type=1):
    shap = im.shape
    im_filt = copy(im)
    for i in range(0,shap[2]):
        if ksig_map is None:
            im_filt[:,:,i] = pywt_filter(im[:,:,i],sig_map=sig_map[:,:,i],opt=opt,nb_scale=nb_scale,k=k,coarse_thresh=coarse_thresh,thresh_type=thresh_type)
        else:
            im_filt[:,:,i] = pywt_filter(im[:,:,i],ksig_map=ksig_map[i],opt=opt,nb_scale=nb_scale,k=k,coarse_thresh=coarse_thresh,thresh_type=thresh_type)
    return im_filt

def pywt_perc_thresh(im,perc,opt='coif5',thresh_type=1,nb_scale=None):
    shap = im.shape






def distance_graph_sorting(field_pos,nb_max=None):
    shap = field_pos.shape
    if nb_max is None:
        nb_max = shap[0]
    graph,dists = knn_interf(field_pos,shap[0]-1)
    list_ind = list()
    i = ind = random.randint(0, shap[0])
    list_ind.append(i)
    nb_points = 1
    while nb_points<nb_max:
        nb_points+=1
        ref_node = list()
        candidates = list()
        l = size(list_ind)
        for k in range(0,l):
            pt = 0
            flag = True
            while pt <shap[0]-1 and flag:
                if graph[list_ind[k],pt] not in list_ind:
                    flag = False
                    candidates.append(graph[list_ind[k],pt])
                    ref_node.append(list_ind[k])
                pt+=1
        p = size(candidates)
        dist_min = 1e16
        ind_opt=0
        for k in range(0,p):
            if dists[ref_node[k],candidates[k]]<dist_min:
                ind_opt = candidates[k]
                dist_min = dists[ref_node[k],candidates[k]]
        list_ind.append(ind_opt)

    return array(list_ind)

def cross_euc_distances(data,histo_en=False,bins=10): # Observation are in rows
    nb_pts = data.shape[0]
    map = zeros((nb_pts,nb_pts))
    hist_vect = zeros((((nb_pts**2+nb_pts)/2),))
    inc = 0
    for i in range(0,nb_pts):
        for j in range(i+1,nb_pts):
            map[i,j] = sqrt(((data[i,:]-data[j,:])**2).sum())
            hist_vect[inc] = map[i,j]
            inc+=1
    map = map+transpose(map)
    if histo_en:
        hist,bin_edges = histogram(hist_vect, bins=bins)
        return map,hist,bin_edges
    else:
        return map,hist_vect



def mesh_estimation(data,part_feat,emb_dim): # We assume that data rows are coefficients corresponding to principal compoents
    import copy
    nb_points = data.shape[0]
    nb_points_mesh = int(nb_points**(double(emb_dim-1)/emb_dim))
    print "Mesh size: ",nb_points_mesh
    tree = list()
    tree.append(range(0,nb_points))
    tree_split = list()
    tree_split.append(1)
    ind = range(0,nb_points)
    tree_hull = list()
    hull3d = ConvexHull(data[:,0:3])
    ind_temp = hull3d.vertices
    #hull = ConvexHull(data[ind_temp,:])
    #tree_hull.append(ind_temp[hull.vertices])
    tree_hull.append(ind_temp)
    part_en = True
    cur_nb_points = 0
    while part_en:
        cur_nb_points = 0
        new_tree = list()
        new_tree_split = list()
        new_tree_hull = list()
        l = len(tree_split)
        part_en = False
        for i in range(0,l):
            if tree_split[i]==1:
                data_in = data[tree[i],:]
                part_i = diffusion_partionning(part_feat[tree[i]],2,nb_neigh=15)
                hull3d_1 = ConvexHull(data_in[part_i[0],0:3])
                hull3d_2 = ConvexHull(data_in[part_i[1],0:3])
                if len(hull3d_1.vertices)==len(part_i[0]) or len(hull3d_2.vertices)==len(part_i[1]):
                    new_tree.append(tree[i])
                    new_tree_split.append(0)
                    new_tree_hull.append(tree_hull[i])
                    cur_nb_points+=len(tree_hull[i])
                else:
                    part_en = True
                    ind_temp = tree[i]
                    new_tree.append(array(tree[i])[part_i[0]])
                    new_tree.append(array(tree[i])[part_i[1]])
                    if len(part_i[0])/2>data.shape[1]:
                        new_tree_split.append(1)
                    else:
                        new_tree_split.append(0)
                    if len(part_i[1])/2>data.shape[1]:
                        new_tree_split.append(1)
                    else:
                        new_tree_split.append(0)
                    ind1 = array(tree[i])[part_i[0]]
                    ind2 = array(tree[i])[part_i[1]]
                    #hull_1 = ConvexHull(data[ind1[hull3d_1.vertices],0:min(len(hull3d_1.vertices)-1,data.shape[1])])
                    #hull_2 = ConvexHull(data[ind2[hull3d_2.vertices],0:min(len(hull3d_2.vertices)-1,data.shape[1])])
                    #new_tree_hull.append(ind1[hull3d_1.vertices][hull_1.vertices])
                    #new_tree_hull.append(ind2[hull3d_2.vertices][hull_2.vertices])
                    new_tree_hull.append(ind1[hull3d_1.vertices])
                    new_tree_hull.append(ind2[hull3d_2.vertices])
                    cur_nb_points = cur_nb_points+len(ind1[hull3d_1.vertices])+len(ind2[hull3d_2.vertices])
            else:
                new_tree.append(tree[i])
                new_tree_hull.append(tree_hull[i])
                new_tree_split.append(tree_split[i])
                cur_nb_points+= len(tree_hull[i])
            if cur_nb_points >nb_points_mesh:
                part_en = False
        tree = copy.deepcopy(new_tree)
        tree_hull = copy.deepcopy(new_tree_hull)
        tree_split = copy.deepcopy(new_tree_split)
        print "Number of cells: ",len(tree)," Number of points in the current hull: ",cur_nb_points

    return tree,tree_hull



# -------------------- math utils ------------------- #

def spread_func(x,a=1.e1): # Increasing function from [0,1] -> ]-Inf,+Inf[
    return a*(x-0.5)

def sigmoid(x,a=1):

    return 1.0/(1+exp(-a*x))

def stretch_sigmoid(x,a=1,b=1.e1):

    return sigmoid(spread_func(x,a=b),a=a)

def stretch_sigmoid_arr(x,a=1,b=1e1):
    nb_pts = len(x)
    out = zeros((nb_pts,))
    for i in range(0,nb_pts):
        out[i] = stretch_sigmoid(double(x[i]),a=a,b=b)

    return out


def remap_arr(u,map_func,param=None,scale_fact=None,umin=None): # remap u according to map_func, which an increasing function from [-1,1] to [-1,1]
    if scale_fact is None:
        scale_fact = 2.0/(u.max()-u.min())
    if umin is None:
        umin = u.min()
    u_rescale = scale_fact*(u-umin)-1.0
    u_remap = (map_func(u_rescale,param=param)+1.0)/scale_fact+umin
    return u_remap,scale_fact,umin

def root_n(u,param=2):
    if param is None:
        param = 2
    return sign(u)*(abs(u)**(1./param))

def inv_root_n(u,param=2):
    if param is None:
        param = 2
    return sign(u)*(abs(u)**param)



def ring_detect(psf,r_init=1):

    psf_out = copy(psf)
    i,j = where(psf==psf.max())
    lin_x = psf[i[0],:]
    lin_x_der = lin_x[0:-1]-lin_x[1:]
    lin_y = psf[:,j[0]]
    lin_y_der = lin_y[0:-1]-lin_y[1:]

    r1 = where(lin_x_der[0:j[0]]>0)
    r1 = j[0]-r1[0][-1]

    r2 = where(lin_x_der[j[0]:]<0)
    r2 = r2[0][0]

    r3 = where(lin_y_der[0:i[0]]>0)
    r3 = i[0]-r3[0][-1]

    r4 = where(lin_y_der[i[0]:]<0)
    r4 = r4[0][0]


    r = mean([r1,r2,r3,r4])
    r = r.astype(int)
    for k in range(i[0]-r,i[0]+r+1):
        for l in range(j[0]-r,j[0]+r+1):
            if sqrt((k-i[0])**2+(l-j[0])**2)<=r:
                psf_out[k,l] = 0

    return psf_out,r

def hist(x,nb_bins=100):
    vect = x.reshape((size(x),))
    hist,interv = histogram(vect,bins=nb_bins)
    return hist,interv

def gen_template(hist,interv,nb_points):

    norm_hist = (double(nb_points)/hist.sum())*hist.astype(double)
    norm_hist = norm_hist.astype(int)

    norm_hist[0] += nb_points-norm_hist.sum()
    output = zeros((nb_points))
    pt=0

    for i in range(0,len(norm_hist)):
        output[pt:pt+norm_hist[i]] = (interv[i+1]-interv[i])*array(range(0,norm_hist[i]))/(norm_hist[i]-1) + interv[i]
        pt+=norm_hist[i]

    return output



def l2_exp(x,a,r,b=None,c=None): # f(x) = x**2 if |x|<r, f(x) = exp(a*x**2)+b*x**2+c otherwise; b and c are chosen so that f is continuous and differentiable

    if b is None:
        b = 1-a*exp(a*r**2)
        c = (r**2 - 1/a)*(1-b)

    output = x*0
    if len(shape(x))==0:
        if abs(x)<r:
            output = x**2
        else:
            output = exp(a*x**2)+b*x**2+c
    else:
        il2 = where(abs(x)<r)
        iexp = where(abs(x)>=r)
        output[il2] = x[il2]**2
        output[iexp] = exp(a*x[iexp]**2)+b*x[iexp]**2+c

    return output,b,c

def log_norm(a,b,u=0.7,min_val = 1e-16):
    if a==0:
        a = min_val
    if b==0:
        b = min_val
    return log(min(abs(a/b),abs(b/a)))/log(u)


def quantizer(im,nb_levels,log_en = True):
    im_out = None
    if log_en:
        im_out = log(copy(im))
    else:
        im_out = copy(im)
    im_out /= abs(im_out).max()
    im_out = floor(nb_levels*im_out)/nb_levels
    if log_en:
        im_out = exp(im_out)
    return im_out

def distances_mat(data): # Samples are stored in the rows
    from numpy import ones,transpose
    nb_samp = data.shape[1]
    norm2_vect = ((data**2).sum(axis=0)).reshape((nb_samp,1))
    ones_vect = ones((1,nb_samp))
    norm2_mat = norm2_vect.dot(ones_vect)
    dot_prod_mat = transpose(data).dot(data)

    return norm2_mat+transpose(norm2_mat)-2*dot_prod_mat


def bar_coord2d(obs,target,cond=1./50,acc=False):
    from numpy import array,ones
    from numpy.linalg import lstsq,norm
    shap = obs.shape
    mat = ones((shap[0]+1,shap[1]))
    for i in range(0,shap[0]):
        mat[i+1,:] = obs[i,:]

    y = ones((shap[0]+1,))
    y[1:] = target
    w = lstsq(mat,y,rcond=cond)[0]
    if acc:
        return w,100*norm(mat.dot(w)-y)/norm(y)
    else:
        return w


def poly_val(x,y,deg):
    nb_monomials = int((deg+1)*(1+ double(deg)/2))
    coeff_vect = zeros((nb_monomials,))

    for i in range(0,deg+1):
        coeff_vect[i] = x**i
    count = deg+1
    for i in range(1,deg+1):
        for j in range(0,deg-i+1):
            coeff_vect[count] = (x**j)*(y**i)
            count +=1
    return coeff_vect,nb_monomials


def eval_vect_poly(x,y,s,deg):

    coeff,nb_monomials = poly_val(x,y,deg)

    return (s.dot(coeff.reshape((nb_monomials,1)))).reshape((s.shape[0],))
