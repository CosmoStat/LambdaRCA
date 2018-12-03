
import numpy as np 
import scipy.signal as scisig
import gc
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from sf_tools.image.shape import Ellipticity
import psf_toolkit as tk
import gaussfitter
from numpy import zeros,size,where,ones,copy,around,double,sinc,random,pi,arange,cos,sin,arccos,transpose,diag,sqrt,arange,floor,exp,array,mean,roots,float64,int,pi,median,rot90,argsort,tile,repeat,squeeze
from numpy.linalg import svd,norm,inv,eigh
import datetime,time
import numpy.ma as npma
from pyflann import *



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)




def transpose_decim(im,decim_fact,av_en=0):
    """ Applies the transpose of the decimation matrix."""
    shap = im.shape
    im_out = np.zeros((shap[0]*decim_fact,shap[1]*decim_fact))

    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            im_out[decim_fact*i,decim_fact*j]=im[i,j]

    if av_en==1:
        siz = decim_fact+1-(decim_fact%2)
        mask = np.ones((siz,siz))/siz**2
        im_out = scisig.fftconvolve(im, mask, mode='same')

    return im_out




def decim(im,d,av_en=1,fft=1):
    """ Decimate image to lower resolution."""

    im_filt=np.copy(im)
    im_d = np.copy(im)
    if d>1:
        if av_en==1:
            siz = d+1-(d%2)
            mask = np.ones((siz,siz))/siz**2
            if fft==1:im_filt = scisig.fftconvolve(im, mask, mode='same')
            else:im_filt = scisig.convolve(im, mask, mode='same')
        n1 = int(np.floor(im.shape[0]/d))
        n2 = int(np.floor(im.shape[1]/d))
        im_d = np.zeros((n1,n2))
        for i in range(n1):
            for j in range(n2):
                im_d[i,j] = im[i*d,j*d]
    if av_en==1:
        return (im_filt,im_d)
    else:
        return im_d



def mad(x):
    """Computes MAD.
    """
    return np.median(abs(x-np.median(x)))

def lineskthresholding(mat,k):
    """ Applies k-thresholding to each line of input matrix.
    
    Calls:
    
    * :func:`utils.kthresholding`
    
    """
    mat_out = np.copy(mat)
    shap = mat.shape
    for j in range(shap[0]):
        mat_out[j,:] = kthresholding(mat[j,:],k)
    return mat_out


def kthresholding(x,k):
    """ Applies k-thresholding (keep only k highest values, set rest to 0).
    """
    k = int(k)
    if k<1:
        print "Warning: wrong k value for k-thresholding"
        k = 1
    if k>len(x):
        return x
    else:
        xout = np.copy(x)*0
        ind = np.argsort(abs(x))
        xout[ind[-k:]] = x[ind[-k:]]
        return xout


def back_tracking_armijo_line_search(x_0, grad, f_old, cost_func, alpha=0.5, beta=0.7,tau=0.6, max_iter=15):
    # beta could be 0.1 or 0.5
    
    newCostBigger = False
    for it in range(max_iter):
        print "alpha "+str(alpha)



        # temp = x_0 - alpha*grad

        # tk.plot_func(temp[:,0,0])        


        f_new = cost_func(x_0 - alpha*grad,count=False,use_cache=True)
        
#        lim = f_old - alpha*beta*sum(np.sum(el**2) for el in grad)
        
#        print "f_new ", f_new
#        print "limiar ", lim
        
        

        if f_new <= f_old - alpha*beta*sum(np.sum(el**2) for el in grad) :
            print "cost now: "+str(f_old)+"  cost next: "+ str(f_new)
            return alpha
        alpha *= tau

    
    f_new = cost_func(x_0 - alpha*grad,count=False)  
    if f_new >= f_old:
        newCostBigger = True
        alpha *= 0.2
        print "did not converge and new cost bigger"
        print "cost now: "+str(f_old)+"  cost next: "+ str(f_new)
        print alpha
    else:
        print "did not converge but new cost smaller"
        print "cost now: "+str(f_old)+"  cost next: "+ str(f_new)
        print alpha


    return alpha



def clipped_zoom(img, zoom_factor,nb_pixels_border = 3, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
        
#        true_index = np.argwhere(out<>0.0)
#        nb_index = true_index.shape[0]
#        border_index = []
#        border_index.extend(true_index[:nb_pixels_border,:])
#        border_index.extend(true_index[(nb_index-nb_pixels_border):,:])
#        border_index = np.array(border_index)
#        
#        avg_pixel = 0.0
#        for i in range(border_index.shape[0]):
#            avg_pixel += out[tuple(border_index[i])]
#        avg_pixel /= border_index.shape[0]
        
        out_old = np.copy(out)
        true_img = out[out<>0.0]
        dime = int(np.sqrt(true_img.shape[0]))
        true_img = true_img.reshape((dime,dime))
        avg_pixel = 0.0
        for i in range(nb_pixels_border):
            for j in range(nb_pixels_border):
                avg_pixel += true_img[i,j] + true_img[-i,-j]
        avg_pixel /= nb_pixels_border*nb_pixels_border*2  
        
#        import pdb; pdb.set_trace()  # breakpoint 3d87ba10 //
        
        out[out==0.0] = avg_pixel
            
        
    
    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left+1:left+zw+1], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img

   
    
#    out[out==0.0] = 1e-8 # 1e-12 exploses the gradient, I guess 1e-9 also

    return out


def given_shift_int(x, shifts): # not tested yet
    # round up shift to integers
    y = np.zeros(x.shape)
    W,H = x.shape
    int_shifts = np.round(shifts).astype(int)
    dx, dy = int_shifts
    print int_shifts
    y[2-dx:W-dx,2-dy:H-dy] = x[2::,2::]
    return y





def shift_y_to_grid_x(y,x):    
    samp = 4.
    sig = .75 * 12 / 0.1 / samp
    Ell_y = Ellipticity(y, sigma=sig)
    Ell_x = Ellipticity(x, sigma=sig)
    theshiftiwant = Ell_x.centroid - Ell_y.centroid
    y = tk.given_shift(y, theshiftiwant)
    return y


def shift_to_center(x):
    samp = 4.
    sig = .75 * 12 / 0.1 / samp
    Ell_x = Ellipticity(x, sigma=sig)
    theshiftiwant = -Ell_x.centroid
    x = tk.given_shift(x, theshiftiwant)
    return x



def im_gauss_nois_est_cube(cube,opt=None,filters=None,return_map=False):
    """
    Estimate sigma mad for a set of images.
    
    #TODO: Note there is clearly something wrong with ``return_map`` since it fills in a ``map`` but 
    does not return it (just the boolean saying it was filled).
    
    Calls:
    
    * :func:`utils.im_gauss_nois_est`
    """
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


def im_gauss_nois_est(im,opt=['-t2','-n2'],filters=None):
    """Compute sigma mad for... What appears to be the first wavelet scale only?
    
    Calls:
    
    * isap.mr_trans_2
    """
    from isap import mr_trans_2
    Result,filters = mr_trans_2(im,filters=filters,opt=opt)
    siz = im.shape
    norm_wav = norm(filters[:,:,0])
    sigma = 1.4826*mad(Result[:,:,0])/norm_wav

    return sigma,filters



def thresholding_3D(x,thresh,thresh_type):
    """Apply thresholding to a set of images (or transport plans I guess).
    
    Calls:
    
    * :func:`utils.thresholding`
    """
    from numpy import copy
    shap = x.shape
    nb_plan = shap[2]
    k=0
    xthresh = copy(x)
    for k in range(0,nb_plan):
        xthresh[:,:,k] = thresholding(copy(x[:,:,k]),thresh[:,:,k],thresh_type)

    return xthresh



def thresholding(x,thresh,thresh_type): 
    """ Performs either soft- (``thresh_type=1``) or hard-thresholding (``thresh_type=0``). Input can be 1D or 2D array.
    """
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



def shift_est(psf_stack): 
    """Estimates shifts (see SPRITE paper, section 3.4.1., subsection 'Subpixel shifts').
    
    Calls:
    
    * gaussfitter.gaussfit
    * :func:`utils.compute_centroid`
    """
    shap = psf_stack.shape
    U = zeros((shap[2],2))
    param=gaussfitter.gaussfit(psf_stack[:,:,0],returnfitimage=False)
    #(centroid_ref,Wc) = compute_centroid(psf_stack[:,:,0],(param[3]+param[4])/2)
    centroid_out = zeros((shap[2],2))
    for i in range(0,shap[2]):
        param=gaussfitter.gaussfit(psf_stack[:,:,i],returnfitimage=False)
        (centroid,Wc) = compute_centroid(psf_stack[:,:,i],(param[3]+param[4])/2)
        U[i,0] = centroid[0,0]-double(shap[0])/2
        U[i,1] = centroid[0,1]-double(shap[1])/2
        centroid_out[i,0]  = centroid[0,0]
        centroid_out[i,1]  = centroid[0,1]
    return U,centroid_out





def compute_centroid(im,sigw=None,nb_iter=4):
    """ Computes centroid.
    
    #TODO: would be interesting to compare with Sam's moments based computation
    
    Calls:
    
    * gaussfitter.gaussfit
    """
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



def shift_ker_stack(shifts,upfact,lanc_rad=4):
    """Generate shifting kernels and rotated shifting kernels.
    
    Calls:
    
    * :func:`utils.lanczos`
    """
    from numpy import rot90
    shap = shifts.shape
    shift_ker_stack = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[0]))
    shift_ker_stack_adj = zeros((2*lanc_rad+1,2*lanc_rad+1,shap[0]))

    for i in range(0,shap[0]):

        uin = shifts[i,:].reshape((1,2))*upfact
        shift_ker_stack[:,:,i] = lanczos(uin,n=lanc_rad)
        shift_ker_stack_adj[:,:,i] = rot90(shift_ker_stack[:,:,i],2)

    return shift_ker_stack,shift_ker_stack_adj



def lanczos(U,n=10,n2=None):
    """Generate Lanczos kernel for a given shift.
    """
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


def flux_estimate_stack(stack,cent=None,rad=4):
    """Estimate flux for a bunch of images.
    
    Calls:
    
    * :func:`utils.flux_estimate`
    """
    shap = stack.shape
    flux = zeros((shap[2],))
    for i in range(0,shap[2]):
        if cent is not None:
            flux[i] = flux_estimate(stack[:,:,i],cent=cent[i,:],rad=rad)
        else:
            flux[i] = flux_estimate(stack[:,:,i],rad=rad)
    return flux


def flux_estimate(im,cent=None,rad=4): # Default value for the flux tunned for Euclid PSF at Euclid resolution
    """Estimate flux for one image (see SPRITE paper, section 3.4.1., subsection 'Photometric flux').
    """
    flux = 0
    if cent is None:
        cent = array(where(im==im.max())).reshape((1,2))
    shap = im.shape
    for i in range(0,shap[0]):
        for j in range(0,shap[1]):
            if sqrt((i-cent[0,0])**2+(j-cent[0,1])**2)<=rad:
                flux = flux+im[i,j]
    return flux



def knn_interf(data,nb_neigh,return_index=False):
    """ Computes closest neighbors. 
    
    #TODO: Probably some costly redundancy with :func:`full_displacement` here. Also is it me or is ``params`` not used?
    """
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



def feat_dist_mat(feat_mat):
    """Computes pairwise distances...?
    
    #TODO: maybe some redundancy with :func:`optim_utils.dist_map_2` here?"""
    shap = feat_mat.shape
    mat_out = zeros((shap[0],shap[0]-1))
    a = array(range(0,shap[0]))
    for i in range(0,shap[0]):
        ind_i = where(a!=i)

        for k in range(0,shap[0]-1):
            mat_out[i,k] = sqrt(sum((feat_mat[i,:]-feat_mat[ind_i[0][k],:])**2))
    return mat_out



def log_sampling(val_min,val_max,nb_samp):
    """Literally ``np.logspace`` I think.
    
    #TODO: you know.
    """
    from  numpy import log,double,array,exp
    lval_min = log(val_min)
    lval_max = log(val_max)
    a = double(array(range(0,nb_samp)))/(nb_samp-1)
    a = a*(lval_max-lval_min) + lval_min
    a = exp(a)
    return a



def kernel_mat_stack_test_unit(mat_stack,mat_test,tol=0):
    """ Computes whatever graph constraint-related quantity :func:`utils.kernel_mat_test_unit` computes
    for a set of matrices.
    
    Calls:
    
    * :func:`utils.kernel_mat_test_unit`
    """
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




def kernel_mat_test_unit(mat,mat_test,tol=0.01):
    """**[???]**
    
    Calls:
    
    * :func:`utils.kernel_ext`
    """
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



def mat_to_cube(mat,n1,n2):
    """ Literally ``np.swapaxes`` I think.
    
    #TODO
    """
    shap = mat.shape
    cube = zeros((n1,n2,shap[0]))
    k=0
    for k in range(0,shap[0]):
        cube[:,:,k] = mat[k,:].reshape(n1,n2)
    return cube



def rand_file_name(ext):
    current_time = datetime.datetime.now().time()
    return 'file'+str(time.clock())+ext


def kernel_ext(mat,tol = 0.01): 
    """ Computes input matrix's kernel, defined as the vector space spanned by the eigenvectors corresponding 
    to 1% of the sum of the squared singular values.
    
    #TODO: this is basically just the SVD, all lines between that and the ``return`` are useless.
    """
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


def cube_svd(cube,nb_comp=None,ind=None,mean_sub=False):
    """ Performs PCA as initialization.
    
    #TODO: replace with Scikit Learn version? Pretty sure this is where RCA crashes with
    too many inputs
    """
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
    U, s, Vt = np.linalg.svd(mat[:,ind],full_matrices=False)
    shap_u = U.shape
    coeff = transpose(U[:,0:nb_comp]).dot(mat)
    approx = U[:,0:nb_comp].dot(coeff)
    comp_cube = U[:,0:nb_comp].reshape((shap[0],shap[1],min(nb_comp,shap[2])))
    approx_cube =  approx.reshape((shap[0],shap[1],shap[2]))
    # if mean_sub:
    #     return coeff,comp_cube,approx_cube,data_mean,centered_data
    # else:
    #     return coeff,comp_cube,approx_cube

    return coeff

# def get_noise_arr(arr):
#     """Estimate noise for each of a set of image.
    
#     Calls:

#     * :func:`utils.cartesian_product`
#     * :func:`utils.get_noise`
#     """
#     shap = arr.shape
#     ind = list()
#     for i in shap[2:]:
#         ind.append(np.arange(0,i))
#     coord = cartesian_product(ind)
#     noise_map = np.ones(arr.shape)
#     s = slice(None) # equivalent to ':
#     for i in range(0,coord.shape[0]):
#         sig = get_noise(arr[(s,s)+tuple(coord[i,:])])
#         noise_map[(s,s)+tuple(coord[i,:])]*=sig

#     return noise_map


# def cartesian_product(arrays):
#     """**[???]***"""
#     broadcastable = np.ix_(*arrays)
#     broadcasted = np.broadcast_arrays(*broadcastable)
#     rows, cols = reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
#     out = np.empty(rows * cols, dtype=broadcasted[0].dtype)
#     start, end = 0, rows
#     for a in broadcasted:
#         out[start:end] = a.reshape(-1)
#         start, end = end, end + rows
#     return out.reshape(cols, rows).T

# def get_noise(im,nb_iter=5,k=3):
#     """ Estimate noise level for one given image through one soft thresholding iterations.
#     See SPRITE paper, appendix... I want to say A.
    
#     Calls:
    
#     * :func:`utils.mad`
#     """
#     sig = 1.4826*mad(im)
#     for i in range(0,nb_iter):
#         im_thresh = im*(abs(im)>k*sig)
#         sig = 1.4826*mad(im-im_thresh)
#     return sig


def non_unif_smoothing_mult_coeff_pos_cp_5(im,src,src_hr,tree,basis,alpha_init,\
                                            theta=0.1,p_smth_mat_inv=None,to=None,eps=0.01,nb_iter=100,tol=0.01,Ainit=None,\
                                            pos_en=False,reg_param=1000,spars_en=True, verbose=True):
    # to is the weight related to the primal variable;
    # basis is a concatenation of the optimal notch filter
    # operator eigenvectors
    shap = src.shape
    shap1 = src_hr.shape
    src_mat = zeros((shap[2],shap[2]))
    for i in range(0,shap[2]):
        for j in range(i+1,shap[2]):
            src_mat[i,j] = (src_hr[:,:,i]*src_hr[:,:,j]).sum()
    src_mat = src_mat+transpose(src_mat)
    for i in range(0,shap[2]):
        src_mat[i,i] = (src[:,:,i]**2).sum()
    U, s, Vt = svd(src_mat,full_matrices=False)
    U, s2, Vt = svd(basis.dot(transpose(basis)),full_matrices=False)
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
    if verbose:
        print "--->> ref res: <<---",rad.sum()
    ref_res = rad.sum()
    cost = 1
    cost_old=0
    for k in range(0,shap[3]):
        U, s, Vt = svd(mat[:,:,k],full_matrices=False)
        spec_rad[k] = s.max()#*(basis[:,k]**2).sum()
    spec_norm=spec_rad.sum()*s2.max()
    shapb = basis.shape
    alpha = copy(alpha_init)*0
    i=0
    t = 1
    alphax = copy(alpha)
    shap_alpha = alpha.shape
    supports = zeros((shap_alpha[0],shap_alpha[1],min(nb_iter,shapb[0])))
    while (i < min(nb_iter,shapb[0])) and (100*abs((cost-cost_old)/cost)>0.01 or cost>1.1*ref_res) :
        A = alpha.dot(basis)
        res = copy(im)
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                res[:,:,k]-=A[l,k]*src[:,:,l,k]
        if verbose:
            print " -------- mse: ",(res**2).sum(),"-----------"
        cost_old = cost
        cost = (res**2).sum()
        temp = Ainit*0
        for k in range(0,shap[3]):
            for l in range(0,shap[2]):
                temp[l,k]+=(src[:,:,l,k]*res[:,:,k]).sum()
        grad = -temp.dot(transpose(basis))
        alphay = alpha - grad/spec_norm
        alphax_old = copy(alphax)
        if spars_en:
            alphax = lineskthresholding(alphay,int(floor(sqrt(i)))+1)
            supports[:,:,i] = copy(alphax)
        else:
            alphax = copy(alphay)
        told = t
        t = (1+sqrt(4*t**2 +1))/2
        lambd = 1 + (told-1)/t
        alpha  = alphax_old + lambd*(alphax-alphax_old)
        supp = where(abs(alpha[0,:])>0)
        i+=1
    mat_out = alpha.dot(basis)
    return mat_out,alpha,supports


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
        output = inv(mat).dot(v)
    return output,mat,v


def man_inv(mat,cond=None):
    U, s, Vt = svd(mat,full_matrices=False)
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




