
import numpy as np
from modopt.signal.positivity import positive
from modopt.signal.noise import thresh
import utils
import scipy
import psf_toolkit as tk
import psf_learning_utils as psflu
# from modopt.opt.proximity import SparseThreshold



def  euclidean_proj_simplex(v, s=1):
        """
        Module to compute projections on the positive simplex

        A positive simplex is a set X = { \mathbf{x} | \sum_i x_i = s, x_i \geq 0 }

        Adrien Gaidon - INRIA - 2011

        Compute the Euclidean projection on a positive simplex

        Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 

        Parameters
        ----------
        v: (n,) numpy array,
        n-dimensional vector to project

        s: int, optional, default: 1,
        radius of the simplex

        Returns
        -------
        w: (n,) numpy array,
        Euclidean projection of v on the simplex

        Notes
        -----
        The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
        Better alternatives exist for high-dimensional sparse vectors (cf. [1])
        However, this implementation still easily scales to millions of dimensions.

        References
        ----------
        [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
        """

        assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
        n, = v.shape  # will raise ValueError if v is not 1-D
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            # best projection: itself!
            return v
        # get the array of cumulative sums of a sorted (decreasing) copy of v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of > 0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - s) / (rho + 1.0)
        # compute the projection by thresholding v using theta
        w = (v - theta).clip(min=0)
        return w

class Simplex(object):
    """Simplex proximity operator

    This class defines a column wise projection onto a given positive simplex

    """

    def __init__(self,mass=1,pos_en=False):
        self.pos_en = pos_en
        self.mass = mass
        self.cost = self.op



    def op(self,data,**kwargs):
        

        temp = self.columns_wise_simplex_proj(data,mass=1) 
        temp[temp==0] = 1e-8 # avoid true zeros
        

        return temp  
    
    # debug
    def op_HOLD(self,data,**kwargs):
        
        print "simplex modif"
        temp = np.copy(data)
#        temp[temp <= 0.0 ] = 1e-10 # avoid true zeros
#        temp = abs(temp)
        temp[temp < 0.0] = abs(temp[temp < 0.0])/1e3
        temp /= np.sum(temp,axis=0)
        

        return temp  


    def columns_wise_simplex_proj(self,mat,mass=None):

        # from simplex_projection import euclidean_proj_simplex
        nb_columns = mat.shape[2]
        nb_atoms = mat.shape[1]
        mat_out = np.zeros(mat.shape)


        if mass is None: #yes
            mass = np.max(0,((mat*(mat>=0)).sum(axis=0)).mean())
        if mass>0:
            for i in range(nb_columns):
                for j in range(nb_atoms):
                    mat_out[:,j,i] = euclidean_proj_simplex(mat[:,j,i],s=mass)

        return mat_out


class LowPass(object):
    def __init__(self,shap,thresh_type='soft',logit=False):
      self.shap = shap
      self.thresh_type = thresh_type
      self.logit = logit
      self.cost = self.op
      
    def op(self,data,extra_factor_LP=1.0,**kwargs):
        data_ori = np.copy(data)
        data = data.reshape((self.shap[0], self.shap[1],2,data.shape[-1]))
        if self.logit:
            data = np.exp(data)/np.sum(np.exp(data), axis = 0)        
        
        fft_data = np.zeros((self.shap[0],self.shap[1],2,data.shape[-1]),dtype='complex128')

        for i in range(data.shape[-1]):
            for at in range(2):
#                temp = scipy.fftpack.fft2(data[:,:,at,i])
                fft_data[:,:,at,i] = scipy.fftpack.fft2(data[:,:,at,i])#np.copy(temp)
           
            
        # Computation of thresholf        
        K = (1.0/(np.prod(self.shap)))*0.5
        norm_small = np.sum(abs(fft_data[:,:,1,:]), axis=(0,1))
        threshold = np.ones((self.shap[0],self.shap[1],data.shape[-1]),dtype='complex128')*K*norm_small*extra_factor_LP
        
        
        
        # Thresholding
        fft_data_thresh  = thresh(fft_data[:,:,0,:], threshold, self.thresh_type)
            
        # Inverse Fourier linear transform
        data_rec = np.zeros(data.shape)
        for i in range(data.shape[-1]):     
            data_rec[:,:,0,i] = np.real(scipy.fftpack.ifft2(fft_data_thresh[:,:,i]))
        
        
        data_rec = data_rec.reshape((self.shap[0]*self.shap[1],2,data.shape[-1]))
        data_rec[:,1,:] = np.copy(data_ori[:,1,:])
        
        # Get the energy back to original 
#        data_rec[:,0,:] = (data_rec[:,0,:]/np.sum(abs(data_rec[:,0,:]), axis=0))*np.sum(abs(data_ori[:,0,:]),axis=0)
                
        if self.logit:
            data_rec = np.log(abs(data_rec))
        
        
        
        
        return   data_rec          


class SparseThreshold(object):

    """Performs sparse threshold in the transform domain.(wavelet domain)
        
    """

    def __init__(self, weights, linear_transf,shap, nsigma_dict,thresh_type='soft',logit=False):
        
        self.linear_transf = linear_transf
        self.weights = weights # assume that they are in the form <nb_filters(exclusing coarse scale),pixels_x,pixels_y,nb_atoms,nb_comp>
        self._thresh_type = thresh_type
        self.cost = self.op
        self.logit = logit
        self.shape = shap
        self.nsigma_dict = nsigma_dict
        
        
    def autoUpdate_weights(self,data):
        self.weights = psflu.get_noise_arr_dict_wvl(data,self.shape)*self.nsigma_dict

    def op(self, data, extra_factor=1.0,**kwargs):
        """Operator Method

        This method returns the input data thresholded by the weights

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray thresholded data

        """
        print "SPARSITY"



        if self.logit:
            data = np.exp(data)/np.sum(np.exp(data), axis = 0)
            
        
        self.autoUpdate_weights(data)

        threshold = self.weights * extra_factor
            
            
        transf_data = self.linear_transf.op(data)


        thresholdable_data = transf_data[:-1,:] #exclude coarse scale <nb_filters(exclusing coarse scale),pixels_x,pixels_y,nb_atoms,nb_comp>


        print "threshold "+ str(np.min(threshold))        
        print "min value in starlet "+ str(np.min(abs(thresholdable_data)))


        # res = thresholdable_data
        res = thresh(thresholdable_data, threshold, self._thresh_type)



        # Put the coarse scale back
        transf_data_thresh = np.zeros(transf_data.shape)
        transf_data_thresh[-1,:] = transf_data[-1,:]
        for i in range(transf_data.shape[0]-1):
            transf_data_thresh[i,:] = res[i,:]



        recovered_data = self.linear_transf.adj_op(transf_data_thresh)

        if self.logit:
            recovered_data = np.log(abs(recovered_data))


        


        return recovered_data


    def update_weights(self,weights):
        

        self.weights = weights




class SparseThreshold_coeff(object):

    """Performs sparse threshold in coefficients.
        
    """

    def __init__(self, weights, thresh_type='hard'):
        self.weights = weights # assume that they are in the form <nb_filters(exclusing coarse scale),pixels_x,pixels_y,nb_atoms,nb_comp>
        self._thresh_type = thresh_type
        self.cost = self.op


    def op(self, data, extra_factor=1.0,**kwargs):
        """Operator Method

        This method returns the input data thresholded by the weights

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray thresholded data

        """

        threshold = self.weights * extra_factor


        res = thresh(data,threshold,self._thresh_type)

        return res



    def update_weights(self,weights):

        self.weights = weights


class KThreshold(object):
    """ KThreshold proximity operator

    This class defines linewise hard threshold operator with variable threshold

    Parameters
    ----------
    iter_func : function
        Input function that calcultates the number of non-zero values to keep in each line at each iteration
        
    Calls:
    
    * :func:`utils.lineskthresholding`

    """
    def __init__(self, iter_func):

        self.iter_func = iter_func
        self.iter = 0
        self.cost = self.op

    def reset_iter(self):
        """Reset iter

        This method sets the iterations counter to zero

        """
        self.iter = 0


    def op(self, data, extra_factor=1.0,**kwargs):
        """Operator

        This method returns the input data thresholded

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray thresholded data

        """


        self.iter += 1

        return utils.lineskthresholding(data,self.iter_func(self.iter))


class Positive(object):


    def __init__(self,transform):
        self.transform = transform
        self.cost = self.op


    def op(self,data,**kwargs):

        data_transf = self.transform.op(data)


        data_proj = positive(data_transf)



        data_rec = self.transform.adj_op(data_proj)

        return data_rec








