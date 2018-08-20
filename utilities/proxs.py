from modopt.signal.positivity import positive
from utils import lineskthresholding
from psf_learning_utils import columns_wise_simplex_proj,columns_wise_simplex_proj_wdl
from modopt.opt.proximity import SparseThreshold
import numpy as np


class KThreshold(object):
    """ KThreshold proximity operator

    This class defines linewise hard threshold operator with variable threshold

    Parameters
    ----------
    iter_func : function
        Input function that calcultates the number of non-zero values to keep in each line at each iteration
        
    CALLS:
    
    * :func:`utils.lineskthresholding`

    """
    def __init__(self, iter_func):

        self.iter_func = iter_func
        self.iter = 0

    def reset_iter(self):
        """Reset iter

        This method sets the iterations counter to zero

        """
        self.iter = 0


    def op(self, data, extra_factor=1.0):
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

        return lineskthresholding(data,self.iter_func(self.iter))
        
        
class Simplex(object):
    """Simplex proximity operator

    This class defines a column wise projection onto a given positive simplex

    CALLS:
    
    * :func:`psf_learning_utils.columns_wise_simplex_proj`
    """

    def __init__(self,mass=1,pos_en=False):
        self.pos_en = pos_en
        self.mass = mass

    def op(self, data, **kwargs):
        """Operator

        This method projects each column of the matrix data onto the simplex
        sum_i v_i = mass, v_i>=0

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray all positive elements from input data

        """

        if self.pos_en:
            return positive(data)
        else:#yes
            return columns_wise_simplex_proj(data,mass=self.mass)


    def op_wdl(self, data, **kwargs):
        if self.pos_en:
            return positive(data)
        else:#yes
            return columns_wise_simplex_proj_wdl(data,mass=self.mass)       
            
class simplex_threshold(object):
    """ Simplex Threshold proximity operator
        This class stacks the proximity operators Simplex and Threshold

    """
    def __init__(self,linop, weights,mass=None,pos_en=False):
        self.linop = linop
        self.weights = weights
        self.thresh = SparseThreshold(self.linop, self.weights)
        self.simplex = Simplex(mass=mass,pos_en=pos_en)

    def update_weights(self, weights):
        """Update weights

        This method update the values of the weights

        Parameters
        ----------
        weights : np.ndarray
            Input array of weights

        """
        self.weights = weights
        self.thresh = SparseThreshold(self.linop, weights)

    def op(self, data, extra_factor=1.0):

        return np.array([self.simplex.op(data[0]),self.thresh.op(data[1],extra_factor=extra_factor)]) #data[0] is the dual transport plan important advection points
