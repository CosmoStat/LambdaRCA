# -*- coding: utf-8 -*-

"""PROXIMITY OPERATORS

This module contains classes of proximity operators for optimisation

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 06/01/2017

"""
import sys
sys.path.append('../utilities')
import numpy as np
from noise import denoise
#from svd import svd_thresh, svd_thresh_coef
from transform import *
from optimisation import *
from psf_learning_utils import columns_wise_simplex_proj
from utils import lineskthresholding

def positivity_operator(data):
    """Positivity operator

    This method preserves only the positive coefficients of the input data

    Parameters
    ----------
    data : np.ndarray
        Input data array
    """

    return data * (data > 0)


class Positive(object):
    """Positivity proximity operator

    This class defines the positivity proximity operator

    """

    def __init__(self):
        pass

    def op(self, data, **kwargs):
        """Operator

        This method preserves only the positive coefficients of the input data

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray all positive elements from input data

        """

        return positivity_operator(data)

class Simplex(object):
    """Simplex proximity operator

    This class defines a column wise projection onto a given positive simplex

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
            return positivity_operator(data)
        else:
            return columns_wise_simplex_proj(data,mass=self.mass)

class Threshold(object):
    """Threshold proximity operator

    This class defines the threshold proximity operator

    Parameters
    ----------
    weights : np.ndarray
        Input array of weights

    """

    def __init__(self, weights):

        self.update_weights(weights)

    def update_weights(self, weights):
        """Update weights

        This method update the values of the weights

        Parameters
        ----------
        weights : np.ndarray
            Input array of weights

        """

        self.weights = weights



    def op(self, data, extra_factor=1.0):
        """Operator

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

        return denoise(data, threshold, 'soft')

class simplex_threshold(object):
    """ Simplex Threshold proximity operator
        This class stacks the proximity operators Simplex and Threshold

    """
    def __init__(self,weights,mass=None,pos_en=False):

        self.thresh = Threshold(weights)
        self.simplex = Simplex(mass=mass,pos_en=pos_en)

    def update_weights(self, weights):
        self.thresh.update_weights(weights)

    def op(self, data, extra_factor=1.0):
        return np.array([self.simplex.op(data[0]),self.thresh.op(data[1],extra_factor=extra_factor)])


class KThreshold(object):
    """ KThreshold proximity operator

    This class defines linewise hard threshold operator with variable threshold

    Parameters
    ----------
    iter_func : function
        Input function that calcultates the number of non-zero values to keep in each line at each iteration

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



class LowRankMatrix(object):
    """Low-rank proximity operator

    This class defines the low-rank proximity operator

    Parameters
    ----------
    thresh : float
        Threshold value
    treshold_type : str {'hard', 'soft'}
        Threshold type (options are 'hard' or 'soft')
    lowr_type : str {'standard', 'ngole'}
        Low-rank implementation (options are 'standard' or 'ngole')
    operator : class
        Operator class ('ngole' only)

    """

    def __init__(self, thresh, thresh_type='soft',
                 lowr_type='standard', operator=None):

        self.thresh = thresh
        self.thresh_type = thresh_type
        self.lowr_type = lowr_type

        self.operator = operator

    def op(self, data, extra_factor=1.0):
        """Operator

        This method returns the input data after the singular values have been
        thresholded

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray SVD thresholded data

        """

        # Update threshold with extra factor.
        threshold = self.thresh * extra_factor

        if self.lowr_type == 'standard':
            data_matrix = svd_thresh(cube2matrix(data), threshold,
                                     thresh_type=self.thresh_type)

        elif self.lowr_type == 'ngole':
            data_matrix = svd_thresh_coef(test_data, self.operator,
                                          threshold,
                                          thresh_type=self.thresh_type)

        new_data = matrix2cube(data_matrix, data.shape[1:])

        # Return updated data.
        return new_data


class ProximityCombo(object):
    """Proximity Combo

    This class defines a combined proximity operator

    Parameters
    ----------
    operators : list
        List of proximity operator class instances

    """

    def __init__(self, operators):

        self.operators = operators

    def op(self, data, extra_factor=1.0):
        """Operator

        This method returns the result of applying all of the proximity
        operators to the data

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray result

        """

        res = np.empty(len(self.operators), dtype=np.ndarray)

        for i in xrange(len(self.operators)):
            res[i] = self.operators[i].op(data[i], extra_factor=extra_factor)

        return res


class SubIter(object):
    """Sub iteration operator

    This class defines the sub-iteration proximity operator

    Parameters
    ----------
    data_shape : tuple
        Shape of input data array
    operator : class
        Proximity operator class
    weights : np.ndarray
        Array of weights
    u_init : np.ndarray
        Initial estimate of u

    """

    def __init__(self, data_shape, operator, weights=None, u_init=None):

        self.operator = operator

        if not isinstance(weights, type(None)):
            self.weights = weights

        if isinstance(u_init, type(None)):
            self.u = np.ones(data_shape)

        self.opt = ForwardBackward(self.u, self.operator,
                                   Threshold(self.weights), auto_iterate=False,
                                   indent_level=2)

    def update_weights(self, weights):
        """Update weights

        This method updates the values of the weights

        Parameters
        ----------
        weights : np.ndarray
            Array of weights

        """

        self.weights = weights

    def update_u(self):
        """Update u

        This method updates the values of u

        """

        self.opt.iterate(100)
        self.u = self.opt.x_final

    def op(self, data):
        """Operator

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray result

        """

        self.update_u()

        new_data = data - self.operator.adj_op(self.u)

        return new_data
