# -*- coding: utf-8 -*-

"""LINEAR OPERATORS

This module contains linear operator classes.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 04/01/2017

"""
import sys
sys.path.append('../utilities')
import numpy as np
from wavelet import *
import sys
sys.path.append('../functions')
from matrix import rotate
from signal import *
from psf_learning_utils import transport_plan_projections_flat_field,transport_plan_projections_flat_field_transpose,\
                                transport_plan_projections_flat_field_transpose_coeff, transport_plan_projections_field_marg,\
                                transport_plan_projections_field_marg_transpose


class Identity(object):
    """Identity operator class

    This is a dummy class that can be used in the optimisation classes

    """

    def __init__(self):

        self.l1norm = 1.0

    def op(self, data, **kwargs):
        """Operator

        Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array
        **kwargs
            Arbitrary keyword arguments

        Returns
        -------
        np.ndarray input data

        """

        return data

    def adj_op(self, data):
        """Adjoint operator

        Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray input data

        """

        return data


class Wavelet(object):
    """Wavelet class

    This class defines the wavelet transform operators

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally an array of 2D images
    wavelet_opt: str, optional
        Additional options for `mr_transform`

    """

    def __init__(self, data, wavelet_opt=None):
        self.y = data
        self.data_shape = shap
        n = data.shape[0]
        self.filters = get_mr_filters(self.data_shape, opt=wavelet_opt)
        self.l1norm = n * np.sqrt(sum((np.sum(np.abs(filter)) ** 2 for
                                       filter in self.filters)))


    def op(self, data):
        """Operator

        This method returns the input data convolved with the wavelet filters

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 2D image

        Returns
        -------
        np.ndarray wavelet convolved data

        """

        return filter_convolve_stack(data, self.filters)

    def adj_op(self, data):
        """Adjoint operator

        This method returns the input data convolved with the wavelet filters
        rotated by 180 degrees

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 3D of wavelet coefficients

        Returns
        -------
        np.ndarray wavelet convolved data

        """

        return filter_convolve_stack(data, self.filters, filter_rot=True)

class Wavelet_bis(object):
    """Wavelet class

    This class defines the wavelet transform operators

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally an array of 2D images
    wavelet_opt: str, optional
        Additional options for `mr_transform`

    """

    def __init__(self, shap, wavelet_opt=None):
        self.data_shape = shap
        #n = shap[-1]
        self.filters = get_mr_filters(self.data_shape, opt=wavelet_opt)
        self.l1norm = np.sqrt(sum((np.sum(np.abs(filter)) ** 2 for
                                       filter in self.filters)))

    def op(self, data):
        """Operator

        This method returns the input data convolved with the wavelet filters

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 2D image

        Returns
        -------
        np.ndarray wavelet convolved data

        """

        return filter_convolve_stack(data, self.filters)

    def adj_op(self, data):
        """Adjoint operator

        This method returns the input data convolved with the wavelet filters
        rotated by 180 degrees

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 3D of wavelet coefficients

        Returns
        -------
        np.ndarray wavelet convolved data

        """

        return filter_convolve_stack(data, self.filters, filter_rot=True)



class LinearCombo(object):
    """Linear combination class

    This class defines a combination of linear transform operators

    Parameters
    ----------
    operators : list
        List of linear operator class instances
    weights : list, optional
        List of weights for combining the linear adjoint operator results

    """

    def __init__(self, operators, weights=None):

        self.operators = operators
        self.weights = weights
        self.l1norm = np.array([operator.l1norm for operator in
                                self.operators])

    def op(self, data):
        """Operator

        This method returns the input data operated on by all of the operators

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 2D image

        Returns
        -------
        np.ndarray linear operation results

        """

        res = np.empty(len(self.operators), dtype=np.ndarray)

        for i in xrange(len(self.operators)):
            res[i] = self.operators[i].op(data)

        return res

    def adj_op(self, data):
        """Adjoint operator

        This method returns the combination of the result of all of the
        adjoint operators. If weights are provided the comibination is the sum
        of the weighted results, otherwise the combination is the mean.

        Parameters
        ----------
        data : np.ndarray
            Input data array, an array of coefficients

        Returns
        -------
        np.ndarray adjoint operation results

        """

        if isinstance(self.weights, type(None)):

            return np.mean([operator.adj_op(x) for x, operator in
                           zip(data, self.operators)], axis=0)

        else:

            return np.sum([weight * operator.adj_op(x) for x, operator,
                          weight in zip(data, self.operators, weights)],
                          axis=0)


class transport_plan_lin_comb(object):
    """transport plan combo class

    This class defines transport plans linear combination operator and its transpose

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally a cube of 2D coupling matrices

    """

    def __init__(self,A,supp,shap):

        self.A = np.copy(A)
        self.supp = supp
        self.shape = shap
        self.mat_norm = np.linalg.svd(self.A, full_matrices=1, compute_uv=0)[0]

    def set_A(self,A_new):
        self.A = np.copy(A_new)
        self.mat_norm = np.linalg.svd(self.A, full_matrices=1, compute_uv=0)[0]

    def op(self, data):
        """Operator

        This method returns linear combinations of the slices of the input cube
        on the support, following the mixing matrix A

        Parameters
        ----------
        data : np.ndarray
            Input data array, a cube of 2D coupling matrices

        Returns
        -------
        np.ndarray

        """

        return transport_plan_projections_flat_field(data,self.supp,self.A)

    def adj_op(self, data):
        """Adjoint operator

        This method returns a coupling matrices cube

        Parameters
        ----------
        data : np.ndarray
            Input data array, a matrix

        Returns
        -------
        np.ndarray cube of coupling matrices

        """

        return transport_plan_projections_flat_field_transpose(data,self.supp,self.A,self.shape)

class transport_plan_lin_comb_coeff(object):
    """transport plan combo class

    This class defines transport plans linear combination operator and its (coefficients related) transpose

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally a cube of 2D coupling matrices

    """

    def __init__(self, P_stack, supp):

        self.P_stack = np.copy(P_stack)
        self.supp = supp
        self.mat_norm = np.linalg.svd(self.P_stack[self.supp[:,0],self.supp[:,1],:], full_matrices=1, compute_uv=0)[0]

    def set_P_stack(self,P_stack_new):
        self.P_stack = np.copy(P_stack_new)
        self.mat_norm = np.linalg.svd(self.P_stack[self.supp[:,0],self.supp[:,1],:], full_matrices=1, compute_uv=0)[0]

    def op(self, data):
        """Operator

        This method returns linear combinations of the slices of the input cube
        on the support, following the mixing matrix A

        Parameters
        ----------
        data : np.ndarray
            Input data array, a cube of 2D coupling matrices

        Returns
        -------
        np.ndarray

        """

        return transport_plan_projections_flat_field(self.P_stack,self.supp,data)

    def adj_op(self, data):
        """Adjoint operator

        This method returns a coupling matrices cube

        Parameters
        ----------
        data : np.ndarray
            Input data array, a matrix

        Returns
        -------
        np.ndarray cube of coupling matrices

        """

        return transport_plan_projections_flat_field_transpose_coeff(data,self.P_stack,self.supp)


class transport_plan_marg_wavelet(object):
    """transport_plan_marg_wavelet class

    This class defines an operator which performs a wavelet transform of a transpose plan first marginal and its transpose

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally a cube of 2D coupling matrices

    """

    def __init__(self,supp,weights_neighbors,neighbors_graph,shap,wavelet_opt=None):

        self.supp = supp
        self.weights_neighbors = weights_neighbors
        self.neighbors_graph = neighbors_graph
        self.shape = shap
        self.wav = Wavelet_bis(shap,wavelet_opt=wavelet_opt)
        self.mat_norm = np.sqrt(shap[0]*shap[1])*self.wav.l1norm

    def op(self, data):
        """Operator

        This method returns the wavelets coefficients of the first marginals of the coupling matrices given as entry

        Parameters
        ----------
        data : np.ndarray
            Input data array, a cube of 2D coupling matrices

        Returns
        -------
        np.ndarray

        """

        return self.wav.op(transport_plan_projections_field_marg(data,self.shape,\
                self.supp,self.neighbors_graph,self.weights_neighbors))

    def adj_op(self, data):
        """Adjoint operator

        This method returns a coupling matrices cube

        Parameters
        ----------
        data : np.ndarray
            Input data array, a matrix

        Returns
        -------
        np.ndarray cube of coupling matrices

        """



        return transport_plan_projections_field_marg_transpose\
                (self.wav.adj_op(data),self.shape,self.supp,self.neighbors_graph,self.weights_neighbors)


class transport_plan_lin_comb_wavelet(object):
    """transport_plan_lin_comb_wavelet class

    This class stacks the operators from transport_plan_marg_wavelet and transport_plan_lin_comb

    """

    def __init__(self,A,supp,weights_neighbors,neighbors_graph,shap,wavelet_opt=None):
        self.lin_comb = transport_plan_lin_comb(A, supp,shap)
        self.marg_wvl = transport_plan_marg_wavelet(supp,weights_neighbors,neighbors_graph,shap,wavelet_opt=wavelet_opt)
        self.mat_norm = np.sqrt(self.lin_comb.mat_norm**2+self.marg_wvl.mat_norm**2)

    def set_A(self,A_new):
        self.lin_comb.set_A(A_new)

    def op(self, data):
        return np.array([self.lin_comb.op(data),self.marg_wvl.op(data)])

    def adj_op(self, data):
        return self.lin_comb.adj_op(data[0])+self.marg_wvl.adj_op(data[1])
