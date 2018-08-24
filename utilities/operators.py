import numpy as np
from psf_learning_utils import transport_plan_projections_flat_field,transport_plan_projections_flat_field_transpose,\
                                transport_plan_projections_flat_field_transpose_coeff, transport_plan_projections_field_marg,\
                                transport_plan_projections_field_marg_transpose,transport_plan_projections_field_marg_wdl,transport_plan_projections_flat_field_wdl,\
                                transport_plan_projections_flat_field_transpose_wdl
from modopt.signal.wavelet import get_mr_filters, filter_convolve_stack


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


    def op_wdl(self,data):
        return transport_plan_projections_flat_field_wdl(data,self.A)


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

    def adj_op_wdl(self, data):
        """get back D_stack
        """
        return transport_plan_projections_flat_field_transpose_wdl(data,self.A)


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

    def __init__(self,polychrom_grad,supp,weights_neighbors,neighbors_graph,shap,w_stack,C,gamma,n_iter_sink,wavelet_opt=None, method='scipy'):
        self.grad = polychrom_grad
        self.supp = supp
        self.weights_neighbors = weights_neighbors
        self.neighbors_graph = neighbors_graph
        self.shape = shap
        self.filters = get_mr_filters(self.shape, opt=wavelet_opt)
        self.l1norm = np.sqrt(sum((np.sum(np.abs(filt)) ** 2 for
                                       filt in self.filters)))
        self.mat_norm = np.sqrt(shap[0]*shap[1])*self.l1norm
        self.method = method
        self.w_stack = w_stack
        self.C = C
        self.gamma = gamma
        self.n_iter_sink = n_iter_sink

    def op_old(self, data):
        """Operator

        This method returns the wavelets coefficients of the first marginals of the coupling matrices given as entry. What does he mean by the first marginals??

        Parameters
        ----------
        data : np.ndarray
            Input data array, a cube of 2D coupling matrices

        Returns
        -------
        np.ndarray
        im_proj <42,42,5> image projection in the first wvl for every component (not multiplied by SED) in the starlet domain

        """

        return filter_convolve_stack(transport_plan_projections_field_marg(data,self.shape,self.supp,self.neighbors_graph,self.weights_neighbors),\
         self.filters, method=self.method)


    def op_wdl(self,data):

    
        temp = transport_plan_projections_field_marg_wdl(data,self.shape,self.w_stack,self.gamma,self.C,self.n_iter_sink)#go step


        
        return filter_convolve_stack(transport_plan_projections_field_marg_wdl(data,self.shape,self.w_stack,self.gamma,self.C,self.n_iter_sink),\
            self.filters, method=self.method)

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


        return transport_plan_projections_field_marg_transpose(filter_convolve_stack(data, self.filters, filter_rot=True, method=self.method),
                                                        self.shape,self.supp,self.neighbors_graph,self.weights_neighbors)

    def adj_op_wdl(self, data):
        # returns atoms used to compute data

        return self.grad.get_atoms()


class transport_plan_lin_comb_wavelet(object):
    """transport_plan_lin_comb_wavelet class

    This class stacks the operators from transport_plan_marg_wavelet and transport_plan_lin_comb

    """

    def __init__(self,polychrom_grad,A,supp,weights_neighbors,neighbors_graph,shap,w_stack,C,gamma,n_iter_sink,wavelet_opt=None):
        self.lin_comb = transport_plan_lin_comb(A, supp,shap)
        self.marg_wvl = transport_plan_marg_wavelet(polychrom_grad,supp,weights_neighbors,neighbors_graph,shap,w_stack,C,gamma,n_iter_sink,wavelet_opt=wavelet_opt)
        self.mat_norm = np.sqrt(self.lin_comb.mat_norm**2+self.marg_wvl.mat_norm**2)

    def set_A(self,A_new):
        self.lin_comb.set_A(A_new) 

    def op_old(self, data):
        return np.array([self.lin_comb.op(data),self.marg_wvl.op(data)])

    def op(self,data):

        return np.array([self.lin_comb.op_wdl(data),self.marg_wvl.op_wdl(data)])

    def adj_op_old(self, data):
        return self.lin_comb.adj_op(data[0])+self.marg_wvl.adj_op(data[1])

    def adj_op(self, data):

        return self.lin_comb.adj_op_wdl(data[0])+self.marg_wvl.adj_op_wdl(data[1])
