import numpy as np
from psf_learning_utils import transport_plan_projections_field,transport_plan_projections_field_transpose,\
transport_plan_projections_field_coeff_transpose,transport_plan_projections_field_transpose_wdl
from modopt.opt.gradient import GradParent, GradBasic
from modopt.math.matrix import PowerMethod
        
        
class polychrom_eigen_psf(GradParent, PowerMethod):
    """Polychromatic eigen PSFs class

    This class defines the operators for a field of undersampled space varying
    polychromatic PSFs. These operators are related to the eigen PSFs estimation.

    Parameters
    ----------
    data : np.ndarray
        Input data array, a array of 2D observed images (i.e. with noise)


    Notes
    -----
    The properties of `GradBasic` and `PowerMethod` are inherited in this class
    
    Calls:
    
    * :func:`psf_learning_utils.transport_plan_projections_field`
    * :func:`psf_learning_utils.transport_plan_projections_field_transpose`
    

    """

    def __init__(self, data, supp, neighbors_graph, weights_neighbors, spectrums, \
                A, flux, sig, ker, ker_rot,D_stack,w_stack,C,gamma,n_iter_sink,D, data_type=float):

        #polychrom_grad

        self._grad_data_type = data_type
        self.obs_data = data
        self.op = self.MX 
        self.trans_op = self.MtX 
        shap = data.shape
        self.shape = (shap[0]*D,shap[1]*D)
        self.D = D
        self.supp = supp
        self.neighbors_graph = neighbors_graph
        self.weights_neighbors = weights_neighbors
        self.spectrums = spectrums
        self.A = np.copy(A)
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot  = ker_rot
        self.D_stack = D_stack
        self.w_stack = w_stack
        self.C = C
        self.gamma = gamma
        self.n_iter_sink = n_iter_sink
     
        PowerMethod.__init__(self, self.trans_op_op, (np.prod(self.shape),np.prod(self.shape),A.shape[0]))
        print " > SPECTRAL RADIUS:\t{}".format(self.spec_rad)
        
        self._current_rec = None # stores latest application of self.MX
        self._current_rec_wdl = None # stores latest application of self.MX_wdl (that includes MX and MtX)

    def set_A(self,A_new,pwr_en=True):
        self.A = np.copy(A_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op_op, (np.prod(self.shape),np.prod(self.shape),self.A.shape[0]))

    def set_flux(self,flux_new,pwr_en=False):
        self.flux = np.copy(flux_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op_op, (np.prod(self.shape),np.prod(self.shape),self.A.shape[0]))

    def get_flux(self):
        return self.flux

    def MX(self, x):
        """MX

        This method calculates the action of the matrix M on the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array, a cube of transportation plans

        Returns
        -------
        np.ndarray result

        """
        self._current_rec = transport_plan_projections_field(x,self.shape,self.supp,self.neighbors_graph\
                ,self.weights_neighbors,self.spectrums,self.A,self.flux,self.sig,self.ker,self.D)
        return self._current_rec

    def MtX_wdl(self,x):
        """
            x: input data array, a cube of 2D images
        """
        self._current_rec_wdl = transport_plan_projections_field_transpose_wdl(x,self.shape,self.A, self.flux, self.sig, self.ker,self.spectrums, self.D_stack,self.w_stack,self.C,self.gamma,self.n_iter_sink)
        #[Mx_stack,Mtx,barys_stack]

        return self._current_rec_wdl[1]


    def MX_wdl(self,x):
        #Do the if self._current_rec_wdl==None thing after finding out wich one is called first in the Condat iterations, MX or MtX        
        self._current_rec_wdl = transport_plan_projections_field_transpose_wdl(self.obs_data,self.shape,self.A, self.flux, self.sig, self.ker,self.spectrums, x,self.w_stack,self.C,self.gamma,self.n_iter_sink)

        return self._current_rec_wdl[0]

    def MtX(self, x):
        """MtX

        This method calculates the action of the transpose of the matrix Mt on
        the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array, a cube of 2D images

        Returns
        -------
        np.ndarray result

        """
        return transport_plan_projections_field_transpose(x,self.supp,self.neighbors_graph,\
                self.weights_neighbors,self.spectrums,self.A,self.flux,self.sig,self.ker_rot,self.D)
                
    def cost(self, x, y=None, verbose=False):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
        the dual variable.)
        """
        if isinstance(self._current_rec, type(None)):
            self._current_rec = self.MX(x)
        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.obs_data) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)
        return cost_val
                
    def get_grad(self, x):
        """Get the gradient step
        This method calculates the gradient step from the input data
        Parameters
        ----------
        x : np.ndarray
            Input data array
            current transport plans
        Returns
        -------
        np.ndarray gradient value
        Notes
        -----
        Calculates M^T (MX - Y)
        """

        self.grad = self.MtX(self.MX(x) - self.obs_data)


    def get_grad_wdl(self,x):
        self.D_stack = x 
        self.grad = self.MtX_wdl(self,x)

    def get_atoms(self):
        return self.D_stack

    def get_current_grad(self):
        return self.grad

class polychrom_eigen_psf_coeff(GradBasic, PowerMethod):
    """Polychromatic eigen PSFs class

    This class defines the operators for a field of undersampled space varying
    polychromatic PSFs. These operators are related to the estimation of weights,
    when there is no spatial constraints.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array, a array of 2D observed images (i.e. with noise)


    Notes
    -----
    The properties of `GradBasic` and `PowerMethod` are inherited in this class
    
    Calls:
    
    * :func:`psf_learning_utils.transport_plan_projections_field`
    * :func:`psf_learning_utils.transport_plan_projections_field_coeff_transpose`

    """
    
        
    def __init__(self, data, supp, neighbors_graph, weights_neighbors, spectrums, \
                P, flux, sig, ker, ker_rot, D, data_type=float):

        self._grad_data_type = data_type
        self.obs_data = data
        self.op = self.MX 
        self.trans_op = self.MtX 
        shap = data.shape
        self.shape = (shap[0]*D,shap[1]*D)
        self.D = D
        self.supp = supp
        self.neighbors_graph = neighbors_graph
        self.weights_neighbors = weights_neighbors
        self.spectrums = spectrums
        self.P = np.copy(P)
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot  = ker_rot
        PowerMethod.__init__(self, self.trans_op_op, (P.shape[-1],spectrums.shape[1]))

        self._current_rec = None # stores latest application of self.MX

    def set_P(self,P_new,pwr_en=True):
        self.P = np.copy(P_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op_op, (self.P.shape[-1],self.spectrums.shape[1]))

    def set_flux(self,flux_new,pwr_en=False):
        self.flux = np.copy(flux_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op_op, (np.prod(self.shape),np.prod(self.shape),self.A.shape[0]))

    def get_flux(self):
        return self.flux

    def MX(self, x):
        """MX

        This method calculates the action of the matrix M on the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array, a cube of transportation plans

        Returns
        -------
        np.ndarray result

        """
        self._current_rec = transport_plan_projections_field(self.P,self.shape,self.supp,self.neighbors_graph\
                            ,self.weights_neighbors,self.spectrums,x,self.flux,self.sig,self.ker,self.D)
        return self._current_rec

    def MtX(self, x):
        """MtX

        This method calculates the action of the transpose of the matrix Mt on
        the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array, a cube of 2D images

        Returns
        -------
        np.ndarray result

        """
        return transport_plan_projections_field_coeff_transpose(x,self.supp,self.neighbors_graph,\
                self.weights_neighbors,self.spectrums,self.P,self.flux,self.sig,self.ker_rot,self.D)
                
    def cost(self, x, y=None, verbose=False):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
        the dual variable.
        """
        if isinstance(self._current_rec, type(None)):
            self._current_rec = self.MX(x)
        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.obs_data) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)
        return cost_val


    def get_grad(self, x):
        """Get the gradient step
        This method calculates the gradient step from the input data
        Parameters
        ----------
        x : np.ndarray
            Input data array
        Returns
        -------
        np.ndarray gradient value
        Notes
        -----
        Calculates M^T (MX - Y)
        """

        self.grad = self.MtX(self.MX(x) - self.obs_data)

    
                


class polychrom_eigen_psf_coeff_graph(GradBasic, PowerMethod):
    """Polychromatic eigen PSFs class

    This class defines the operators for a field of undersampled space varying
    polychromatic PSFs. These operators are related to the estimation of weights,
    when the graph constraint is activated (i.e. weights are further factorized
    by the spatial constraints matrix).

    Parameters
    ----------
    data : np.ndarray
        Input data array, a array of 2D observed images (i.e. with noise)


    Notes
    -----
    The properties of `GradBasic` and `PowerMethod` are inherited in this class
    
    Calls:
    
    * :func:`psf_learning_utils.transport_plan_projections_field`
    * :func:`psf_learning_utils.transport_plan_projections_field_coeff_transpose`
    
    """

    def __init__(self, data, supp, neighbors_graph, weights_neighbors, spectrums, \
                P, flux, sig, ker, ker_rot, D, basis,D_stack,w_stack,C,gamma,n_iter_sink, data_type=float):

    #polychrom_grad_coeff

        self._grad_data_type = data_type
        self.obs_data = data
        self.op = self.MX 
        self.trans_op = self.MtX 
        shap = data.shape
        self.shape = (shap[0]*D,shap[1]*D)
        self.D = D
        self.supp = supp
        self.neighbors_graph = neighbors_graph
        self.weights_neighbors = weights_neighbors
        self.spectrums = spectrums
        self.P = np.copy(P)
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot  = ker_rot
        self.basis = basis
        self.D_stack = D_stack
        self.w_stack = w_stack
        self.C = C
        self.gamma = gamma
        self.n_iter_sink = n_iter_sink
        
        PowerMethod.__init__(self, self.trans_op_op, (P.shape[-1],self.basis.shape[0]))

        self._current_rec = None # stores latest application of self.MX

    def set_P(self,P_new,pwr_en=True):
        self.P = np.copy(P_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op_op, (self.P.shape[-1],self.basis.shape[0]))

    def set_flux(self,flux_new,pwr_en=False):
        self.flux = np.copy(flux_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op_op, (self.P.shape[-1],self.basis.shape[0]))

    def get_flux(self):
        return self.flux


    def MX(self, x):
        """MX

        This method calculates the action of the matrix M on the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array, a cube of transportation plans

        Returns
        -------
        np.ndarray result

        """
        self._current_rec = transport_plan_projections_field(self.P,self.shape,self.supp,self.neighbors_graph\
                            ,self.weights_neighbors,self.spectrums,x.dot(self.basis),self.flux,self.sig,self.ker,self.D)
        return self._current_rec

    def MtX(self, x):
        """MtX

        This method calculates the action of the transpose of the matrix Mt on
        the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array, a cube of 2D images

        Returns
        -------
        np.ndarray result

        """
        return transport_plan_projections_field_coeff_transpose(x,self.supp,self.neighbors_graph,\
                self.weights_neighbors,self.spectrums,self.P,self.flux,self.sig,self.ker_rot,self.D).dot(np.transpose(self.basis))
                
    def cost(self, x, y=None, verbose=False):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
        the dual variable.
        """
        if isinstance(self._current_rec, type(None)):
            self._current_rec = self.MX(x)
        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.obs_data) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)
        return cost_val
                
    def get_grad(self, x):
        """Get the gradient step
        This method calculates the gradient step from the input data
        Parameters
        ----------
        x : np.ndarray
            Input data array
        Returns
        -------
        np.ndarray gradient value
        Notes
        -----
        Calculates M^T (MX - Y)
        """

        self.grad = self.MtX(self.MX(x) - self.obs_data)
