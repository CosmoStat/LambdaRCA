import numpy as np
from psf_learning_utils import transport_plan_projections_field,\
transport_plan_projections_field_coeff_transpose
from modopt.opt.gradient import GradParent, GradBasic
from modopt.math.matrix import PowerMethod
import sys
sys.path.append('../baryOT')
import logOT_bary as ot
import time
import C_wrapper as Cw
        
        
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

        self._current_rec_MtX = None # stores latest application of self.MX_wdl (that includes MX and MtX)
        self._current_rec_MX = None # stores latest application of self.MX
        self._current_rec = None

        self.spec_rad = 30.9930631176
        self.inv_spec_rad = 0.03226528453
        # self.spec_rad = 20.0
        # self.inv_spec_rad = 0.05
        PowerMethod.__init__(self, self.trans_op, (np.prod(self.shape),D_stack.shape[1],A.shape[0]))
        print " > SPECTRAL RADIUS:\t{}".format(self.spec_rad)
        
        
    def set_A(self,A_new,pwr_en=True):
        self.A = np.copy(A_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op, (np.prod(self.shape),self.D_stack.shape[1],self.A.shape[0]))

    def set_flux(self,flux_new,pwr_en=False):
        self.flux = np.copy(flux_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op, (np.prod(self.shape),self.D_stack.shape[1],self.A.shape[0]))

    def get_flux(self):
        return self.flux

    

    def MtX_init(self,x):
        a = self.MtX(x)
        b = self.MtX(x,y=self.obs_data)

        return a-b 


    def MtX(self,x,y=None):
        """
            x: input data array, a cube of 2D images, residuous MX-Y. x not used.
        """

        dictionary = self.D_stack
        if isinstance(y, type(None)):
            y = np.zeros(self.obs_data.shape)
            dictionary = x

        tic = time.time()
        self._current_rec_MtX = ot.Theano_wdl_MtX(self.A,self.spectrums,self.flux,self.sig,self.ker,dictionary,self.w_stack,self.C,self.gamma,self.n_iter_sink,y)
        toc = time.time()

        print str((toc-tic)/60.0) + " min"



        # self._current_rec_MtX = Cw.call_WDL(A=self.A,spectrums=self.spectrums,flux=self.flux,sig=self.sig,ker=self.ker,rot_ker=self.ker_rot, D_stack=self.D_stack,
        #     w_stack=self.w_stack,gamma=self.gamma,n_iter_sink=self.n_iter_sink,y=y,N=self.D_stack.shape[0],func="--MtX_wdl")




        return self._current_rec_MtX[0]


    def MX(self,x):

        temp = ot.Theano_wdl_MX(self.A,self.spectrums,self.flux,self.sig,self.ker,x,self.w_stack,self.C,self.gamma,self.n_iter_sink)

        # temp = Cw.call_WDL(A=self.A, spectrums=self.spectrums,flux=self.flux,sig=self.sig,ker=self.ker,rot_ker=self.ker_rot,D_stack=x,w_stack=self.w_stack,
        #     gamma=self.gamma,n_iter_sink=self.n_iter_sink,N=x.shape[0], func="--MX_wdl")

        self._current_rec_MX = temp[0]

        return self._current_rec_MX




    def MX_old(self, x):
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

    def MtX_old(self, x):
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
        if isinstance(self._current_rec_MtX, type(None)):
            self._current_rec = self.MX(x)
        else:
            self._current_rec = self._current_rec_MtX[1]

        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.obs_data) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)
        return cost_val
                
    def get_grad_old(self, x):
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


    def get_grad(self,x):


        self.D_stack_old = self.D_stack
        self.D_stack = x 
        self.grad = self.MtX(x,y=self.obs_data)

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
                P, flux, sig, ker, ker_rot, D, basis,D_stack,w_stack,C,gamma,n_iter_sink,polychrom_grad,A, data_type=float):

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
        self.polychrom_grad = polychrom_grad
        self.A_ini = A


        self._current_rec = None # stores latest application of self.MX
        self._current_rec_MX = None
        self._current_rec_MtX = None
        self._current_x = self.A_ini.dot(np.transpose(self.basis))



        self.spec_rad = 5.595602793283493  
        self.inv_spec_rad = 0.1787117558     
        PowerMethod.__init__(self, self.trans_op, (D_stack.shape[-1],self.basis.shape[0]))




    def set_D_stack(self,D_stack_new,pwr_en=True):
        self.D_stack = np.copy(D_stack_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op, (self.D_stack.shape[-1],self.basis.shape[0]))

    def set_P(self,P_new,pwr_en=True):
        self.P = np.copy(P_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op, (self.P.shape[-1],self.basis.shape[0]))


    def set_flux(self,flux_new,pwr_en=False):
        self.flux = np.copy(flux_new)
        if pwr_en:
            PowerMethod.__init__(self, self.trans_op, (self.D_stack.shape[-1],self.basis.shape[0]))

    def get_flux(self):
        return self.flux


    def MX_old(self, x):
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


    def MX(self, x):
        # x: <5,5*100> 
        # basis: <5*100, 100>

        self._current_rec_MX = ot.Theano_coeff_MX(x.dot(self.basis),self.spectrums,self.polychrom_grad._current_rec_MtX[2],self.flux,self.sig,self.ker)
        
        return self._current_rec_MX

    def MtX_old(self, x):
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

    def MtX(self, x, y=None): # the x here is not used. MX - observed data is taken as default (the loss function)

        variable = self._current_x 
        if isinstance(y, type(None)):
            y = np.zeros(self.obs_data.shape)
            variable  = x

       
        self._current_rec_MtX = ot.Theano_coeff_MtX(variable.dot(self.basis),self.spectrums,self.polychrom_grad._current_rec_MtX[2],self.flux,self.sig,self.ker,y) #[MtX_coeff_, MX_coeff]


        

        return self._current_rec_MtX[0].dot(np.transpose(self.basis))


                
    def cost(self, x, y=None, verbose=False):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
        the dual variable.
        """
        if isinstance(self._current_rec_MtX, type(None)): #check if cost is not called outside modopt, e.g. in optim_outils
            self._current_rec = self.MX(x)
        else:
            self._current_rec = self._current_rec_MtX[1]

        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.obs_data) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)
        return cost_val
                
    def get_grad_old(self, x):
        """Get the gradient step
        This method calculates the gradient step from the input data
        Parameters
        ----------
        x : np.ndarray
            Input data array
        Ret
        -------
        np.ndarray gradient value
        Notes
        -----
        Calculates M^T (MX - Y)
        """

        # np.save("/Users/rararipe/Documents/Data/test_coeff_MtX_quickgen_22x22pixels_5lbdas10pos_25_07_2018/barycenters.npy", multi_spec_comp_mat)
        # np.save("/Users/rararipe/Documents/Data/test_coeff_MtX_quickgen_22x22pixels_5lbdas10pos_25_07_2018/A.npy", x.dot(self.basis))

        self.grad = self.MtX(self.MX(x) - self.obs_data)


        # temp3 = transport_plan_projections_field_coeff_transpose(temp1,self.supp,self.neighbors_graph,self.weights_neighbors,self.spectrums,self.P,self.flux,self.sig,self.ker_rot,self.D)

        # np.save("/Users/rararipe/Documents/Data/test_coeff_MtX_quickgen_22x22pixels_5lbdas10pos_25_07_2018/gradA.npy", temp3)

    def get_grad(self, x):
        """Get the gradient step
        This method calculates the gradient step from the input data
        Parameters
        ----------
        x : np.ndarray
            Input data array
        Ret
        -------
        np.ndarray gradient value
        Notes
        -----
        Calculates M^T (MX - Y)
        """


        self._current_x  = x
        self.grad = self.MtX(x, y= self.obs_data)





