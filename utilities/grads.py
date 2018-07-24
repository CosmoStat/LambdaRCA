import numpy as np
from psf_learning_utils import transport_plan_projections_field,transport_plan_projections_field_transpose,\
transport_plan_projections_field_coeff_transpose
from modopt.opt.gradient import GradBasic
from modopt.math.matrix import PowerMethod
        
        
class polychrom_eigen_psf(GradBasic, PowerMethod):
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

    """

    def __init__(self, data, supp, neighbors_graph, weights_neighbors, spectrums, \
                A, flux, sig, ker, ker_rot, D):

        self.y = data
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
        
        self.op = self.MX 
        self.trans_op = self.MtX 
        PowerMethod.__init__(self, self.trans_op_op, (np.prod(self.shape),np.prod(self.shape),A.shape[0]))

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
        return transport_plan_projections_field(x,self.shape,self.supp,self.neighbors_graph\
                ,self.weights_neighbors,self.spectrums,self.A,self.flux,self.sig,self.ker,self.D)

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


class polychrom_eigen_psf_coeff(GradBasic, PowerMethod):
    """Polychromatic eigen PSFs class

    This class defines the operators for a field of undersampled space varying
    polychromatic PSFs. These operators are related to the eigen PSFs weights estimation.

    Parameters
    ----------
    data : np.ndarray
        Input data array, a array of 2D observed images (i.e. with noise)


    Notes
    -----
    The properties of `GradBasic` and `PowerMethod` are inherited in this class

    """

    def __init__(self, data, supp, neighbors_graph, weights_neighbors, spectrums, \
                P, flux, sig, ker, ker_rot, D):

        self.y = data
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
        
        self.op = self.MX 
        self.trans_op = self.MtX 
        PowerMethod.__init__(self, self.trans_op_op, (P.shape[-1],spectrums.shape[1]))


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
        return transport_plan_projections_field(self.P,self.shape,self.supp,self.neighbors_graph\
                ,self.weights_neighbors,self.spectrums,x,self.flux,self.sig,self.ker,self.D)

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


class polychrom_eigen_psf_coeff_graph(GradBasic, PowerMethod):
    """Polychromatic eigen PSFs class

    This class defines the operators for a field of undersampled space varying
    polychromatic PSFs. These operators are related to the eigen PSFs weights estimation.

    Parameters
    ----------
    data : np.ndarray
        Input data array, a array of 2D observed images (i.e. with noise)


    Notes
    -----
    The properties of `GradBasic` and `PowerMethod` are inherited in this class

    """

    def __init__(self, data, supp, neighbors_graph, weights_neighbors, spectrums, \
                P, flux, sig, ker, ker_rot, D, basis):

        self.y = data
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
        
        self.op = self.MX 
        self.trans_op = self.MtX 
        PowerMethod.__init__(self, self.trans_op_op, (P.shape[-1],self.basis.shape[0]))


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
        return transport_plan_projections_field(self.P,self.shape,self.supp,self.neighbors_graph\
                ,self.weights_neighbors,self.spectrums,x.dot(self.basis),self.flux,self.sig,self.ker,self.D)

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
