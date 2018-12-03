
import numpy as np
from modopt.signal.wavelet import get_mr_filters, filter_convolve_stack,filter_convolve
import time

class dict_wavelet_transform(object):
    """transport_plan_marg_wavelet class

    This class defines an operator which performs a wavelet transform of a dictionary

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally a cube of dictionary atoms

    """

    def __init__(self,shap,nb_comp,wavelet_opt=None, method='scipy'):

        self.nb_comp = nb_comp
        self.shape = shap
        self.method = method
        if wavelet_opt is not None:
            self.filters = get_mr_filters(self.shape, opt=wavelet_opt)
        else:
            self.filters = get_mr_filters(self.shape)
        # self.filters = get_mr_filters(self.shape)
        
    def op_single(self,data):
        # Receives 2d image
        res = filter_convolve_stack(data.reshape((1,self.shape[0],self.shape[1])),self.filters, method=self.method)
        res = res.squeeze()
        sum_scales = np.sum(res,axis = 0)
        coarse = data - sum_scales
        complete = np.zeros((res.shape[0]+1,self.shape[0],self.shape[1]))        
        complete[-1,:,:] = coarse
        complete[:-1,:,:] = res
        
        return complete
    def adj_op_single(self,data):
        reduction = np.sum(data, axis=0)
        return reduction
    def op(self,data):

        res = []
        for d in range(data.shape[1]):
            # res.append(filter_convolve_stack(data[:,d,:].swapaxes(0,1).reshape((data.shape[2],self.shape[0],self.shape[1])),self.filters))
            res.append(filter_convolve_stack(data[:,d,:].swapaxes(0,1).reshape((data.shape[2],self.shape[0],self.shape[1])),self.filters, method=self.method))

        res = np.array(res)

        res_final = res
        res_final = res.swapaxes(1,2).swapaxes(2,3).swapaxes(3,4).swapaxes(0,1).swapaxes(1,2).swapaxes(2,3) # put in usual format <nb_filters,pixels_x,pixels_y,nb_atoms,nb_comp>


        sum_scales = np.sum(res_final,axis = 0)
        coarse = data.reshape((self.shape[0],self.shape[1],data.shape[1],data.shape[2])) - sum_scales

        complete = np.zeros((res_final.shape[0]+1,res_final.shape[1],res_final.shape[2],res_final.shape[3],res_final.shape[4]))

        complete[-1,:,:,:,:] = coarse
        complete[:-1,:,:,:,:] = res_final

        return complete


    def adj_op(self,data):

        reduction =  np.sum(data, axis=0)

        return reduction.reshape((self.shape[0]*self.shape[1],data.shape[3],data.shape[4]))


    def l1norm(self):

        return  self.nb_comp * np.sqrt(sum((np.sum(np.abs(filter_i)) ** 2 for filter_i in self.filters)))


class RCA_wavelet_transform(object):
    """transport_plan_marg_wavelet class

    This class defines an operator which performs a wavelet transform of a dictionary

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally a cube of dictionary atoms

    """

    def __init__(self,shap,nb_comp,wavelet_opt=None, method='scipy'):

        self.nb_comp = nb_comp
        self.shape = shap
        self.method = method
        if wavelet_opt is not None:
            self.filters = get_mr_filters(self.shape, opt=wavelet_opt)
        else:
            self.filters = get_mr_filters(self.shape)
        # self.filters = get_mr_filters(self.shape)

    def op(self,data):

        # res.append(filter_convolve_stack(data[:,d,:].swapaxes(0,1).reshape((data.shape[2],self.shape[0],self.shape[1])),self.filters))
        res = filter_convolve_stack(data.swapaxes(0,1).reshape((data.shape[1],self.shape[0],self.shape[1])),self.filters, method=self.method)


        res_final = res
        res_final = res.swapaxes(0,1).swapaxes(1,2).swapaxes(2,3) # put in usual format <nb_filters,pixels_x,pixels_y,nb_comp>



        sum_scales = np.sum(res_final,axis = 0)
        coarse = data.reshape((self.shape[0],self.shape[1],data.shape[1])) - sum_scales

        complete = np.zeros((res_final.shape[0]+1,res_final.shape[1],res_final.shape[2],res_final.shape[3]))

        complete[-1,:,:,:] = coarse
        complete[:-1,:,:,:] = res_final

        return complete


    def adj_op(self,data):

        reduction =  np.sum(data, axis=0)

        return reduction.reshape((self.shape[0]*self.shape[1],data.shape[3]))


    def l1norm(self):

        return  self.nb_comp * np.sqrt(sum((np.sum(np.abs(filter_i)) ** 2 for filter_i in self.filters)))


class TransformCombo(object):


    def __init__(self,transfs):
        self.transfs = transfs


    def op(self,data):

        res = np.empty(len(self.transfs), dtype=np.ndarray)


        for i in range(len(self.transfs)):
            res[i] = self.transfs[i].op(data[i])

        return res


    def adj_op(self,data):

        res = np.empty(len(self.transfs), dtype=np.ndarray)


        for i in range(len(self.transfs)):
            res[i] = self.transfs[i].adj_op(data[i])

        return res



class Apply_matrix(object):


    def __init__(self,A):
        '''
            Parameters
            ----------
            A: 2d matrix where the first dimension corresponds to different components.
        '''
        self.A = A

    def set_matrix(self,A):
        self.A = A


    def op(self,data):
        '''Condense data.
        '''


        return data.dot(self.A)

    def adj_op(self,data):
        '''
        Parameters
        ----------
        data: 2d matrix

        Output
        ------
        res: <nb_comp,N> 2d matrix of eigen-components
        '''
        # return data.dot(np.transpose(A))
        print "inverting A.."

        tic = time.time()
        res = np.linalg.solve(self.A.dot(np.transpose(self.A)), self.A.dot(np.transpose(data)))
        toc = time.time()
        # print str((toc-tic)/60.0) + "min"
        
        res = np.transpose(res)

        return res













        
        