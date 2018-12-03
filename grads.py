import numpy as np
from modopt.opt.gradient import GradParent, GradBasic #try to erase these two lines latter
from modopt.math.matrix import PowerMethod
import time
import C_wrapper as omp
import psf_learning_utils as psflu 
from modopt.math.matrix import PowerMethod
import utils
import psf_toolkit as tk
#import logOT_bary as  ot # 2 min to load in total

        
class polychrom_eigen_psf(GradParent, PowerMethod):

    def __init__(self,A,spectrums,flux,sig,ker,ker_rot,D_stack,w_stack, gamma, n_iter_sink,stars,data_type=float,useTheano=False,logit=False):
        self._grad_data_type = data_type
        self.A = A
        self.spectrums = spectrums
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot = ker_rot
        self.D_stack = D_stack
        self.Dlog_stack = None
        self.logit = logit
        if logit:
#            D_stack[D_stack< 0.0] = 1e-9
            D_stack = abs(D_stack)
            self.Dlog_stack = np.log(D_stack)
        self.w_stack = w_stack
        self.gamma = gamma
        self.n_iter_sink = n_iter_sink
        self.stars = stars
        self.shap = (stars.shape[0]*2,stars.shape[1]*2)
        self.N = self.shap[0]*self.shap[1]

        self.MtMX__MtMX = None
        self.MtMX__MX = None
        self.MtMX__barys = None
        self.MX__MX = None
        self.MX__barys = None
        self.compute__barys_barys = None
        self._current_rec = None
        self.debug_grads = []
        self.iter = 0

        self.min_dict = None
        self.lin_l1norm = 0
        self.compute_grad__iter = 0
        self.get_grad__iter  = 0
        self.gamma_update__iter = 0

        self.costs = []
        self.steps = []
        self.D_stack_energy = []

        self.alpha = 1.0
        self.n_iter = 0
        
        self.x_old = None

        self.useTheano = useTheano

        if self.useTheano:
#            self.C = ot.myAssymetricWellCost(self.shap[0], self.shap[1])
            self.C = ot.EuclidCost(self.shap[0], self.shap[1])


        # PowerMethod.__init__(self, self.compute_MtMX_pm, D_stack.shape,verbose=True)
        # self.spec_rad = pm.spec_rad
        # self.inv_spec_rad = pm.inv_spec_rad
        
    def set_n_iter(self,n_iter_new):
        self.n_iter = n_iter_new

    def reset_costs(self):
        self.costs = []

    def reset_steps(self):
        self.steps = []

    def reset_D_stack_energy(self):
        self.D_stack_energy = []

    def set_A(self,A_new):
        self.A = np.copy(A_new)
    

    def set_Dlog_stack(self,Dlog_new):
        self.Dlog_stack = np.copy(Dlog_new)
        

    def set_D_stack(self,D_stack_new):
        self.D_stack = np.copy(D_stack_new)
        

    def set_flux(self,flux_new):
        self.flux = np.copy(flux_new)


    def set_min_dict(self, min_dict_new):
        self.min_dict = min_dict_new

    def set_lin_comb_l1norm(self,new_norm):
        self.lin_l1norm = new_norm

    def reset_iterations(self):
        self.compute_grad__iter = 0
        self.get_grad__iter  = 0
        self.gamma_update__iter = 0


    def get_flux(self):
        return self.flux


    def MtX_noise(self,D_stack=None):
        x = self.D_stack

        if D_stack is not None:
            x = D_stack

        a = self.compute_MtMX(stars=np.zeros(self.stars.shape),D_stack=x)
        b = self.compute_MtMX(D_stack=x)

        return a-b 


    # def compute_MtMX_pm(self,x):
    #     x = abs(x)
    #     y = np.zeros(self.stars.shape)
    #     print "MtMX.."
    #     tic = time.time()
    #     res = omp.call_WDL(A=self.A,spectrums=self.spectrums,flux=self.flux,sig=self.sig,ker=self.ker,rot_ker=self.ker_rot, 
    #         D_stack=x,w_stack=self.w_stack,gamma=self.gamma,n_iter_sink=self.n_iter_sink,y=y,N=self.N,func='--MtX_wdl')
    #     toc = time.time()
    #     print "Done in: " + str((toc-tic)/60.0) + " min"

    #     return res[0]

    def compute_MtMX(self,stars=None,D_stack=None,Dlog_stack=None,count=False):

        y = self.stars
        if self.logit:
            x = self.Dlog_stack
        else:
            x = self.D_stack

        if stars is not None:
            y = stars

        if D_stack is not None:
            x = D_stack
#            x[x< 0.0] = 1e-9
            x = abs(x)
            

        if Dlog_stack is not None:
            x = Dlog_stack

        # Sadly, do abs of D_stack so that it doesn't explode. The projection on the simplex is not enough, it works in convergence but not at every iteration specifically.


        if self.useTheano:
            print "theano MtMX.."
            if self.logit:
                tic = time.time()
                res  = ot.Theano_wdlLog_MtX(self.A,self.spectrums,self.flux,self.sig,self.ker,x,self.w_stack,self.C,self.gamma,self.n_iter_sink,self.stars)
                toc = time.time()
            else:
                tic = time.time()
                res  = ot.Theano_wdl_MtX(self.A,self.spectrums,self.flux,self.sig,self.ker,x,self.w_stack,self.C,self.gamma,self.n_iter_sink,self.stars)
                toc = time.time()
            print "Done in: " + str((toc-tic)/60.0) + " min" # 10 min, for 80 objs 8 wvls

        else:
            print "MtMX.."    
            if self.logit:
                tic = time.time()
                res = omp.call_WDL(A=self.A,spectrums=self.spectrums,flux=self.flux,sig=self.sig,ker=self.ker,rot_ker=self.ker_rot, 
                    Dlog_stack=x,w_stack=self.w_stack,gamma=self.gamma,n_iter_sink=self.n_iter_sink,y=y,N=self.N,func='--MtX_wdl')
                toc = time.time()
            else:
                tic = time.time()
                res = omp.call_WDL(A=self.A,spectrums=self.spectrums,flux=self.flux,sig=self.sig,ker=self.ker,rot_ker=self.ker_rot, 
                    D_stack=x,w_stack=self.w_stack,gamma=self.gamma,n_iter_sink=self.n_iter_sink,y=y,N=self.N,func='--MtX_wdl')
                toc = time.time()
            print "Done in: " + str((toc-tic)/60.0) + " min" # between 4 and 10 min for 80 objs 8 wvls


        


        self.MtMX__MtMX = res[0]
        self.MtMX__MX = res[1]


        return self.MtMX__MtMX



    def compute_MX(self, x, x_in_log=False):

        if x_in_log:
            D_stack = psflu.logitTonormal(x)
        else:
#            x[x<0.0] = 1e-9
            x = abs(x)
            D_stack = x

        if self.useTheano:
            print "tehano MX.."
            tic = time.time()
            res = ot.Theano_wdl_MX(self.A,self.spectrums,self.flux,self.sig,self.ker,D_stack,self.w_stack,self.C,self.gamma,self.n_iter_sink)
            toc = time.time()
            print "Done in: " + str((toc-tic)/60.0) + " min" # 0.2 min for 80 obj, 8 wvls
        else:
            print "MX.."
            tic = time.time()
            res = omp.call_WDL(A=self.A,spectrums=self.spectrums,flux=self.flux,sig=self.sig,ker=self.ker,rot_ker=self.ker_rot, 
                D_stack=D_stack,w_stack=self.w_stack,gamma=self.gamma,n_iter_sink=self.n_iter_sink,N=self.N,func='--MX_wdl')
            toc = time.time()
            print "Done in: " + str((toc-tic)/60.0) + " min" # 3 min for 80 objs , 8 wvls
        self.MX__MX = res[0]
        self.MX__barys = res[1]

        

        # print ">>>> Energy of reconstructed stars: ", np.sum(abs(self.MtMX__MX),axis=(0,1))

        return self.MX__MX



    def compute_barys(self,D_stack=None,w_stack=None):

        dic = self.D_stack
        w = self.w_stack

        if D_stack is not None:
            dic = D_stack
        if w_stack is not None:
            w = w_stack

        if self.useTheano:
            print "theano Computing barycenters.."
            tic = time.time()
            self.compute_barys__barys = ot.Theano_bary(dic,w,self.gamma,self.C,self.n_iter_sink)
            toc = time.time()
            print "Done in: " + str((toc-tic)/60.0) + " min"  # 1.39 min, 80 stars 8 wvls
        else:
            print "Computing barycenters.."
            tic = time.time()
            self.compute_barys__barys = omp.call_WDL(D_stack=dic,w_stack=w,gamma=self.gamma,n_iter_sink=self.n_iter_sink,
                func="--bary",N=self.N,remove_files=True)
            toc = time.time()
            print "Done in: " + str((toc-tic)/60.0) + " min" # 0.16 min, 80 stars 8 wvls

        return self.compute_barys__barys


    def compute_grad(self,stars=None,D_stack=None,Dlog_stack=None,use_cache_grad=False,count=False):

        if not use_cache_grad:
            y = self.stars
            dic = self.D_stack
            dicLog = self.Dlog_stack


            if stars is not None:
                y = stars

            if D_stack is not None:
                dic = D_stack

            if Dlog_stack is not None:
                dicLog = Dlog_stack

            if self.logit:
                print "LOGIT"
                res = self.compute_MtMX(stars=y,Dlog_stack=dicLog,count=count)
                res[:,0,:] *= 1.0
            else:
                res = self.compute_MtMX(stars=y,D_stack=dic,count=count)
        
        else:

            res = self.MtMX__MtMX





        return res



    def get_grad(self,x): #active command, in modopt style


        print "get_grad called"
        
                
        if self.logit:
            self.D_stack_energy.append(np.sum(abs(psflu.logitTonormal(x)), axis=0))
            print "===========================> Dictionary energy ", np.sum(abs(psflu.logitTonormal(x)), axis=0)
            self.grad = self.compute_grad(Dlog_stack = x,use_cache_grad=True,count=True)
        else:
            self.D_stack_energy.append(np.sum(abs(x), axis=0))
            print "===========================> Dictionary energy ", np.sum(abs(x), axis=0)
            self.grad = self.compute_grad(D_stack = x,use_cache_grad=True,count=True)
            
         


        return self.grad


    def cost(self, x, y=None, verbose=False,count=True,use_cache=False):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
        the dual variable.)
        """
#        import pdb; pdb.set_trace()  # breakpoint 3d87ba10 //
        if use_cache:
            if isinstance(self.MX__MX , type(None)):
                self._current_rec = self.compute_MX(x,x_in_log=self.logit)
            elif not np.equal(x,self.x_old).all(): 
                self._current_rec = self.compute_MX(x,x_in_log=self.logit)
            else:
                self._current_rec = self.MX__MX
        else:
           self._current_rec = self.compute_MX(x,x_in_log=self.logit) 

        self.x_old = x
#        self._current_rec = self.compute_MX(x,x_in_log=self.logit)

        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.stars) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)

        if count :
            self.costs.append(cost_val)
        return cost_val
    


    def gamma_update(self,gamma):



        print "gamma update called"


        if self.min_dict is not None:
            x_0 = np.copy(self.min_dict._x_new)
        else:
            if self.logit:
                x_0 = np.copy(self.Dlog_stack)
            else:
                x_0 = np.copy(self.D_stack)
        if self.logit:
            grad = self.compute_grad(Dlog_stack=x_0)
        else:
            grad = self.compute_grad(D_stack=x_0)
        f_0 = self.cost(x_0,count=False,use_cache=True) # this will be computed twice :/

        if self.logit:
            beta = 0.1
#            self.alpha = 1.0
        else:
            beta = 0.1
#            self.alpha = 0.2
        
        gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=self.alpha, beta=beta)
        max_gamma = 1.0
        min_lmbda_fake = 0.7 # max gamma is 1.0
        if self.min_dict is not None:
            if self.n_iter == 0:
                lmbda = max(min_lmbda_fake,gamma)
                if gamma/lmbda <= max_gamma/15.0: # around 7 %
                    lmbda /= 2.0 
                self.min_dict.extra_factor_LP = 1.0
            else:
                lmbda = np.copy(gamma)
                self.min_dict.extra_factor_LP = gamma
            self.min_dict._lambda_param = lmbda
        return gamma

    
    # def sig_tau_update_old(self,sigma):

    #     print "sigma update called"

    #     if self.min_dict is not None:
    #         x_0 = np.copy(self.min_dict._x_new)

    #     else:
    #         x_0 = np.copy(self.D_stack)
            
    #     grad = self.compute_grad(D_stack=x_0)
    #     f_0 = self.cost(x_0,count=False) # this will be computed twice :/

    #     if self.logit:
    #         beta = 0.1
    #         alpha = 60.0
    #     else:
    #         beta = 0.7
    #         alpha = 1.0
    #     L_est = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=alpha, beta=beta)

    #     sigma = 1.0/(L_est + self.lin_l1norm)
    #     print "lin norm ",str(self.lin_l1norm)
    #     print "final sigma ",str(sigma)
    #     tau = np.copy(sigma)

    #     # Check convergence condition
    #     if 1.0/tau - sigma*self.lin_l1norm**2 >= L_est/2.0 :
    #         print "good combination of parameters"
    #     else:
    #         print "attention: bad combination "

    #     self.steps.append(sigma)

    #     if self.min_dict is not None:
    #         self.min_dict._tau = tau

    #     return sigma



    def sig_tau_update(self,sigma,x=None):

        print "sigma update called"
        print " "


        if self.min_dict is not None and x is None:
            x_0 = np.copy(self.min_dict._x_new)
        elif x is not None:
            x_0 = x            
        else:
            if self.logit:
                x_0 = np.copy(self.Dlog_stack)
            else:
                x_0 = np.copy(self.D_stack)
            
        if self.logit:
            grad = self.compute_grad(Dlog_stack=x_0)
        else:
            grad = self.compute_grad(D_stack=x_0)
        f_0 = self.cost(x_0,count=False,use_cache=True) # this will be computed twice :/
        if self.logit:
            beta = 0.1
#            self.alpha = 1.0
        else:
            beta = 0.07
#            self.alpha = 0.2
        gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost,alpha=self.alpha, beta=beta)

        sigma = gamma
        tau = np.copy(sigma)

        self.steps.append(sigma)
        
        max_sigma = 1.0
        min_lmbda_fake = 0.7 # max gamma is 1.0
        if self.min_dict is not None:
            if self.n_iter == 0:
                lmbda = max(min_lmbda_fake,gamma)
                if gamma/lmbda <= max_sigma/15.0: # around 7 %
                    lmbda /= 2.0 
                self.min_dict.extra_factor_LP = 1.0
            else:
                lmbda = np.copy(gamma)
                self.min_dict.extra_factor_LP = gamma
            self.min_dict._tau = lmbda

        
        if self.min_dict is not None:
            self.min_dict._tau = tau

        return sigma
    # def lambda_update(self,lbda):

    #     gamma = self.min_dict.gamma

    #     if lbda > 0 and lbda < 1.0:


class polychrom_eigen_psf_RCA(GradParent, PowerMethod):
    def __init__(self,A,flux,sig,ker,ker_rot,S_stack,stars,D=2,data_type=float):
        self._grad_data_type = float
        self.A = A
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot = ker_rot
        self.stars = stars
        self.shap = (stars.shape[0]*D,stars.shape[1]*D)
        self.N = self.shap[0]*self.shap[1]
        self.S_stack = S_stack
        self.MtX__MtX = None
        self.MX__MX = None

        self.min_dict = None
        self.lin_l1norm = None
        self.costs = []
        self.steps = []


    def set_A(self, new_A):
        self.A = new_A

    def set_S_stack(self,new_S_stack):
        self.S_stack = new_S_stack

    def set_flux(self,flux_new):
        self.flux = np.copy(flux_new)

    def get_flux(self):
        return self.flux

    def MtX(self,mx):        

        self.MtX__MtX = psflu.MtX_RCA(mx,self.A,self.flux,self.sig,self.ker_rot)

        return self.MtX__MtX

    def MX(self,S_stack):

        self.MX__MX = psflu.MX_RCA(S_stack,self.A,self.flux,self.sig,self.ker)

        return self.MX__MX 

    def compute_grad(self,S_stack,use_cache_grad=False):

        if not use_cache_grad:
            self.grad = self.MtX(self.MX(S_stack) - self.stars)





        return self.grad


    def set_min_dict(self, min_dict_new):
        self.min_dict = min_dict_new


    def set_lin_comb_l1norm(self,new_norm):
        self.lin_l1norm = new_norm

    def reset_costs(self):
        self.costs = []

    def reset_steps(self):
        self.steps = []


    def MtX_noise(self):
            
        return self.MtX(self.stars)

    def get_grad(self,S_stack):

        return self.compute_grad(S_stack,use_cache_grad=True)


    def cost(self, x, y=None, verbose=False,count=True,use_cache=False):

        self._current_rec = self.MX(x)

        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.stars) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)

        if count:
            print "loss counted"
            self.costs.append(cost_val)
        
        return cost_val


    def gamma_update(self,gamma):


        print "gamma update called"

        if self.min_dict is not None:
            x_0 = np.copy(self.min_dict._x_new)
        else:
            x_0 = np.copy(self.S_stack)

        grad = self.compute_grad(x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=1.0)

        return gamma



    def sig_tau_update(self,sigma):

        print "sigma update called"

        if self.min_dict is not None:
            x_0 = np.copy(self.min_dict._x_new)

        else:
            x_0 = np.copy(self.S_stack)
            
        grad = self.compute_grad(x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        L_est = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=0.2)

        sigma = L_est
        tau = np.copy(sigma)

        # Check convergence condition
        
        self.steps.append(sigma)

        if self.min_dict is not None:
            self.min_dict._tau = tau

        return sigma



    def sig_tau_update_old(self,sigma):

        print "sigma update called"

        if self.min_dict is not None:
            x_0 = np.copy(self.min_dict._x_new)

        else:
            x_0 = np.copy(self.S_stack)
            
        grad = self.compute_grad(x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        L_est = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=1.0)


        sigma = 1.0/(L_est + self.lin_l1norm)
        tau = np.copy(sigma)

        # Check convergence condition
        if 1.0/tau - sigma*self.lin_l1norm**2 >= L_est/2.0 :
            print "good combination of parameters"
        else:
            print "attention: bad combination "

        self.steps.append(sigma)

        if self.min_dict is not None:
            self.min_dict._tau = tau

        return sigma








class polychrom_eigen_psf_wrapper(GradParent, PowerMethod):

    def __init__(self,A_svd_mix,spectrums,flux,sig,ker,ker_rot,mix_stack,w_stack, gamma, n_iter_sink,stars,D=2,data_type=np.ndarray):
        self._grad_data_type = data_type
        self.A_mix = A_svd_mix
        self.spectrums = spectrums
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot = ker_rot
        self.w_stack = w_stack
        self.gamma = gamma
        self.n_iter_sink = n_iter_sink
        self.stars = stars
        self.shap = (stars.shape[0]*2,stars.shape[1]*2)
        self.N = self.shap[0]*self.shap[1]
        self.mix_stack = mix_stack
        self.nb_comp_chrom = mix_stack[0].shape[-1]

        self.Min_wdl = polychrom_eigen_psf(A_svd_mix[0],spectrums,flux,sig,ker,ker_rot,mix_stack[0],w_stack, gamma, n_iter_sink,stars)
        self.Min_RCA = polychrom_eigen_psf_RCA(A_svd_mix[1],flux,sig,ker,ker_rot,mix_stack[1],stars)

        self.min_dict = None
        self.lin_l1norm = None

        self.compute_grad_wdl = None
        self.compute_grad_RCA = None
        self.MX__wdl = None
        self.MX__RCA = None
        self.MX__MX = None
        self.costs = []
        self.steps = []

    def set_A_mix(self,A_mix_new):
        self.A_mix = np.copy(A_mix_new)
        self.Min_wdl.set_A(A_mix_new[0])
        self.Min_RCA.set_A(A_mix_new[1])

    def set_mix_stack(self,mix_stack_new):
        self.mix_stack = np.copy(mix_stack_new)
        self.Min_wdl.set_D_stack(mix_stack_new[0])
        self.Min_RCA.set_S_stack(mix_stack_new[1])


    def set_flux(self,flux_new):
        self.flux = np.copy(flux_new)
        self.Min_wdl.set_flux(flux_new)
        self.Min_RCA.set_flux(flux_new)

    def set_min_dict(self, min_dict_new):
        self.min_dict = min_dict_new

    def set_lin_comb_l1norm(self,new_norm):
        self.lin_l1norm = new_norm

    def reset_costs(self):
        self.costs = []


    def reset_steps(self):
        self.steps = []


    def MtX_noise(self,mix_stack=None):
        
        mix = self.mix_stack
        if mix_stack is not None:
            mix = mix_stack
            

        a = self.Min_wdl.compute_grad(stars=np.zeros(self.stars.shape),D_stack=mix[0])
        b = self.Min_wdl.compute_grad(D_stack=mix[0])
        res_wdl = a - b
        res_RCA = self.Min_RCA.MtX(self.stars)

        res = np.empty(2,dtype=np.ndarray)
        res[0] = np.copy(res_wdl)
        res[1] = np.copy(res_RCA)



        return res


    def MX(self,mix_stack=None):

        if mix_stack is not None:
            mix = mix_stack
        else:
            mix = self.mix_stack

        self.MX__wdl = self.Min_wdl.compute_MX(mix[0])
        self.MX__RCA = self.Min_RCA.MX(mix[1])

        self.MX__MX = self.MX__wdl + self.MX__RCA

        return self.MX__MX


    def compute_grad(self,mix_stack,use_cache_grad=False):

        if not use_cache_grad:

            self.compute_grad_wdl = self.Min_wdl.compute_grad(D_stack=mix_stack[0])
            self.compute_grad_RCA = self.Min_RCA.compute_grad(mix_stack[1])
   
            res = np.empty(2,dtype=np.ndarray)
            res[0] = np.copy(self.compute_grad_wdl)
            res[1] = np.copy(self.compute_grad_RCA)
            # res = [ np.copy(res_dict), np.copy(res_RCA)]
            

            self.grad = res
        
        
        return self.grad


    def get_grad(self,mix_stack):

        return self.compute_grad(mix_stack,use_cache_grad=True)


    def cost(self, x, y=None, verbose=False,count=True):

        self._current_rec = self.MX(mix_stack = x)

        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.stars) ** 2
        if verbose:
            print " > MIN(X):\t{0}\t{1}".format(np.min(x[0]), np.min(x[1]))
            print " > Current cost: {}".format(cost_val)

        if count:
            print "loss counted"
            self.costs.append(cost_val)
        
        return cost_val


    def gamma_update(self,gamma):


        print "gamma update called"

        if self.min_dict is not None:
            x_0 = np.copy(self.min_dict._x_new)
        else:
            x_0 = np.copy(self.mix_stack)

        grad = self.compute_grad(x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost)

        return gamma


    def sig_tau_update(self,sigma):

        print "sigma update called"

        if self.min_dict is not None:
            x_0 = np.copy(self.min_dict._x_new)

        else:
            x_0 = np.copy(self.mix_stack)
            
        grad = self.compute_grad(x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        L_est = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=0.2)

        sigma = L_est
        tau = np.copy(sigma)

        # Check convergence condition
        self.steps.append(sigma)

        if self.min_dict is not None:
            self.min_dict._tau = tau

        return sigma


    def sig_tau_update_old(self,sigma):

        print "sigma update called"

        if self.min_dict is not None:
            x_0 = np.copy(self.min_dict._x_new)

        else:
            x_0 = np.copy(self.mix_stack)
            
        grad = self.compute_grad(x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        L_est = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=1.0)

        sigma = 1.0/(L_est + self.lin_l1norm)
        tau = np.copy(sigma)

        # Check convergence condition
        if 1.0/tau - sigma*self.lin_l1norm**2 >= L_est/2.0 :
            print "good combination of parameters"
        else:
            print "attention: bad combination "

        self.steps.append(sigma)

        if self.min_dict is not None:
            self.min_dict._tau = tau

        return sigma


    def compute_barys(self,D_stack=None,w_stack=None):

        if D_stack is not None:
            dic = D_stack
        else:
            dic = self.mix_stack[0]
        if w_stack is not None:
            w = w_stack
        else:
            w = self.w_stack

        self.compute_barys__barys = self.Min_wdl.compute_barys(D_stack=dic,w_stack=w)

        return self.compute_barys__barys



# class polychrom_eigen_psf_mixed_comp(GradParent, PowerMethod):

#     def __init__(self,A,spectrums,flux,sig,ker,ker_rot,mix_stack,w_stack, gamma, n_iter_sink,stars,data_type=np.ndarray):
#         self._grad_data_type = data_type
#         self.A = A
#         self.spectrums = spectrums
#         self.flux = flux
#         self.sig = sig
#         self.ker = ker
#         self.ker_rot = ker_rot
#         self.w_stack = w_stack
#         self.gamma = gamma
#         self.n_iter_sink = n_iter_sink
#         self.stars = stars
#         self.shap = (stars.shape[0]*2,stars.shape[1]*2)
#         self.N = self.shap[0]*self.shap[1]
#         self.mix_stack = mix_stack

#         self.nb_comp_chrom = mix_stack[0].shape[-1]

#         self.MtMX_dict__MtMX = None
#         self.MtMX_dict__MX = None
#         self.MtMX_dict__barys = None
#         self.MX_dict__MX = None
#         self.MX_dict__barys = None
#         self.compute__barys_barys = None
#         self.compute_grad__resRCA = None
#         self.compute_grad__res = None
#         self._current_rec = None
#         self.compute_MX__MX = None
#         self.debug_grads = []
#         self.iter = 0

#         self.min_dict = None
#         self.lin_l1norm = 0
#         self.compute_grad__iter = 0
#         self.get_grad__iter  = 0
#         self.gamma_update__iter = 0


#         # PowerMethod.__init__(self, self.compute_MtMX_pm, D_stack.shape,verbose=True)
#         # self.spec_rad = pm.spec_rad
#         # self.inv_spec_rad = pm.inv_spec_rad

#     def set_A(self,A_new):
#         self.A = np.copy(A_new)


#     def set_mix_stack(self,mix_stack_new):
#         self.mix_stack = np.copy(mix_stack_new)


#     def set_flux(self,flux_new):
#         self.flux = np.copy(flux_new)


#     def set_min_dict(self, min_dict_new):
#         self.min_dict = min_dict_new

#     def set_lin_comb_l1norm(self,new_norm):
#         self.lin_l1norm = new_norm

#     def reset_iterations(self):
#         self.compute_grad__iter = 0
#         self.get_grad__iter  = 0
#         self.gamma_update__iter = 0


#     def get_flux(self):
#         return self.flux

#     def get_D_stack(self):
#         return self.mix_stack[0]

#     def get_S_stack(self):
#         return self.mix_stack[1]



#     def MtX_noise(self,mix_stack=None):
        
#         mix = self.mix_stack
#         if mix_stack is not None:
#             mix = mix_stack
            

#         a = self.compute_grad(stars=np.zeros(self.stars.shape),mix_stack=mix)
#         b = self.compute_grad(mix_stack=mix)


#         return a-b 


#     def compute_MtMX_pm(self,x):
#         x = abs(x)
#         y = np.zeros(self.stars.shape)
#         print "MtMX.."
#         tic = time.time()
#         res = omp.call_WDL(A=self.A,spectrums=self.spectrums,flux=self.flux,sig=self.sig,ker=self.ker,rot_ker=self.ker_rot, 
#             D_stack=x,w_stack=self.w_stack,gamma=self.gamma,n_iter_sink=self.n_iter_sink,y=y,N=self.N,func='--MtX_wdl')
#         toc = time.time()
#         print "Done in: " + str((toc-tic)/60.0) + " min"

#         return res[0]

#     def compute_MtMX_dict(self,stars=None,D_stack=None):

#         y = self.stars
#         dic = self.mix_stack[0]

#         if stars is not None:
#             y = stars

#         if D_stack is not None:
#             dic = D_stack
            
#         # Sadly, do abs of D_stack so that it doesn't explode. The projection on the simplex is not enough, it works in convergence but not at every iteration specifically.

#         dic = abs(dic)


#         print "MtMX.."
#         tic = time.time()
#         res = omp.call_WDL(A=self.A[:self.nb_comp_chrom,:],spectrums=self.spectrums,flux=self.flux,sig=self.sig,ker=self.ker,rot_ker=self.ker_rot, 
#             D_stack=dic,w_stack=self.w_stack,gamma=self.gamma,n_iter_sink=self.n_iter_sink,y=y,N=self.N,func='--MtX_wdl')
#         toc = time.time()
#         print "Done in: " + str((toc-tic)/60.0) + " min"

#         self.MtMX_dict__MtMX = res[0]
#         self.MtMX_dict__MX = res[1]


#         return self.MtMX_dict__MtMX


#     def compute_MtX_RCA(self,mx):        

#         res = psflu.MtX_RCA(mx,self.A[self.nb_comp_chrom:,:],self.flux,self.sig,self.ker_rot)

#         return res


#     def compute_MX_RCA(self,S_stack):

#         res = psflu.MX_RCA(S_stack,self.A[self.nb_comp_chrom:,:],self.flux,self.sig,self.ker,D=2)

#         return res


#     def compute_MX_dict(self, D_stack):

#         print "MX.."
#         tic = time.time()
#         res = omp.call_WDL(A=self.A[:self.nb_comp_chrom,:],spectrums=self.spectrums,flux=self.flux,sig=self.sig,ker=self.ker,rot_ker=self.ker_rot, 
#             D_stack=D_stack,w_stack=self.w_stack,gamma=self.gamma,n_iter_sink=self.n_iter_sink,N=self.N,func='--MX_wdl')
#         toc = time.time()
#         print "Done in: " + str((toc-tic)/60.0) + " min"
#         self.MX_dict__MX = res[0]
#         self.MX_dict__barys = res[1]

#         # print ">>>> Energy of reconstructed stars: ", np.sum(abs(self.MtMX__MX),axis=(0,1))

#         return self.MX_dict__MX

#     def compute_MX(self,mix_stack):

#         dic = mix_stack[0]
#         S = mix_stack[1]

#         self.compute_MX_MX = self.compute_MX_dict(dic) + self.compute_MX_RCA(S)

#         return self.compute_MX_MX


#     def compute_barys(self,D_stack=None,w_stack=None):

#         dic = self.mix_stack[0]
#         w = self.w_stack

#         if D_stack is not None:
#             dic = D_stack
#         if w_stack is not None:
#             w = w_stack

#         print "Computing barycenters.."
#         tic = time.time()
#         self.compute_barys__barys = omp.call_WDL(D_stack=dic,w_stack=w,gamma=self.gamma,n_iter_sink=self.n_iter_sink,
#             func="--bary",N=self.N,remove_files=True)
#         toc = time.time()
#         print "Done in: " + str((toc-tic)/60.0) + " min"

#         return self.compute_barys__barys


#     def compute_grad(self,stars=None,mix_stack=None,use_cache_grad=False):


#         self.compute_grad__iter += 1

#         if not use_cache_grad:
#             y = self.stars
#             dic = self.mix_stack[0]
#             S = self.mix_stack[1]
            
#             if stars is not None:
#                 y = stars

#             if mix_stack is not None:
#                 dic = mix_stack[0]
#                 S = mix_stack[1]

#             res_dict = self.compute_MtMX_dict(stars=y,D_stack=dic)
#             res_RCA = self.compute_MtX_RCA(self.compute_MX_RCA(S) - y)
#             self.compute_grad__resRCA = np.copy(res_RCA)
   
#                res = np.empty(2,dtype=np.ndarray)
#                res[0] = np.copy(res_dict)
#                res[1] = np.copy(res_RCA)
#             # res = [ np.copy(res_dict), np.copy(res_RCA)]
            

#             self.compute_grad__res = res
        
        
#         return self.compute_grad__res



#     def get_grad(self,mix_stack): #active command, in modopt style


#         print "get_grad called"

#         self.get_grad__iter += 1

#         self.grad = self.compute_grad(mix_stack = mix_stack,use_cache_grad=True)

#         return self.grad


#     def cost(self, x, y=None, verbose=False):
#         """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
#         the dual variable.)
#         """
#         # if isinstance(self.MtMX__MX, type(None)):
#         #     self._current_rec = self.compute_MX(x)
#         # else:
#         #     self._current_rec = self.MtMX__MX

#         self._current_rec = self.compute_MX(x)

#         cost_val = 0.5 * np.linalg.norm(self._current_rec - self.stars) ** 2
#         if verbose:
#             print " > MIN(X):\t{0}\t{1}".format(np.min(x[0]), np.min(x[1]))
#             print " > Current cost: {}".format(cost_val)
#         return cost_val


#     def gamma_update(self,gamma):


#         self.gamma_update__iter += 1

#         print "gamma update called"


#         if self.min_dict is not None:
#             x_0 = np.copy(self.min_dict._x_new)
#         else:
#             x_0 = np.copy(self.mix_stack)

#         grad = self.compute_grad(mix_stack=x_0)
#         f_0 = self.cost(x_0) # this will be computed twice :/
#         gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=1.0)

#         return gamma

    
#     def sig_tau_update(self,sigma):

#         print "sigma update called"

#         if self.min_dict is not None:
#             x_0 = np.copy(self.min_dict._x_new)

#         else:
#             x_0 = np.copy(self.mix_stack)
            
#         grad = self.compute_grad(mix_stack=x_0)
#         f_0 = self.cost(x_0) # this will be computed twice :/
#         L_est = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=1.0)

#         sigma = 1.0/(L_est + self.lin_l1norm)
#         tau = np.copy(sigma)

#         # Check convergence condition
#         if 1.0/tau - sigma*self.lin_l1norm**2 >= L_est/2.0 :
#             print "good combination of parameters"
#         else:
#             print "attention: bad combination "

#         if self.min_dict is not None:
#             self.min_dict._tau = tau

#         return sigma
#     # def lambda_update(self,lbda):

#     #     gamma = self.min_dict.gamma

#     #     if lbda > 0 and lbda < 1.0:


class polychrom_eigen_psf_coeff_A(GradParent, PowerMethod):

    def __init__(self,A,spectrums,flux,sig,ker,ker_rot,D_stack,w_stack, gamma, n_iter_sink,stars,barycenters=None,data_type=float):
        self._grad_data_type = data_type
        self.A = A
        self.spectrums = spectrums
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot = ker_rot
        self.D_stack = D_stack
        self.w_stack = w_stack
        self.gamma = gamma
        self.n_iter_sink = n_iter_sink
        self.stars = stars
        self.shap = (stars.shape[0]*2,stars.shape[1]*2)
        self.N = self.shap[0]*self.shap[1]

        self.compute_barys__barys = None
        self.MtX__MtX = None
        self.MX__MX = None
        self.barycenters = barycenters


        self.min_coef = None
        self.compute_grad__iter = 0
        self.get_grad__iter  = 0
        self.gamma_update__iter = 0


        self.costs = []
        self.steps = []

        # PowerMethod.__init__(self, self.compute_MtMX_pm, alpha.shape,verbose=True)

    def reset_costs(self):
        self.costs = []

    def reset_steps(self):
        self.steps = []

    def set_A(self,A_new):
        self.A = np.copy(A_new)


    def set_D_stack(self,D_stack_new):
        self.D_stack = np.copy(D_stack_new)

    def set_flux(self,flux_new):
        self.flux = np.copy(flux_new)

    def get_flux(self):
        return self.flux


    def set_min_coef(self, min_coef_new):
        self.min_coef = min_coef_new


    def set_barycenters(self, barycenters_new):
        self.barycenters = np.copy(barycenters_new)

    def reset_iterations(self):
        self.compute_grad__iter = 0
        self.get_grad__iter  = 0
        self.gamma_update__iter = 0

    def get_barycenters(self):
        return self.barycenters

    def compute_barys(self,D_stack=None,w_stack=None): # recover it from grads dict if it gets to slow
        
        dic = self.D_stack
        w = self.w_stack

        if D_stack is not None:
            dic = D_stack
        if w_stack is not None:
            w = w_stack

        print "Computing barycenters.."
        tic = time.time()
        self.compute_barys__barys = omp.call_WDL(D_stack=dic,w_stack=w,gamma=self.gamma,n_iter_sink=self.n_iter_sink,
            func="--bary",N=self.N,remove_files=True)
        toc = time.time()
        print "Done in: " + str((toc-tic)/60.0) + " min"

        return self.compute_barys__barys


    def compute_MtX(self,mx,use_cache_bary=False):


        if use_cache_bary and self.barycenters is not None :
            barycenters = self.barycenters
        else:
            barycenters = self.compute_barys()
        A = psflu.MtX_coeff_graph(mx,barycenters,self.spectrums,self.sig,self.flux,self.ker_rot)

        self.MtX__MtX = A

        return self.MtX__MtX

    def MX(self):
        return self.compute_MX()

    def compute_MX(self,A=None,D_stack=None,w_stack=None,barycenters=None,use_cache_bary=False):

        a = self.A
        dic = self.D_stack
        w = self.w_stack

        if A is not None:
            a = A
        if D_stack is not None:
            dic = D_stack
        if w_stack is not None:
            w = w_stack

        if use_cache_bary and self.barycenters is not None :
            barys = self.barycenters
        elif barycenters is not None:
            barys = barycenters
        else:
            barys = self.compute_barys(dic,w)
        self.MX__MX = psflu.MX(a,barys,self.spectrums,self.sig,self.flux,self.ker)

        # print ">>>> Energy of reconstructed stars: ", np.sum(abs(self.MX__MX),axis=(0,1))

        return self.MX__MX


    def compute_MtMX_pm(self,x):


        temp =  self.compute_MtX(self.compute_MX(A=x,use_cache_bary=True),use_cache_bary=True)

        return temp

    def compute_grad(self,A=None,use_cache_grad=False):

        if not use_cache_grad:
            a = self.A
            if A is not None:
                a = A
            self.grad = self.compute_MtX(self.compute_MX(A=a,use_cache_bary=True)-self.stars,use_cache_bary=True)
             
        return self.grad

    def get_grad(self,A):

        print "get_grad COEF A"

        res = self.compute_grad(A=A,use_cache_grad=True)

        return res


    def cost(self, x, y=None, verbose=False,count=True,use_cache=False):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
        the dual variable.)
        """
        self._current_rec = self.compute_MX(A=x,use_cache_bary=True)

        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.stars) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)

        if count:
            self.costs.append(cost_val)
            
            
#        import pdb; pdb.set_trace()  # breakpoint bc817e81 //

        return cost_val



    def gamma_update(self,gamma):


        if self.min_coef is not None:
            print "got x new"
            x_0 = np.copy(self.min_coef._x_new)
        else:
            x_0 = np.copy(self.A)

        grad = self.compute_grad(A=x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost)

        self.steps.append(gamma)

        return gamma



class polychrom_eigen_psf_coeff_graph(GradParent, PowerMethod):

    def __init__(self,alpha,basis,spectrums,flux,sig,ker,ker_rot,D_stack,w_stack, gamma, n_iter_sink,stars,barycenters=None,data_type=float):
        self._grad_data_type = data_type
        self.alpha = alpha
        self.basis = basis
        self.spectrums = spectrums
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot = ker_rot
        self.D_stack = D_stack
        self.w_stack = w_stack
        self.gamma = gamma
        self.n_iter_sink = n_iter_sink
        self.stars = stars
        self.shap = (stars.shape[0]*2,stars.shape[1]*2)
        self.N = self.shap[0]*self.shap[1]

        self.compute_barys__barys = None
        self.MtX__MtX = None
        self.MX__MX = None
        self.barycenters = barycenters


        self.min_coef = None
        self.compute_grad__iter = 0
        self.get_grad__iter  = 0
        self.gamma_update__iter = 0


        self.costs = []
        self.steps = []

        # PowerMethod.__init__(self, self.compute_MtMX_pm, alpha.shape,verbose=True)

    def reset_costs(self):
        self.costs = []

    def reset_steps(self):
        self.steps = []

    def set_alpha(self,alpha_new):
        self.alpha = np.copy(alpha_new)


    def set_D_stack(self,D_stack_new):
        self.D_stack = np.copy(D_stack_new)

    def set_flux(self,flux_new):
        self.flux = np.copy(flux_new)

    def get_flux(self):
        return self.flux


    def set_min_coef(self, min_coef_new):
        self.min_coef = min_coef_new


    def set_barycenters(self, barycenters_new):
        self.barycenters = np.copy(barycenters_new)

    def reset_iterations(self):
        self.compute_grad__iter = 0
        self.get_grad__iter  = 0
        self.gamma_update__iter = 0

    def get_barycenters(self):
        return self.barycenters

    def compute_barys(self,D_stack=None,w_stack=None): # recover it from grads dict if it gets to slow
        
        dic = self.D_stack
        w = self.w_stack

        if D_stack is not None:
            dic = D_stack
        if w_stack is not None:
            w = w_stack

        print "Computing barycenters.."
        tic = time.time()
        self.compute_barys__barys = omp.call_WDL(D_stack=dic,w_stack=w,gamma=self.gamma,n_iter_sink=self.n_iter_sink,
            func="--bary",N=self.N,remove_files=True)
        toc = time.time()
        print "Done in: " + str((toc-tic)/60.0) + " min"

        return self.compute_barys__barys


    def compute_MtX(self,mx,use_cache_bary=False):


        if use_cache_bary and self.barycenters is not None :
            barycenters = self.barycenters
        else:
            barycenters = self.compute_barys()
        A = psflu.MtX_coeff_graph(mx,barycenters,self.spectrums,self.sig,self.flux,self.ker_rot)

        self.MtX__MtX = A.dot(np.transpose(self.basis))*1.0/(1.0*A.shape[0])

        return self.MtX__MtX

    def MX(self):
        return self.compute_MX()

    def compute_MX(self,alpha=None,D_stack=None,w_stack=None,barycenters=None,use_cache_bary=False):

        a = self.alpha
        dic = self.D_stack
        w = self.w_stack

        if alpha is not None:
            a = alpha
        if D_stack is not None:
            dic = D_stack
        if w_stack is not None:
            w = w_stack

        if use_cache_bary and self.barycenters is not None :
            barys = self.barycenters
        elif barycenters is not None:
            barys = barycenters
        else:
            barys = self.compute_barys(dic,w)
        self.MX__MX = psflu.MX(a.dot(self.basis),barys,self.spectrums,self.sig,self.flux,self.ker)

        # print ">>>> Energy of reconstructed stars: ", np.sum(abs(self.MX__MX),axis=(0,1))

        return self.MX__MX


    def compute_MtMX_pm(self,x):


        temp =  self.compute_MtX(self.compute_MX(alpha=x,use_cache_bary=True),use_cache_bary=True)

        return temp

    def compute_grad(self,alpha=None,use_cache_grad=False):


        self.compute_grad__iter += 1

        if not use_cache_grad:
            a = self.alpha
            if alpha is not None:
                a = alpha
            self.grad = self.compute_MtX(self.compute_MX(alpha=a,use_cache_bary=True)-self.stars,use_cache_bary=True)
             
        return self.grad

    def get_grad(self,alpha):

        print "get_grad called"

        self.get_grad__iter += 1

        res = self.compute_grad(alpha=alpha,use_cache_grad=True)

        return res


    def cost(self, x, y=None, verbose=False,count=True,use_cache=False):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
        the dual variable.)
        """
        self._current_rec = self.compute_MX(alpha=x,use_cache_bary=True)

        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.stars) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)

        if count:
            self.costs.append(cost_val)
        return cost_val



    def gamma_update(self,gamma):


        if self.min_coef is not None:
            print "got x new"
            x_0 = np.copy(self.min_coef._x_new)
        else:
            x_0 = np.copy(self.alpha)

        grad = self.compute_grad(alpha=x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost)

        self.steps.append(gamma)

        return gamma

class polychrom_eigen_psf_coeff_graph_RCA(GradParent, PowerMethod):

    def __init__(self,alpha,basis,flux,sig,ker,ker_rot,stars,S_stack,D=2,data_type=float):
        self._grad_data_type = float
        self.alpha = alpha
        self.basis = basis
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot = ker_rot
        self.stars = stars
        self.shap = (stars.shape[0]*D,stars.shape[1]*D)
        self.N = self.shap[0]*self.shap[1]
        self.S_stack = S_stack
        self.MtX__MtX = None
        self.MX__MX = None

        self.min_coef = None
        self.costs = []
        self.steps = []

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    def set_S_stack(self,new_S_stack):
        self.S_stack = new_S_stack

    def set_flux(self,flux_new):
        self.flux = np.copy(flux_new)

    def get_flux(self):
        return self.flux

    def MtX(self,mx,S_stack=None):


        if S_stack  is not None :
            S = S_stack
        else:
            S = self.S_stack

        A = psflu.MtX_coef_graph_RCA(mx,S,self.sig,self.flux,self.ker_rot)

        self.MtX__MtX = A.dot(np.transpose(self.basis))*1.0/(1.0*A.shape[0])


        return self.MtX__MtX


    def MX(self,alpha, S_stack=None):

        if S_stack  is not None :
            S = S_stack
        else:
            S = self.S_stack

        self.MX__MX = psflu.MX_RCA(S,alpha.dot(self.basis),self.flux,self.sig,self.ker)


        return self.MX__MX

    def compute_grad(self,alpha, S_stack=None,use_cache_grad=False):

        if not use_cache_grad:
            if S_stack is not None:
                self.grad =  self.MtX(self.MX(alpha,S_stack=S_stack) - self.stars,S_stack=S_stack)
            else:
                self.grad =  self.MtX(self.MX(alpha) - self.stars)

        return self.grad


    def set_min_coef(self, min_coef_new):
        self.min_coef = min_coef_new

    def reset_costs(self):
        self.costs = []

    def reset_steps(self):
        self.steps = []

    def get_grad(self,alpha):

        print "get_grad called"

        return self.compute_grad(alpha,use_cache_grad=True)


    def cost(self, x, y=None, verbose=False,count=True):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
        the dual variable.)
        """
        self._current_rec = self.MX(x)

        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.stars) ** 2
        if verbose:
            print " > MIN(X):\t{}".format(np.min(x))
            print " > Current cost: {}".format(cost_val)

        if count:
            print "loss counted"
            self.costs.append(cost_val)
        
        return cost_val


    def gamma_update(self,gamma):


        if self.min_coef is not None:
            print "got x new"
            x_0 = np.copy(self.min_coef._x_new)
        else:
            x_0 = np.copy(self.alpha)

        grad = self.compute_grad(x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost)

        self.steps.append(gamma)

        return gamma



class polychrom_eigen_psf_coeff_graph_wrapper(GradParent, PowerMethod):

    def __init__(self,alpha_mix,basis_mix,spectrums,flux,sig,ker,ker_rot,mix_stack,w_stack, gamma, n_iter_sink,stars,barycenters=None,data_type=np.ndarray):    
        self._grad_data_type = data_type
        self.alpha_mix = alpha_mix
        self.basis_mix = basis_mix
        self.spectrums = spectrums
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot = ker_rot
        self.mix_stack = mix_stack
        self.w_stack = w_stack
        self.gamma = gamma
        self.n_iter_sink = n_iter_sink
        self.stars = stars
        self.shap = (stars.shape[0]*2,stars.shape[1]*2)
        self.N = self.shap[0]*self.shap[1]

        self.Min_wdl = polychrom_eigen_psf_coeff_graph(alpha_mix[0],basis_mix[0],spectrums,flux,sig,ker,ker_rot,mix_stack[0],w_stack, 
            gamma, n_iter_sink,stars,barycenters=barycenters)
        self.Min_RCA = polychrom_eigen_psf_coeff_graph_RCA(alpha_mix[1],basis_mix[1],flux,sig,ker,ker_rot,stars,mix_stack[1])

        self.min_coef = None
        self.compute_grad__wdl = None
        self.compute_grad__RCA = None
        self.grad = None
        self.MX__wdl = None
        self.MX__RCA = None
        self.MX__MX = None

        self.costs = []
        self.steps = []

    def reset_costs(self):
        self.costs = []

    def reset_steps(self):
        self.steps = []
    def set_alpha_mix(self,alpha_new):
        self.alpha_mix = np.copy(alpha_new)
        self.Min_wdl.set_alpha(alpha_new[0])
        self.Min_RCA.set_alpha(alpha_new[1])

    def set_mix_stack(self,mix_stack_new):
        self.mix_stack = np.copy(mix_stack_new)
        self.Min_wdl.set_D_stack(mix_stack_new[0])
        self.Min_RCA.set_S_stack(mix_stack_new[1])

    def set_flux(self,flux_new):
        self.flux = np.copy(flux_new)
        self.Min_wdl.set_flux(flux_new)
        self.Min_RCA.set_flux(flux_new)

    def get_flux(self):
        return self.flux


    def set_min_coef(self, min_coef_new):
        self.min_coef = min_coef_new


    def set_barycenters(self, barycenters_new):
        self.barycenters = np.copy(barycenters_new)
        self.Min_wdl.set_barycenters(barycenters_new)


    def MX(self,alpha_mix=None):
        if alpha_mix is not None:
            a = alpha_mix
        else:
            a = self.alpha_mix
        self.MX__wdl = self.Min_wdl.compute_MX(alpha=a[0],use_cache_bary=True)
        self.MX__RCA = self.Min_RCA.MX(a[1])

        self.MX__MX = self.MX__wdl + self.MX__RCA

        return self.MX__MX


    def compute_grad(self,alpha_mix,use_cache_grad=False):

        if not use_cache_grad:
            self.compute_grad__wdl = self.Min_wdl.compute_grad(alpha=alpha_mix[0])
            self.compute_grad__RCA = self.Min_RCA.compute_grad(alpha_mix[1])
            
            res = np.empty(2, dtype=np.ndarray) 
            res[0] = np.copy(self.compute_grad__wdl)
            res[1] = np.copy(self.compute_grad__RCA)

            self.grad = res
 
        return self.grad

    def get_grad(self,alpha_mix):

        print "get_grad called"

        return self.compute_grad(alpha_mix,use_cache_grad=True)


    def cost(self, x, y=None, verbose=False,count=True):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
        the dual variable.)
        """
        self._current_rec = self.MX(alpha_mix = x)

        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.stars) ** 2
        if verbose:
            print " > MIN(X):\t{0}\t{1}".format(np.min(x[0]), np.min(x[1]))
            print " > Current cost: {}".format(cost_val)

        if count:
            print "loss counted"
            self.costs.append(cost_val)
        
        return cost_val


    def gamma_update(self,gamma):


        if self.min_coef is not None:
            print "got x new"
            x_0 = np.copy(self.min_coef._x_new)
        else:
            x_0 = np.copy(self.alpha_mix)

        grad = self.compute_grad(x_0)
        f_0 = self.cost(x_0,count=False) # this will be computed twice :/
        gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost)

        self.steps.append(gamma)

        return gamma





# class polychrom_eigen_psf_coeff_graph_mix(GradParent, PowerMethod):

#     def __init__(self,alpha_mix,basis,spectrums,flux,sig,ker,ker_rot,mix_stack,w_stack, gamma, n_iter_sink,stars,barycenters=None,data_type=np.ndarray):
#         self._grad_data_type = data_type
#         self.alpha_mix = alpha_mix
#         self.basis = basis
#         self.spectrums = spectrums
#         self.flux = flux
#         self.sig = sig
#         self.ker = ker
#         self.ker_rot = ker_rot
#         self.mix_stack = mix_stack
#         self.w_stack = w_stack
#         self.gamma = gamma
#         self.n_iter_sink = n_iter_sink
#         self.stars = stars
#         self.shap = (stars.shape[0]*2,stars.shape[1]*2)
#         self.N = self.shap[0]*self.shap[1]
#         self.nb_comp_chrom = mix_stack[0].shape[-1]

#         self.compute_barys__barys = None
#         self.MtX_dict__MtX = None
#         self.MX_dict__MX = None
#         self.MtX_RCA__MtX = None
#         self.MX_RCA__MX = None
#         self.MX__MX = None 
#         self.compute_grad__resdict = None
#         self.compute_grad__resRCA = None
#         self.barycenters = barycenters


#         self.min_coef = None
#         self.compute_grad__iter = 0
#         self.get_grad__iter  = 0
#         self.gamma_update__iter = 0

#         # PowerMethod.__init__(self, self.compute_MtMX_pm, alpha.shape,verbose=True)


#     def set_alpha_mix(self,alpha_new):
#         self.alpha_mix = np.copy(alpha_new)


#     def set_mix_stack(self,mix_stack_new):
#         self.mix_stack = np.copy(mix_stack_new)

#     def set_flux(self,flux_new):
#         self.flux = np.copy(flux_new)

#     def get_flux(self):
#         return self.flux


#     def set_min_coef(self, min_coef_new):
#         self.min_coef = min_coef_new


#     def set_barycenters(self, barycenters_new):
#         self.barycenters = np.copy(barycenters_new)

#     def reset_iterations(self):
#         self.compute_grad__iter = 0
#         self.get_grad__iter  = 0
#         self.gamma_update__iter = 0

#     def get_barycenters(self):
#         return self.barycenters

#     def compute_barys(self,D_stack=None,w_stack=None): # recover it from grads dict if it gets to slow
        
#         dic = self.mix_stack[0]
#         w = self.w_stack

#         if D_stack is not None:
#             dic = D_stack
#         if w_stack is not None:
#             w = w_stack

#         print "Computing barycenters.."
#         tic = time.time()
#         self.compute_barys__barys = omp.call_WDL(D_stack=dic,w_stack=w,gamma=self.gamma,n_iter_sink=self.n_iter_sink,
#             func="--bary",N=self.N,remove_files=True)
#         toc = time.time()
#         print "Done in: " + str((toc-tic)/60.0) + " min"

#         return self.compute_barys__barys


#     def compute_MtX_dict(self,mx,use_cache_bary=False):

        

#         if use_cache_bary and self.barycenters is not None :
#             barycenters = self.barycenters
#         else:
#             barycenters = self.compute_barys()
#         A = psflu.MtX_coeff_graph(mx,barycenters,self.spectrums,self.sig,self.flux,self.ker_rot)

#         self.MtX_dict__MtX = A.dot(np.transpose(self.basis))*1.0/(1.0*A.shape[0])


#         return self.MtX_dict__MtX


#     def compute_MX_dict(self,alpha_mix,D_stack=None,w_stack=None,barycenters=None,use_cache_bary=False):

#         dic = self.mix_stack[0]
#         w = self.w_stack

        
#         if D_stack is not None:
#             dic = D_stack
#         if w_stack is not None:
#             w = w_stack

#         if use_cache_bary and self.barycenters is not None :
#             barys = self.barycenters
#         elif barycenters is not None:
#             barys = barycenters
#         else:
#             barys = self.compute_barys(D_stack=dic,w_stack=w)
#         self.MX_dict__MX = psflu.MX(alpha_mix[0].dot(self.basis),barys,self.spectrums,self.sig,self.flux,self.ker)

#         # print ">>>> Energy of reconstructed stars: ", np.sum(abs(self.MX__MX),axis=(0,1))

#         return self.MX_dict__MX


#     def compute_MtX_RCA(self,mx,S_stack=None):

#         S = self.mix_stack[1]
#         if S_stack  is not None :
#             S = S_stack

#         A = psflu.MtX_coef_graph_RCA(mx,S,self.sig,self.flux,self.ker_rot,D=2)



#         self.MtX_RCA__MtX = A.dot(np.transpose(self.basis))*1.0/(1.0*A.shape[0])


#         return self.MtX_RCA__MtX



#     def compute_MX_RCA(self,alpha_mix, S_stack=None):

#         S = self.mix_stack[1]

#         if S_stack is not None:
#             S = S_stack



#         self.MX_RCA__MX = psflu.MX_RCA(S,alpha_mix[1].dot(self.basis),self.flux,self.sig,self.ker)


#         return self.MX_RCA__MX


#     def compute_MX(self,alpha_mix=None):

#         a = self.alpha_mix

#         if alpha_mix is not None:
#             a = alpha_mix



#         self.MX__MX = self.compute_MX_dict(a,use_cache_bary=True) + self.compute_MX_RCA(a)


#         return self.MX__MX 

#     # def compute_MtMX_pm(self,x):


#     #     temp =  self.compute_MtX(self.compute_MX(alpha=x,use_cache_bary=True),use_cache_bary=True)

#     #     return temp

#     def compute_grad(self,alpha_mix=None,use_cache_grad=False):


#         self.compute_grad__iter += 1

#         if not use_cache_grad:
#             a = self.alpha_mix
#             if alpha_mix is not None:
#                 a = alpha_mix
#             self.compute_grad__resdict = self.compute_MtX_dict(self.compute_MX(alpha_mix = a)-self.stars,use_cache_bary=True)
#             self.compute_grad__resRCA =  self.compute_MtX_RCA(self.compute_MX(alpha_mix = a)-self.stars)

#             res = np.empty(2, dtype=np.ndarray) 
#             res[0] = np.copy(self.compute_grad__resdict)
#             res[1] = np.copy(self.compute_grad__resRCA)

#             self.grad = res




#         return self.grad

#     def get_grad(self,alpha_mix):

#         print "get_grad called"


#         self.get_grad__iter += 1

#         res = self.compute_grad(alpha_mix=alpha_mix,use_cache_grad=True)

#         return res


#     def cost(self, x, y=None, verbose=False):
#         """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` can feed
#         the dual variable.)
#         """
#         self._current_rec = self.compute_MX(alpha_mix = x)

#         cost_val = 0.5 * np.linalg.norm(self._current_rec - self.stars) ** 2
#         if verbose:
#             print " > MIN(X):\t{0}\t{1}".format(np.min(x[0]), np.min(x[1]))
#             print " > Current cost: {}".format(cost_val)
#         return cost_val


#     def gamma_update(self,gamma):


#         self.gamma_update__iter += 1


#         if self.min_coef is not None:
#             print "got x new"
#             x_0 = np.copy(self.min_coef._x_new)
#         else:
#             x_0 = np.copy(self.alpha_mix)


#         grad = self.compute_grad(alpha_mix=x_0)
#         f_0 = self.cost(x_0) # this will be computed twice :/
#         gamma = utils.back_tracking_armijo_line_search(x_0, grad, f_0, self.cost, alpha=1.0)

#         return gamma






