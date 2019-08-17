import psf_toolkit as tk
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time as time
import sys
import matplotlib.gridspec as gridspec
import C_wrapper as omp
import grads as grads
from modopt.opt.cost import costObj
import modopt.opt.algorithms as optimalg
from modopt.opt.linear import Identity
from modopt.opt.linear import LinearCombo
from modopt.opt.proximity import ProximityCombo, IdentityProx
from modopt.signal.wavelet import get_mr_filters, filter_convolve
import proxs
import transforms
import psf_learning_utils as psflu
import linear
import utils
import os
import optim_utils
from numpy import zeros,size,where,ones,copy,around,double,sinc,random,pi,arange,cos,sin,arccos,transpose,diag,sqrt,arange,floor,exp,array,mean,roots,float64,int,pi,median,rot90,argsort,tile,repeat,squeeze
import scipy.signal as scisig
sys.path.append('opt')
import algorithms as modoptAlgorithms 


save_path = '/Users/mschmitz/Desktop/BecaV2/output_firstgo'
if not os.path.exists(save_path):
        os.makedirs(save_path)

result_path = save_path +"/result"
if not os.path.exists(result_path):
        os.makedirs(result_path)


def reconstruct_stars_est_2euclidres(psfs,SEDs):
    nb_obj = psfs.shape[2]
    W = psfs.shape[0]
    stars2euclidres = np.zeros((W,W,nb_obj))

    for obj in range(nb_obj):
        stars2euclidres[:,:,obj] = np.sum(psfs[:,:,obj,:] * SEDs[:,obj].reshape(1,1,-1), axis=-1)

    return stars2euclidres

def field_reconstruction_wdl(barycenters,A,shap,S_stack=None): #TO DO: implement that in theano
    nb_bands = barycenters.shape[2]
    nb_obj = A.shape[1]
    mono_chromatic_psf = np.zeros((shap[0],shap[1],nb_obj,nb_bands))
    nb_comp_chrom =  A.shape[0]
    if S_stack is not None:
        nb_comp_chrom = nb_comp_chrom - S_stack.shape[-1]



    for v in range(nb_bands):
        mono_chromatic_psf_temp = barycenters[:,:,v].dot(A[:nb_comp_chrom,:])
        for k in range(nb_obj):
            mono_chromatic_psf[:,:,k,v] = mono_chromatic_psf_temp[:,k].reshape((shap[0],shap[1]))
    

    if S_stack is not None:
        spatial_psf = S_stack.dot(A[nb_comp_chrom:,:])
        for v in range(nb_bands):
            mono_chromatic_psf[:,:,:,v] += spatial_psf.reshape((shap[0],shap[1],nb_obj))
    
    return mono_chromatic_psf

def field_reconstruction_RCA(S_stack,A,shap):
    psfs = np.zeros((shap[0],shap[1],A.shape[1]))
    temp = S_stack.dot(A)
    for obj in range(A.shape[1]):
        psfs[:,:,obj] = temp[:,obj].reshape(shap)
    return psfs


def field_reconstruction_mix(barycenters,A_chrom,shap,S_stack,A_rca):

    psfs_wdl = field_reconstruction_wdl(barycenters,A_chrom,shap)
    psfs_rca = field_reconstruction_RCA(S_stack,A_rca,shap)

    psfs = np.copy(psfs_wdl)
    for wvl in range(barycenters.shape[2]):
        psfs[:,:,:,wvl] += psfs_rca

    return psfs

def mse(x,y,N): 
    """    Parameters
        ----------
        x,y: multidimensional arrays with "pixels" at first 2 dimensions
    """ 
    return np.sum((x - y)**2,axis=(0,1))/N


def euclidean_cost(x,y):
    """    Parameters
        ----------
        x,y: cubic arrays with "pixels" at first 2 dimensions
    """ 
    # c = 0.0
    # for i in range(x.shape[-1]):
    #     c += 0.5*(np.sum((x[:,:,i]-y[:,:,i])**2))

    c = 0.5 * np.linalg.norm(x - y) ** 2
        

    return c



def relative_mse(y,x):
    """Parameters
       ----------
       x,y: multidimensional arrays with "pixels" at first 2 dimensions. y is the truth.

    """

    # return np.linalg.norm(x - y) ** 2/np.linalg.norm(y)
    return np.sum((x - y)**2,axis=(0,1))/np.sum(y**2,axis=(0,1))


def plot_dic(D_stack):
    for i in range(D_stack.shape[-1]):
        for at in range(D_stack.shape[1]):
            tk.plot_func(D_stack[:,at,i])

#%%
def plot_func(im, wind=False, cmap='gist_stern', norm=None, cutoff=5e-4,
                title='',cb=True):
    if cmap in ['sam','Sam']:
        cmap = Samcmap
        boundaries = np.arange(cutoff, np.max(im), 0.0001)
        norm = BoundaryNorm(boundaries, plt.cm.get_cmap(name=cmap).N)
    if len(im.shape) == 2:
        if not wind:
            plt.imshow(im, cmap=cmap, norm=norm,
                       interpolation='Nearest')
        else:
            vmin, vmax = wind
            plt.imshow(im, cmap=cmap, norm=norm,
                       interpolation='Nearest', vmin=vmin, vmax=vmax)
    else:
        sqrtN = int(np.sqrt(im.shape[0]))
        if not wind:
            plt.imshow(im.reshape(sqrtN,sqrtN), cmap=cmap, norm=norm,
                       interpolation='Nearest')
        else:
            vmin, vmax = wind
            plt.imshow(im.reshape(sqrtN,sqrtN), cmap=cmap, norm=norm, 
                       interpolation='Nearest', vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    if cb:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
#%%
def plot_sideByside(x,path=None):
    """Parameters
       ----------
       x: 4 dimensions array, where:
               dim 0:  iteration
               dim 1,2: pixels
               dim 3: wvl
    """

    fig = plt.figure()
    gs1 = gridspec.GridSpec(x.shape[0],x.shape[-1])
    gs1.update(wspace=0.05, hspace=0.0) # set the spacing between axes. 
    for i in range(x.shape[0]):
        for j in range(x.shape[-1]):
            ax1 = plt.subplot(gs1[i*x.shape[-1]+j])
            plot_func(x[i,:,:,j])
            plt.xlabel(r'${}$'.format(int(j)),fontsize=3)

    if path is not None:
        plt.savefig(path, format='jpeg', dpi=1000)
    plt.show()


def becafunc(stars, lbdas, spectrums,
             shifts, flux, sig, ker, ker_rot,
             alph, basis,
             all_lbdas=None):
    """ all_lbdas is needed if gt_wvlAll=True (you recompute
    """
    # stuff
    shap = (stars.shape[0]*2,stars.shape[1]*2)
    shap_lr = stars.shape
    nb_obj = stars.shape[-1]
    nb_wvl = lbdas.size

    # PARAMETERS! PARAMETERS ALL ABOUT!
    D=2
    opt_shift = ['-t2','-n2']
    nsig_shift_est=4
    # lbdaRCA parameters
    nb_atoms = 2 # this is actually a constant
    nb_comp = 3
    nb_comp_chrom = 3
    feat_init = "super_res_zout" #"ground_truth"
    feat_init_RCA = "super_res"
    gamma = 0.3
    n_iter_sink = 10
    list_iterations_dict = [6,4,3]
    # Gradient descent parameters / alternate optimization
    n_iter = 2 #6
    max_iter_FB_dict = 200 #4# rca Fred is 200 for each
    max_iter_FB_coef = 200 # 20
    list_iterations_coef = [30,30,30]
    n_iter = len(list_iterations_coef)
    alg = "genFB" #genFB
    logit = False
    gt_wvlAll = False #[BIGMORGANTAG]True
    
    fb_gamma_param_dict = 1.0
    fb_lambda_param_dict = 1.0 # must be in ]0,1[
    fb_gamma_param_coef = 1.0
    fb_lambda_param_coef = 1.0 # must be in ]0,1[
    
    if logit:
        alpha_step = [10.0,10.0,10.0] # 0.07 1.0

    else:
        alpha_step = [1.0,1.0,1.0] # 0.07 1.0

    # initialize A
    A = utils.cube_svd(stars,nb_comp=nb_comp)
    A = abs(A)

    # set up barycentric weights
    t = (lbdas-lbdas.min()).astype(float)/(lbdas.max()-lbdas.min())
    w_stack = np.array([t + 1e-10, 1 - t - 1e-10]).T

    if gt_wvlAll:
        t_wvlAll = (all_lbdas-all_lbdas.min()).astype(float)/(all_lbdas.max()-all_lbdas.min())
        w_stack_wvlAll = np.array([t_wvlAll + 1e-10, 1 - t_wvlAll - 1e-10]).T
   
    # First guess initialization
    first_guesses = psflu.SR_first_guesses_rnd(stars,shifts,nb_comp)
    D_stack = psflu.D_stack_first_guess(shap,nb_obj,nb_comp_chrom,feat_init,sr_first_guesses=first_guesses,logit=logit)

    D_stack = abs(D_stack)
    D_stack = D_stack/np.sum(D_stack, axis=0)

    Dlog_stack = np.log(D_stack)

    D_stack_0 = np.copy(D_stack)
    #np.save(save_path+'/D_stack_0.npy', D_stack)
    #np.save(save_path+'/spectrums.npy', spectrums)

    # Indicators 
    MSE_rel_nor = []
    MSE_rel_nor_alternates = []
    MSE_rel_nor_integrated = []
    MSE_rel_nor_integrated_alternates = []
    loss_inners = []
    steps_inners = []
    D_stack_energy = []
    loss_alternates = []
    loss_iters = []
    MSE_rel_nor_stars_2euclidres = []
    MSE_rel_nor_stars_2euclidres_alternates = []
    np.save(save_path+'/A.npy',A)

    # Set up gradients
    Wdl_comp = grads.polychrom_eigen_psf(A,spectrums,flux,sig,ker,ker_rot,D_stack,w_stack, gamma, n_iter_sink,stars,logit=logit)
    Cost_comp = costObj([Wdl_comp])
    barycenters = Wdl_comp.compute_barys()
    Wdl_coef = grads.polychrom_eigen_psf_coeff_graph(alph,basis,spectrums,flux,sig,ker,ker_rot,D_stack,w_stack, gamma, n_iter_sink,stars,barycenters=barycenters)
    Wdl_coef_A = grads.polychrom_eigen_psf_coeff_A(A,spectrums,flux,sig,ker,ker_rot,D_stack,w_stack, gamma, n_iter_sink,stars,barycenters=barycenters)
    Cost_coef = costObj([Wdl_coef])
    Cost_coef_A = costObj([Wdl_coef_A])
    np.save(save_path+'/barycenters_0.npy', barycenters)

    # Set up prox's
    nsigma_dict = 3.0
    nsigma_RCA = 1.5
    noise_map_dict = psflu.get_noise_arr_dict_wvl(D_stack,shap)*nsigma_dict
    print "THRSHOLD_estimated: ",str(np.min(noise_map_dict)) 

    Wavelet_transf_dict = transforms.dict_wavelet_transform(shap,nb_comp_chrom)
    Lin_comb_wavelet_dict = linear.lin_comb(Wavelet_transf_dict)

    Sparse_prox_dict = proxs.SparseThreshold(noise_map_dict, Lin_comb_wavelet_dict,shap,nsigma_dict,logit=logit)
    Simplex_prox_dict = proxs.Simplex()
    LowPass_prox_dict = proxs.LowPass(shap,logit=logit)

    iter_func = lambda x: np.floor(np.sqrt(x))

    Sparse_prox_coef = proxs.KThreshold(iter_func)
    proxs_coef = [Sparse_prox_coef] 

    if logit:
        proxs_comp = [Sparse_prox_dict,LowPass_prox_dict]
    else:
        proxs_comp = [Simplex_prox_dict,Sparse_prox_dict,LowPass_prox_dict]

    # Iteration zero
    if gt_wvlAll:        
        barycenters_wvlAll = Wdl_comp.compute_barys(w_stack=w_stack_wvlAll)
        psf_est = field_reconstruction_wdl(barycenters_wvlAll,A,shap)
    else:
        psf_est = field_reconstruction_wdl(barycenters,A,shap)

    # Compute current estimates (all commented are just diagnostics I believe)
    """psf_est_shift = psflu.shift_PSF_to_gt(psf_est,gt_PSFs)
    psf_est_integrated_shift_nor = np.sum(psf_est_shift,axis=3)/np.sum(abs(np.sum(psf_est_shift,axis=3)),axis=(0,1))
    psf_est_shift_nor = psf_est_shift/np.sum(abs(psf_est_shift),axis=(0,1))
    MSE_rel_nor_alternates.append(relative_mse(gt_PSFs,psf_est_shift_nor))
    MSE_rel_nor.append(relative_mse(gt_PSFs,psf_est_shift_nor))"""
    Wdl_coef.set_barycenters(barycenters)
    obs_est = Wdl_coef.MX()
    """if gt_wvlAll:  
        stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,all_spectrums)
    else:
        stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,spectrums)
    stars_est_2euclidres_nor = stars_est_2euclidres/np.sum(abs(stars_est_2euclidres), axis=(0,1))"""
    
    """MSE_rel_nor_integrated_alternates.append(relative_mse(gt_integrated_nor,psf_est_integrated_shift_nor))
    MSE_rel_nor_integrated.append(relative_mse(gt_integrated_nor,psf_est_integrated_shift_nor))
    MSE_rel_nor_stars_2euclidres_alternates.append(relative_mse(gt_stars_2euclidrec,stars_est_2euclidres_nor))
    MSE_rel_nor_stars_2euclidres.append(relative_mse(gt_stars_2euclidrec,stars_est_2euclidres_nor))"""

    res_rec = euclidean_cost(obs_est,stars) 

    np.save(save_path+'/obs_est_0.npy', obs_est)
    #np.save(save_path+'/stars_est_2euclidres_0.npy', stars_est_2euclidres)

    print " > Current cost: {}".format(res_rec)
    loss_inners.append([res_rec])
    loss_alternates.append(res_rec)
    loss_iters.append(res_rec)

    # go I think
    tic_ext = time.time()
    for i in tqdm(range(n_iter)):
        print ">>>>>>>>>>>>>>>>>>>> Iteration " + str(i)
        if not os.path.exists(save_path+'/iter_'+str(i)):
            os.makedirs(save_path+'/iter_'+str(i))

        print "-------------------- Steps coefficients set-up -----------------------"
        Wdl_coef.min_coef = None
        if i ==0:
            fb_gamma_param_coef = Wdl_coef_A.gamma_update(fb_gamma_param_coef)
        else:
            fb_gamma_param_coef = Wdl_coef.gamma_update(fb_gamma_param_coef) # or maybe pass the entire update_step_function aqui
        
        print "-------------------- Coefficients estimation -----------------------"
        # Coefficients update
        if i == 0:
           min_coef = optimalg.GenForwardBackward(A,Wdl_coef_A,[IdentityProx()],Cost_coef_A,auto_iterate=False,
            gamma_param=fb_gamma_param_coef,lambda_param =fb_lambda_param_coef, gamma_update=Wdl_coef_A.gamma_update)
        else:
            min_coef = optimalg.GenForwardBackward(alph,Wdl_coef,proxs_coef,Cost_coef,auto_iterate=False,
            gamma_param=fb_gamma_param_coef,lambda_param =fb_lambda_param_coef, gamma_update=Wdl_coef.gamma_update)
        

        if i == 0:
            Wdl_coef_A.set_min_coef(min_coef) 
            Wdl_coef_A.reset_costs()
            Wdl_coef_A.reset_steps()
        else:
            Wdl_coef.set_min_coef(min_coef) 
            Wdl_coef.reset_costs()
            Wdl_coef.reset_steps()
        tic = time.time()
        if list_iterations_coef is None:
            min_coef.iterate(max_iter=max_iter_FB_coef)
        else:
            min_coef.iterate(max_iter=list_iterations_coef[i])
        toc = time.time()
        print "Done in: " + str((toc-tic)/60.0) + " min"

        # Update
        if i == 0:
            A = min_coef.x_final
            alph =  A.dot(np.transpose(basis))/(1.0*nb_comp)
        else:
            alph = min_coef.x_final
            A = alph.dot(basis)
    
        Wdl_coef.set_alpha(alph)
        Wdl_comp.set_A(alph.dot(basis))

        obs_est = Wdl_coef.MX()
        if gt_wvlAll:        
            barycenters_wvlAll = Wdl_comp.compute_barys(w_stack=w_stack_wvlAll)
            psf_est = field_reconstruction_wdl(barycenters_wvlAll,A,shap)
        else:
            psf_est = field_reconstruction_wdl(barycenters,A,shap)
       
        # diagnostics again I think
        """psf_est_shift = psflu.shift_PSF_to_gt(psf_est,gt_PSFs)
        # psf_est_inner_shift = 
        psf_est_integrated_shift_nor = np.sum(psf_est_shift,axis=3)/np.sum(abs(np.sum(psf_est_shift,axis=3)),axis=(0,1))
        psf_est_shift_nor = psf_est_shift/np.sum(abs(psf_est_shift),axis=(0,1))
        MSE_rel_nor_alternates.append(relative_mse(gt_PSFs,psf_est_shift_nor))
        if gt_wvlAll:  
            stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,all_spectrums)
        else:
            stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,spectrums)
        stars_est_2euclidres_nor = stars_est_2euclidres/np.sum(abs(stars_est_2euclidres), axis=(0,1))
   
        MSE_rel_nor_integrated_alternates.append(relative_mse(gt_integrated_nor,psf_est_integrated_shift_nor))
        MSE_rel_nor_stars_2euclidres_alternates.append(relative_mse(gt_stars_2euclidrec,stars_est_2euclidres_nor))
        energy_est = np.sum(abs(psf_est),axis=(0,1))
        print "PSF est energy: ",energy_est
        np.save(save_path+'/iter_'+str(i)+'/stars_est_2euclidres_2.npy', stars_est_2euclidres)
        np.save(save_path+'/iter_'+str(i)+'/psf_est_2.npy', psf_est)
        np.save(save_path+'/iter_'+str(i)+'/obs_est_2.npy', obs_est)
        np.save(save_path+'/iter_'+str(i)+'/psf_est_integrated_shift_nor_2.npy', psf_est_integrated_shift_nor)
        if problem_type == 'lbdaRCA':
            np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_nor_2.npy', psf_est_shift_nor)
            np.save(save_path+'/iter_'+str(i)+'/alpha_2.npy', alph)
            np.save(save_path+'/iter_'+str(i)+'/A_2.npy', A)
            np.save(save_path+'/iter_'+str(i)+'/D_stack_2.npy', D_stack)
            np.save(save_path+'/iter_'+str(i)+'/barycenters_2.npy', barycenters)
            np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_2.npy', psf_est_shift)
        elif problem_type == 'mix':
            np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_nor_2.npy', psf_est_shift_nor)
            np.save(save_path+'/iter_'+str(i)+'/A_chrom_2.npy', A_mix[0])
            np.save(save_path+'/iter_'+str(i)+'/D_stack_2.npy', mix_stack[0])
            np.save(save_path+'/iter_'+str(i)+'/S_stack_2.npy', mix_stack[1])
            np.save(save_path+'/iter_'+str(i)+'/A_rca_2.npy', A_mix[1])
            np.save(save_path+'/iter_'+str(i)+'/barycenters_2.npy', barycenters)
            np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_2.npy', psf_est_shift)
        elif problem_type == 'RCA':
            np.save(save_path+'/iter_'+str(i)+'/alpha_2.npy', alph)
            np.save(save_path+'/iter_'+str(i)+'/A_2.npy', alph.dot(basis))
            np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_2.npy', psf_integrated_shift)
            np.save(save_path+'/iter_'+str(i)+'/S_stack_2.npy', S_stack)"""

        # Bilan
        res_rec = euclidean_cost(obs_est,stars)
        print " > Cost in test_gd: {}".format(res_rec)
        print "Cost from dict"
        Wdl_comp.cost(D_stack, verbose=True,count=False)
        print "Cost from coef please be the same"
        Wdl_coef.cost(alph, verbose=True,count=False)
        if i ==0:
            loss_inners[0] += Wdl_coef_A.costs
            steps_inners.append(Wdl_coef_A.steps)
        else:
            loss_inners.append(Wdl_coef.costs)
            steps_inners.append(Wdl_coef.steps)
        loss_alternates.append(res_rec)
        
        
        print "-------------------- Steps dictionary set-up -----------------------"
        Wdl_comp.min_coef = None
        Wdl_comp.alpha = alpha_step[i] # se quiser mudar alpha, mudar la em cima na lista
        fb_gamma_param_dict = Wdl_comp.gamma_update(fb_gamma_param_dict)
        if i==0:
            fb_lambda_param_dict = 1.0
        else:
            fb_lambda_param_dict = np.copy(fb_gamma_param_dict)   
            
        print "-------------------- Dictionary estimation -----------------------"
        # Dictionary update
        
        if logit:
            min_dict = modoptAlgorithms.GenForwardBackward(Dlog_stack,Wdl_comp,proxs_comp,cost=Cost_comp,auto_iterate=False,
                gamma_param=fb_gamma_param_dict,lambda_param =fb_lambda_param_dict, gamma_update=Wdl_comp.gamma_update,logit=logit,weights=[0.7,0.3]) 
        else:
            min_dict = modoptAlgorithms.GenForwardBackward(D_stack,Wdl_comp,proxs_comp,cost=Cost_comp,auto_iterate=False,
                gamma_param=fb_gamma_param_dict,lambda_param =fb_lambda_param_dict, gamma_update=Wdl_comp.gamma_update,weights=[0.2,0.4,0.4])
        
        Wdl_comp.set_n_iter(i)
        Wdl_comp.set_min_dict(min_dict)
        Wdl_comp.reset_costs()
        Wdl_comp.reset_steps()
        Wdl_comp.reset_D_stack_energy()
        tic = time.time()
        if list_iterations_dict is None:
            min_dict.iterate(max_iter=max_iter_FB_dict)
        else:
            min_dict.iterate(max_iter=list_iterations_dict[i])
        toc = time.time()
        print "Done in: " + str((toc-tic)/60.0) + " min"    
        
        # Update
        if logit:
            Dlog_stack = min_dict.x_final
            D_stack = psflu.logitTonormal(Dlog_stack)
            Wdl_comp.set_Dlog_stack(Dlog_stack)
        else:
            D_stack = min_dict.x_final
            # Force the projection in simplex
            D_stack[D_stack< 0.0] = 1e-9

        Wdl_comp.set_D_stack(D_stack)
        print "Dictionary energy: ",np.sum(abs(D_stack), axis=0)
        Wdl_coef.set_D_stack(D_stack)
        
        barycenters = Wdl_comp.compute_barys()
        Wdl_coef.set_barycenters(barycenters)


        # Weights update
        noise_map_dict = psflu.get_noise_arr_dict_wvl(D_stack,shap)*nsigma_dict
        print "THRSHOLD_estimated: ",str(np.min(noise_map_dict))
        Sparse_prox_dict.update_weights(noise_map_dict)
        
        if logit:
            D_stack = psflu.logitTonormal(Dlog_stack)
            Wdl_comp.set_Dlog_stack(Dlog_stack)
        else:
            D_stack /= np.sum(abs(D_stack),axis=0)             
        Wdl_comp.set_D_stack(D_stack)
        Wdl_coef.set_D_stack(D_stack)
        print "Dictionary energy: ",np.sum(abs(D_stack), axis=0)
        
        #Compute barycenters
        barycenters = Wdl_comp.compute_barys()
        Wdl_coef.set_barycenters(barycenters) 
               
        # Evaluation
        if logit:
              obs_est = Wdl_comp.compute_MX(Dlog_stack) 
        else:              
              obs_est = Wdl_comp.compute_MX(D_stack)
        if gt_wvlAll:        
            barycenters_wvlAll = Wdl_comp.compute_barys(w_stack=w_stack_wvlAll)
            psf_est = field_reconstruction_wdl(barycenters_wvlAll,A,shap)
        else:
            psf_est = field_reconstruction_wdl(barycenters,A,shap)

        # diagnostics
        """psf_est_shift = psflu.shift_PSF_to_gt(psf_est,gt_PSFs)
        psf_est_integrated_shift_nor = np.sum(psf_est_shift,axis=3)/np.sum(abs(np.sum(psf_est_shift,axis=3)),axis=(0,1))
        psf_est_shift_nor = psf_est_shift/np.sum(abs(psf_est_shift),axis=(0,1))
        MSE_rel_nor_alternates.append(relative_mse(gt_PSFs,psf_est_shift_nor))

        if gt_wvlAll:  
            stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,all_spectrums)
        else:
            stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,spectrums)
        stars_est_2euclidres_nor = stars_est_2euclidres/np.sum(abs(stars_est_2euclidres), axis=(0,1))

        MSE_rel_nor_integrated_alternates.append(relative_mse(gt_integrated_nor,psf_est_integrated_shift_nor))
        MSE_rel_nor_stars_2euclidres_alternates.append(relative_mse(gt_stars_2euclidrec,stars_est_2euclidres_nor))
        
        energy_est = np.sum(abs(psf_est),axis=(0,1))
        print "PSF est energy: ",energy_est

        np.save(save_path+'/iter_'+str(i)+'/stars_est_2euclidres_1.npy', stars_est_2euclidres)
        np.save(save_path+'/iter_'+str(i)+'/psf_est_1.npy', psf_est)
        np.save(save_path+'/iter_'+str(i)+'/obs_est_1.npy', obs_est)
        np.save(save_path+'/iter_'+str(i)+'/psf_est_integrated_shift_nor_1.npy', psf_est_integrated_shift_nor)
    
        np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_nor_1.npy', psf_est_shift_nor)
        np.save(save_path+'/iter_'+str(i)+'/alpha_1.npy', alph)
        np.save(save_path+'/iter_'+str(i)+'/A_1.npy', A)
        np.save(save_path+'/iter_'+str(i)+'/D_stack_1.npy', D_stack)
        np.save(save_path+'/iter_'+str(i)+'/barycenters_1.npy', barycenters)
        np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_1.npy', psf_est_shift)"""

        # Bilan
        res_rec = euclidean_cost(obs_est,stars)
        print " > Cost in test_gd: {}".format(res_rec)

        print "Cost from dict"
        if logit:
            res_rec_dict = Wdl_comp.cost(Dlog_stack, verbose=True,count=False)
        else:
            res_rec_dict = Wdl_comp.cost(D_stack, verbose=True,count=False)
        print "Cost from coef please be the same"
        Wdl_coef.cost(alph, verbose=True,count=False)


        D_stack_energy += Wdl_comp.D_stack_energy
        if i ==0:
            loss_inners[0]+= Wdl_comp.costs
        else:
            loss_inners.append(Wdl_comp.costs)
        steps_inners.append(Wdl_comp.steps)
        loss_alternates.append(res_rec_dict)
        
        loss_iters.append(res_rec)
        
        """MSE_rel_nor.append(relative_mse(gt_PSFs,psf_est_shift/np.sum(psf_est_shift,axis=(0,1))))

        MSE_rel_nor_integrated.append(relative_mse(gt_integrated_nor,psf_est_integrated_shift_nor))
        MSE_rel_nor_stars_2euclidres.append(relative_mse(gt_stars_2euclidrec,stars_est_2euclidres_nor))"""
        
        # Flux update
        flux_new = (obs_est*stars).sum(axis=(0,1))/(obs_est**2).sum(axis=(0,1))
        print "Flux correction: ",flux_new
        Wdl_comp.set_flux(Wdl_comp.get_flux() * flux_new)
        Wdl_coef.set_flux(Wdl_coef.get_flux() * flux_new)
        
    # WRAP UP AND SAVE
    toc_ext = time.time()
    print "Total time taken "+ str((toc_ext-tic_ext)/60.0) + " min"
    print "Time per iteration "+ str((toc_ext-tic_ext)/60.0/n_iter) + " min"




    """
    # Build integrated error
    stars_est_2euclidres_nor = stars_est_2euclidres/np.sum(abs(stars_est_2euclidres),axis=(0,1))
    error_integrated_energy = np.sum(abs(stars_est_2euclidres_nor - gt_stars_2euclidrec), axis=(0,1))

    # Build chromatic error
    error = psf_est_shift_nor - gt_PSFs
    error_energy = np.sum(abs(error), axis=(0,1))




    MSE_rel_nor = np.array(MSE_rel_nor)
    MSE_rel_nor_alternates = np.array(MSE_rel_nor_alternates)
    steps_inners = np.array(steps_inners)
    loss_inners_save = np.array([x for y in loss_inners for x in y]) # flatten list before saving
    loss_alternates = np.array(loss_alternates)
    loss_iters = np.array(loss_iters)
    MSE_rel_nor_integrated_alternates = np.array(MSE_rel_nor_integrated_alternates)
    MSE_rel_nor_integrated = np.array(MSE_rel_nor_integrated)
    MSE_rel_nor_stars_2euclidres = np.array(MSE_rel_nor_stars_2euclidres)
    MSE_rel_nor_stars_2euclidres_alternates = np.array(MSE_rel_nor_stars_2euclidres_alternates)"""
    D_stack_energy = np.array(D_stack_energy)

    np.save(result_path+'/D_stack_energy.npy', D_stack_energy)
    """np.save(result_path+'/steps_inners.npy', steps_inners)
    np.save(result_path+'/MSE_rel_nor.npy', MSE_rel_nor)
    np.save(result_path+'/MSE_rel_nor_alternates.npy', MSE_rel_nor_alternates)
    np.save(result_path+'/loss_inners.npy', loss_inners_save) # descomentar depois de ter transformado em numpy array
    np.save(result_path+'/loss_iters.npy', loss_iters)
    np.save(result_path+'/loss_alternates.npy', loss_alternates)
    np.save(result_path+'/MSE_rel_nor_integrated_alternates.npy', MSE_rel_nor_integrated_alternates)
    np.save(result_path+'/MSE_rel_nor_integrated.npy', MSE_rel_nor_integrated)
    np.save(result_path+'/MSE_rel_nor_stars_2euclidres.npy', MSE_rel_nor_stars_2euclidres)
    np.save(result_path+'/MSE_rel_nor_stars_2euclidres_alternates.npy', MSE_rel_nor_stars_2euclidres_alternates)"""

    if gt_wvlAll:
        np.save(result_path+'/barycenters_wvlAll.npy',barycenters_wvlAll)
    np.save(result_path+'/D_stack.npy',D_stack)
    np.save(result_path+'/barycenters.npy',barycenters)
    np.save(result_path+'/A.npy',A)
    np.save(result_path+'/psf_est.npy',psf_est)
    np.save(result_path+'/obs_est.npy',obs_est)
    """np.save(result_path+'/stars_est_2euclidres_nor.npy',stars_est_2euclidres_nor)
    np.save(result_path+'/stars_est_2euclidres.npy',stars_est_2euclidres)
    np.save(result_path+'/psf_est_shift_nor.npy',psf_est_shift_nor)"""
    np.save(result_path+'/shifts.npy',shifts)
    np.save(result_path+'/flux.npy',flux)

    np.save(result_path+'/SEDs.npy',spectrums)
    np.save(result_path+'/lambdas.npy',lbdas)
    #np.save(result_path+'/gt_PSFs.npy',gt_PSFs)
    np.save(result_path+'/D_stack_0.npy',D_stack_0)
    """np.save(result_path+'/error_integrated_energy.npy',error_integrated_energy)
    np.save(result_path+'/gt_stars_2euclidrec.npy',gt_stars_2euclidrec)
    np.save(result_path+'/error.npy',error)
    np.save(result_path+'/error_energy.npy',error_energy)"""

class LambdaRCA(object):
    """ Color Resolved Components Analysis.
    
    Parameters
    ----------
    n_comp: int
        Number of components to learn.
    upfact: int
        Upsampling factor. Default is 1 (no superresolution).
    ksig: float
        Value of :math:`k` for the thresholding in Starlet domain (taken to be 
        :math:`k\sigma`, where :math:`\sigma` is the estimated noise standard deviation.)
    n_scales: int
        Number of Starlet scales to use for the sparsity constraint. Default is 3. Unused if
        ``filters`` are provided.
    ksig_init: float
        Similar to ``ksig``, for use when estimating shifts and noise levels, as it might 
        be desirable to have it set higher than ``ksig``. Unused if ``shifts`` are provided 
        when running :func:`RCA.fit`. Default is 5.
    filters: np.ndarray
        Optional filters to the transform domain wherein eigenPSFs are assumed to be sparse;
        convolution by them should amount to applying :math:`\Phi`. Optional; if not provided, the
        Starlet transform with `n_scales` scales will be used.
    verbose: bool or int
        If True, will only output RCA-specific lines to stdout. If verbose is set to 2,
        will run ModOpt's optimization algorithms in verbose mode. 
        
    """
    def __init__(self, n_comp, upfact=1, ksig=3, n_scales=3, ksig_init=5, filters=None, 
                 verbose=2):
        self.n_comp = n_comp
        self.upfact = upfact
        self.ksig = ksig
        self.ksig_init = ksig_init
        
        if filters is None:
            # option strings for mr_transform
            self.opt = ['-t2', '-n{}'.format(n_scales)]
            self.default_filters = True
        else:
            self.Phi_filters = filters
            self.default_filters = False
        self.verbose = verbose
        if self.verbose > 1:
            self.modopt_verb = True
        else:
            self.modopt_verb = False
        self.is_fitted = False
        
    def fit(self, obs_data, obs_pos, SEDs, lbdas, all_lbdas=None,
            obs_weights=None, S=None, VT=None, alpha=None,
            shifts=None, sigs=None, psf_size=None, psf_size_type='fwhm',
            flux=None, nb_iter=2, nb_subiter_S=200, nb_reweight=0, 
            nb_subiter_weights=None, n_eigenvects=5, graph_kwargs={}):
        """ Fits lambdaRCA to observed star field.
        
        Parameters
        ----------
        obs_data: np.ndarray
            Observed data.
        obs_pos: np.ndarray
            Corresponding positions.
        obs_weights: np.ndarray
            Corresponding weights. Can be either one per observed star, or contain pixel-wise values. Masks can be
            handled via binary weights. Default is None (in which case no weights are applied). Note if fluxes and
            shifts are not provided, weights will be ignored for their estimation. Noise level estimation only removes 
            bad pixels (with weight strictly equal to 0) and otherwise ignores weights.
        S: np.ndarray
            First guess (or warm start) eigenPSFs :math:`S`. Default is ``None``.
        VT: np.ndarray
            Matrix of concatenated graph Laplacians. Default is ``None``.
        alpha: np.ndarray
            First guess (or warm start) weights :math:`\\alpha`, after factorization by ``VT``. Default is ``None``.
        shifts: np.ndarray
            Corresponding sub-pixel shifts. Default is ``None``; will be estimated from
            observed data if not provided.
        sigs: np.ndarray
            Estimated noise levels. Default is ``None``; will be estimated from data
            if not provided.
        psf_size: float
            Approximate expected PSF size in pixels; will be used for the size of the Gaussian window for centroid estimation.
            ``psf_size_type`` determines the convention used for this size (default is FWHM).
            Ignored if ``shifts`` are provided. Default is Gaussian sigma of 7.5 pixels.
        psf_size_type: str
            Can be any of ``'R2'``, ``'fwhm'`` or ``'sigma'``, for the size defined from quadrupole moments, full width at half maximum
            (e.g. from SExtractor) or 1-sigma width of the best matching 2D Gaussian. Default is ``'fwhm'``.
        flux: np.ndarray
            Flux levels. Default is ``None``; will be estimated from data if not provided.
        nb_iter: int
            Number of overall iterations (i.e. of alternations). Note the weights do not
            get updated the last time around, so they actually get ``nb_iter-1`` updates.
            Default is 2.
        nb_subiter_S: int
            Maximum number of iterations for :math:`S` updates. If ModOpt's optimizers achieve 
            internal convergence, that number may (and often is) not reached. Default is
            200.
        nb_reweight: int 
            Number of reweightings to apply during :math:`S` updates. See equation (33) in RCA paper. 
            Default is 0.
        nb_subiter_weights: int
            Maximum number of iterations for :math:`\\alpha` updates. If ModOpt's optimizers achieve 
            internal convergence, that number may (and often is) not reached. Default is None;
            if not provided, will be set to ``2*nb_subiter_S`` (as it was in RCA v1). 
        n_eigenvects: int
            Maximum number of eigenvectors to consider per :math:`(e,a)` couple. Default is ``None``;
            if not provided, *all* eigenvectors will be considered, which can lead to a poor
            selection of graphs, especially when data is undersampled. Ignored if ``VT`` and
            ``alpha`` are provided.
        graph_kwargs: dictionary
            List of optional kwargs to be passed on to the :func:`utils.GraphBuilder`.
        """
        
        self.obs_data = np.copy(obs_data)
        self.shap = self.obs_data.shape
        self.im_hr_shape = (self.upfact*self.shap[0],self.upfact*self.shap[1],self.shap[2])
        self.obs_pos = obs_pos
        if obs_weights is None:
            self.obs_weights = np.ones(self.shap) #/ self.shap[2]
        elif obs_weights.shape == self.shap:
            self.obs_weights = obs_weights / np.expand_dims(np.sum(obs_weights,axis=2), 2) * self.shap[2]
        elif obs_weights.shape == (self.shap[2],):
            self.obs_weights = obs_weights.reshape(1,1,-1) / np.sum(obs_weights) * self.shap[2]
        else:
            raise ValueError(
            'Shape mismatch; weights should be of shape {} (for per-pixel weights) or {} (per-observation)'.format(
                             self.shap, self.shap[2:]))
        if S is None:
            self.S = np.zeros(self.im_hr_shape[:2] + (self.n_comp,))
        else:
            self.S = S
        self.VT = VT
        self.alpha = alpha
        self.shifts = shifts
        if shifts is None:
            self.psf_size = self._set_psf_size(psf_size, psf_size_type)
        self.sigs = sigs
        self.flux = flux
        self.nb_iter = nb_iter
        self.nb_subiter_S = nb_subiter_S
        if nb_subiter_weights is None:
            nb_subiter_weights = 2*nb_subiter_S
        self.nb_subiter_weights = nb_subiter_weights
        self.nb_reweight = nb_reweight
        self.n_eigenvects = n_eigenvects
        self.graph_kwargs = graph_kwargs
            
        if self.verbose:
            print('Running basic initialization tasks...')
        self._initialize()
        if self.verbose:
            print('... Done.')
        if self.VT is None or self.alpha is None:
            if self.verbose:
                print('Constructing graph constraint...')
            self._initialize_graph_constraint()
            if self.verbose:
                print('... Done.')
        else:
            self.A = self.alpha.dot(self.VT)
        becafunc(obs_data, lbdas, SEDs,
             self.shifts, self.flux, self.sigs, self.shift_ker_stack,self.shift_ker_stack_adj,
             self.alpha, self.VT,
             all_lbdas=all_lbdas)
        self.is_fitted = True
        
    def _set_psf_size(self, psf_size, psf_size_type):
        """ Handles different "size" conventions."""
        if psf_size is not None:
            if psf_size_type == 'fwhm':
                return psf_size / (2*np.sqrt(2*np.log(2)))
            elif psf_size_type == 'R2':
                return np.sqrt(psf_size / 2)
            elif psf_size_type == 'sigma':
                return psf_size
            else:
                raise ValueError('psf_size_type should be one of "fwhm", "R2" or "sigma"')
        else:
            print('''WARNING: neither shifts nor an estimated PSF size were provided to RCA;
the shifts will be estimated from the data using the default Gaussian
window of 7.5 pixels.''')
            return 7.5
  
    def _initialize(self):
        """ Initialization tasks related to noise levels, shifts and flux. Note it includes
        renormalizing observed data, so needs to be ran even if all three are provided."""
        if self.default_filters:
            init_filters = get_mr_filters(self.shap[:2], opt=self.opt, coarse=True)
        else:
            init_filters = self.Phi_filters
        # noise levels
        if self.sigs is None:
            transf_data = utils.apply_transform(self.obs_data, init_filters)
            transf_mask = utils.transform_mask(self.obs_weights, init_filters[0])
            sigmads = np.array([1.4826*utils.mad(fs[0],w) for fs,w in zip(transf_data,
                                                      utils.reg_format(transf_mask))])
            self.sigs = sigmads / np.linalg.norm(init_filters[0])
        else:
            self.sigs = np.copy(self.sigs)
        self.sig_min = np.min(self.sigs)
        # intra-pixel shifts
        if self.shifts is None:
            thresh_data = np.copy(self.obs_data)
            cents = []
            for i in range(self.shap[2]):
                # don't allow thresholding to be over 80% of maximum observed pixel
                nsig_shifts = min(self.ksig_init, 0.8*self.obs_data[:,:,i].max()/self.sigs[i])
                thresh_data[:,:,i] = utils.HardThresholding(thresh_data[:,:,i], nsig_shifts*self.sigs[i])
                cents += [utils.CentroidEstimator(thresh_data[:,:,i], sig=self.psf_size)]
            self.shifts = np.array([ce.return_shifts() for ce in cents])
        self.shift_ker_stack,self.shift_ker_stack_adj = utils.shift_ker_stack(self.shifts,
                                                                              self.upfact)
        # flux levels
        if self.flux is None:
            #TODO: could actually pass on the centroids to flux estimator since we have them at this point
            self.flux = utils.flux_estimate_stack(self.obs_data,rad=4)
        self.flux_ref = np.median(self.flux)
        # Normalize noise levels observed data
        self.sigs /= self.sig_min
        self.obs_data /= self.sigs.reshape(1,1,-1)
    
    def _initialize_graph_constraint(self):
        gber = utils.GraphBuilder(self.obs_data, self.obs_pos, self.obs_weights, self.n_comp, 
                                  n_eigenvects=self.n_eigenvects, verbose=self.verbose,
                                  **self.graph_kwargs)
        self.VT, self.alpha, self.distances = gber.VT, gber.alpha, gber.distances
        self.sel_e, self.sel_a = gber.sel_e, gber.sel_a
        self.A = self.alpha.dot(self.VT)
    
def main():
    load_path = '/Users/mschmitz/Documents/PhD/Teaching/Rebeca/Morgan_kit/Data/QuickestGenerator/full_res_70lbdas_80train300test/train/PickleSEDs/'
    stars = np.load(load_path+'stars.npy')
    fov = np.load(load_path+'fov.npy')
    all_lbdas = np.load(load_path+'all_lbdas.npy')
    all_spectrums = np.load(load_path+'all_SEDs.npy')
    load_path_seds = load_path + 'Interp_12wvls/'
    spectrums = np.load(load_path_seds+'SEDs.npy')
    lbdas = np.load(load_path_seds+'lbdas.npy')
    
    """beca_path = '/Users/mschmitz/Documents/PhD/Teaching/Rebeca/Morgan_kit/Data/lbdaRCA_results/42x42pixels_12lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4' 
    ker = np.load(beca_path+'/ker.npy')
    ker_rot = np.load(beca_path+'/ker_rot.npy')
    shifts = np.load(beca_path+'/shifts.npy')
    sig = np.load(beca_path+'/sig.npy')
    flux = np.load(beca_path+'/flux.npy')
    
    alph = np.load(beca_path+'/alph_0.npy')
    basis = np.load(beca_path+'/basis.npy')"""

    lbdarca = LambdaRCA(3)
    lbdarca.fit(stars, fov, spectrums, lbdas, all_lbdas=all_lbdas)
    
if __name__ == "__main__":
    main()



