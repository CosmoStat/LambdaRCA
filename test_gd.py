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
# tic = time.time() 
# import logOT_bary as ot
# toc = time.time()
# print "Theano compilation in " + str((toc-tic)/60.0) + " min"


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


def plot_func(im, wind=False, cmap='gist_stern', norm=None, cutoff=5e-4,
                title=''):
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
    plt.xticks([])
    plt.yticks([])

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



print "Loading data.."



# first_guesses = np.load('/Users/rararipe/Documents/Data/lbdaRCA_wdl/Fake_SEDs/morgan_stars/first_guesses.npy')
first_guesses = np.load('/Users/rararipe/Documents/Data/QuickestGenerator/FirstGuess_500/first_guesses.npy')

# Load data
load_path = '/Users/rararipe/Documents/Data/QuickestGenerator/full_res_70lbdas_80train300test/train/PickleSEDs/'
stars = np.load(load_path+'stars.npy')
fov = np.load(load_path+'fov.npy')
gt_PSFs = np.load(load_path + 'PSFs_2euclidrec.npy')
gt_stars_2euclidrec = np.load(load_path+'stars_2euclidrec_gt.npy')
all_lbdas = np.load(load_path+'all_lbdas.npy')
all_spectrums = np.load(load_path+'all_SEDs.npy')

# Load interpolated spectrum for learning
load_path_seds = load_path + 'Interp_6wvls/'
spectrums = np.load(load_path_seds+'SEDs.npy')
lbdas = np.load(load_path_seds+'lbdas.npy')

shifts = None
sig = None
flux = None
centroids=None
list_iterations_dict = None
list_iterations_coef = None
gt_PSFs = gt_PSFs.swapaxes(2,3)
gt_PSFs = gt_PSFs/np.sum(gt_PSFs, axis=(0,1)) # normalize
gt_stars_2euclidrec /= np.sum(abs(gt_stars_2euclidrec),axis = (0,1))

shap = (stars.shape[0]*2,stars.shape[1]*2)
shap_lr = stars.shape
nb_obj = stars.shape[-1]
nb_wvl = lbdas.size


# Parameters
D=2
opt_shift = ['-t2','-n2']
nsig_shift_est=4
# lbdaRCA parameters
nb_atoms = 2 # this is actually a constant
nb_comp = 3
nb_comp_chrom = 3
#feat_init = "ground_truth"
feat_init = "super_res_zout"
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
Fred = False
alg = "genFB" #genFB
logit = False
gt_wvlAll = True

if logit:
    alpha_step = [10.0,10.0,10.0] # 0.07 1.0

else:
    alpha_step = [1.0,1.0,1.0] # 0.07 1.0

fb_gamma_param_dict = 1.0
fb_lambda_param_dict = 1.0 # must be in ]0,1[
fb_gamma_param_coef = 1.0
fb_lambda_param_coef = 1.0 # must be in ]0,1[

#save_path = '/Users/rararipe/Documents/Data/GradientDescent_output/testes_aleatorios_0p6'
#save_path = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_8lbdas80pos'


# Logit
#save_path = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_8lbdas80pos_3chrom0RCA_sr_zout1zin1p2_coef_dict_sigmaEqualsLinTrace_alpha321p5beta0p1_absA_3it753dict303030coef_weight4dict1p5coef_noFluxUpdate_logitCondatSparsity_sink15'
# ClassicCondat
save_path = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_6lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1pBeta0p1_abssvdASR50_3it643dict30coef_weight4dict1p5coef_FluxUpdate_genFB_sink10_unicornio_lbdaEquals1p0_LowPass_W0p20p40p4'

#save_path = '/Users/rararipe/Documents/Data/GradientDescent_output/trueSEDs/42x42pixels_8lbdas80pos_3chrom0RCA_sr_zout0p6zin1p2_coef_dict_sigmaEqualsLinTrace_alpha1beta0p5_absA_3it10dict5030coef_weight4dict1p5coef_FluxUpdate_Condat_sink10_THEANO_low_pass'
if not os.path.exists(save_path):
        os.makedirs(save_path)

result_path = save_path +"/result"
if not os.path.exists(result_path):
        os.makedirs(result_path)



nb_comp_RCA = nb_comp - nb_comp_chrom

if nb_comp_RCA == 0:
    problem_type = 'lbdaRCA'
elif nb_comp_chrom == 0:
    problem_type = 'RCA'
else:
    problem_type = 'mix'


print "------------------- Noise estimation ------------------------"
centroids = None
if sig is None:
    sig,filters = utils.im_gauss_nois_est_cube(np.copy(stars),opt=opt_shift)

print "------------------- Shifts estimation ------------------------"
if shifts is None:
    map = np.ones(stars.shape)
    for i in range(0,nb_obj):
        map[:,:,i] *= nsig_shift_est*sig[i]
    print 'Shifts estimation...'
    psf_stack_shift = utils.thresholding_3D(np.copy(stars),map,0)
    shifts,centroids = utils.shift_est(psf_stack_shift)
    print 'Done...'
else:
    print "---------- /!\ Warning: shifts provided /!\ ---------"
ker,ker_rot = utils.shift_ker_stack(shifts,D)
sig /=sig.min()
for k in range(0,nb_obj):
    stars[:,:,k] = stars[:,:,k]/sig[k]


print " ------ ref energy: ",(stars**2).sum()," ------- "
if flux is None:
    flux = utils.flux_estimate_stack(np.copy(stars),rad=4)



print "-------------------- Spatial constraint setting -----------------------"

if problem_type in ['lbdaRCA','RCA']:
    e_opt,p_opt,weights,comp_temp,data,basis,alph  = optim_utils.analysis(stars,0.1*np.prod((shap_lr[0],shap_lr[1],nb_obj))*sig.min()**2,fov,nb_max=nb_comp)
elif problem_type == 'mix':
    e_opt,p_opt,weights,comp_temp,data,basis_chrom,alph_chrom  = optim_utils.analysis(stars,0.1*np.prod((shap_lr[0],shap_lr[1],nb_obj))*sig.min()**2,fov,nb_max=nb_comp_chrom)
    e_opt,p_opt,weights,comp_temp,data,basis_RCA,alph_RCA  = optim_utils.analysis(stars,0.1*np.prod((shap_lr[0],shap_lr[1],nb_obj))*sig.min()**2,fov,nb_max=nb_comp_RCA)
    #debug
    e_opt,p_opt,weights,comp_temp,data,basis,alph  = optim_utils.analysis(stars,0.1*np.prod((shap_lr[0],shap_lr[1],nb_obj))*sig.min()**2,fov,nb_max=nb_comp)

    alph_mix = np.empty(2,dtype=np.ndarray)
    alph_mix[0] = np.copy(alph_chrom)
    alph_mix[1] = np.copy(alph_RCA)

    basis_mix = np.empty(2,dtype=np.ndarray)
    basis_mix[0] = np.copy(basis_chrom)
    basis_mix[1] = np.copy(basis_RCA)


if Fred:
    print "Contructing PSF tree..."
    nb_neighs = nb_obj-1
    neigh,dists = utils.knn_interf(fov,nb_neighs)
    print "Done..."
    dist_weight_deg = 1
    dists_unsorted = utils.feat_dist_mat(fov)
    dist_med = utils.median(dists)
    dist_weights = (dist_med/dists_unsorted)**dist_weight_deg
    dist_weigths = dist_weights/dist_weights.max()



print "------------- Coeff init ------------" # So the idea is that we start with abs(alph.dot(basis)) for D_stack and svd for S_stack. And then alph and basis for everybody
if problem_type == 'RCA':
    A = utils.cube_svd(stars,nb_comp=nb_comp)
    # A = abs(A)
    # A[4:,:] = np.copy(-A[4:,:])
    #alph = A.dot(np.transpose(basis))/(1.0*nb_comp)

elif problem_type == 'lbdaRCA':
    print "hello"
    # A = (alph.dot(basis))
    
    #======== USUAL ============
    A = utils.cube_svd(stars,nb_comp=nb_comp)
    A = abs(A)
    #===========================
    
    # A[2:,:] = np.copy(-A[2:,:])
    
    # A = -A
    
    # Uniform
#    A = np.ones((nb_comp_chrom,nb_obj))*1.0/nb_comp_chrom
#
#
#    alph = A.dot(np.transpose(basis))/(1.0*nb_comp)
    
    # Random
#    A = np.random.normal(0.0, 1.0, (nb_comp,nb_obj))


elif problem_type == 'mix':
    # A_svd_chrom = utils.cube_svd(stars,nb_comp=nb_comp_chrom)
    # A_svd_RCA = utils.cube_svd(stars,nb_comp=nb_comp_RCA)

    A = utils.cube_svd(stars,nb_comp=nb_comp)

    A_svd_chrom = abs(A[:nb_comp_chrom,:])
    # A_svd_RCA = A[nb_comp_chrom:,:]
    A_svd_RCA = A[:nb_comp_RCA,:]
    A_mix = np.empty(2,dtype=np.ndarray)
    A_mix[0] = np.copy(A_svd_chrom)
    A_mix[1] = np.copy(A_svd_RCA)

    # DEBUG  copiando versao que funciona
    alph_mix[0] = np.copy(A_svd_chrom.dot(np.transpose(basis))/(1.0*nb_comp)) #TO DO: qual das comps aqui? nb_comp? or nb_comp_chrom?
    alph_mix[1] = np.copy(A_svd_RCA.dot(np.transpose(basis))/(1.0*nb_comp))

    basis_mix[0] = np.copy(basis)
    basis_mix[1] = np.copy(basis)


# # Load initial guess
# load_path = '/Users/rararipe/Documents/Data/lbdaRCA_wdl/Fake_SEDs/42x42pixels_8lbdas10pos_10comp_sr/' 
# A = np.load(load_path+'A_first.npy')
# ker = np.load(load_path+'ker.npy')
# ker_rot = np.load(load_path+'ker_rot.npy')
# shifts = np.load(load_path+'shifts.npy')
# sig = np.load(load_path+'sig.npy')
# flux = np.load(load_path+'flux.npy')
# alph = np.load(load_path+'alph.npy')
# basis = np.load(load_path+'basis.npy')





np.save(save_path+'/gt_stars_2euclidrec.npy',gt_stars_2euclidrec)
np.save(save_path+'/ker.npy',ker)
np.save(save_path+'/ker_rot.npy',ker_rot)
np.save(save_path+'/shifts.npy',shifts)
np.save(save_path+'/sig.npy',sig)
np.save(save_path+'/flux.npy',flux)


if problem_type in ['lbdaRCA','RCA']:
    np.save(save_path+'/alph_0.npy',alph)
    np.save(save_path+'/basis.npy',basis)
    np.save(save_path+'/A_0.npy',alph.dot(basis))
#    np.save(save_path+'/A_svd.npy',A)
elif problem_type == 'mix':
    np.save(save_path+'/alph_chrom_0.npy',alph_mix[0])
    np.save(save_path+'/alph_rca_0.npy',alph_mix[1])
    np.save(save_path+'/basis_chrom_0.npy',basis_mix[0])
    np.save(save_path+'/basis_RCA_0.npy',basis_mix[1])
    np.save(save_path+'/A_chrom_0.npy',alph_mix[0].dot(basis_mix[0]))
    np.save(save_path+'/A_rca_0.npy',alph_mix[1].dot(basis_mix[1]))
    np.save(save_path+'/A_svd_chrom.npy',A_mix[0])
    np.save(save_path+'/A_svd_rca.npy',A_mix[1])





np.save(save_path+'/gt_PSFs.npy', gt_PSFs)
np.save(save_path+'/stars.npy', stars)












t = (lbdas-lbdas.min()).astype(float)/(lbdas.max()-lbdas.min())
w_stack = np.array([t + 1e-10, 1 - t - 1e-10]).T

if gt_wvlAll:
    t_wvlAll = (all_lbdas-all_lbdas.min()).astype(float)/(all_lbdas.max()-all_lbdas.min())
    w_stack_wvlAll = np.array([t_wvlAll + 1e-10, 1 - t_wvlAll - 1e-10]).T
   

print "-------------------- First guess initialization -----------------------"

# DEBUG
# A[:nb_comp_chrom,:] = abs(A[:nb_comp_chrom,:])
# alph = A.dot(np.transpose(basis))/(1.0*nb_comp)

#first_guesses = psflu.SR_first_guesses(stars,shifts)


#first_guesses,A = psflu.SR_first_guesses_kmeans(stars,fov,shifts,nb_comp)
#first_guesses,A = psflu.SR_first_guesses_radial(stars,fov,shifts,nb_comp)
first_guesses = psflu.SR_first_guesses_rnd(stars,shifts,nb_comp)

if problem_type in ['lbdaRCA','mix']:
    if feat_init == 'ground_truth':
        D_stack = psflu.D_stack_first_guess(shap,nb_obj,nb_comp_chrom,feat_init,gt_PSFs=gt_PSFs)
    else:
        D_stack = psflu.D_stack_first_guess(shap,nb_obj,nb_comp_chrom,feat_init,sr_first_guesses=first_guesses,logit=logit)

    D_stack = abs(D_stack)
    D_stack = D_stack/np.sum(D_stack, axis=0)

    Dlog_stack = np.log(D_stack)

    D_stack_0 = np.copy(D_stack)
    np.save(save_path+'/D_stack_0.npy', D_stack)
    np.save(save_path+'/spectrums.npy', spectrums)


if problem_type == 'mix':
    S_stack = psflu.S_stack_first_guess(shap,nb_comp_RCA,feat_init=feat_init_RCA,sr_first_guesses=first_guesses)
    np.save(save_path+'/S_stack_0.npy', S_stack)

    mix_stack = np.empty(2,dtype=np.ndarray)
    if logit:
        mix_stack[0] = np.copy(Dlog_stack)
    else:
        mix_stack[0] = np.copy(D_stack)
    mix_stack[1] =  np.copy(S_stack)
    # mix_stack = [np.copy(D_stack), np.copy(S_stack)] 
    # alph_mix = np.empty(2,dtype=np.ndarray)
    # alph_mix[0] = np.copy(alph[:nb_comp_chrom,:])
    # alph_mix[1] = np.copy(alph[nb_comp_chrom:,:])
    # alph_mix = [np.copy(alph[:nb_comp_chrom,:]), np.copy(alph[nb_comp_chrom:,:])]
    np.save(save_path+'/spectrums.npy', spectrums)


if problem_type == 'RCA':    
    S_stack = psflu.S_stack_first_guess(shap,nb_comp_RCA,feat_init=feat_init_RCA,sr_first_guesses=first_guesses)
    np.save(save_path+'/S_stack_0.npy', S_stack)



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


if problem_type in ['lbdaRCA','RCA']:
    np.save(save_path+'/A.npy',A)


#import pdb; pdb.set_trace()  # breakpoint 3d87ba10 //
#
# Set-up
# print "-------------------- Spatial constraint setting -----------------------"
# e_opt,p_opt,weights,comp_temp,data,basis,alph  = analysis(im_stack,0.1*prod(shap_obs)*sig.min()**2,field_pos,nb_max=nb_comp)

if problem_type == 'lbdaRCA':
    Wdl_comp = grads.polychrom_eigen_psf(A,spectrums,flux,sig,ker,ker_rot,D_stack,w_stack, gamma, n_iter_sink,stars,logit=logit)
    Cost_comp = costObj([Wdl_comp])
    barycenters = Wdl_comp.compute_barys()
    Wdl_coef = grads.polychrom_eigen_psf_coeff_graph(alph,basis,spectrums,flux,sig,ker,ker_rot,D_stack,w_stack, gamma, n_iter_sink,stars,barycenters=barycenters)
    Wdl_coef_A = grads.polychrom_eigen_psf_coeff_A(A,spectrums,flux,sig,ker,ker_rot,D_stack,w_stack, gamma, n_iter_sink,stars,barycenters=barycenters)
    Cost_coef = costObj([Wdl_coef])
    Cost_coef_A = costObj([Wdl_coef_A])
    np.save(save_path+'/barycenters_0.npy', barycenters)

if problem_type == 'RCA':
    Wdl_comp = grads.polychrom_eigen_psf_RCA(A,flux,sig,ker,ker_rot,S_stack,stars)
    Cost_comp = costObj([Wdl_comp])
    Wdl_coef = grads.polychrom_eigen_psf_coeff_graph_RCA(alph,basis,flux,sig,ker,ker_rot,stars,S_stack)
    Cost_coef = costObj([Wdl_coef])


if problem_type == 'mix':
    Wdl_comp = grads.polychrom_eigen_psf_wrapper(A_mix,spectrums,flux,sig,ker,ker_rot,mix_stack,w_stack, gamma, n_iter_sink,stars)
    Cost_comp = costObj([Wdl_comp])
    barycenters = Wdl_comp.compute_barys()
    Wdl_coef = grads.polychrom_eigen_psf_coeff_graph_wrapper(alph_mix,basis_mix,spectrums,flux,sig,ker,ker_rot,mix_stack,w_stack, gamma, n_iter_sink,stars,barycenters=barycenters)
    Cost_coef = costObj([Wdl_coef])
    np.save(save_path+'/barycenters_0.npy', barycenters)









## DEBUG : test grads RCA with from scipy.optimize import check_grad check_grad(func, grad, [1.5, -1.5])









print "-------------------- Proxs set-up -----------------------"

nsigma_dict = 3.0
nsigma_RCA = 1.5

if problem_type in ['lbdaRCA','RCA']: #though it is only used in RCA
    AS_transform = transforms.Apply_matrix(A)
elif problem_type == 'mix':
    AS_transform = transforms.Apply_matrix(alph_mix[1].dot(basis_mix[1]))


if problem_type == 'mix':
    # ini_all = Wdl_comp.MtX_noise()
    # ini_n_dict = ini_all[0]
    # ini_n_RCA = ini_all[1]
    noise_map_RCA = psflu.get_noise_arr_RCA_wvl(S_stack,shap)*nsigma_RCA
    noise_map_dict = psflu.get_noise_arr_dict_wvl(D_stack,shap)*nsigma_dict

elif problem_type == 'lbdaRCA':
    noise_map_dict = psflu.get_noise_arr_dict_wvl(D_stack,shap)*nsigma_dict
    print "THRSHOLD_estimated: ",str(np.min(noise_map_dict)) 

    # if logit:
    #     noise_map_dict = psflu.get_noise_arr_dict_wvl(Dlog_stack,shap)*nsigma_dict
    # noise_map_dict = psflu.get_noise_arr_dict_wvl(Wdl_comp.MtX_noise(),shap)*nsigma_dict

elif problem_type == 'RCA':
    noise_map_RCA = psflu.get_noise_arr_RCA_wvl(S_stack,shap)*nsigma_RCA




if problem_type in ['lbdaRCA','mix']:
    Wavelet_transf_dict = transforms.dict_wavelet_transform(shap,nb_comp_chrom)
    Lin_comb_wavelet_dict = linear.lin_comb(Wavelet_transf_dict)

elif  problem_type == 'RCA': #condition might change when I implement this constraint to lbdaRCA also
    Lin_comb_AS_RCA = linear.lin_comb(AS_transform)




if problem_type in ['RCA','mix']:
    Wavelet_transf_RCA = transforms.RCA_wavelet_transform(shap,nb_comp_RCA)
    Lin_comb_wavelet_RCA = linear.lin_comb(Wavelet_transf_RCA) 


#Proxs

if problem_type in ['lbdaRCA','mix']:
    Sparse_prox_dict = proxs.SparseThreshold(noise_map_dict, Lin_comb_wavelet_dict,shap,nsigma_dict,logit=logit)
    Simplex_prox_dict = proxs.Simplex()
    LowPass_prox_dict = proxs.LowPass(shap,logit=logit)

if problem_type in ['RCA','mix']:
    Sparse_prox_RCA =  proxs.SparseThreshold(noise_map_RCA, Lin_comb_wavelet_RCA,shap,nsigma_dict)

if problem_type in ['RCA','mix']:
    Simplex_prox_RCA = proxs.Positive(AS_transform)

if problem_type == 'mix':
    Sparse_prox_mix = ProximityCombo([Sparse_prox_dict,Sparse_prox_RCA]) 
    Simplex_prox_mix = ProximityCombo([Simplex_prox_dict,IdentityProx()])
    # Simplex_prox_mix = ProximityCombo([Simplex_prox_dict,Simplex_prox_RCA])






iter_func = lambda x: np.floor(np.sqrt(x))

if problem_type == 'mix':
    Sparse_prox_coef = ProximityCombo([proxs.KThreshold(iter_func),proxs.KThreshold(iter_func)])
if problem_type in ['lbdaRCA','RCA']:
    Sparse_prox_coef = proxs.KThreshold(iter_func)

proxs_coef = [Sparse_prox_coef] 



if problem_type == 'lbdaRCA':
    if logit:
        proxs_comp = [Sparse_prox_dict,LowPass_prox_dict]
        # proxs_comp = [IdentityProx()]
    else:
#        proxs_comp = [Simplex_prox_dict,Sparse_prox_dict]
        proxs_comp = [Simplex_prox_dict,Sparse_prox_dict,LowPass_prox_dict]
elif problem_type == 'mix':
    proxs_comp = [Simplex_prox_mix,Sparse_prox_mix]
elif problem_type == 'RCA':
    proxs_comp = [Simplex_prox_RCA,Sparse_prox_RCA]


# if alg == "genFB":
#     fb_gamma_param_dict = 1.0
#     fb_lambda_param_dict = 1.0 # must be in ]0,1[
#     fb_gamma_param_coef = 1.0
#     fb_lambda_param_coef = 0.6 # must be in ]0,1[

    

if alg == "condat":
    print "-------------------- Dual variable set-up -----------------------"
    
    if problem_type in ['lbdaRCA','RCA']:
        prox_comp_primal = Identity()
    elif problem_type == 'mix': 
        prox_comp_primal = ProximityCombo([IdentityProx(),IdentityProx()])

    # noise_shap = list(noise_map_dict.shape)
    # noise_shap[0] += 1
    # dual_var = [np.zeros(D_stack.shape), np.zeros(tuple(noise_shap))] 
    # dual_var = np.array([np.zeros(D_stack.shape), np.zeros(D_stack.shape)])



    if problem_type == 'lbdaRCA':
        Wdl_comp.set_lin_comb_l1norm(Lin_comb_wavelet_dict.get_l1norm())
    elif problem_type =='mix':
        filters = Wavelet_transf_dict.filters
        filters = np.concatenate((filters, Wavelet_transf_RCA.filters), axis=0)
        l1norm = nb_comp * np.sqrt(sum((np.sum(np.abs(filter_i)) ** 2 for filter_i in filters)))
        Wdl_comp.set_lin_comb_l1norm(l1norm)
    elif problem_type == 'RCA':
        Wdl_comp.set_lin_comb_l1norm(Lin_comb_wavelet_RCA.get_l1norm())

    # Lin_comb_condat = LinearCombo([Identity(), Identity()],weights=[0.4,0.6]) # the linear operator is already included on proxs. By these means we can switch betueen Condat and genFB more easily.
    id_list = []
    for j in range(len(proxs_comp)):
        id_list.append(Identity())
    Lin_comb_condat = LinearCombo(id_list,weights=[0.3,0.4,0.3])
#    if logit:
#        Lin_comb_condat = LinearCombo([Identity()])
#    else:
#        Lin_comb_condat = LinearCombo([Identity(), Identity()])
    
    prox_comp_dual =  ProximityCombo(proxs_comp)
    
    dual_var = np.empty(len(proxs_comp), dtype=np.ndarray) 

    if problem_type == 'lbdaRCA':
        for i in range(len(proxs_comp)):
            dual_var[i] = np.zeros(D_stack.shape)
    elif problem_type =='mix':
        for i in range(len(proxs_comp)):
            dual_var[i] = np.zeros(mix_stack.shape)
    elif problem_type == 'RCA':
        for i in range(len(proxs_comp)):
            dual_var[i] = np.zeros(S_stack.shape)

    condat_rho_dict = 0.8
    condat_sigma_dict = 1.0
    condat_tau_dict = np.copy(condat_sigma_dict)







print "Done."
# find out a nice value to start and then how to update during iterations inside FB
# fb_gamma_param_dict = 
# fb_lambda_param_dict = 








# Iteration zero

gt_integrated_nor= np.sum(gt_PSFs,axis=3)/np.sum(abs(np.sum(gt_PSFs,axis=3)),axis=(0,1))
np.save(save_path+'/gt_integrated_nor.npy', gt_integrated_nor)


if problem_type == 'lbdaRCA':
    if gt_wvlAll:        
        barycenters_wvlAll = Wdl_comp.compute_barys(w_stack=w_stack_wvlAll)
        psf_est = field_reconstruction_wdl(barycenters_wvlAll,A,shap)
    else:
        psf_est = field_reconstruction_wdl(barycenters,A,shap)
elif problem_type =='mix':
    psf_est = field_reconstruction_mix(barycenters,A_mix[0],shap,S_stack,A_mix[1])
elif problem_type == 'RCA':
    psf_est = field_reconstruction_RCA(S_stack,A,shap)



if problem_type in ['lbdaRCA','mix']:    
    psf_est_shift = psflu.shift_PSF_to_gt(psf_est,gt_PSFs)
    psf_est_integrated_shift_nor = np.sum(psf_est_shift,axis=3)/np.sum(abs(np.sum(psf_est_shift,axis=3)),axis=(0,1))
    psf_est_shift_nor = psf_est_shift/np.sum(abs(psf_est_shift),axis=(0,1))
    MSE_rel_nor_alternates.append(relative_mse(gt_PSFs,psf_est_shift_nor))
    MSE_rel_nor.append(relative_mse(gt_PSFs,psf_est_shift_nor))
    Wdl_coef.set_barycenters(barycenters)
    obs_est = Wdl_coef.MX()
    if gt_wvlAll:  
        stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,all_spectrums)
    else:
        stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,spectrums)
    stars_est_2euclidres_nor = stars_est_2euclidres/np.sum(abs(stars_est_2euclidres), axis=(0,1))
if problem_type == 'RCA':
    psf_integrated_shift = psflu.shift_PSFinter_to_gt(psf_est,gt_stars_2euclidrec)
    psf_est_integrated_shift_nor = psf_integrated_shift/np.sum(abs(psf_integrated_shift),axis=(0,1))
    obs_est = Wdl_coef.MX(A.dot(np.transpose(basis))/(1.0*nb_comp_RCA))
    stars_est_2euclidres = np.copy(psf_integrated_shift)
    stars_est_2euclidres_nor = np.copy(psf_est_integrated_shift_nor)



MSE_rel_nor_integrated_alternates.append(relative_mse(gt_integrated_nor,psf_est_integrated_shift_nor))
MSE_rel_nor_integrated.append(relative_mse(gt_integrated_nor,psf_est_integrated_shift_nor))
MSE_rel_nor_stars_2euclidres_alternates.append(relative_mse(gt_stars_2euclidrec,stars_est_2euclidres_nor))
MSE_rel_nor_stars_2euclidres.append(relative_mse(gt_stars_2euclidrec,stars_est_2euclidres_nor))

res_rec = euclidean_cost(obs_est,stars) 

np.save(save_path+'/obs_est_0.npy', obs_est)
np.save(save_path+'/stars_est_2euclidres_0.npy', stars_est_2euclidres)

print " > Current cost: {}".format(res_rec)
loss_inners.append([res_rec])
loss_alternates.append(res_rec)
loss_iters.append(res_rec)




tic_ext = time.time()
for i in tqdm(range(n_iter)):
    print ">>>>>>>>>>>>>>>>>>>> Iteration " + str(i)
    if not os.path.exists(save_path+'/iter_'+str(i)):
        os.makedirs(save_path+'/iter_'+str(i))




    # raise ValueError('Check gradient of mix_stack[1]  VERSAO MASTER with small data. Pq nao ta mudando quase nada???????')
 
    
    print "-------------------- Steps coefficients set-up -----------------------"
    Wdl_coef.min_coef = None
    if i ==0:
        fb_gamma_param_coef = Wdl_coef_A.gamma_update(fb_gamma_param_coef)
    else:
        fb_gamma_param_coef = Wdl_coef.gamma_update(fb_gamma_param_coef) # or maybe pass the entire update_step_function aqui
    # if i == 0:
    #     print "first iter of coefficients"
    #     fb_gamma_param_coef = fb_gamma_param_coef/1000.0 # because the proxs will cause the cost to increase a lot at first iteration of all

    
    print "-------------------- Coefficients estimation -----------------------"
    # Coefficients update
    if not Fred:
        if problem_type in ['lbdaRCA','RCA']:
            if i == 0:
               min_coef = optimalg.GenForwardBackward(A,Wdl_coef_A,[IdentityProx()],Cost_coef_A,auto_iterate=False,
                gamma_param=fb_gamma_param_coef,lambda_param =fb_lambda_param_coef, gamma_update=Wdl_coef_A.gamma_update)
            else:
                min_coef = optimalg.GenForwardBackward(alph,Wdl_coef,proxs_coef,Cost_coef,auto_iterate=False,
                gamma_param=fb_gamma_param_coef,lambda_param =fb_lambda_param_coef, gamma_update=Wdl_coef.gamma_update)
        elif problem_type == 'mix':
            min_coef = optimalg.GenForwardBackward(alph_mix,Wdl_coef,proxs_coef,Cost_coef,auto_iterate=False,
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
            if problem_type in ['lbdaRCA','RCA']:
                A = min_coef.x_final
                alph =  A.dot(np.transpose(basis))/(1.0*nb_comp)
        else:
            if problem_type in ['lbdaRCA','RCA']:
                alph = min_coef.x_final
                A = alph.dot(basis)
            elif problem_type == 'mix':
                alph_mix = min_coef.x_final
                A_mix[0] = alph_mix[0].dot(basis_mix[0])
                A_mix[1] = alph_mix[1].dot(basis_mix[1])
                # alph = np.concatenate((alph_mix[0], alph_mix[1]), axis=0)
            if problem_type == 'RCA':
                AS_transform.set_matrix(A)
    
    elif problem_type == 'RCA' :
        comp_lr = zeros((int(shap[0]/2),int(shap[1]/2),nb_comp_RCA,nb_obj))
        for l in range(nb_comp_RCA):
            for p in range(nb_obj):
                comp_lr[:,:,l,p] = (flux[i]/sig[i])*utils.decim(scisig.fftconvolve(S_stack[:,l].reshape(shap),ker[:,:,i],mode='same'),D,av_en=0)

        n_max = n_iter-1
        if i < n_max:
            weights_k = None
            weights_k,alph,supports = utils.non_unif_smoothing_mult_coeff_pos_cp_5(stars,comp_lr,S_stack.reshape((shap[0],shap[1],nb_comp_RCA)),neigh,basis,alph,\
                nb_iter=max_iter_FB_coef*2,tol=0.1,Ainit=A)
            for l in range(nb_comp):
                a = sqrt((weights_k[l,:]**2).sum())
                if a>0:
                    S_stack[:,l] *= a
                    weights_k[l,:] /= a

            A = np.copy(weights_k)
            # alph = A.dot(np.transpose(basis))/(1.0*nb_comp_RCA)


    # CHANGE comp_mix to alpha_mix !!!!!!! it is positive even without the constraint, but pay attention!!
    if problem_type in ['lbdaRCA','RCA']:
        Wdl_coef.set_alpha(alph)
        Wdl_comp.set_A(alph.dot(basis))
    elif problem_type == 'mix':
        Wdl_coef.set_alpha_mix(alph_mix)
        Wdl_comp.set_A_mix(A_mix)

    # Weights update
    if problem_type =='mix':
        # ini_all = Wdl_comp.MtX_noise()
        # ini_n_dict = ini_all[0]
        # ini_n_RCA = ini_all[1]
        noise_map_dict = psflu.get_noise_arr_dict_wvl(mix_stack[0],shap)*nsigma_dict
        noise_map_RCA = psflu.get_noise_arr_RCA_wvl(mix_stack[1],shap)*nsigma_RCA
        Sparse_prox_dict.update_weights(noise_map_dict) 
        Sparse_prox_RCA.update_weights(noise_map_RCA)
#    elif problem_type == 'lbdaRCA': # no need to update after updating A because I actually use D_stack and not Mt(stars)
#        noise_map_dict = psflu.get_noise_arr_dict_wvl(Wdl_comp.MtX_noise(),shap)*nsigma_dict
#        Sparse_prox_dict.update_weights(noise_map_dict) 
    elif problem_type == 'RCA':
        noise_map_RCA = psflu.get_noise_arr_RCA_wvl(S_stack,shap)*nsigma_RCA
        Sparse_prox_RCA.update_weights(noise_map_RCA)
    
        
    

    # Evaluation
    if problem_type == 'lbdaRCA':
        obs_est = Wdl_coef.MX()
        if gt_wvlAll:        
            barycenters_wvlAll = Wdl_comp.compute_barys(w_stack=w_stack_wvlAll)
            psf_est = field_reconstruction_wdl(barycenters_wvlAll,A,shap)
        else:
            psf_est = field_reconstruction_wdl(barycenters,A,shap)
    elif problem_type =='mix':
        obs_est = Wdl_coef.MX()
        psf_est = field_reconstruction_mix(barycenters,A_mix[0],shap,mix_stack[1],A_mix[1])
    elif problem_type =='RCA':
        obs_est = Wdl_coef.MX(alph)
        psf_est = field_reconstruction_RCA(S_stack,alph.dot(basis),shap)


    if problem_type in ['lbdaRCA','mix']:
        psf_est_shift = psflu.shift_PSF_to_gt(psf_est,gt_PSFs)
        # psf_est_inner_shift = 
        psf_est_integrated_shift_nor = np.sum(psf_est_shift,axis=3)/np.sum(abs(np.sum(psf_est_shift,axis=3)),axis=(0,1))
        psf_est_shift_nor = psf_est_shift/np.sum(abs(psf_est_shift),axis=(0,1))
        MSE_rel_nor_alternates.append(relative_mse(gt_PSFs,psf_est_shift_nor))
        if gt_wvlAll:  
            stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,all_spectrums)
        else:
            stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,spectrums)
        stars_est_2euclidres_nor = stars_est_2euclidres/np.sum(abs(stars_est_2euclidres), axis=(0,1))
    if problem_type == 'RCA':
        psf_integrated_shift = psflu.shift_PSFinter_to_gt(psf_est,gt_stars_2euclidrec)
        psf_est_integrated_shift_nor = psf_integrated_shift/np.sum(abs(psf_integrated_shift),axis=(0,1))
        stars_est_2euclidres = np.copy(psf_integrated_shift)
        stars_est_2euclidres_nor = np.copy(psf_est_integrated_shift_nor)

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
        np.save(save_path+'/iter_'+str(i)+'/S_stack_2.npy', S_stack)

    # Bilan
    res_rec = euclidean_cost(obs_est,stars)
    print " > Cost in test_gd: {}".format(res_rec)
    if problem_type == 'lbdaRCA':
        print "Cost from dict"
        Wdl_comp.cost(D_stack, verbose=True,count=False)
        print "Cost from coef please be the same"
        Wdl_coef.cost(alph, verbose=True,count=False)
    elif problem_type =='mix':
        print "Cost from dict"
        Wdl_comp.cost(mix_stack, verbose=True,count=False)
        print "Cost from coef please be the same"
        Wdl_coef.cost(alph_mix, verbose=True,count=False)
    elif problem_type =='RCA':
        print "Cost from dict"
        Wdl_comp.cost(S_stack, verbose=True,count=False)
        print "Cost from coef please be the same"
        Wdl_coef.cost(alph, verbose=True,count=False)
    if i ==0:
        loss_inners[0] += Wdl_coef_A.costs
        steps_inners.append(Wdl_coef_A.steps)
    else:
        loss_inners.append(Wdl_coef.costs)
        steps_inners.append(Wdl_coef.steps)
    loss_alternates.append(res_rec)
    
    
    #    import pdb; pdb.set_trace()  # breakpoint bc817e81 //
    print "-------------------- Steps dictionary set-up -----------------------"
    Wdl_comp.min_coef = None
    Wdl_comp.alpha = alpha_step[i] # se quiser mudar alpha, mudar la em cima na lista
    if alg in ["genFB","FB"]:
        fb_gamma_param_dict = Wdl_comp.gamma_update(fb_gamma_param_dict)
        if i==0:
            fb_lambda_param_dict = 1.0
        else:
            fb_lambda_param_dict = np.copy(fb_gamma_param_dict)    
#        lmbda = np.copy(fb_gamma_param_dict)*1.8 #almbda of first inner iteration, podemos ousar pq tem menos chance de dar ruim
#        if lmbda > 1.0:
#            lmbda = 1.0
#        fb_lambda_param_dict = np.copy(lmbda)
        
    if alg=="condat":
        if problem_type == 'lbdaRCA':
            if logit:
                condat_sigma_dict = Wdl_comp.sig_tau_update(condat_sigma_dict,x=Dlog_stack)
            else:
                condat_sigma_dict = Wdl_comp.sig_tau_update(condat_sigma_dict,x=D_stack) 
        else:
            condat_sigma_dict = Wdl_comp.sig_tau_update(condat_sigma_dict)
        if i==0:
            condat_tau_dict = 1.0
        else:
            condat_tau_dict = np.copy(condat_sigma_dict)

    print "-------------------- Dictionary estimation -----------------------"
    # Dictionary update
    
    if alg=="genFB":
        if problem_type == 'lbdaRCA':
            if logit:
                min_dict = modoptAlgorithms.GenForwardBackward(Dlog_stack,Wdl_comp,proxs_comp,cost=Cost_comp,auto_iterate=False,
                    gamma_param=fb_gamma_param_dict,lambda_param =fb_lambda_param_dict, gamma_update=Wdl_comp.gamma_update,logit=logit,weights=[0.7,0.3]) # check again the effects of these weights weights=[0.1, 0.9]
            else:
                min_dict = modoptAlgorithms.GenForwardBackward(D_stack,Wdl_comp,proxs_comp,cost=Cost_comp,auto_iterate=False,
                    gamma_param=fb_gamma_param_dict,lambda_param =fb_lambda_param_dict, gamma_update=Wdl_comp.gamma_update,weights=[0.2,0.4,0.4])
        elif problem_type =='mix':
            min_dict = optimalg.GenForwardBackward(mix_stack,Wdl_comp,proxs_comp,cost=Cost_comp,auto_iterate=False,
                gamma_param=fb_gamma_param_dict,lambda_param =fb_lambda_param_dict, gamma_update=Wdl_comp.gamma_update) # check again the effects of these weights weights=[0.1, 0.
        elif problem_type == 'RCA':
            min_dict = optimalg.GenForwardBackward(S_stack,Wdl_comp,proxs_comp,cost=Cost_comp,auto_iterate=False,
                gamma_param=fb_gamma_param_dict,lambda_param =fb_lambda_param_dict, gamma_update=Wdl_comp.gamma_update)
    if alg=="FB":
        if problem_type == 'lbdaRCA':
            if logit:
                min_dict = modoptAlgorithms.ForwardBackward(Dlog_stack,Wdl_comp,Sparse_prox_dict,cost=Cost_comp,auto_iterate=False,
                    beta_param=fb_gamma_param_dict,lambda_param =fb_lambda_param_dict, beta_update=Wdl_comp.gamma_update ) 
    if alg=="condat":
        if problem_type == 'lbdaRCA':
            if logit:
                min_dict = modoptAlgorithms.Condat(Dlog_stack,dual_var,Wdl_comp,prox_comp_primal,prox_comp_dual,linear=Lin_comb_condat,cost=Cost_comp,auto_iterate=False,
                    rho=condat_rho_dict,sigma =condat_sigma_dict, tau = condat_tau_dict, sigma_update=Wdl_comp.sig_tau_update)
            else:
                min_dict = modoptAlgorithms.Condat(D_stack,dual_var,Wdl_comp,prox_comp_primal,prox_comp_dual,linear=Lin_comb_condat,cost=Cost_comp,auto_iterate=False,
                    rho=condat_rho_dict,sigma =condat_sigma_dict, tau = condat_tau_dict, sigma_update=Wdl_comp.sig_tau_update)
        elif problem_type =='mix':
            min_dict = optimalg.Condat(mix_stack,dual_var,Wdl_comp,prox_comp_primal,prox_comp_dual,linear=Lin_comb_condat,cost=Cost_comp,auto_iterate=False,
                rho=condat_rho_dict,sigma =condat_sigma_dict, tau = condat_tau_dict, sigma_update=Wdl_comp.sig_tau_update)
        elif problem_type == 'RCA':
            min_dict = optimalg.Condat(S_stack,dual_var,Wdl_comp,prox_comp_primal,prox_comp_dual,linear=Lin_comb_condat,cost=Cost_comp,auto_iterate=False,
                rho=condat_rho_dict,sigma =condat_sigma_dict, tau = condat_tau_dict, sigma_update=Wdl_comp.sig_tau_update)
    Wdl_comp.set_n_iter(i)
    Wdl_comp.set_min_dict(min_dict)
    Wdl_comp.reset_costs()
    Wdl_comp.reset_steps()
    if problem_type == 'lbdaRCA':
        Wdl_comp.reset_D_stack_energy()
    tic = time.time()
    if list_iterations_dict is None:
        min_dict.iterate(max_iter=max_iter_FB_dict)
    else:
        min_dict.iterate(max_iter=list_iterations_dict[i])
    toc = time.time()
    print "Done in: " + str((toc-tic)/60.0) + " min"    
    
    # Update
    if problem_type == 'lbdaRCA':
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
    elif problem_type =='mix':
        mix_stack = min_dict.x_final
        # Force the projection in simplex
        mix_stack[0][mix_stack[0]< 0.0] = 1e-9
        D_stack = mix_stack[0]
        S_stack = mix_stack[1]
        print "Dictionary energy: ",np.sum(abs(mix_stack[0]), axis=0)
        print "Free components energy: ",np.sum(abs(mix_stack[1]), axis=0)
        Wdl_comp.set_mix_stack(mix_stack)
        Wdl_coef.set_mix_stack(mix_stack)
    elif problem_type == 'RCA':
        S_stack = min_dict.x_final
        Wdl_comp.set_S_stack(S_stack)
        Wdl_coef.set_S_stack(S_stack)
    


    if problem_type in ['mix','lbdaRCA']:
        barycenters = Wdl_comp.compute_barys()
        Wdl_coef.set_barycenters(barycenters)

    # # Weights update
    # noise_map_dict = psflu.get_noise_arr_dict_wvl(Wdl_comp.MtX_noise(),shap)*nsigma_dict
    # Sparse_prox_dict.update_weights(noise_map_dict) 


    # Weights update
    if problem_type =='mix':
        # ini_all = Wdl_comp.MtX_noise()
        # ini_n_dict = ini_all[0]
        # ini_n_RCA = ini_all[1]
        noise_map_dict = psflu.get_noise_arr_dict_wvl(mix_stack[0],shap)*nsigma_dict
        noise_map_RCA = psflu.get_noise_arr_RCA_wvl(mix_stack[1],shap)*nsigma_RCA
        Sparse_prox_dict.update_weights(noise_map_dict) 
        Sparse_prox_RCA.update_weights(noise_map_RCA)
    elif problem_type == 'lbdaRCA':
        noise_map_dict = psflu.get_noise_arr_dict_wvl(D_stack,shap)*nsigma_dict
        print "THRSHOLD_estimated: ",str(np.min(noise_map_dict))
        Sparse_prox_dict.update_weights(noise_map_dict)
#        Lin_comb_condat.weights = [0.6,0.4]
               
    elif problem_type == 'RCA':
        noise_map_RCA = psflu.get_noise_arr_RCA_wvl(Wdl_comp.MtX_noise(),shap)*nsigma_RCA
        Sparse_prox_RCA.update_weights(noise_map_RCA)
        
       
        
        
        
    print "Optimal dictionary"        
    tk.plot_func(D_stack[:,0,0])
    tk.plot_func(D_stack[:,1,0])
    # Denoise guesses and normalize (right? can I do that?)
    if problem_type == 'lbdaRCA':
        if logit:
#            Dlog_stack =  Sparse_prox_dict.op(Dlog_stack)
            D_stack = psflu.logitTonormal(Dlog_stack)
            Wdl_comp.set_Dlog_stack(Dlog_stack)
        else:
#            D_stack = Sparse_prox_dict.op(D_stack) 
            D_stack /= np.sum(abs(D_stack),axis=0)             
        Wdl_comp.set_D_stack(D_stack)
        Wdl_coef.set_D_stack(D_stack)
        print "Dictionary energy: ",np.sum(abs(D_stack), axis=0)
        print "Normalized guess"        
        tk.plot_func(D_stack[:,0,0])
        tk.plot_func(D_stack[:,1,0])

#    import pdb; pdb.set_trace()  # breakpoint 3d87ba10 //
    
    #Compute barycenters
    if problem_type == 'lbdaRCA':
        barycenters = Wdl_comp.compute_barys()
#        barycenters =  Wdl_comp.MX__barys
        Wdl_coef.set_barycenters(barycenters) 

           
    # Evaluation
    if problem_type == 'lbdaRCA':
        if logit:
              obs_est = Wdl_comp.compute_MX(Dlog_stack) 
        else:              
              obs_est = Wdl_comp.compute_MX(D_stack)
        if gt_wvlAll:        
            barycenters_wvlAll = Wdl_comp.compute_barys(w_stack=w_stack_wvlAll)
            psf_est = field_reconstruction_wdl(barycenters_wvlAll,A,shap)
        else:
            psf_est = field_reconstruction_wdl(barycenters,A,shap)
    elif problem_type =='mix':
        obs_est = Wdl_comp._current_rec
        psf_est = field_reconstruction_mix(barycenters,A_mix[0],shap,mix_stack[1],A_mix[1])
    elif problem_type =='RCA':
        obs_est = Wdl_comp._current_rec
        psf_est = field_reconstruction_RCA(S_stack,alph.dot(basis),shap)



    
    if problem_type in ['lbdaRCA','mix']:
        psf_est_shift = psflu.shift_PSF_to_gt(psf_est,gt_PSFs)
        psf_est_integrated_shift_nor = np.sum(psf_est_shift,axis=3)/np.sum(abs(np.sum(psf_est_shift,axis=3)),axis=(0,1))
        psf_est_shift_nor = psf_est_shift/np.sum(abs(psf_est_shift),axis=(0,1))
        MSE_rel_nor_alternates.append(relative_mse(gt_PSFs,psf_est_shift_nor))

        if gt_wvlAll:  
            stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,all_spectrums)
        else:
            stars_est_2euclidres = reconstruct_stars_est_2euclidres(psf_est_shift,spectrums)
        stars_est_2euclidres_nor = stars_est_2euclidres/np.sum(abs(stars_est_2euclidres), axis=(0,1))

    if problem_type == 'RCA':
        psf_integrated_shift = psflu.shift_PSFinter_to_gt(psf_est,gt_stars_2euclidrec)
        psf_est_integrated_shift_nor = psf_integrated_shift/np.sum(abs(psf_integrated_shift),axis=(0,1))
        stars_est_2euclidres = np.copy(psf_integrated_shift)
        stars_est_2euclidres_nor = np.copy(psf_est_integrated_shift_nor)

    MSE_rel_nor_integrated_alternates.append(relative_mse(gt_integrated_nor,psf_est_integrated_shift_nor))
    MSE_rel_nor_stars_2euclidres_alternates.append(relative_mse(gt_stars_2euclidrec,stars_est_2euclidres_nor))
    
    energy_est = np.sum(abs(psf_est),axis=(0,1))
    print "PSF est energy: ",energy_est

    np.save(save_path+'/iter_'+str(i)+'/stars_est_2euclidres_1.npy', stars_est_2euclidres)
    np.save(save_path+'/iter_'+str(i)+'/psf_est_1.npy', psf_est)
    np.save(save_path+'/iter_'+str(i)+'/obs_est_1.npy', obs_est)
    np.save(save_path+'/iter_'+str(i)+'/psf_est_integrated_shift_nor_1.npy', psf_est_integrated_shift_nor)
    if problem_type == 'lbdaRCA':
        np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_nor_1.npy', psf_est_shift_nor)
        np.save(save_path+'/iter_'+str(i)+'/alpha_1.npy', alph)
        np.save(save_path+'/iter_'+str(i)+'/A_1.npy', A)
        np.save(save_path+'/iter_'+str(i)+'/D_stack_1.npy', D_stack)
        np.save(save_path+'/iter_'+str(i)+'/barycenters_1.npy', barycenters)
        np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_1.npy', psf_est_shift)
    elif problem_type == 'mix':
        np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_nor_1.npy', psf_est_shift_nor)
        np.save(save_path+'/iter_'+str(i)+'/A_chrom_1.npy', A_mix[0])
        np.save(save_path+'/iter_'+str(i)+'/D_stack_1.npy', mix_stack[0])
        np.save(save_path+'/iter_'+str(i)+'/S_stack_1.npy', mix_stack[1])
        np.save(save_path+'/iter_'+str(i)+'/A_rca_1.npy', A_mix[1])
        np.save(save_path+'/iter_'+str(i)+'/barycenters_1.npy', barycenters)
        np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_1.npy', psf_est_shift)
    elif problem_type == 'RCA':
        np.save(save_path+'/iter_'+str(i)+'/alpha_1.npy', alph)
        np.save(save_path+'/iter_'+str(i)+'/A_1.npy', alph.dot(basis))
        np.save(save_path+'/iter_'+str(i)+'/psf_est_shift_1.npy', psf_integrated_shift)
        np.save(save_path+'/iter_'+str(i)+'/S_stack_1.npy', psf_integrated_shift)

    # Bilan
    res_rec = euclidean_cost(obs_est,stars)
    print " > Cost in test_gd: {}".format(res_rec)

    if problem_type == 'lbdaRCA':
        print "Cost from dict"
        if logit:
            res_rec_dict = Wdl_comp.cost(Dlog_stack, verbose=True,count=False)
        else:
            res_rec_dict = Wdl_comp.cost(D_stack, verbose=True,count=False)
        print "Cost from coef please be the same"
        Wdl_coef.cost(alph, verbose=True,count=False)
    elif problem_type =='mix':
        print "Cost from dict"
        res_rec_dict = Wdl_comp.cost(mix_stack, verbose=True,count=False)
        print "Cost from coef please be the same"
        Wdl_coef.cost(alph_mix, verbose=True,count=False)
    elif problem_type =='RCA':
        print "Cost from dict"
        res_rec_dict = Wdl_comp.cost(S_stack, verbose=True,count=False)
        print "Cost from coef please be the same"
        Wdl_coef.cost(alph, verbose=True,count=False)


    if problem_type == 'lbdaRCA':
        D_stack_energy += Wdl_comp.D_stack_energy
    if i ==0:
        loss_inners[0]+= Wdl_comp.costs
    else:
        loss_inners.append(Wdl_comp.costs)
    steps_inners.append(Wdl_comp.steps)
    loss_alternates.append(res_rec_dict)
    
    
    
    
    
    

#    import pdb; pdb.set_trace()  # breakpoint 3d87ba10 //
    # End of alternate turn  
    loss_iters.append(res_rec)
    if problem_type in ['lbdaRCA','mix']:    
        MSE_rel_nor.append(relative_mse(gt_PSFs,psf_est_shift/np.sum(psf_est_shift,axis=(0,1))))

    MSE_rel_nor_integrated.append(relative_mse(gt_integrated_nor,psf_est_integrated_shift_nor))
    MSE_rel_nor_stars_2euclidres.append(relative_mse(gt_stars_2euclidrec,stars_est_2euclidres_nor))
    
    # Flux update
    flux_new = (obs_est*stars).sum(axis=(0,1))/(obs_est**2).sum(axis=(0,1))
    print "Flux correction: ",flux_new
    Wdl_comp.set_flux(Wdl_comp.get_flux() * flux_new)
    Wdl_coef.set_flux(Wdl_coef.get_flux() * flux_new)
    
    
    
    
     
    



     

    


    # # #COST WITH NEW FLUX
    # # #Evaluation
    # # Evaluation
    # if problem_type == 'lbdaRCA':
    #     obs_est = Wdl_coef.MX()
    #     psf_est = field_reconstruction_wdl(barycenters,A,shap)
    # elif problem_type =='mix':
    #     obs_est = Wdl_coef.MX()
    #     psf_est = field_reconstruction_mix(barycenters,A_mix[0],shap,mix_stack[1],A_mix[1])
    # elif problem_type =='RCA':
    #     obs_est = Wdl_coef.MX(alph)
    #     psf_est = field_reconstruction_RCA(S_stack,alph.dot(basis),shap)

    # res_rec = euclidean_cost(obs_est,stars)
    # print "COST WITH NEW FLUX (to the next iteration)"    
    # print " > Cost in test_gd: {}".format(res_rec)

    # #DEBUG (check if two ways to compute cost are equal)
    # if nb_comp_RCA == 0:
    #     print "Cost from dict"
    #     Wdl_comp.cost(D_stack, verbose=True)
    #     print "Cost from coef please be the same"
    #     Wdl_coef.cost(alph, verbose=True)
    # else:
    #     print "Cost from dict"
    #     Wdl_comp.cost(mix_stack, verbose=True)
    #     print "Cost from coef please be the same"
    #     Wdl_coef.cost(alph_mix, verbose=True)



toc_ext = time.time()
print "Total time taken "+ str((toc_ext-tic_ext)/60.0) + " min"
print "Time per iteration "+ str((toc_ext-tic_ext)/60.0/n_iter) + " min"



MSE_rel_nor = np.array(MSE_rel_nor)
MSE_rel_nor_alternates = np.array(MSE_rel_nor_alternates)
steps_inners = np.array(steps_inners)
loss_inners_save = np.array([x for y in loss_inners for x in y]) # flatten list before saving
loss_alternates = np.array(loss_alternates)
loss_iters = np.array(loss_iters)
MSE_rel_nor_integrated_alternates = np.array(MSE_rel_nor_integrated_alternates)
MSE_rel_nor_integrated = np.array(MSE_rel_nor_integrated)
MSE_rel_nor_stars_2euclidres = np.array(MSE_rel_nor_stars_2euclidres)
MSE_rel_nor_stars_2euclidres_alternates = np.array(MSE_rel_nor_stars_2euclidres_alternates)
D_stack_energy = np.array(D_stack_energy)

np.save(result_path+'/D_stack_energy.npy', D_stack_energy)
np.save(result_path+'/steps_inners.npy', steps_inners)
np.save(result_path+'/MSE_rel_nor.npy', MSE_rel_nor)
np.save(result_path+'/MSE_rel_nor_alternates.npy', MSE_rel_nor_alternates)
np.save(result_path+'/loss_inners.npy', loss_inners_save) # descomentar depois de ter transformado em numpy array
np.save(result_path+'/loss_iters.npy', loss_iters)
np.save(result_path+'/loss_alternates.npy', loss_alternates)
np.save(result_path+'/MSE_rel_nor_integrated_alternates.npy', MSE_rel_nor_integrated_alternates)
np.save(result_path+'/MSE_rel_nor_integrated.npy', MSE_rel_nor_integrated)
np.save(result_path+'/MSE_rel_nor_stars_2euclidres.npy', MSE_rel_nor_stars_2euclidres)
np.save(result_path+'/MSE_rel_nor_stars_2euclidres_alternates.npy', MSE_rel_nor_stars_2euclidres_alternates)

if gt_wvlAll:
    np.save(result_path+'/barycenters_wvlAll.npy',barycenters_wvlAll)
np.save(result_path+'/D_stack.npy',D_stack)
np.save(result_path+'/barycenters.npy',barycenters)
np.save(result_path+'/A.npy',A)
np.save(result_path+'/psf_est.npy',psf_est)
np.save(result_path+'/obs_est.npy',obs_est)
np.save(result_path+'/stars_est_2euclidres_nor.npy',stars_est_2euclidres_nor)
np.save(result_path+'/stars_est_2euclidres.npy',stars_est_2euclidres)
np.save(result_path+'/psf_est_shift_nor.npy',psf_est_shift_nor)
np.save(result_path+'/shifts.npy',shifts)
np.save(result_path+'/flux.npy',flux)

np.save(result_path+'/SEDs.npy',spectrums)
np.save(result_path+'/fov.npy',fov)
np.save(result_path+'/lambdas.npy',lbdas)
np.save(result_path+'/gt_PSFs.npy',gt_PSFs)




raise ValueError('Check gradient of mix_stack[1] with small data. Pq nao ta mudando quase nada???????')


#%%    

fig = plt.figure()
plt.plot(range(MSE_rel_nor_stars_2euclidres_alternates.shape[0]), MSE_rel_nor_stars_2euclidres_alternates)
plt.title(r'integrated MSE  across alternate scheme')
plt.show()

# plot MSE integrated
fig = plt.figure()
plt.plot(range(nb_obj), MSE_rel_nor_stars_2euclidres[-1,:] )
plt.xlabel(r'Objects')
plt.ylabel(r'relative MSE integrated PSF')
plt.title('Final')

for wvl in range(MSE_rel_nor_alternates.shape[2]):
    fig = plt.figure()
    plt.plot(range(MSE_rel_nor_alternates.shape[0]), MSE_rel_nor_alternates[:,:,wvl]) # <iter,objs, wvls>
    plt.title(r'MSE at wvl {} across alternate scheme'.format(wvl))
    plt.show()
    fig = plt.figure()
    plt.plot(range(nb_obj), MSE_rel_nor[-1,:,wvl] )
    plt.xlabel(r'Objects')
    plt.ylabel(r'relative MSE')
    plt.title('Final')

   

for j in range(len(loss_inners)):
    fig = plt.figure()
    plt.plot(range(len(loss_inners[j])), loss_inners[j])
    plt.title(r'Loss at each descent on the gradient')
    plt.show()

    fig = plt.figure()
    plt.plot(range(len(loss_inners[j])), np.log(loss_inners[j]))
    plt.title(r'Loss at each descent on the gradient (log)')
    plt.show()

fig = plt.figure()
plt.plot(range(len(loss_alternates)), loss_alternates)
plt.title(r'Loss at each alternate scheme')
plt.show()



fig = plt.figure()
plt.plot(range(len(loss_iters)), loss_iters)
plt.title(r'Loss at each overal optimization scheme')
plt.show()

for it in range(len(steps_inners)):
    fig = plt.figure()
    plt.plot(range(len(steps_inners[it])), steps_inners[it])
    plt.title(r'Steps')
    plt.show()

#%%

if problem_type in ['mix','lbdaRCA']:
    for i in range(D_stack.shape[-1]):
        print "===============comp " + str(i)
        for at in range(2):
            print "atom "+ str(at)
            print "first guess"
            tk.plot_func(D_stack_0[:,at,i])
            print "Final,  ", np.sum(abs(D_stack[:,at,i]))
            tk.plot_func(D_stack[:,at,i])
#            print "res"
#            tk.plot_func(D_stack[:,at,i] - D_stack_0[:,at,i])
            
tk.plot_func(A)


#%% D_stack energy
for it in range(n_iter):
    fig = plt.figure() 
    plt.plot(range(D_stack_energy.shape[0]), D_stack_energy[:,:,it] )
    print D_stack_energy[:,:,it]
    plt.xlabel(r'Iterations')
    plt.ylabel(r'Dictionary integrated energy')            


#%%
# Build integrated error
stars_est_2euclidres_nor = stars_est_2euclidres/np.sum(abs(stars_est_2euclidres),axis=(0,1))
error_integrated_energy = np.sum(abs(stars_est_2euclidres_nor - gt_stars_2euclidrec), axis=(0,1))

# plot error energy
fig = plt.figure()
plt.plot(range(nb_obj), error_integrated_energy )
plt.xlabel(r'Objects')
plt.ylabel(r'Error integrated energy')

# Build chromatic error
if problem_type in ['mix','lbdaRCA']:
    error = psf_est_shift_nor - gt_PSFs
    error_energy = np.sum(abs(error), axis=(0,1))

#%%
for obj in range(nb_obj):
    print "estimated, ", np.sum(abs(stars_est_2euclidres[:,:,obj]))
    tk.plot_func((stars_est_2euclidres_nor[:,:,obj]))
    print "gt "
    tk.plot_func(gt_stars_2euclidrec[:,:,obj])
    # show error
    print "error, ", error_integrated_energy[obj]
    tk.plot_func(stars_est_2euclidres_nor[:,:,obj] - gt_stars_2euclidrec[:,:,obj])
    
#%%
psf_est_nor = psf_est/np.sum(abs(psf_est),axis=(0,1))
for pos in range(gt_PSFs.shape[2]):
    print ">>>>>>> pos "+ str(pos)
    for wvl in range(gt_PSFs.shape[-1]):
        print "wvl "+str(wvl)
        print "estimated ",np.sum(abs(psf_est[:,:,pos,wvl])) 
        tk.plot_func(psf_est_nor[:,:,pos,wvl], wind=(0,np.max(gt_PSFs[:,:,pos,wvl])))
        print "gt "
        tk.plot_func(gt_PSFs[:,:,pos,wvl])
        print "error ", error_energy[pos,wvl]
        tk.plot_func(error[:,:,pos,wvl])

#%%
for obj in range(nb_obj):
    print np.sum(abs(stars[:,:,obj]))
    tk.plot_func(stars[:,:,obj])
#%%

for wvl in range(nb_wvl):
    print "wvl "+str(wvl)
    for obj in range(nb_obj):
        tk.plot_func(abs(psf_est[:,:,obj,wvl]))

for comp in range(nb_comp_RCA):
    tk.plot_func((S_stack[:,comp]))

for comp in range(nb_comp_RCA):
    tk.plot_func((mix_stack[1][:,comp]))

for comp in range(nb_comp):
    for at in range(2):
        tk.plot_func(D_stack[:,at,comp])



for obj in range(nb_obj):
    tk.plot_func(psf_est[:,:,obj])




for obj in range(nb_obj):
    tk.plot_func(obs_est[:,:,obj])



for comp in range(nb_comp_chrom):
    for at in range(2):
        tk.plot_func(D_stack[:,at,comp])





