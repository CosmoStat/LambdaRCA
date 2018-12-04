import numpy as np 
import scipy.signal as scisig
import utils as utils
import gc
import transforms
import psf_toolkit as tk
import matplotlib.pyplot as plt
from sf_tools.image.shape import Ellipticity
from sklearn.cluster import KMeans


def MtX_coeff_graph(stars,barycenters,spectrums,sig,flux,ker_rot,D=2):
    nb_comp = barycenters.shape[1]
    nb_wvl = spectrums.shape[0]
    nb_obj = stars.shape[-1]
    N = barycenters.shape[0]

    psf_est = np.zeros((N,nb_obj))
    for i in range(nb_obj):
        psf_temp = (flux[i]/sig[i])*scisig.convolve(utils.transpose_decim(stars[:,:,i],D),ker_rot[:,:,i],mode='same')
        psf_est[:,i] = psf_temp.reshape((N,))


    A = np.zeros((nb_comp,nb_obj))
    for i in range(nb_obj):
        Si = np.zeros((N,nb_comp))
        for k in range(0,nb_comp):
            Si[:,k] = (barycenters[:,k,:].dot(spectrums[:,i].reshape((nb_wvl,1)))).reshape((N,))
        A[:,i] = (np.transpose(Si).dot(psf_est[:,i].reshape((N,1)))).reshape((nb_comp,))
    gc.collect()

    return A


def MX(A,barycenters,spectrums,sig,flux,ker,D=2):
    nb_comp = A.shape[0]
    nb_wvl = spectrums.shape[0]
    nb_obj = A.shape[1]
    N = barycenters.shape[0]
    shap = (int(np.sqrt(N)),int(np.sqrt(N)))

    mono_chromatic_psf = np.zeros((N,nb_obj,nb_wvl))

    for v in range(nb_wvl):
        mono_chromatic_psf[:,:,v] = barycenters[:,:,v].dot(A)

    stars_est = np.zeros((shap[0]/D,shap[1]/D,nb_obj))

    for i in range(nb_obj):
        stars_temp = (mono_chromatic_psf[:,i,:].dot(spectrums[:,i].reshape((nb_wvl,1)))).reshape(shap)
        stars_est[:,:,i] = (flux[i]/sig[i])*utils.decim(scisig.fftconvolve(stars_temp,ker[:,:,i],mode='same'),D,av_en=0)
    gc.collect()


    return stars_est


def get_noise_arr_dict_wvl(D_stack,shap):

    Wavelet_transf_dict = transforms.dict_wavelet_transform(shap, D_stack.shape[-1])
    D_stack_scales = Wavelet_transf_dict.op(D_stack)#<nb_filters,pixels_x,pixels_y,nb_atoms,nb_comp>

    nb_scales = D_stack_scales.shape[0]
    nb_comp = D_stack.shape[-1]

    noise_map = np.ones((nb_scales-1,shap[0],shap[1], 2, nb_comp))
    # mads = np.zeros((nb_scales-1, 2, nb_comp))
    for f in range(nb_scales-1):
        for comp in range(nb_comp):
            for at in range(2):
                mad =  utils.mad(D_stack_scales[f,:,:,at,comp])
                noise_map[f,:,:,at,comp] *= mad

    
    return noise_map



def get_noise_arr_RCA_wvl(S_stack,shap):

    Wavelet_transf_RCA = transforms.RCA_wavelet_transform(shap, S_stack.shape[-1])
    S_stack_scales = Wavelet_transf_RCA.op(S_stack)#<nb_filters,pixels_x,pixels_y,nb_atoms,nb_comp>

    nb_scales = S_stack_scales.shape[0]
    nb_comp = S_stack.shape[-1]

    noise_map = np.ones((nb_scales-1,shap[0],shap[1],nb_comp))
    # mads = np.zeros((nb_scales-1, 2, nb_comp))
    for f in range(nb_scales-1):
        for comp in range(nb_comp):
            mad =  utils.mad(S_stack_scales[f,:,:,comp])
            noise_map[f,:,:,comp] *= mad
    
    return noise_map




def D_stack_first_guess(shap,nb_im,nb_comp,feat_init,sr_first_guesses=None,gt_PSFs=None):
    nb_atoms = 2
    p = shap[0]*shap[1] # number of pixels
    D_stack = []
    nb_pop = int(nb_im/nb_comp)
    sr_star_all = []
    for i in range(nb_comp):
        if feat_init == "uniform":
            Ys = np.ones((p,nb_atoms)) / p
        elif feat_init == "random":
            Ys = np.random.rand(p,nb_atoms) #<2,42x42>
            Ys = (Ys.T / np.sum(Ys, axis = 1)).T #normalize the total of mass in each line

        elif feat_init == "zoom":

            in_fact = 1.3
            out_fact = 0.5
            Ys = np.zeros((p,nb_atoms))   
            # guess = utils.gauss_convolve(stars_first_guess[:,:,i],sig=0.7) #TO DO: add random shift and noise
            guess = stars_first_guess[:,:,i]
            zIn = utils.clipped_zoom(guess,in_fact).reshape(-1)
            zOut = guess.reshape(-1)
            # zOut = utils.clipped_zoom(guess,out_fact).reshape(-1)

            Ys[:,0] = zIn / np.sum(zIn, axis = 0) #normalize the total of mass in each line
            Ys[:,1] = zOut/ np.sum(zOut, axis = 0)

        elif feat_init == "ground_truth":

            Ys = np.zeros((p,nb_atoms)) 
            guess = gt_PSFs[:,:,i,:]
            low = guess[:,:,0]
            high = guess[:,:,-1]
            high_shift = utils.shift_y_to_grid_x(high,low)
            Ys[:,0] = np.copy(high_shift.reshape(-1)/np.sum(high_shift))
            Ys[:,1] = np.copy(low.reshape(-1)/np.sum(low))
            tk.plot_func(Ys[:,0])
            tk.plot_func(Ys[:,1])

        elif feat_init == "zoom_2":
            in_fact = 1.3
            out_fact = 0.5

            # Denoise

            guess = (flux[i]/sig[i])*scisig.convolve(utils.transpose_decim(im_stack[:,:,i],2),ker_rot[:,:,i],mode='same')

            Ys = np.zeros((p,nb_atoms))   
            
            zIn = utils.clipped_zoom(guess,in_fact).reshape(-1)
            zOut = guess.reshape(-1)
            # zOut = utils.clipped_zoom(guess,out_fact).reshape(-1)

            Ys[:,0] = zIn / np.sum(zIn, axis = 0) #normalize the total of mass in each line
            Ys[:,1] = zOut/ np.sum(zOut, axis = 0)


        # elif feat_init == "super_res":
        #     # Change stars to Louis format
        #     Y = im_stack[:,:,i*nb_pop:(i+1)*nb_pop].swapaxes(2,1).swapaxes(0,1)
        #     deci_matrix = utils.D2(D,shap_obs[0])

        #     rel_shifs = [shifts[i*nb_pop]-shifts[i*nb_pop+n] for n in range(nb_pop)]
        #     res = utils.first_guess_integer(Y,nb_pop,shap[0],deci_matrix,shifts[i*nb_pop:(i+1)*nb_pop])
        #     sr_star_shifted = res[1]
        #     # Denoise
        #     # image_rec = call_mr_filter(image_noisy, opt=['-t', '1', '-s', str(n_sigma)])

        #     # Centralize
        #     sr_star = scisig.convolve(sr_star_shifted,ker_rot[:,:,i*nb_pop],mode='same')


            
        #     sr_star_all.append(sr_star)


        #     # Zoom
        #     in_fact = 1.3
        #     out_fact = 0.5
        #     Ys = np.zeros((p,nb_atoms)) 
        #     guess = sr_star
        #     zIn = utils.clipped_zoom(guess,in_fact).reshape(-1)
        #     zOut = guess.reshape(-1)
        #     # zOut = utils.clipped_zoom(guess,out_fact).reshape(-1)

        #     Ys[:,0] = zIn / np.sum(zIn, axis = 0) #normalize the total of mass in each line
        #     Ys[:,1] = zOut/ np.sum(zOut, axis = 0)

        # 

        # elif feat_init == "super_res":
        #     # Zoom
        #     in_fact = 1.2
        #     out_fact = 0.6
        #     Ys = np.zeros((p,nb_atoms)) 
        #     guess = sr_first_guesses[i]

        #     zIn = utils.clipped_zoom(guess,in_fact).reshape(-1)
        #     # zOut = utils.clipped_zoom(guess,out_fact).reshape(-1)
        #     zOut = guess.reshape(-1)

        #     Ys[:,0] = zIn / np.sum(zIn, axis = 0) #normalize the total of mass in each line
        #     Ys[:,1] = zOut/ np.sum(zOut, axis = 0)


        elif feat_init == "super_res_zout":
            # Zoom
            in_fact = 1.2
            out_fact = 0.6
            Ys = np.zeros((p,nb_atoms)) 
            guess = abs(sr_first_guesses[i])

            zIn = utils.clipped_zoom(guess,in_fact)

            zOut = utils.clipped_zoom(guess,out_fact)
#            zOut = guess
#            zOut_shift = utils.shift_y_to_grid_x(zOut,zIn)
            zIn_shift = utils.shift_y_to_grid_x(zIn,zOut)




            Ys[:,0] = np.copy(zIn_shift.reshape(-1) / np.sum(zIn_shift)) #normalize the total of mass in each line
            Ys[:,1] = np.copy(zOut.reshape(-1)/ np.sum(zOut))
            
          
        
            
            
        elif feat_init == "super_res_zout_RED":
            in_fact = 1.4
            out_fact = 0.8
            Ys = np.zeros((p,nb_atoms)) 
            guess = sr_first_guesses[i]
    
            zIn = utils.clipped_zoom(guess,in_fact) 
            tk.plot_func(zIn,title="before")
            zOut = utils.clipped_zoom(guess,out_fact)
            
            samp = 4.
            sig = .75 * 12 / 0.1 / samp
            Ell_zIn = Ellipticity(zIn, sigma=sig)
            cent = Ell_zIn.centroid
            r = 7.0
            nb_pixels_border = 3
            avg_pixel = 0.0
            for m in range(nb_pixels_border):
                for j in range(nb_pixels_border):
                    avg_pixel += zIn[m,j] + zIn[-m,-j]
            avg_pixel /= nb_pixels_border*nb_pixels_border*2 
            mask = np.zeros((shap[0],shap[1]))
            for row in range(shap[0]):
                for col in range(shap[1]):
                    if ((row-cent[0])**2 + (col-cent[1])**2) >=r and zIn[row,col]  > 70*avg_pixel:
                        mask[row,col] =1.0
                        zIn[row,col] *=2.0
            tk.plot_func(mask,title="mask")   
            tk.plot_func(zIn,title="later")            
            zIn_shift = utils.shift_y_to_grid_x(zIn,zOut)
            # zOut = guess.reshape(-1)
     
            Ys[:,0] = np.copy(zIn_shift.reshape(-1) / np.sum(zIn_shift)) #normalize the total of mass in each line
            Ys[:,1] = np.copy(zOut.reshape(-1)/ np.sum(zOut))
         
            


        elif feat_init == "guess_bkup":
            Ys = D_first[:,:,i]


        elif feat_init == "sr_semi_free":
            Ys = np.zeros((p,nb_atoms)) 
            if i < 0:
                 # Zoom
                in_fact = 2.0
                out_fact = 0.5
                guess = sr_first_guesses[i]

                zIn = utils.clipped_zoom(guess,in_fact)
                zOut = utils.clipped_zoom(guess,out_fact)
                # zOut_shift = utils.shift_y_to_grid_x(zOut,zIn)
                zIn_shift = utils.shift_y_to_grid_x(zIn,zOut)
                # zOut = guess.reshape(-1)

                Ys[:,0] = np.copy(zIn_shift.reshape(-1) / np.sum(zIn_shift)) #normalize the total of mass in each line
                Ys[:,1] = np.copy(zOut.reshape(-1)/ np.sum(zOut))
            else:
                one = np.random.normal(0.0, 2.0, shap[0]*shap[1])
                two = np.random.normal(0.0, 2.0, shap[0]*shap[1])
                Ys[:,0] = one/np.sum(abs(one))
                Ys[:,1] = two/np.sum(abs(two))


        D_stack.append(np.copy(Ys))

    D_stack = np.array(D_stack)
    D_stack = D_stack.swapaxes(0,1).swapaxes(1,2)

    return D_stack






def shift_PSF_to_gt(psf_est,gt_PSFs):
    psf_est_shifted = np.zeros(psf_est.shape)
    for obj in range(psf_est.shape[2]):
        for wvl in range(psf_est.shape[-1]):
            psf_est_shifted[:,:,obj,wvl] = utils.shift_y_to_grid_x(psf_est[:,:,obj,wvl],gt_PSFs[:,:,obj,wvl])
    return psf_est_shifted

def shift_PSFinter_to_gt(psf_est,gt_PSFs):
    psf_est_shifted = np.zeros(psf_est.shape)

    for obj in range(psf_est.shape[-1]):
            psf_est_shifted[:,:,obj] = utils.shift_y_to_grid_x(psf_est[:,:,obj],gt_PSFs[:,:,obj])
    return psf_est_shifted



def S_stack_first_guess(shap,nb_comp,feat_init="random",sr_first_guesses=None):

    S_stack = np.zeros((shap[0]*shap[1],nb_comp))
    
    if feat_init == "random":
        for i in range(nb_comp):
            temp = np.random.normal(0.0, 2.0, shap[0]*shap[1])
            temp = temp/np.sum(temp)
            S_stack[:,i] = np.copy(temp)

    if feat_init == "super_res":
        for i in range(nb_comp):
            if i < sr_first_guesses.shape[0]:
                S_stack[:,i] = np.copy(sr_first_guesses[i].reshape(-1)/np.sum(abs(sr_first_guesses[i])))
            else:
                temp = np.random.normal(0.0, 2.0, shap[0]*shap[1])
                S_stack[:,i] = np.copy(temp/np.sum(abs(temp)))

     
    return S_stack 


def MtX_RCA(mx,A,flux,sig,ker_rot,D=2):

    nb_comp = A.shape[0]
    nb_obj = A.shape[1]
    N = D*mx.shape[0]*D* mx.shape[1]

    psf_est = np.zeros((N,nb_obj))
    for i in range(nb_obj):
        psf_temp = (flux[i]/sig[i])*scisig.convolve(utils.transpose_decim(mx[:,:,i],D),ker_rot[:,:,i],mode='same')
        psf_est[:,i] = psf_temp.reshape((N,))


    S_stack = A.dot(psf_est.swapaxes(0,1))
    S_stack = S_stack.swapaxes(0,1) # back to format <N, nb_comp>
    
    return S_stack


def MX_RCA(S_stack,A,flux,sig,ker,D=2):

    nb_comp = A.shape[0]
    nb_obj = A.shape[1]
    N = S_stack.shape[0]
    shap = (int(np.sqrt(N)),int(np.sqrt(N)))



    combined_psf = S_stack.dot(A)

    stars_est = np.zeros((shap[0]/D,shap[1]/D,nb_obj))



    for i in range(nb_obj):
        stars_est[:,:,i] = (flux[i]/sig[i])*utils.decim(scisig.fftconvolve(combined_psf[:,i].reshape(shap),ker[:,:,i],mode='same'),D,av_en=0)
    gc.collect()


    return stars_est



def MtX_coef_graph_RCA(stars,S_stack,sig,flux,ker_rot,D=2):
    nb_comp = S_stack.shape[-1]
    nb_obj = stars.shape[-1]
    N = S_stack.shape[0]

    psf_est = np.zeros((N,nb_obj))
    for i in range(nb_obj):
        psf_temp = (flux[i]/sig[i])*scisig.convolve(utils.transpose_decim(stars[:,:,i],D),ker_rot[:,:,i],mode='same')
        psf_est[:,i] = psf_temp.reshape((N,))


    A = np.transpose(S_stack).dot(psf_est)



    return A



def logitTonormal(Dlog_stack):
        return np.exp(Dlog_stack)/np.sum(np.exp(Dlog_stack), axis = 0)

def normalTologit(D_stack):
        return np.log(abs(D_stack))



def SnA(LRobs, shifts):
    nbpix, _, nobj = LRobs.shape
    # round up shift to integers
    int_shifts = np.round(shifts).astype(int)
    # zero pad data
    zerop = np.zeros((2*nbpix,2*nbpix,nobj))
    zerop[::2, ::2] = LRobs
    # initialize HR pic and pixel counter
    HR = np.zeros((2*(nbpix+2), 2*(nbpix+2)))
    HR_P = np.zeros((2*(nbpix+2), 2*(nbpix+2)))

    lim = nbpix*2 +2
    for j in range(nobj):
        # shift...
        dx, dy = int_shifts[j]
        # ... and add.
        HR[2-dx:lim-dx, 2-dy:lim-dy] += zerop[:,:,j]
        # (also count activated pixels)
        HR_P[2-dx:lim-dx:2, 2-dy:lim-dy:2] += 1
    # if np.any(HR_P[:-1,:-1]==0): # last row and column will always be 0
    #     raise ValueError('Get on out of here with these shifts')
    # recrop and normalize
    temp = HR_P[2:-2,2:-2]
    temp[temp==0] = 1
#    return HR[2:-2,2:-2] / HR_P[2:-2,2:-2]
    return HR[2:-2,2:-2] / temp

def SR_first_guesses(stars,shifts):
    np.random.seed(17471)
    group_idx = np.arange(80)
    np.random.shuffle(group_idx)
    # split in 3 groups and create first guesse
    first_guesses = np.array([SnA(stars[:,:,group_idx[:26]], shifts[group_idx[:26]]),
                          SnA(stars[:,:,group_idx[26:52]], shifts[group_idx[26:52]]),
                          SnA(stars[:,:,group_idx[52:]], shifts[group_idx[52:]])])

    print " SR first guesses"
    for fg in first_guesses:
        tk.plot_func(fg)
        print np.any(fg==0)
    
    return first_guesses


def SR_first_guesses_kmeans(stars,fov,shifts,nb_comp): 
    nb_obj = stars.shape[-1]
    kmeans = KMeans(n_clusters=nb_comp, random_state=0).fit(fov)
    labels = kmeans.predict(fov)
    plt.scatter(fov[:, 0], fov[:, 1], c=labels)
    groups = []
    for i in range(nb_comp):
        groups.append([])
    for obj in range(nb_obj):
        groups[labels[obj]].append(obj)
    
    sr_stars = []
    for i in range(nb_comp):
        print len(groups[i])," stars"
        sr_stars.append(SnA(stars[:,:,groups[i]],shifts[groups[i]]))
    
    sr_stars = np.array(sr_stars)
    
    for sr in sr_stars:
        tk.plot_func(sr)
        
        
    A = np.zeros((nb_comp,nb_obj))
    for i in range(nb_comp):
        A[i,groups[i]] = 1.0
    
    tk.plot_func(A)   
    
    
    return sr_stars,A  



def SR_first_guesses_radial(stars,fov,shifts,nb_comp): 
    nb_obj = stars.shape[-1]
    
    min_el_per_grp = 15
    W = 0.3 # inches?
    H = 0.25
    center = [0.0,0.775]
    r_last = min(W,H)/2 
    
    segment = r_last/(nb_comp-1)
    r_list = []
    for i in range(nb_comp-1):
        r_list.append(segment*(i+1))

    groups = []
    for i in range(nb_comp):
        groups.append([])
    for obj in range(nb_obj):
        sector = 0
        r_obj = np.sqrt((fov[obj,0] - center[0])**2 + (fov[obj,1] - center[1])**2)
    #    if r_obj > 0.125:
    #        print "LAST SECTOR"
        
        # Find sector
        not_found = True
        j = int((len(r_list)-1)/2) # middle of the list
        while not_found:
            if r_obj < r_list[j]:
                j -= 1
                if j < 0:
                    not_found = False
            elif r_obj >= r_list[j]:
                if r_obj < r_list[j+1]:
                    not_found = False
                else:
                    j +=1
                    if j == len(r_list)-1:
                        not_found = False
        sector = j+1
    
        # Put at the sector
        groups[sector].append(obj)   
    
    # Check if all groups have minimum stars requirement (NOT OPTIMAL)
    for i in range(nb_comp):
        if len(groups[i]) < min_el_per_grp:
            deficit = min_el_per_grp - len(groups[i])
            # go through neighbor groups to find a donator
            neighbors = [i-1,i+1]
            for j in neighbors:
                if j < 0:
                    continue
                sobra = len(groups[j]) - min_el_per_grp
                if sobra > 0:
                    gain = min(sobra,deficit)
                    groups[i].extend([groups[j].pop(el) for el in range(gain)])
                    deficit -= gain
                    if deficit == 0:
                        break  

    
    sr_stars = []
    for i in range(nb_comp):
        print len(groups[i])," stars"
        sr_stars.append(SnA(stars[:,:,groups[i]],shifts[groups[i]]))
    
    sr_stars = np.array(sr_stars)
    
    for sr in sr_stars:
        tk.plot_func(sr)
        
        
    plt.figure()
    plt.plot(fov[groups[0],0], fov[groups[0],1], '.',color='red')
    plt.plot(fov[groups[1],0], fov[groups[1],1], '.',color='blue') 
    plt.plot(fov[groups[2],0], fov[groups[2],1], '.',color='green')  
    
    
    A = np.zeros((nb_comp,nb_obj))
    for i in range(nb_comp):
        A[i,groups[i]] = 1.0
    
    tk.plot_func(A)   
          
    
    
    return sr_stars,A  


#def SR_first_guess_FOV_splitRadial():
    
    
    
    
    
    
    
    
    