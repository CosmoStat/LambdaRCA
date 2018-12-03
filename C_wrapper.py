
from __future__ import division
from builtins import zip
import numpy as np
from os import remove,rmdir
from subprocess import check_call,check_output
from datetime import datetime
import time
from modopt.interface.errors import is_executable, warn
import sys
import os
import shutil



def call_WDL(A=None,spectrums=None,flux=None,sig=None,ker=None,rot_ker=None, D_stack=None,Dlog_stack=None,w_stack=None,gamma=None,
    n_iter_sink=None,y=None,N=None,barycenters=None,func='', opt='', path='./',remove_files=True): 

    executable = './app_dictionary_learning'

    # Make sure mr_transform is installed.
    is_executable(executable)


     # Specify directories to save temporary files . ADD 2 to everybody when working in parallel (at the wend of .._warapper and ..._output)

    # Create a unique string using the current date and time.
    # unique_string = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    
    unique_string = str(time.time())

    varDir = "/Users/rararipe/Documents/src/C/data/lbdaRCA_wrapper/variables"+unique_string
    if not os.path.exists(varDir):
        os.makedirs(varDir)
    dictDir = varDir+"/dictionary" 
    if not os.path.exists(dictDir):
        os.makedirs(dictDir)
    deltaDir = varDir+"/delta" 
    if not os.path.exists(deltaDir):
        os.makedirs(deltaDir)
    kerDir = varDir+"/ker"
    if not os.path.exists(kerDir):
        os.makedirs(kerDir)
    rkerDir = varDir+"/rot_ker" 
    if not os.path.exists(rkerDir):
        os.makedirs(rkerDir)
    inDir = "/Users/rararipe/Documents/src/C/data/lbdaRCA_wrapper/observed_stars"+unique_string
    if not os.path.exists(inDir):
        os.makedirs(inDir)
    baryDir = varDir+"/barycenters"
    if not os.path.exists(baryDir):
        os.makedirs(baryDir)
    output_path = "/Users/rararipe/Documents/Data/lbdaRCA_wdl/wrapper_output"+unique_string
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    logit = False
    W = H = int(np.sqrt(N))

    if A is not None: 
        nb_obj = A.shape[1]
        nb_comp = A.shape[0]
        N_lr = int(N/4)
        W_lr = int(np.sqrt(N_lr))
        ker_dim = ker.shape[0]
        nb_wvl = spectrums.shape[0]
        np.savetxt(varDir+"/A.csv", A, delimiter=",")
        np.savetxt(varDir+"/spectrums.csv", spectrums, delimiter=",") 
        np.savetxt(varDir+"/flux.csv", flux, delimiter=",")
        np.savetxt(varDir+"/sig.csv", sig, delimiter=",")
        for obj in range(nb_obj):
            np.savetxt(kerDir+"/ker_"+format(obj, '05d')+".csv", ker[:,:,obj], delimiter=",")
        for obj in range(nb_obj):
            np.savetxt(rkerDir+"/rker_"+format(obj, '05d')+".csv", rot_ker[:,:,obj], delimiter=",")

    if D_stack is not None:
        nb_comp = D_stack.shape[2]
        nb_atoms = D_stack.shape[1]
        nb_wvl = w_stack.shape[0]
        for i in range(nb_comp):
            np.savetxt(dictDir+"/D_"+str(i)+ ".csv", D_stack[:,:,i], delimiter=",")
        np.savetxt(varDir+"/w_stack.csv", w_stack, delimiter=",")

    if Dlog_stack is not None:
        logit = True
        nb_comp = Dlog_stack.shape[2]
        nb_atoms = Dlog_stack.shape[1]
        nb_wvl = w_stack.shape[0]
        for i in range(nb_comp):
            np.savetxt(deltaDir+"/delta_"+str(i)+ ".csv", Dlog_stack[:,:,i], delimiter=",")
        np.savetxt(varDir+"/w_stack.csv", w_stack, delimiter=",")

    if barycenters is not None:
        for comp in range(nb_comp):
            np.savetxt(baryDir+"/bary_"+str(i)+ ".csv", barycenters[:,comp,:], delimiter=",")

    if y is not None:
        
        for obj in range(nb_obj):
           np.savetxt(inDir+"/star_"+format(obj, '05d')+".csv", y[:,:,obj], delimiter=",")


    if isinstance(opt, str):
        opt = opt.split()




    
    try:


        if func == "--MtX_wdl":

            if logit:
                check_call([executable] + ["-i",inDir,"-iv",varDir,"-idLog",deltaDir,"-iw", "w_stack.csv","-iA", 
                "A.csv","-ik",kerDir,"-irk",rkerDir,"-isp", "spectrums.csv", "-if", "flux.csv", "-isi", "sig.csv","-g",str(gamma),
                "-n",str(n_iter_sink),"-o",output_path,"-W", str(W), "-H",str(H),"-P",str(nb_obj),"-nb_comp",str(nb_comp),"-nb_atoms",str(nb_atoms),"-nb_wvl",
                str(nb_wvl),"-ker_dim", str(ker_dim), "--MtX_wdl"])

            else:
                check_call([executable] + ["-i",inDir,"-iv",varDir,"-id",dictDir,"-iw", "w_stack.csv","-iA", 
                    "A.csv","-ik",kerDir,"-irk",rkerDir,"-isp", "spectrums.csv", "-if", "flux.csv", "-isi", "sig.csv","-g",str(gamma),
                    "-n",str(n_iter_sink),"-o",output_path,"-W", str(W), "-H",str(H),"-P",str(nb_obj),"-nb_comp",str(nb_comp),"-nb_atoms",str(nb_atoms),"-nb_wvl",
                    str(nb_wvl),"-ker_dim", str(ker_dim), "--MtX_wdl"])

        if func == "--MX_wdl":
            # print executable,"-iv ",varDir,"-id ",dictDir,"-iw ", "w_stack.csv ","-iA ", "A.csv ","-ik ",kerDir,"-irk ",rkerDir,"-isp ", "spectrums.csv ", "-if ", "flux.csv ", "-isi ", "sig.csv ","-g ",str(gamma),"-n ",str(n_iter_sink),"-o ",output_path,"-W ", str(W), "-H ",str(H),"-P ",str(nb_obj),"-nb_comp ",str(nb_comp),"-nb_atoms ",str(nb_atoms),"-nb_wvl ",str(nb_wvl)," -ker_dim ", str(ker_dim), "--MX_wdl"


            check_call([executable] + ["-iv",varDir,"-id",dictDir,"-iw", "w_stack.csv","-iA", 
                "A.csv","-ik",kerDir,"-irk",rkerDir,"-isp", "spectrums.csv", "-if", "flux.csv", "-isi", "sig.csv","-g",str(gamma),
                "-n",str(n_iter_sink),"-o",output_path,"-W", str(W), "-H",str(H),"-P",str(nb_obj),"-nb_comp",str(nb_comp),"-nb_atoms",str(nb_atoms),"-nb_wvl",
                str(nb_wvl),"-ker_dim", str(ker_dim), "--MX_wdl"])

        if func == "--bary":
            

            check_output([executable] + ["-iv",varDir,"-id",dictDir,"-iw", "w_stack.csv","-g",str(gamma),
                "-n",str(n_iter_sink),"-o",output_path,"-W", str(W), "-H",str(H),"-nb_comp",str(nb_comp),"-nb_atoms",str(nb_atoms),"-nb_wvl",
                str(nb_wvl), "--bary"])



    except Exception:

        warn('{} failed to run with the options provided.'.format(executable))

        if func == "--MtX_wdl" or func == "--MX_wdl" or func == "--MX_coeff" or func == "--MtX_coeff": 
            remove(varDir+"/A.csv")
            remove(varDir+"/spectrums.csv")
            remove(varDir+"/flux.csv")
            remove(varDir+"/sig.csv")
            for obj in range(nb_obj):
                remove(kerDir+"/ker_"+format(obj, '05d')+".csv")
                remove(rkerDir+"/rker_"+format(obj, '05d')+".csv")
            
        
        if func == "--MtX_wdl" or func == "--MX_wdl" or func == "--bary":
            remove(varDir+"/w_stack.csv")
            if logit:
                for i in range(nb_comp): 
                    remove(deltaDir+"/delta_"+str(i)+ ".csv")
            else:
                for i in range(nb_comp): 
                    remove(dictDir+"/D_"+str(i)+ ".csv")

        if func=="--MtX_wdl" or func=="--MtX_coeff":
            for obj in range(nb_obj):
                remove(inDir+"/star_"+format(obj, '05d')+".csv")

        if func=="MC_coeff" or func=="MtX_coeff":
            for comp in range(nb_comp):
                remove(baryDir+"/bary_"+str(i)+ ".csv")


        # rmdir(varDir)
        # rmdir(kerDir)
        # rmdir(rkerDir)
        # rmdir(dicDir)
        # rmdir(deltaDir)
        # rmdir(inDir)
        # rmdir(baryDir)


        if  os.path.exists(varDir):
            shutil.rmtree(varDir)
        if  os.path.exists(kerDir):
            shutil.rmtree(kerDir)
        if  os.path.exists(rkerDir):
            shutil.rmtree(rkerDir)
        if  os.path.exists(dictDir):
            shutil.rmtree(dictDir)
        if  os.path.exists(deltaDir):
            shutil.rmtree(deltaDir)
        if  os.path.exists(inDir):
            shutil.rmtree(inDir)
        if  os.path.exists(baryDir):
            shutil.rmtree(baryDir)
        if  os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        sys.exit(0)




    # else:



    # Retrieve barycenters.
    if func == "--MtX_wdl" or func == "--MX_wdl" or  func == "--bary" :
        C_barys = []
        for i in range(nb_comp):
            C_barys.append(np.genfromtxt(output_path+"/C_bary_"+str(i)+".csv" , delimiter=',')) 
        C_barys = np.array(C_barys) #<nb_comp,nb_wvl,N>
        C_barys = C_barys.swapaxes(1,2).swapaxes(0,1) #<N,nb_comp,nb_wvl>

    #Retrieve MX
    if func == "--MtX_wdl" or func == "--MX_wdl" or  func == "--MX_coeff" or  func == "--MtX_coeff" :
        MX_all = np.genfromtxt(output_path+"/C_MX_convolved_deci.csv" , delimiter=',')
        C_MX = np.zeros((W_lr,W_lr,nb_obj))
        for i in range(nb_obj):
            C_MX[:,:,i] = MX_all[N_lr*i:N_lr*(i+1)].reshape(W_lr, W_lr)



    #Retrieve MtX

    
    if func == "--MtX_wdl":
        C_grads_all = np.genfromtxt(output_path+"/gradients.csv" , delimiter=',')
        if logit:
            C_grads = np.zeros(Dlog_stack.shape)
        else:
            C_grads = np.zeros(D_stack.shape)
        for i in range(nb_comp):
                for a in range(nb_atoms):
                    C_grads[:,a,i] = C_grads_all[(i*nb_atoms+a)*N:(i*nb_atoms+a+1)*N]





    # Remove temporary files

    if remove_files:
        if func == "--MtX_wdl" or func == "--MX_wdl" or func == "--MX_coeff" or func == "--MtX_coeff":        
            remove(varDir+"/A.csv")
            remove(varDir+"/spectrums.csv")
            remove(varDir+"/flux.csv")
            remove(varDir+"/sig.csv")
            for obj in range(nb_obj):
                remove(kerDir+"/ker_"+format(obj, '05d')+".csv")
                remove(rkerDir+"/rker_"+format(obj, '05d')+".csv")
        
        if func == "--MtX_wdl" or func == "--MX_wdl" or func == "--bary":
            remove(varDir+"/w_stack.csv")
            if logit:
                for i in range(nb_comp): 
                    remove(deltaDir+"/delta_"+str(i)+ ".csv")
            else:
                for i in range(nb_comp): 
                    remove(dictDir+"/D_"+str(i)+ ".csv")

        if func=="--MtX_wdl" or func=="--MtX_coeff":
            for obj in range(nb_obj):
                remove(inDir+"/star_"+format(obj, '05d')+".csv")

        if func=="--MC_coeff" or func=="--MtX_coeff":
            for comp in range(nb_comp):
                remove(baryDir+"/bary_"+str(i)+ ".csv")


        
    
        if  os.path.exists(varDir):
            shutil.rmtree(varDir)
        if  os.path.exists(kerDir):
            shutil.rmtree(kerDir)
        if  os.path.exists(rkerDir):
            shutil.rmtree(rkerDir)
        if  os.path.exists(dictDir):
            shutil.rmtree(dictDir)
        if  os.path.exists(deltaDir):
            shutil.rmtree(deltaDir)
        if  os.path.exists(inDir):
            shutil.rmtree(inDir)
        if  os.path.exists(baryDir):
            shutil.rmtree(baryDir)
        if  os.path.exists(output_path):
            shutil.rmtree(output_path)


    # Return C results
    if func == "--MtX_wdl":
        return C_grads,C_MX,C_barys


    if func == "--MX_wdl":
        return C_MX,C_barys

    if func == "--bary":

        return C_barys

        










def call_MtX(A,spectrums,flux,sig,ker,rot_ker, D_stack,w_stack,C,gamma,
    n_iter_sink,y, opt='', path='./',remove_files=True): 


    
    
    executable = '../utilities/app_dictionary_learning'

    # Make sure mr_transform is installed.
    is_executable(executable)




    nb_comp = D_stack.shape[2]
    nb_obj = A.shape[1]
    N = D_stack.shape[0]
    N_lr = int(D_stack.shape[0]/4)
    nb_atoms = D_stack.shape[1]



    # Specify directories to save temporary files 

    varDir = "/Users/rararipe/Documents/src/C/data/lbdaRCA_wrapper_2/variables"
    dictDir = "/Users/rararipe/Documents/src/C/data/lbdaRCA_wrapper_2/variables/dictionary" 
    kerDir = "/Users/rararipe/Documents/src/C/data/lbdaRCA_wrapper_2/variables/ker" 
    rkerDir = "/Users/rararipe/Documents/src/C/data/lbdaRCA_wrapper_2/variables/rot_ker" 
    inDir = "/Users/rararipe/Documents/src/C/data/lbdaRCA_wrapper_2/observed_stars"
    output_path = "/Users/rararipe/Documents/Data/lbdaRCA_wdl/wrapper_output_2"



    for i in range(nb_comp):
        np.savetxt(dictDir+"/D_"+str(i)+ ".csv", D_stack[:,:,i], delimiter=",")


    np.savetxt(varDir+"/w_stack.csv", w_stack, delimiter=",")
    np.savetxt(varDir+"/A.csv", A, delimiter=",")
    np.savetxt(varDir+"/spectrums.csv", spectrums, delimiter=",")
    np.savetxt(varDir+"/flux.csv", flux, delimiter=",")
    np.savetxt(varDir+"/sig.csv", sig, delimiter=",")


    for obj in range(nb_obj):
        np.savetxt(kerDir+"/ker_"+str(obj)+".csv", ker[:,:,obj], delimiter=",")

    for obj in range(nb_obj):
        np.savetxt(inDir+"/star_"+str(obj)+".csv", y[:,:,obj], delimiter=",")

    for obj in range(nb_obj):
        np.savetxt(rkerDir+"/rker_"+str(obj)+".csv", rot_ker[:,:,obj], delimiter=",")




    if isinstance(opt, str):
        opt = opt.split()

    
    try:


        check_call([executable] + ["-i",inDir,"-iv",varDir,"-id",dictDir,"-iw", "w_stack.csv","-iA", "A.csv","-ik",kerDir,"-irk",rkerDir,"-isp", "spectrums.csv", "-if", "flux.csv", "-isi", "sig.csv","-g",str(gamma),"-n",str(n_iter_sink),"-o",output_path, "--MtX_wdl"])



    except Exception:

        warn('{} failed to run with the options provided.'.format(executable))
        

    else:



        # Retrieve barycenters.
        C_barys = []
        for i in range(nb_comp):
            C_barys.append(np.genfromtxt(output_path+"/C_bary_"+str(i)+".csv" , delimiter=',')) 
        C_barys = np.array(C_barys) #<nb_comp,nb_wvl,N>
        C_barys = C_barys.swapaxes(1,2).swapaxes(0,1) #<N,nb_comp,nb_wvl>

        #Retrieve MX
        MX_all = np.genfromtxt(output_path+"/C_MX_convolved_deci.csv" , delimiter=',')
        C_MX = np.zeros(y.shape)
        for i in range(nb_obj):
            C_MX[:,:,i] = MX_all[N_lr*i:N_lr*(i+1)].reshape(y.shape[0], y.shape[1])




        #Retrieve MtX
        C_grads_all = np.genfromtxt(output_path+"/gradients.csv" , delimiter=',')
        C_grads = np.zeros(D_stack.shape)
        for i in range(nb_comp):
                for a in range(nb_atoms):
                    C_grads[:,a,i] = C_grads_all[(i*nb_atoms+a)*N:(i*nb_atoms+a+1)*N]



        # Return the C results.
        return C_grads,C_MX,C_barys













