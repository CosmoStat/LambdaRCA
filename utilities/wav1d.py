'''
Created on Mar 30, 2015

@author: mjiang
'''
import numpy as np
import scipy.signal as psg
import param as pm

def scaleFilter(wname,wtype):
    if wtype == 1:
        if wname =='haar' or wname == 'db1':
            F = np.array([0.5,0.5])
            
        elif wname == 'db2':
            F = np.array([0.34150635094622,0.59150635094587,0.15849364905378,-0.09150635094587])
            
        elif wname == 'db3':
            F = np.array([0.23523360389270,0.57055845791731,0.32518250026371,-0.09546720778426,\
                          -0.06041610415535,0.02490874986589])
            
        elif wname == 'db4':
            F = np.array([0.16290171402562,0.50547285754565,0.44610006912319,-0.01978751311791,\
                          -0.13225358368437,0.02180815023739,0.02325180053556,-0.00749349466513])
            
        elif wname == 'db5':
            F = np.array([0.11320949129173,0.42697177135271,0.51216347213016,0.09788348067375,\
                           -0.17132835769133,-0.02280056594205,0.05485132932108,-0.00441340005433,\
                           -0.00889593505093,0.00235871396920])
        return F
    elif wtype == 2:
        if wname == '9/7':
            Df = np.array([0.0267487574110000,-0.0168641184430000,-0.0782232665290000,0.266864118443000,\
                           0.602949018236000,0.266864118443000,-0.0782232665290000,-0.0168641184430000,\
                           0.0267487574110000])
            Rf = np.array([-0.0456358815570000,-0.0287717631140000,0.295635881557000,0.557543526229000,\
                           0.295635881557000,-0.0287717631140000,-0.0456358815570000])
        
        return (Rf,Df)
 
def orthWavFilter(F):       
    p = 1
#     h1 = np.copy(F)
    Lo_R = np.sqrt(2)*F/np.sum(F)
#     Lo_R = F/np.sqrt(np.sum(F**2))
    Hi_R = np.copy(Lo_R[::-1])
    first = 2-p%2
#     print first 
#     print tmp
    Hi_R[first::2] = -Hi_R[first::2]
    Hi_D=np.copy(Hi_R[::-1])
    Lo_D=np.copy(Lo_R[::-1])
    return (Lo_D,Hi_D,Lo_R,Hi_R)


def biorWavFilter(Rf,Df):
    lr = len(Rf)
    ld = len(Df)
    lmax = max(lr,ld)
    if lmax%2:
        lmax += 1
    Rf = np.hstack([np.zeros((lmax-lr)/2),Rf,np.zeros((lmax-lr+1)/2)])
    Df = np.hstack([np.zeros((lmax-ld)/2),Df,np.zeros((lmax-ld+1)/2)])
    
    [Lo_D1,Hi_D1,Lo_R1,Hi_R1] = orthWavFilter(Df)
    [Lo_D2,Hi_D2,Lo_R2,Hi_R2] = orthWavFilter(Rf)
    
    return (Lo_D1,Hi_D1,Lo_R1,Hi_R1,Lo_D2,Hi_D2,Lo_R2,Hi_R2)

def wavFilters(wname,wtype,mode):
    if wtype == 1:
        F = scaleFilter(wname,1)
        (Lo_D,Hi_D,Lo_R,Hi_R) = orthWavFilter(F)
    elif wtype == 2:
        (Rf,Df) = scaleFilter(wname,2)
        [Lo_D,Hi_D1,Lo_R1,Hi_R,Lo_D2,Hi_D,Lo_R,Hi_R2] = biorWavFilter(Rf,Df)
    if mode =='d':
        return (Lo_D,Hi_D)
    elif mode =='r':
        return (Lo_R,Hi_R)
    elif mode == 'l':
        return (Lo_D,Lo_R)
    elif mode == 'h':
        return (Hi_D,Hi_R)
        

def wavOrth1d(sig,nz,wname='haar',wtype=1):
    N = np.size(sig)
    scale = nz
    if scale > np.ceil(np.log2(N))+1:
        print "Too many decomposition scales! The decomposition scale will be set to default value:4!"
        scale = 4
    if scale < 1:
        print "Decomposition scales should be greater than 1! The decomposition scale will be set to default value:1!"
        scale = 1        
     
    if scale == 0:
        wt = np.copy(sig)
        band = np.array([N-1])
    else:            
        (h0,g0) = wavFilters(wname,wtype,'d')
        lf = np.size(h0)   
        x = np.copy(sig)
        wt = np.array([])
        band = np.zeros(scale+1)
        band[-1] = N
        end = N
        start = 1
        for sc in np.arange(scale-1):
#             start = np.ceil(float(end)/2)
            lsig = np.size(x)
            end = lsig + lf - 1
            lenExt = lf - 1
            xExt = np.lib.pad(x, (lenExt,lenExt), 'symmetric')
            app = np.convolve(xExt,h0,'valid')
            x = np.copy(app[start:end:2])
            detail = np.convolve(xExt,g0,'valid')
            wt = np.hstack([detail[start:end:2],wt])     
            band[-2-sc] = len(detail[start:end:2])
        wt = np.hstack([x,wt]) 
        band[0] = len(x)  
    return (wt,band)
        
def iwavOrth1d(wt,band,wname='haar',wtype=1):
    if np.size(band) == 1:
        sig = np.copy(wt)
        return sig
    else:
        (h1,g1) = wavFilters(wname,wtype,'r')
        sig = np.copy(wt[:band[0]])
        start = band[0]
#         lf = np.size(h1)
        for sc in np.arange(np.size(band)-2):
            last = start+band[sc+1]
            detail = np.copy(wt[start:last])
            lsig = 2*np.size(sig)
#             s = lsig - lf + 2
            s = band[sc+2]
            appInt = np.zeros(lsig-1)
            appInt[::2] = np.copy(sig)
            appInt = np.convolve(appInt,h1,'full')
            first = np.floor(float(np.size(appInt) - s)/2.)
            last = np.size(appInt) - np.ceil(float(np.size(appInt) - s)/2.)
            appInt = appInt[first:last]            
            detailInt = np.zeros(lsig-1)
            detailInt[::2] = np.copy(detail)
            detailInt = np.convolve(detailInt,g1,'full')
            detailInt = detailInt[first:last]           
            sig = appInt + detailInt 
            start = last          
        return sig

#######################################################
############### Starlet 1d ############################
#######################################################
def test_ind(ind,N):
    res = ind
    if ind < 0 : 
        res = -ind
        if res >= N: 
            res = 2*N - 2 - ind
    if ind >= N : 
        res = 2*N - 2 - ind
        if res < 0:
            res = -ind
    return res
    

def b3splineTrans(sig_in,step):
    n = np.size(sig_in)
    sig_out = np.zeros(n)
    c1 = 1./16
    c2 = 1./4
    c3 = 3./8
    
    for i in np.arange(n):
        il = test_ind(i-step,n)
        ir = test_ind(i+step,n)
        il2 = test_ind(i-2*step,n)
        ir2 = test_ind(i+2*step,n)
        sig_out[i] = c3 * sig_in[i] + c2 * (sig_in[il] + sig_in[ir]) + c1 * (sig_in[il2] + sig_in[ir2])
    
    return sig_out

def b3spline_fast(step_hole):
    c1 = 1./16
    c2 = 1./4
    c3 = 3./8
    length = 4*step_hole+1
    kernel1d = np.zeros(length)
    kernel1d[0] = c1
    kernel1d[-1] = c1
    kernel1d[step_hole] = c2
    kernel1d[-1-step_hole] = c2
    kernel1d[2*step_hole] = c3
    return kernel1d

def star1d(sig,scale,fast = True,gen2=True,normalization=False):
    n = np.size(sig)
    ns = scale
    # Normalized transfromation
    head = 'star1d_gen2' if gen2 else 'star1d_gen1'
    if normalization and (pm.trHead != head):
        pm.trHead = head
        pm.trTab = nsNorm(n,ns,gen2)
    wt = np.zeros((ns,n))
    step_hole = 1
    sig_in = np.copy(sig)
    
    for i in np.arange(ns-1):
        if fast:
            kernel1d = b3spline_fast(step_hole)
            sig_pad = np.lib.pad(sig_in, (2*step_hole,2*step_hole), 'reflect')
            sig_out = psg.convolve(sig_pad, kernel1d, mode='valid')
        else:
            sig_out = b3splineTrans(sig_in,step_hole)
            
        if gen2:
            if fast:
                sig_pad = np.lib.pad(sig_out, (2*step_hole,2*step_hole), 'reflect')
                sig_aux = psg.convolve(sig_pad, kernel1d, mode='valid')
            else:
                sig_aux = b3splineTrans(sig_out,step_hole)
            wt[i] = sig_in - sig_aux
        else:        
            wt[i] = sig_in - sig_out
            
        if normalization:
            wt[i] /= pm.trTab[i]
        sig_in = np.copy(sig_out)
        step_hole *= 2
        
    wt[ns-1] = np.copy(sig_out)
    if normalization:
        wt[ns-1] /= pm.trTab[ns-1]
    
    return wt

   
def istar1d(wtOri,fast=True,gen2=True,normalization=False):
    (ns,n) = np.shape(wtOri)
    wt = np.copy(wtOri)
    # Unnormalization step
    head = 'star1d_gen2' if gen2 else 'star1d_gen1'   
    if normalization:
        if pm.trHead != head:
            pm.trHead = head
            pm.trTab = nsNorm(n,ns,gen2)
        for i in np.arange(ns):
            wt[i] *= pm.trTab[i]
    
    if gen2:
        '''
        h' = h, g' = Dirac
        '''
        step_hole = pow(2,ns-2)
        sigRec = np.copy(wt[ns-1])
        for k in np.arange(ns-2,-1,-1):            
            if fast:
                kernel1d = b3spline_fast(step_hole)
                sig_pad = np.lib.pad(sigRec, (2*step_hole,2*step_hole), 'reflect')
                sig_out = psg.convolve(sig_pad, kernel1d, mode='valid')
            else:
                sig_out = b3splineTrans(sigRec,step_hole)
            sigRec = sig_out + wt[k]
            step_hole /= 2            
    else:
        '''
        h' = Dirac, g' = Dirac
        '''
        sigRec = np.sum(wt,axis=0)
#         '''
#         h' = h, g' = Dirac + h
#         '''
#         sigRec = np.copy(wt[ns-1])
#         step_hole = pow(2,ns-2)
#         for k in np.arange(ns-2,-1,-1):
#             if fast:
#                 kernel1d = b3spline_fast(step_hole)
#                 sig_pad = np.lib.pad(sigRec, (2*step_hole,2*step_hole), 'reflect')
#                 sigRec = psg.convolve(sig_pad, kernel1d, mode='valid')
#                 wt_pad = np.lib.pad(wt[k], (2*step_hole,2*step_hole), 'reflect')
#                 sig_out = psg.convolve(wt_pad, kernel1d, mode='valid')
#             else:
#                 sigRec = b3splineTrans(sigRec,step_hole)
#                 sig_out = b3splineTrans(wt[k],step_hole)
#             sigRec += wt[k]+sig_out
#             step_hole /= 2
    return sigRec        

def adstar1d(wtOri,fast=True,gen2=True,normalization=False):
    (ns,n) = np.shape(wtOri)
    wt = np.copy(wtOri)
    # Unnormalization step
    # !Attention: wt is not the original wt after unnormalization
    head = 'star1d_gen2' if gen2 else 'star1d_gen1' 
    if normalization:
        if pm.trHead != head:
            pm.trHead = head
            pm.trTab = nsNorm(n,ns,gen2)
        for i in np.arange(ns):
            wt[i] *= pm.trTab[i]
     
    sigRec = np.copy(wt[ns-1])
    step_hole = pow(2,ns-2)
    for k in np.arange(ns-2,-1,-1):
        if fast:
            kernel1d = b3spline_fast(step_hole)
            sig_pad = np.lib.pad(sigRec, (2*step_hole,2*step_hole), 'reflect')
            sigRec = psg.convolve(sig_pad, kernel1d, mode='valid')
            wt_pad = np.lib.pad(wt[k], (2*step_hole,2*step_hole), 'reflect')
            sig_out = psg.convolve(wt_pad, kernel1d, mode='valid')
            if gen2:
                sig_pad = np.lib.pad(sig_out, (2*step_hole,2*step_hole), 'reflect')
                sig_out2 = psg.convolve(sig_pad, kernel1d, mode='valid')
                sigRec += wt[k] -sig_out2
            else: sigRec += wt[k] -sig_out
        else:
            sigRec = b3splineTrans(sigRec,step_hole)
            sig_out = b3splineTrans(wt[k],step_hole)
            if gen2:
                sig_out2 = b3splineTrans(sig_out,step_hole)
                sigRec += wt[k] -sig_out2
            else: sigRec += wt[k]-sig_out
        step_hole /= 2
    return sigRec

def nsNorm(nx,nz,gen2=True):
    sig = np.zeros(nx)
    sig[nx/2] = 1                      
    wt = star1d(sig,nz,fast=True,gen2=gen2,normalization=False)      
    tmp = wt**2
    tabNs = np.sqrt(np.sum(tmp,1)) 
    head = 'star1d_gen2' if gen2 else 'star1d_gen1' 
    if pm.trHead != head:
        pm.trHead = head
        pm.trTab = tabNs     
    return tabNs