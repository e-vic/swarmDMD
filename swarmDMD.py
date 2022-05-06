# swarmDMD
# Emma Hansen  
# March 2022

import numpy as np
from functools import partial
import gc
from swarmDMD_functions import *
from os import path
import time
import itertools as it

def swarmDMD_propagate(N,rho,wide,gif,R,method,T_init,re_init,waterfall,milling,data_input):
    print('entered main loop')
    r = data_input[0]
    datatype = data_input[1]
    T_beg = data_input[2][0]
    T_end = data_input[2][1]
    eta = data_input[3]	

    r_name = str(r).replace('.','')
    eta_name = str(eta).replace('.','')

    print('eta is: '+eta_name)
    print('r is: '+r_name)

    if milling == 1:
        millname = '_milling'
    else:
        millname = ''

    # Import Data
    data = np.load('Data/SwarmModel/vicsek'+millname+'_N'+str(N)+'_eta'+eta_name+'_W'+str(wide)+'_rho'+str(rho)+'_r'+r_name+'.npz') # milling load
    print('GT data imported')
    TN = np.shape(data['X'])[1]
    print('num time steps: '+str(TN))
    L = data['L']
    dt = data['dt']
    print('time step: '+str(dt))

    X3d = data['X']
    X = np.concatenate((data['X'][:,:,0],data['X'][:,:,1]),axis=0)
    X_dot = position_difference(L,X[:,:-1],X[:,1:])/dt
    X_dot3d = np.stack((X_dot[:N,:],X_dot[N:,:]),axis=2)

    print('data initialised')

    # read and write file names
    filename = 'swarmDMD_datatype'+str(datatype)+'_N'+str(N)+'_eta'+eta_name+'_W'+str(wide)+'_rho'+str(rho)+'_r'+r_name+'_R'+str(R)+'_Tend'+str(T_end)+'Tbeg'+str(T_beg)+'_method'+method+'_T_start'+str(T_init)+'_ReInit'+str(re_init)+'_WF'+str(waterfall)+millname

    try:
        save_filename = find_name(filename)
    except Exception as e: print(e)
    print('file name will be: '+str(save_filename))

    # DMD Analysis	

    X1_bar,X2_bar,X_dot0 = setup(X,R,T_beg,T_end,dt,datatype,L,method)
    print('DMD setup complete')

    # DMD
    try:
        K = DMD(X1_bar,X2_bar,R)
        print('K computed')
    except Exception as e: print(e)
    _, K_svals, _ = np.linalg.svd(K)
        
    
    # Simulate with swarmDMD
    simdt = dt
    x0 = np.copy(X[:,T_init]) # use for same initial positions as training data
    v0 = np.copy(X_dot0[:,T_init])

    try:
        if re_init == 0:
            X_DMD3d, X_DMD, _ = get_xdmdc(x0, v0, K, simdt,N,TN-T_init,datatype,L,method)
            DMD_range = TN-T_init
        else:
            super_steps = (TN-T_init)//re_init
            if not waterfall:
                DMD_range = re_init*super_steps
                X_DMD = np.zeros((2*N,DMD_range))
                for k in range(super_steps):
                    _, X_DMD_temp, _ = get_xdmdc(X[:,T_init+k*re_init], X_dot0[:,T_init+k*re_init], K, simdt,N,re_init,datatype,L,method)
                    X_DMD[:,(T_init+k*re_init):(T_init+k*re_init+re_init)] = X_DMD_temp
                    print('column indices: '+str(T_init+k*re_init)+', '+str(T_init+k*re_init+re_init))
                X_DMD3d = np.stack((X_DMD[0:N,:],X_DMD[N:2*N,:]),axis=2)
            else:
                X_DMD = np.zeros((2*N,waterfall,super_steps)) 
                for k in range(super_steps):
                    _, X_DMD_temp, _ = get_xdmdc(X[:,T_init+k*re_init], X_dot0[:,T_init+k*re_init], K, simdt,N,waterfall,datatype,L,method)
                    X_DMD[:,:,k] = X_DMD_temp
        
        np.savez_compressed('Data/'+method+'/'+save_filename,X_DMD=X_DMD,K_svals=K_svals)
    except Exception as e: print(e)
    print('DMD result saved')

    del K

    if gif and not re_init:
        num_frames = 500 # max is TN
        frames_start = 500 
        get_gif(save_filename,N,L,X_DMD3d,TN,num_frames,frames_start,dt)



if __name__ == '__main__':	
    # Simulation settings
    milling = 1 # options: 0 (off), 1 (on)
    gif = 1 # options: 0 (off), 1 (on)
    method = 'simple' # available methods: simple, FO_cartesian, FO_polar
    wide = 0 # to make domain wider than initial condition
    width = 1.1 #  multiplier for domain width
    re_init = 0 # OPTIONS  0: regular, no re-initialisation; int: re-initialise every int time steps, minimum is 2: (one time step for the initialisation, one time step for propagation)
    waterfall = 0 # 0: waterfall reinitialisation OFF, int: waterfall reinitialisation ON for length of int, reinitialised every re_init time steps
    R = 8 # number of modes to use, integer greater than 0

    # Define data/methods to use
    N = 100  # number of agents
    rho = 2.5 # density
    
    if wide:
        L0 = np.sqrt(N/rho) # length of domain of agent initial positions
        L = width*L0 # length of domain square side 
    else:
        L0 = np.sqrt(N/rho)
        L = L0

    A = L0**2
    r_av = (1/(N/A))**(1/2)

    eta = [0.08726646259971647] # choose the noise in the GT model, currently accepted are 0, pi/12, 0.08726646259971647 (only for milling simulation)

    # rs = [0.2*r_av,r_av,2*r_av] #20%, 100%, and 200% of r_av
    # rs = [r_av]
    rs = [1.] # USE ONLY FOR MILLING

    Datatypes = [13] # The possible datatypes are 1-21, but 13,20, and 21 are the ones focussed on for simple, cartesian, and polar resp. 
    T_pairs = [[0,50]]	# Choose the interval of time to learn with
    T_init = 0
    

    big_list = [rs,Datatypes,T_pairs,eta]
    parameters = list(it.product(*big_list))

    print('variables initialised')

    for params in parameters:
        print('running in series')
        func = partial(swarmDMD_propagate,N,rho,wide,gif,R,method,T_init,re_init,waterfall,milling)
        func(params)