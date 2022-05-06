# swarmDMD Analysis
# Emma Hansen
# July 2020

# Description
#	Analyzes the DMD and ground truth data for comparison, but does not output plots. Analyses does: average density matrix, polarisation, angular momentum, absolute angular momentum, average minimum distance.
#------------------------------#

import numpy as np
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing
from functools import partial
import gc
from swarmDMD_functions import *
from os import path
import time
import itertools as it

# Functions
def main_loop(N,rho,wide,focal_range,R,method,T_init,re_init,T_pred,waterfall,milling,data_input):
	print('entered main loop')
	r = data_input[0]
	datatype = data_input[1]
	T_beg = data_input[2][0]
	T_end = data_input[2][1]
	eta = data_input[3]	

	bin_width = r/4.
	bin_freq = r/4.

	num_bins = int(np.floor(2*r/bin_freq))
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

	print('GT data initialised')	
	F = np.size(focal_range)
	
	# read and write file names
	GT_filename = 'vicsekAnalysis'+millname+'_N'+str(N)+'_eta'+eta_name+'_W'+str(wide)+'_rho'+str(rho)+'_r'+r_name+'_Focal'+str(F)
	filename = 'swarmDMD_datatype'+str(datatype)+'_N'+str(N)+'_eta'+eta_name+'_W'+str(wide)+'_rho'+str(rho)+'_r'+r_name+'_R'+str(R)+'_Tend'+str(T_end)+'Tbeg'+str(T_beg)+'_method'+method+'_T_start'+str(T_init)+'_ReInit'+str(re_init)+'_WF'+str(waterfall)+millname
	filenameAnalysis = 'swarmDMDAnalysis_datatype'+str(datatype)+'_N'+str(N)+'_eta'+eta_name+'_W'+str(wide)+'_rho'+str(rho)+'_r'+r_name+'_R'+str(R)+'_Tend'+str(T_end)+'Tbeg'+str(T_beg)+'_Focal'+str(F)+'_method'+method+'_T_start'+str(T_init)+'_ReInit'+str(re_init)+'_WF'+str(waterfall)+millname

	try:
		save_filename = find_name(filename)
		save_filenameAnalysis = find_name(filenameAnalysis)
	except Exception as e: print(e)
	print('GT analysis file name: ' + str(save_filename))
	print('DMD analsysis file name: '+str(save_filenameAnalysis))


	# Ground Truth Analysis	
	bins = np.zeros((2*num_bins+1,2*num_bins+1))
	binsP = np.zeros((2*num_bins+1,2*num_bins+1))
	bins_percent = np.zeros((2*num_bins+1,2*num_bins+1))
	binsP_percent = np.zeros((2*num_bins+1,2*num_bins+1))
	D = np.zeros((TN,1))
	P = np.linalg.norm(np.sum(X_dot3d,axis=0),axis=1)/np.sum(np.linalg.norm(X_dot3d,axis=2),axis=0)
	Mang = np.absolute(np.sum(np.cross(X3d[:,:-1,:],X_dot3d[:,:,:],axis=2),axis=0)/(np.sum(np.linalg.norm(X_dot3d,axis=2)*np.linalg.norm(X3d[:,:-1,:],axis=2),axis=0)))
	Mabs = np.absolute(np.sum(np.absolute(np.cross(X3d[:,:-1,:],X_dot3d[:,:,:],axis=2)),axis=0)/(np.sum(np.linalg.norm(X_dot3d,axis=2)*np.linalg.norm(X3d[:,:-1,:],axis=2),axis=0)))

	try:
		for t in range(TN):
			D[t] = np.sum(np.amin(get_interagent(X,L,t)+np.diag(np.diag(100*np.ones((N,N)))),axis=0))/N
	except Exception as e: print(e)
	print('GT analysis complete')
	
	# DMD Analysis	
	if re_init == 0:
		DMD_range = TN-T_init
	else:
		super_steps = (TN-T_init)//re_init
		if not waterfall:
			DMD_range = re_init*super_steps

	# Load saved swarmDMD propagation
	swarmDMD_data = np.load('Data/'+method+'/'+save_filename+'.npz')
	X_DMD = swarmDMD_data['X_DMD']
	X_DMD3d = np.stack((X_DMD[0:N,:],X_DMD[N:2*N,:]),axis=2)
	print('DMD data loaded')

	if not waterfall:
		X_DMD_dot = position_difference(L,X_DMD[:,:-1],X_DMD[:,1:])/dt
		X_DMD_dot3d = np.stack((X_DMD_dot[:N,:],X_DMD_dot[N:,:]),axis=2)
		
		print('X_DMD_dot initialised')

		bins_DMD = np.zeros((2*num_bins+1,2*num_bins+1))
		bins_DMDP = np.zeros((2*num_bins+1,2*num_bins+1))
		bins_DMD_percent = np.zeros((2*num_bins+1,2*num_bins+1))
		bins_DMDP_percent = np.zeros((2*num_bins+1,2*num_bins+1))
		D_DMD = np.zeros((TN,1))
		P_DMD = np.linalg.norm(np.sum(X_DMD_dot3d,axis=0),axis=1)/np.sum(np.linalg.norm(X_DMD_dot3d,axis=2),axis=0)
		Mang_DMD = np.absolute(np.sum(np.cross(X_DMD3d[:,:-1,:],X_DMD_dot3d[:,:,:],axis=2),axis=0)/(np.sum(np.linalg.norm(X_DMD_dot3d,axis=2)*np.linalg.norm(X_DMD3d[:,:-1,:],axis=2),axis=0)))
		Mabs_DMD = np.absolute(np.sum(np.absolute(np.cross(X_DMD3d[:,:-1,:],X_DMD_dot3d[:,:,:],axis=2)),axis=0)/(np.sum(np.linalg.norm(X_DMD_dot3d,axis=2)*np.linalg.norm(X_DMD3d[:,:-1,:],axis=2),axis=0)))
		print('P, Mang, Mabs computed')

		# DMD Analysis
		try:
			for t in range(DMD_range): # compute for entire time period
				D[t] = np.sum(np.amin(get_interagent(X,L,t+T_init)+np.diag(np.diag(100*np.ones((N,N)))),axis=0))/N
				D_DMD[t] = np.sum(np.amin(get_interagent(X_DMD,L,t)+np.diag(np.diag(100*np.ones((N,N)))),axis=0))/N
		except Exception as e:print(e)
		print('D computed for DMD')		

		# compute average agent density during training period
		T_start_adjusted = np.maximum(T_beg,T_init)
		j = 0
		k = 0
		try:
			for t in range(T_start_adjusted,T_end):
				for f in range(F):
					local_bin, agent_count = get_LocalBin(dt,L,t,focal_range[f],X3d,num_bins,bin_freq,bin_width)
					local_bin = np.reshape(local_bin,(2*num_bins+1,2*num_bins+1))
					local_bin_DMD, agent_count_DMD = get_LocalBin(dt,L,t-T_start_adjusted,focal_range[f],X_DMD3d,num_bins,bin_freq,bin_width)
					local_bin_DMD = np.reshape(local_bin_DMD,(2*num_bins+1,2*num_bins+1))
					bins = bins + local_bin
					bins_DMD = bins_DMD + local_bin_DMD

					if agent_count == 0:
						bins_percent = bins_percent + local_bin
						j=j+1
					else:
						bins_percent = bins_percent + np.nan_to_num(100*local_bin/agent_count)
					if agent_count_DMD == 0:
						bins_DMD_percent = bins_DMD_percent + local_bin_DMD
						k = k+1
					else:
						bins_DMD_percent = bins_DMD_percent + np.nan_to_num(100*local_bin_DMD/agent_count_DMD)
			mean_bin = bins # just return the count
			mean_bin_DMD = bins_DMD

		except Exception as e:print(e)

		print('average agent density during training period computed')

		# compute average agent density post-training over T_pred number of time steps
		T_end_adjusted = np.maximum(T_end,T_init)
		m = 0
		n = 0
		for t in range(T_end_adjusted,T_end_adjusted+T_pred):
			for f in range(F):
				local_bin, agent_count = get_LocalBin(dt,L,t,focal_range[f],X3d,num_bins,bin_freq,bin_width)
				local_bin = np.reshape(local_bin,(2*num_bins+1,2*num_bins+1))
				local_bin_DMD, agent_count_DMD = get_LocalBin(dt,L,t-T_init,focal_range[f],X_DMD3d,num_bins,bin_freq,bin_width)
				local_bin_DMD = np.reshape(local_bin_DMD,(2*num_bins+1,2*num_bins+1))
				binsP = binsP + local_bin
				bins_DMDP = bins_DMDP + local_bin_DMD
				if agent_count == 0:
					binsP_percent = binsP_percent + local_bin
					m = m+1
				else:
					binsP_percent = binsP_percent + np.nan_to_num(100*local_bin/agent_count)
				if agent_count_DMD == 0:
					bins_DMDP_percent = bins_DMDP_percent + local_bin_DMD
					n = n+1
				else:
					try:
						bins_DMDP_percent = bins_DMDP_percent + np.nan_to_num(100*local_bin_DMD/agent_count_DMD)
					except Exception as e:
						print(agent_count_DMD)
						print(e)
		mean_bin_P = binsP # just the count
		mean_bin_DMD_P = bins_DMDP
		
		print('average agent density post-training over T_pred number of time steps computed')


		try:
			np.savez('Data/SwarmModel/'+GT_filename,D=D,P=P,Mang=Mang,Mabs=Mabs,mean_bin=mean_bin,mean_bin_P=mean_bin_P,bin_width=bin_width,bin_freq=bin_freq,num_bins=num_bins,bins_percent=bins_percent,binsP_percent=binsP_percent,training_zero_count=j,prediction_zero_count=m)
			np.savez('Data/'+method+'/'+save_filenameAnalysis,D_DMD=D_DMD,P_DMD=P_DMD,Mang_DMD=Mang_DMD,Mabs_DMD=Mabs_DMD,mean_bin_DMD=mean_bin_DMD,mean_bin_DMD_P=mean_bin_DMD_P,num_bins=num_bins,bin_freq=bin_freq,bin_width=bin_width,bins_DMD_percent=bins_DMD_percent,bins_DMDP_percent=bins_DMDP_percent,training_zero_count=k,prediction_zero_count=n)
		except Exception as e:print(e)
		print('analyses saved')
	else:
		print('entering waterfall analysis')
		D_DMD = np.zeros((super_steps,waterfall))
		P_DMD = np.zeros((super_steps,waterfall-1))
		Mang_DMD = np.zeros((super_steps,waterfall-1))
		Mabs_DMD = np.zeros((super_steps,waterfall-1))
		print('waterfall init complete')
		print('num supersteps: ',str(super_steps))
		for k in range(super_steps):
			X_DMD_dot = position_difference(L,X_DMD[:,:-1,k],X_DMD[:,1:,k])/dt
			X_DMD_dot3d = np.stack((X_DMD_dot[:N,:],X_DMD_dot[N:,:]),axis=2)
			X_DMD3d = np.stack((X_DMD[:N,:,k],X_DMD[N:,:,k]),axis=2)
			print('derivative calculated')
			try:
				print(k)
				print(np.shape(X_DMD3d[:,:-1,:]))
				print(np.shape(X_DMD_dot3d[:,:,:]))
				P_DMD[k,:] = np.linalg.norm(np.sum(X_DMD_dot3d,axis=0),axis=1)/np.sum(np.linalg.norm(X_DMD_dot3d,axis=2),axis=0)
				Mang_DMD[k,:] = np.absolute(np.sum(np.cross(X_DMD3d[:,:-1,:],X_DMD_dot3d[:,:,:],axis=2),axis=0)/(np.sum(np.linalg.norm(X_DMD_dot3d,axis=2)*np.linalg.norm(X_DMD3d[:,:-1,:],axis=2),axis=0)))
				Mabs_DMD[k,:] = np.absolute(np.sum(np.absolute(np.cross(X_DMD3d[:,:-1,:],X_DMD_dot3d[:,:,:],axis=2)),axis=0)/(np.sum(np.linalg.norm(X_DMD_dot3d,axis=2)*np.linalg.norm(X_DMD3d[:,:-1,:],axis=2),axis=0)))

				for t in range(waterfall):
					D_DMD[k,t] = np.sum(np.amin(get_interagent(X_DMD[:,:,k],L,t)+np.diag(np.diag(100*np.ones((N,N)))),axis=0))/N
			except Exception as e:
				print('analysis')
				print(e)
			# print('D computed for DMD')

		np.savez('Data/'+method+'/'+save_filenameAnalysis,D_DMD=D_DMD,P_DMD=P_DMD,Mang_DMD=Mang_DMD,Mabs_DMD=Mabs_DMD)
		print('waterfall analysis saved')

# Main Code
if __name__ == '__main__':
	milling = 1 # options: 0 (off), 1 (on)
	gif = 0 # options: 0 (off), 1 (on)
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

	rs = [0.2*r_av,r_av,2*r_av] #20%, 100%, and 200% of r_av
	# rs = [r_av]
	rs = [1.] # USE ONLY FOR MILLING

	Datatypes = [13] # The possible datatypes are 1-21, but 13,20, and 21 are the ones focussed on for simple, cartesian, and polar resp. 
	T_pairs = [[0,50]]	# Choose the interval of time to learn with
	T_init = 0
	T_pred = 50 # number of time steps to use to analyse prediction capabilities

	np.random.seed(2)
	focal_range = np.random.randint(0,high=N,size=N) # agents to include in density distribution analysis, max value: N

	big_list = [rs,Datatypes,T_pairs,eta]
	parameters = list(it.product(*big_list))

	print('variables initialised')

	try:
		for params in parameters:
			print('running in series')
			func = partial(main_loop,N,rho,wide,focal_range,R,method,T_init,re_init,T_pred,waterfall,milling)
			func(params)
	except Exception as e: print(e)	

	
