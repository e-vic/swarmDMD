# mDMDc Functions
# Emma Hansen
# July 2020
#--------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.gridspec as gridspec
import sklearn.cluster as skcl
from pygifsicle import optimize
import csv

# Funtions
def find_name(name):
	"""locates filesave name in csv list. using this because filenames were getting too long to upload to github or be used in jupyter"""
	with open('filename_list.csv','r') as csv_file: # check to see if an entry already exists
		csv_reader = csv.reader(csv_file, delimiter=',')
		filename = 'FILENOTFOUND'
		for row in csv_reader:
			if row[0] == name:
				filename = row[1]
				break
			last_filename = row[1]
		csv_file.close()
		
	if filename == 'FILENOTFOUND': # create an entry if one doesn't
		with open('filename_list.csv','a',newline='') as csv_file:
			# writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer = csv.writer(csv_file, delimiter=',')
			writer.writerow([name,str(int(last_filename)+1)])
			filename = str(int(last_filename)+1)
			csv_file.close()
			
	return filename 

def get_svd(A):
	"""Calculates SVD of matri A and returns U,S (vector of singular values), and V """
	U,S,V = np.linalg.svd(A,full_matrices=False)
	V = V.conj().T
	return U,S,V

def setup(X,R,T_beg,T_end,dt,datatype,L,method):
	"""Sets up data matrices for mDMDc based on the selected datatype and training time """
	N = int(np.shape(X)[0]/2)
	TN = int(np.shape(X)[1])
	X1 = X[:,:-1]
	X2 = X[:,1:]
	X_dot = position_difference(L,X1,X2)/dt
	S = np.linalg.norm(np.stack((X_dot[:N,:],X_dot[N:,:]),axis = 2),axis = 2)
	theta = np.arctan2(X_dot[N:,:],X_dot[:N,:])

	if datatype == 11:
		Y = X[:,0:TN-1]
	elif datatype == 12:
		Y = X_dot
	elif datatype == 13:
		interagent = get_interagent(X,L,range(TN))
		Y = interagent[:,:-1]
	elif datatype == 15:
		Y = theta
	elif datatype == 17:
		try:
			Y = get_rel_speed(X_dot,range(TN-1))
		except Exception as e: print(e)
	elif datatype == 18:
		try:
			Y = get_rel_theta(theta,range(TN-1))
		except Exception as e: print(e)
	elif datatype == 19:
		try:
			Y = get_rel_vel(X_dot,range(TN-1))	
		except Exception as e: print(e)	
	elif datatype == 20:
		# relative position and relative velocity - use for cartesian simulations
		temp = get_interagent(X,L,range(TN))
		try:
			Y = np.concatenate((temp[:,:-1],get_rel_vel(X_dot,range(TN-1))),axis=0) #relative position and velocity
		except Exception as e: print(e)
	elif datatype == 21:
		# relative position, relative speed, relative heading - use for polar simulations
		Y = np.concatenate((get_interagent(X[:,:TN-1],L,range(TN-1)),get_rel_speed(S,range(TN-1)),get_rel_theta(theta,range(TN-1))),axis=0) # relative position, speed, and angle
	print('Y set')

	if method == 'simple':
		Y = Y[:,T_beg:T_end]
		Y_prime = position_difference(L,X1,X2)/dt
		Y_prime = Y_prime[:,T_beg:T_end]
	elif method == 'FO_cartesian':
		Y = Y[:,T_beg:T_end]
		Y_prime = position_difference(L,X[:,1:-1],X[:,2:]) - dt * X_dot[:,:-1]
		Y_prime = Y_prime[:,T_beg:T_end]
	elif method == 'FO_polar':
		Y  = Y[:,T_beg:T_end]
		Y_prime = position_difference(L,X[:,1:-1],X[:,2:]) - dt * np.concatenate((S[:,:-1],S[:,:-1]),axis = 0) * np.concatenate((np.cos(theta[:,:-1]),np.sin(theta[:,:-1])))
		Y_prime = Y_prime[:,T_beg:T_end] 
	elif method == 'SO_cartesian':
		Y = Y[:,T_beg:T_end]
		Y_prime = X_dot[:,1:] - X_dot[:,:-1]
		Y_prime = Y_prime[:,T_beg:T_end]
	elif method == 'SO_polar':
		Y = Y[:,T_beg:T_end]
		Y_prime = np.concatenate((S[:,1:],theta[:,1:]),axis = 0) - np.concatenate((S[:,:-1],theta[:,:-1]),axis = 0)
		Y_prime = Y_prime[:,T_beg:T_end]
	
	return Y, Y_prime, X_dot

def DMD(Y,Y_prime,R):
	"""Calculates the 'feedback' matrix K for the modified DMDc """
	U1,S1,V1 = get_svd(Y)
	U1 = U1[:,0:R]
	S1 = S1[0:R]
	V1 = V1[:,0:R]
	K = Y_prime @ V1 @ np.diag(np.reciprocal(S1)) @ U1.conj().T
	
	return K

def split(array, nrows, ncols):
	"""Split a matrix into sub-matrices."""

	r, h = array.shape
	return (array.reshape(h//nrows, nrows, -1, ncols)
				 .swapaxes(1, 2)
				 .reshape(-1, nrows, ncols))

def get_interactionM(K,x_bar,N):
	interactionM = np.multiply(K,x_bar.T)
	interactionM = np.absolute(interactionM[:N,:]) + np.absolute(interactionM[N:,:])
	p = int(np.shape(interactionM)[1]/N)
	
	interactionM = np.absolute(split(interactionM,N,N))
	interactionM = interactionM.sum(axis=0)
	return interactionM

def get_xdmdc(x0, v0, K,dt,N,TN,datatype,L,method):
	"""Propagates the agent positions in time using K from mDMDc. Updated 09/20 to include periodic boundary condition """
	print('entering get_xdmdc')
	X_DMD = np.empty((2*N,TN))
	X_DMD[:,0] = np.copy(x0)
	x_dmd1 = np.copy(x0)
	interaction = np.empty((N,N,TN))
	v1 = np.copy(v0) # new with addition of "method"
	s1 = np.linalg.norm(np.stack((v1[:N],v1[N:]),axis = 1),axis = 1) # speed
	
	print('shape of x0 is '+str(np.shape(x0)))
	
	theta0 = np.arctan2(v0[N:2*N],v0[0:N])
	theta1 = np.copy(theta0)
	atansincos0 = np.arctan(np.sin(theta0)/np.cos(theta0))

	print('shape of theta0 is: '+str(np.shape(theta0)))
	print('shape of s1 is: '+str(np.shape(s1)))
	
	if datatype == 11:
		x1_bar = x0
	elif datatype == 12:
		x1_bar = v0
	elif datatype == 13:
		interagent = get_interagent(x0,L)
		x1_bar = interagent
	elif datatype == 15:
		x1_bar = theta0
	elif datatype == 17:
		try:
			x1_bar = get_rel_speed(v0)
		except Exception as e: print(e)
	elif datatype == 18:
		try:
			x1_bar = get_rel_theta(theta0)
		except Exception as e: print(e)
	elif datatype == 19:
		try:
			x1_bar = get_rel_vel(v0)	
		except Exception as e: print(e)
	elif datatype == 20:
		try:
			x1_bar = np.concatenate((get_interagent(x0,L),get_rel_vel(v0)),axis=0) #relative position and velocity
		except Exception as e: print(e)
	elif datatype == 21:
		try:
			x1_bar = np.concatenate((get_interagent(x0,L),get_rel_speed(s1),get_rel_theta(theta0)),axis=0) # relative position, speed, and angle
		except Exception as e: print(e)

	interaction[:,:,0] = get_interactionM(K,x1_bar,N)
	
	print('xdmdc initialized, entering time loop')
	for k in range(1,TN):

		if method == 'simple':
			x_dmd2 = x_dmd1 + dt * K @ x1_bar

		elif method == 'FO_cartesian':
			x_dmd2 = x_dmd1 + (dt * v1) +  (K @ x1_bar)

		elif method == 'FO_polar':
			x_dmd2 = x_dmd1 + dt * np.concatenate((s1,s1),axis = 0) * np.concatenate((np.cos(theta1),np.sin(theta1))) + (K @ x1_bar)
		
		# Periodic Boundary Condition
		boundaryT_index = np.where(x_dmd2>L)
		boundaryB_index = np.where(x_dmd2<0)
		
		x_dmd2[boundaryT_index] = x_dmd2[boundaryT_index] - L*(x_dmd2[boundaryT_index]//L)
		x_dmd2[boundaryB_index] = x_dmd2[boundaryB_index] - L*(x_dmd2[boundaryB_index]//L)
		
		# Check bounds obeyed
		boundaryT_index = np.where(x_dmd2>L)
		boundaryB_index = np.where(x_dmd2<0)
		
		v1 = position_difference(L,x_dmd1,x_dmd2)/dt


		s1 = np.linalg.norm(np.stack((v1[:N],v1[N:]),axis = 1),axis = 1) # speed
		try:
			theta1 = np.arctan2(v1[N:2*N],v1[0:N])
		except Exception as e: print(e)
		
		atansincos = np.arctan(np.sin(theta1)/np.cos(theta1))	

		if datatype == 11:
			x1_bar = x_dmd2
		elif datatype == 12:
			x1_bar = v1
		elif datatype == 13:
			interagent = get_interagent(x_dmd2,L)
			x1_bar = interagent
		elif datatype == 15:
			x1_bar = theta1
		elif datatype == 17:
			x1_bar = get_rel_speed(v1)
		elif datatype == 18:
			x1_bar = get_rel_theta(theta1)
		elif datatype == 19:
			x1_bar = get_rel_vel(v1)
		elif datatype == 20:
			x1_bar = np.concatenate((get_interagent(x_dmd2,L),get_rel_vel(v1)),axis=0) #relative position and velocity
		elif datatype == 21:
			x1_bar = np.concatenate((get_interagent(x_dmd2,L),get_rel_speed(s1),get_rel_theta(theta1)),axis=0) # relative position, speed, and angle	

		x_dmd1 = np.copy(x_dmd2)
		interaction[:,:,k] = get_interactionM(K,x1_bar,N)
		
		X_DMD[:,k] = np.real(x_dmd2[0:2*N])
	
	X_DMD3d = np.stack((X_DMD[0:N,:],X_DMD[N:2*N,:]),axis=2)
	print('xdmdc complete')
	return X_DMD3d, X_DMD, interaction
	
def get_interagent(X,L,T=-1):
	"""Calculates the interagent distances for all agents at time t, if t is specified. """
	N = int(np.shape(X)[0]/2)
	if isinstance(T,int):
		if T == -1:
			interagentx = np.empty((N**2))
			interagenty = np.empty((N**2))
			for i in range(N):
				interagentx[i*N:(i+1)*N] = np.minimum(np.absolute((X[i]+L) - X[:N]),np.minimum(np.absolute(X[i] - X[:N]),np.absolute((X[i]-L) - X[:N])))
				interagenty[i*N:(i+1)*N] = np.minimum(np.absolute((X[i+N]+L) - X[N:2*N]),np.minimum(np.absolute(X[i+N] - X[N:2*N]), np.absolute((X[i+N]-L) - X[N:2*N])))
			interagent = np.linalg.norm(np.stack((interagentx,interagenty),axis=1),axis=1)
		else:
			interagentx = np.empty((N**2))
			interagenty = np.empty((N**2))
			for i in range(N):
				interagentx[i*N:(i+1)*N] = np.minimum(np.absolute((X[i,T]+L) - X[:N,T]),np.minimum(np.absolute(X[i,T] - X[:N,T]),np.absolute((X[i,T]-L) - X[:N,T])))
				interagenty[i*N:(i+1)*N] = np.minimum(np.absolute((X[i+N,T]+L) - X[N:2*N,T]),np.minimum(np.absolute(X[i+N,T] - X[N:2*N,T]), np.absolute((X[i+N,T]-L) - X[N:2*N,T])))
			interagent = np.reshape(np.linalg.norm(np.stack((interagentx,interagenty),axis=1),axis=1),(N,N))
	else:
		X3d = np.stack((X[:N,:],X[N:2*N,:]),axis=2)
		TN = int(np.shape(X)[1])
		interagentx = np.empty((N**2,TN))
		interagenty = np.empty((N**2,TN))
		for i in range(N):
			interagentx[i*N:(i+1)*N,:] = np.minimum(np.absolute((X3d[i,:,0]+L) - X3d[:,:,0]),np.minimum(np.absolute(X3d[i,:,0] - X3d[:,:,0]),np.absolute((X3d[i,:,0]-L) - X3d[:,:,0])))
			interagenty[i*N:(i+1)*N,:] = np.minimum(np.absolute((X3d[i,:,1]+L) - X3d[:,:,1]),np.minimum(np.absolute(X3d[i,:,1] - X3d[:,:,1]),np.absolute((X3d[i,:,1]-L) - X3d[:,:,1])))
		interagent = np.linalg.norm(np.stack((interagentx,interagenty),axis=2),axis=2)
	return interagent

def get_rel_vel(X_dot,T=-1):
	"""Calculates the interagent distances for all agents at time t, if t is specified. """
	N = int(np.shape(X_dot)[0]/2)
	if isinstance(T,int):
		if T == -1:
			velx = np.empty((N**2))
			vely = np.empty((N**2))
			for i in range(N):
				velx[i*N:(i+1)*N] = np.absolute((X_dot[i]) - X_dot[:N])
				vely[i*N:(i+1)*N] = np.absolute((X_dot[i+N]) - X_dot[N:2*N])
			vel = np.concatenate((velx,vely),axis=0)
		else:
			velx = np.empty((N**2))
			vely = np.empty((N**2))
			for i in range(N):
				velx[i*N:(i+1)*N] = np.absolute((X_dot[i,T]) - X_dot[:N,T])
				vely[i*N:(i+1)*N] = np.absolute((X_dot[i+N,T]) - X_dot[N:2*N,T])
			vel = np.stack((np.reshape(velx,(N,N)),np.reshape(velx,(N,N))),axis=2) # returns two matrices stacked
	else:
		X_dot3d = np.stack((X_dot[:N,:],X_dot[N:2*N,:]),axis=2)
		TN = int(np.shape(X_dot)[1])
		velx = np.empty((N**2,TN))
		vely = np.empty((N**2,TN))
		for i in range(N):
			velx[i*N:(i+1)*N,:] = np.absolute((X_dot3d[i,:,0]) - X_dot3d[:,:,0])
			vely[i*N:(i+1)*N,:] = np.absolute((X_dot3d[i,:,1]) - X_dot3d[:,:,1])
		vel = np.concatenate((velx,vely),axis=0)
	return vel

def get_rel_theta(theta,T=-1):
	"""Calculates the interagent distances for all agents at time t, if t is specified. """
	N = int(np.shape(theta)[0])
	if isinstance(T,int):
		if T == -1:
			thetarel = np.empty((N**2))
			for i in range(N):
				thetarel[i*N:(i+1)*N] = np.absolute((theta[i]) - theta)

		else:
			thetarel = np.empty((N**2))
			for i in range(N):
				thetarel[i*N:(i+1)*N] = np.absolute((theta[i,T]) - theta[T])
	else:
		TN = int(np.shape(theta)[1])
		thetarel = np.empty((N**2,TN))
		for i in range(N):
			thetarel[i*N:(i+1)*N,:] = np.absolute((theta[i,:]) - theta)
	return thetarel

def get_rel_speed(S,T=-1):
	"""Calculates the relative speeds for all agents at time T, if T is specified. """
	N = int(np.shape(S)[0])
	if isinstance(T,int):
		if T == -1:
			Srel = np.empty((N**2))
			for i in range(N):
				Srel[i*N:(i+1)*N] = np.absolute((S[i]) - S)

		else:
			Srel = np.empty((N**2))
			for i in range(N):
				Srel[i*N:(i+1)*N] = np.absolute((S[i,T]) - S[T])
	else:
		TN = int(np.shape(S)[1])
		Srel = np.empty((N**2,TN))
		for i in range(N):
			Srel[i*N:(i+1)*N,:] = np.absolute((S[i,:]) - S)
	return Srel

def position_difference(L,X1,X2):
	"""Calculates the distance between points in a periodic boundary environment. 
	Thanks wikipedia https://en.wikipedia.org/wiki/Periodic_boundary_conditions"""
	num_dim = X1.ndim # vector or matix?
	diff = X2-X1

	if num_dim == 1:
		jumps_forward = np.where(diff > L*0.5)
		jumps_backward = np.where(diff <= -L*0.5)

		diff[jumps_forward] = diff[jumps_forward] - L
		diff[jumps_backward] = diff[jumps_backward] + L
	else:
		for i in range(np.shape(diff)[1]):
			jumps_forward = np.where(diff[:,i] > L*0.5)
			jumps_backward = np.where(diff[:,i] <= -L*0.5)

			diff[jumps_forward,i] = diff[jumps_forward,i] - L
			diff[jumps_backward,i] = diff[jumps_backward,i] + L
	return diff

def recentre(X,L,x_offset,y_offset):
	TN = int(np.shape(X)[1])
	N = int(np.shape(X)[0]/2)
	X_new = np.zeros(np.shape(X))
	for t in range(TN):
		x = np.concatenate([X[:N,t]+x_offset,X[N:,t]+y_offset],axis=0)

		boundaryT_index = np.where(x>L)
		boundaryB_index = np.where(x<0)

		x[boundaryT_index] = x[boundaryT_index] - L*(x[boundaryT_index]//L)
		x[boundaryB_index] = x[boundaryB_index] - L*(x[boundaryB_index]//L)

		X_new[:,t] = x
	return X_new
	
def focal_rotate(t,focal,L,dt,X):
	N = int(np.shape(X)[0]/2)
	X_dot = position_difference(L,X[:,:-1],X[:,1:])/dt
	X_heading = np.arctan2(X_dot[N:,:],X_dot[:N,:])

	X = recentre(X,L,(L/2)-X[focal,t],(L/2)-X[N+focal,t])
	distance = np.linalg.norm(np.stack([X[focal,t]-X[:N,t],X[N+focal,t]-X[N:,t]],axis=1),axis=1)

	focal_heading = (np.pi/2) - X_heading[focal,t]
	theta = np.arctan2(X[N:,t] - X[N+focal,t],X[:N,t] - X[focal,t])

	X_rot = np.concatenate([X[focal,t]+distance*np.cos(focal_heading+theta),X[N+focal,t]+distance*np.sin(focal_heading+theta)])

	boundaryT_index = np.where(X_rot>L)
	boundaryB_index = np.where(X_rot<0)

	X_rot[boundaryT_index] = X_rot[boundaryT_index] - L*(X_rot[boundaryT_index]//L)
	X_rot[boundaryB_index] = X_rot[boundaryB_index] - L*(X_rot[boundaryB_index]//L)

	return X_rot

def get_LocalBin(dt,L,t,focal,X,num_bins,bin_freq,bin_width):
	"""Determine number of agents in bins around the focal agent f at time step t.
	   t - current time step
	   f - index of focal agent in focal_range
	   focal - corresponding index of focal agent in swarm
	   X - positions of all agents in '3D' form (ie. X_3D from main code) """
	
	N = np.shape(X)[0]

	local_bin = np.zeros((2*num_bins+1,2*num_bins+1)) # intialize bins
	local_bin = local_bin.flatten() # map array of bins to a vector
	focal_loc = X[focal,t,:] # location of focal agent
	x_upper = focal_loc[0] + (num_bins*bin_freq + bin_width) 
	x_lower = focal_loc[0] - (num_bins*bin_freq + bin_width)
	y_upper = focal_loc[1] + (num_bins*bin_freq + bin_width)
	y_lower = focal_loc[1] - (num_bins*bin_freq + bin_width)

	bin_x,bin_y = np.meshgrid(np.linspace(focal_loc[0] - num_bins*bin_freq,focal_loc[0] + num_bins*bin_freq,2*num_bins+1),np.linspace(focal_loc[1] - num_bins*bin_freq,focal_loc[1] + num_bins*bin_freq,2*num_bins+1)) # x and y coordinates of the centres of each bin

	# all_agents = X[:,t,:]
	all_agents = focal_rotate(t,focal,L,dt,np.concatenate([X[:,:,0],X[:,:,1]],axis=0))
	all_agents = np.stack([all_agents[:N],all_agents[N:]],axis=1)
	all_agents = np.delete(all_agents,focal,0) #remove the focal agent
	x_locs = np.logical_and(all_agents[:,0]>x_lower, all_agents[:,0]<x_upper) # finds indices of agents within the binning region 
	y_locs = np.logical_and(all_agents[:,1]>y_lower, all_agents[:,1]<y_upper) # "" 
	locs = np.logical_and(x_locs,y_locs) 
	local_agents = all_agents[locs,:]

	for b in range(np.size(local_bin)):
		bin_x_upper = bin_x.flatten()[b] + bin_width
		bin_x_lower = bin_x.flatten()[b] - bin_width
		bin_y_upper = bin_y.flatten()[b] + bin_width
		bin_y_lower = bin_y.flatten()[b] - bin_width

		in_bin_x = np.logical_and(local_agents[:,0]>bin_x_lower,local_agents[:,0]<bin_x_upper)
		in_bin_y = np.logical_and(local_agents[:,1]>bin_y_lower,local_agents[:,1]<bin_y_upper)
		in_bin = np.logical_and(in_bin_x,in_bin_y)

		local_bin[b] = sum(in_bin)
	return local_bin, np.sum(locs)


def get_gif(gif_name,N,L,X,TN,num_frames,frames_start,dt):
	print('starting gif creation')
	t_min = frames_start
	
	# Attaching 3D axis to the figure
	fig = plt.figure(figsize=(8,8))
	ax = plt.axes()
	plt.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'
	# Setting the axes properties
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_xlim((0,L))
	ax.set_ylim((0,L))

	text = ax.text(0, L*1.03, '', fontsize=15)
	
	print('axes defined')
	print('TN is '+str(TN))
	print('N is '+str(N))
	print('range is '+str(len(X[:, 0, 0])))

	# Create a list of line objects
	lines = [ax.plot(X[i_agent, 0, 0], X[i_agent, 0, 1], marker='o', linestyle='none',markersize=5)[0] for i_agent in range(len(X[:, 0, 0]))]
	lines2 = [ax.plot(X[i_agent, 0, 0], X[i_agent, 0, 1], marker='o', linestyle='none',markersize=15)[0] for i_agent in range(len(X[:, 0, 0]))]
	
	print('lines defined')

	# Start animation
	traj_ani = animation.FuncAnimation(fig, update_trajectory, frames=num_frames, fargs=(X, lines, lines2, text, dt, t_min), interval=dt,
									   repeat=False, blit=False)

	print('animation complete')

	FFwriter = animation.FFMpegWriter(fps=30)
	traj_ani.save('Results/'+gif_name+'.mp4', writer=FFwriter)
	print('gif saved')

def update_trajectory(current_t, x, lines, lines2, text, dt, t_min):
	trail_length = 25 
	text.set_text('Time: '+str(np.round(dt*(current_t+t_min),2)))
	for i_agent in range(len(x[:,0,0])):
		lines[i_agent].set_data(x[i_agent, max(current_t-trail_length, 0):current_t, 0],
								x[i_agent, max(current_t-trail_length, 0):current_t, 1])
		lines2[i_agent].set_data(x[i_agent, max(current_t-trail_length, 0), 0],
								x[i_agent, max(current_t-trail_length, 0), 1])
	return lines