## Vicsek Swarm Model - Milling
## Emma Hansen
## Updated: April 2022

# Load Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing
from functools import partial
import itertools as it
from scipy.interpolate import interp1d

def wrap_position(L,agent):
    boundaryT_index = np.where(agent>L)
    boundaryB_index = np.where(agent<0)

    agent[boundaryT_index] = agent[boundaryT_index] - L*(agent[boundaryT_index]//L)
    agent[boundaryB_index] = agent[boundaryB_index] - L*(agent[boundaryB_index]//L)
    
    return agent

def getDifference(b1, b2):
	"""code from https://rosettacode.org/wiki/Angle_difference_between_two_bearings#Python"""
	r = (b2 - b1) % (2*np.pi)
	r[r >= np.pi] -= 2*np.pi
	return r

# Update Swarm
def make_swarm(x0,theta0,N,dt,TN,eta,rho,v,omega_max,phi,r,L,dynamicstype):
	X = np.empty((N,TN,2))
	Theta = np.empty((N,TN))
	
	X[:,0,:] = x0
	Theta[:,0:2] = theta0
	x1 = np.copy(x0)
	x2 = np.copy(x0)
	theta1 = np.copy(theta0)
	theta2 = np.copy(theta0)

	for k in range(1,TN):
		for i in range(N):
			dtheta = eta*(np.random.rand(1,1)-0.5)
			dist = np.linalg.norm(x1[i]-x1,axis=1)
			index = np.where(dist<r)
			index2 = np.copy(index)
			index3 = np.copy(index)
			if np.absolute(x1[i][0] - L) < r: # approaching right boundary
				p2 = np.array([0 - np.absolute(x1[i][0] - L),x1[i][1]])
				dist2 = np.linalg.norm(p2-x1,axis=1)
				index2 = np.where(dist2<r)
			elif np.absolute(x1[i][0]) < r: # approaching left boundary
				p2 = np.array([L + np.absolute(x1[i][0]),x1[i][1]])
				dist2 = np.linalg.norm(p2-x1,axis=1)
				index2 = np.where(dist2<r)
			
			if np.absolute(x1[i][1] - L) < r: # approaching upper boundary
				p3 = np.array([x1[i][0], 0 - np.absolute(x1[i][1] - L)])
				dist3 = np.linalg.norm(p3-x1,axis=1)
				index3 = np.where(dist3<r)
			elif np.absolute(x1[i][1]) < r: #approaching lower boundary
				p3 = np.array([x1[i][0], L + np.absolute(x1[i][1])])
				dist3 = np.linalg.norm(p3-x1,axis=1)
				index3 = np.where(dist3<r)
			
			index = np.unique(np.concatenate((index[0],index2[0],index3[0])))
			if dynamicstype == 'standard':
				theta_avg = np.arctan2(np.mean(np.sin(theta1[index])),np.mean(np.cos(theta1[index])))
				theta2[i] = theta_avg + dtheta

			elif dynamicstype == 'milling':
				p_diff = x1 - x1[i]
				rel_angle = np.abs(getDifference(np.arctan2(p_diff[:,1],p_diff[:,0]),theta1[i]))
				rel_angle[i] = 0.
				ind_theta = np.where(rel_angle<=(phi/2))
				index_theta = np.intersect1d(index,ind_theta)

				theta_avg = np.arctan2(np.mean(np.sin(theta1[index_theta])),np.mean(np.cos(theta1[index_theta])))
				theta_diff = getDifference(theta1[i],theta_avg)

				if np.abs(theta_diff) < omega_max*dt:
					theta2[i] = theta_avg + dtheta
				elif theta_diff >= omega_max*dt:
					theta2[i] = theta1[i] + omega_max*dt + dtheta
				else:
					theta2[i] = theta1[i] - omega_max*dt + dtheta

				wrap_reference = 0
				wrap_array = np.array([wrap_reference, theta2[i][0]])
				theta2[i][0] = np.unwrap(wrap_array)[-1]
	
		vi = v*np.concatenate((np.cos(theta2),np.sin(theta2)),axis=1)
		x2 = x1 + vi*dt
	
		# Periodic Boundary Condition
		boundaryT_index = np.where(x2[:,1]>L)
		boundaryB_index = np.where(x2[:,1]<0)
		boundaryL_index = np.where(x2[:,0]<0)
		boundaryR_index = np.where(x2[:,0]>L)
		
		x2[boundaryT_index,1] = x2[boundaryT_index,1] - L
		x2[boundaryB_index,1] = x2[boundaryB_index,1] + L
		x2[boundaryL_index,0] = x2[boundaryL_index,0] + L
		x2[boundaryR_index,0] = x2[boundaryR_index,0] - L
	
		# Store and update positions
		X[:,k,:] = x2
		Theta[:,k] = theta2[:,0]
	
		x1 = np.copy(x2)
		theta1 = np.copy(theta2)
	return X,Theta
	
# Plot Result
def update_trajectory(current_t, x, lines, lines2, text, dt, t_min):
	trail_length = 13
	text.set_text('Time: '+str(np.round(dt*(current_t+t_min),2)))
	for i_agent in range(len(x[:,0,0])):
		lines[i_agent].set_data(x[i_agent, max(current_t-trail_length+t_min, 0):current_t+t_min, 0],
								x[i_agent, max(current_t-trail_length+t_min, 0):current_t+t_min, 1])
		lines2[i_agent].set_data(x[i_agent, max(current_t-trail_length+t_min, 0), 0],
								x[i_agent, max(current_t-trail_length+t_min, 0), 1])
	return lines,text

def get_gif(N,eta_name,wide,rho,r_name,L,X,TN,dynamicstype,dt):
	# Attaching 3D axis to the figure
	print('Creating video.')
	fig = plt.figure(figsize=(8,8))
	ax = plt.axes()
	plt.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'

	if TN < 500:
		t_max = TN
		t_min = 0
	else:
		t_max = 500 # =t, set this to 0 to plot the whole time period, otherwise it'll plot for time steps t:end
		t_min = TN-500

	# Setting the axes properties
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_xlim((0,L))
	ax.set_ylim((0,L))

	text = ax.text(0, L*1.03, '', fontsize=15)

	# Create a list of line objects
	lines = [ax.plot(X[i_agent, 0, 0], X[i_agent, 0, 1], marker='o', linestyle='none',markersize=1*(20/L))[0] for i_agent in range(len(X[:, 0, 0]))]
	lines2 = [ax.plot(X[i_agent, 0, 0], X[i_agent, 0, 1], marker='o', linestyle='none',markersize=5*(15/L))[0] for i_agent in range(len(X[:, 0, 0]))]

	# Start animation
	traj_ani = animation.FuncAnimation(fig, update_trajectory, frames=t_max, fargs=(X, lines, lines2, text, dt,t_min), interval=dt,
									   repeat=False, blit=False) # only print frames from t_min to t_max

	gif_name = 'vicsek_'+dynamicstype+'_N'+str(N)+'_eta'+eta_name+'_W'+str(wide)+'_rho'+str(rho)+'_r'+r_name
	FFwriter = animation.FFMpegWriter(fps=30)
	traj_ani.save('Results/SwarmModel/'+gif_name+'.mp4', writer=FFwriter)
	# traj_ani.save('Results/SwarmModel/'+gif_name, writer='pillow')

def par_func(x0,theta0,N,dt,TN,rho,v,omega_max,phi,L,wide,dynamicstype,L0,downsample_milling,sample_freq,dt_new,data_input):
	eta = data_input[0]
	r = data_input[1]

	# try:
	X,Theta = make_swarm(x0,theta0,N,dt,TN,eta,rho,v,omega_max,phi,r,L,dynamicstype)
	print('swarm created.')
	if downsample_milling:
		print('downsampling started')
		agent_indices = range(0,np.shape(X)[0],sample_freq)
		X = X[agent_indices,:,:]
		N = len(agent_indices)
		t = np.arange(0,TN,dt)
		t_new = np.arange(0,TN-1,dt_new)

		X_stack = np.concatenate((X[:,:,0],X[:,:,1]),axis=0)
		for ts in range(1,TN):
			# at each time steps adjusts agents' positions (if necessary) to behave as if in a properly continuous domain and not finite one with periodic boundary condition
			X_diff = X_stack[:,ts]-X_stack[:,ts-1]
			diff_locs = np.argwhere(np.abs(X_diff)>(L/2))
			for i in diff_locs[:,0]:
				if X_diff[i] < 0 :
					X_stack[i,ts:] = X_stack[i,ts:] + L
				else:
					X_stack[i,ts:] = X_stack[i,ts:] - L

		f = interp1d(t,X_stack,axis=1)
		X_stack_new = f(t_new)
		X_stack_new = wrap_position(L,X_stack_new) 
		X = np.stack((X_stack_new[:N,:],X_stack_new[N:,:]),axis=2) 
		dt = dt_new
		print('downsampling finished')


	# except Exception as e: print(e)
	r_name = str(r).replace('.','')
	eta_name = str(eta).replace('.','')
	
	filename = 'vicsek_'+dynamicstype+'_N'+str(N)+'_eta'+eta_name+'_W'+str(wide)+'_rho'+str(rho)+'_r'+r_name
	np.savez('Data/SwarmModel/'+filename,X=X,L=L,L0=L0,r=r,dt=dt,eta=eta,v=v,dynamics=dynamicstype,omega_max=omega_max)
	print('Simulation saved.')
	try:
		get_gif(N,eta_name,wide,rho,r_name,L,X,TN,dynamicstype,dt)
		print('Video saved.')
	except Exception as e: print(e)


# Main Code
if __name__ == '__main__':
	# Parameters
	N = 1000 # number of agents
	dt = 1 # time step size
	TN = 1000 # number of time steps

	rho = 2.5 # density 
	ratio_speedang = 1.03
	phi = np.pi*(4/3)

	# Simulation settings and names
	dynamicstype = 'milling' # options: standard, milling
	downsample_milling = 1 # choose to reduce the number of agents and change the time step in the milling sim before saving
	sample_freq = 5 # keep every n'th agent
	dt_new = 0.1
				
	wide = 0 # to make domain wider than initial condition
	if wide:
		L0 = np.sqrt(N/rho) # length of domain of agent initial positions
		L = 1.5*L0 # length of domain square side 
	else:
		L0 = np.sqrt(N/rho)
		L = L0

	A = L0**2
	r_av = (1/(N/A))**(1/2)

	Rs = [1*(L/20)] # as chosen in paper
	omega_max = (np.pi/18) 
	v = ratio_speedang*Rs[0]*omega_max/dt 
	eta = [(L/20)*0.5*omega_max/dt] # PARAMETER FROM PAPER

	print('eta is: '+str(eta[0]))
	print('omega max is: '+str(omega_max))
	print('speed is: '+str(v))

	big_list = [eta,Rs]
	parameters = list(it.product(*big_list))
	print('parameters are:')
	print(parameters)

	# Initialize Swarm
	np.random.seed(1)
	if wide:
		x0 = L0*np.random.rand(N,2) + L/2
	else:
		x0 = L*np.random.rand(N,2) # initial the agent positions as: x in [0,L], y in[0,L]
		
	theta0 = 2*np.pi*(np.random.rand(N,1)-0.5) # intialize agent directions as: theta in [-pi,pi]
							
	func = partial(par_func,x0,theta0,N,dt,TN,rho,v,omega_max,phi,L,wide,dynamicstype,L0,downsample_milling,sample_freq,dt_new)
	for i in range(len(parameters)):
		func(parameters[i])