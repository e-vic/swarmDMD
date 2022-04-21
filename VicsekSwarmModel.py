## Vicsek Swarm Model
## Emma Hansen
## Updated: June 2020

# Load Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
# from multiprocessing import Pool
# from multiprocessing import Process
# import multiprocessing
from functools import partial
import itertools as it

# Update Swarm
def make_swarm(x0,theta0,N,dt,TN,eta,v,r,L):
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
			index2 = index[:]
			index3 = index[:]
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
			theta2[i] = np.arctan2(np.mean(np.sin(theta1[index])),np.mean(np.cos(theta1[index]))) + dtheta
		
	
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
def update_trajectory(current_t, x, lines, lines2):
	trail_length = 13
	for i_agent in range(len(x[:,0,0])):
		lines[i_agent].set_data(x[i_agent, max(current_t-trail_length, 0):current_t, 0],
								x[i_agent, max(current_t-trail_length, 0):current_t, 1])
		lines2[i_agent].set_data(x[i_agent, max(current_t-trail_length, 0), 0],
								x[i_agent, max(current_t-trail_length, 0), 1])
	return lines

def get_gif(N,eta_name,wide,rho,r_name,L,X,TN,dt):
	# Attaching 3D axis to the figure
	fig = plt.figure(figsize=(8,8))
	ax = plt.axes()
	plt.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'

	# Setting the axes properties
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_xlim((0,L))
	ax.set_ylim((0,L))

	# Create a list of line objects
	lines = [ax.plot(X[i_agent, 0, 0], X[i_agent, 0, 1], marker='o', linestyle='none',markersize=5)[0] for i_agent in range(len(X[:, 0, 0]))]
	lines2 = [ax.plot(X[i_agent, 0, 0], X[i_agent, 0, 1], marker='o', linestyle='none',markersize=15)[0] for i_agent in range(len(X[:, 0, 0]))]

	# Start animation
	traj_ani = animation.FuncAnimation(fig, update_trajectory, frames=TN, fargs=(X, lines, lines2), interval=dt,
									   repeat=False, blit=False)

	gif_name = 'vicsek_N'+str(N)+'_eta'+eta_name+'_W'+str(wide)+'_rho'+str(rho)+'_r'+r_name
	FFwriter = animation.FFMpegWriter(fps=30)
	traj_ani.save('Results/SwarmModel/'+gif_name+'.mp4', writer=FFwriter)


def par_func(x0,theta0,N,dt,TN,rho,v,L,wide,L0,data_input):
	eta = data_input[0]
	r = data_input[1]
	r_name = str(r).replace('.','')
	eta_name = str(eta).replace('.','')
	filename = 'vicsek_N'+str(N)+'_eta'+eta_name+'_W'+str(wide)+'_rho'+str(rho)+'_r'+r_name

	X,_ = make_swarm(x0,theta0,N,dt,TN,eta,v,r,L)
	np.savez('Data/SwarmModel/'+filename,X=X,L=L,L0=L0,r=r,dt=dt,eta=eta,v=v)
	
	get_gif(N,eta_name,wide,rho,r_name,L,X,TN,dt)
	


# Main Code
if __name__ == '__main__':
	# Parameters
	N = 10 # number of agents
	dt = 0.1 # time step size
	TN = 300 # number of time steps

	# eta = [0,np.pi/12] #0deg and 15deg 
	eta = [0]
	rho = 16 # density
	v = 0.03 # agent speed

	# Simulation settings and names
	wide = 0 # to make domain wider than initial condition
	if wide:
		L0 = np.sqrt(N/rho) # length of domain of agent initial positions
		L = 1.5*L0 # length of domain square side 
	else:
		L0 = np.sqrt(N/rho)
		L = L0

	A = L0**2
	r_av = (1/(N/A))**(1/2)

	# Rs = [0.2*r_av,r_av,2*r_av] #20%, 100%, and 200% of the average interaction radius
	Rs = [r_av]

	big_list = [eta,Rs]
	parameters = list(it.product(*big_list))

	# Initialize Swarm
	np.random.seed(1)
	if wide:
		x0 = L0*np.random.rand(N,2) + L/2
	else:
		np.random.seed(4)
		x0 = L*np.random.rand(N,2) # initial the agent positions as: x in [0,L], y in[0,L]

	theta0 = 2*np.pi*(np.random.rand(N,1)-0.5) # intialize agent directions as: theta in [-pi,pi]

	num_procs = np.size(Rs)
	if num_procs > 28:
		num_procs = 28

	func = partial(par_func,x0,theta0,N,dt,TN,rho,v,L,wide,L0)
	for i in range(len(parameters)):
		func(parameters[i])

	# pool = multiprocessing.Pool(processes = num_procs)
	# func = partial(par_func,x0,theta0,N,dt,TN,rho,v,L,wide,L0)
	# pool.map_async(func,parameters)
	# pool.close()
	# pool.join()
