import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import trange
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def find_peaks(data:np.ndarray, neighborhood_size: int = 5, threshold: float = 0.1) -> np.ndarray:
	""" Find the peaks in a 2D image using the local maximum filter. """


	data_max = filters.maximum_filter(data, neighborhood_size)
	maxima = (data == data_max)
	data_min = filters.minimum_filter(data, neighborhood_size)
	diff = ((data_max - data_min) > threshold)
	maxima[diff == 0] = 0

	labeled, num_objects = ndimage.label(maxima)
	slices = ndimage.find_objects(labeled)
	x, y = [], []
	for dy,dx in slices:
		x_center = (dx.start + dx.stop - 1)/2
		x.append(x_center)
		y_center = (dy.start + dy.stop - 1)/2    
		y.append(y_center)

	peaks = np.array([y,x]).T.astype(int)

	return peaks, data[peaks[:,0], peaks[:,1]]


def plot_path(path: np.ndarray, axs = None, title: str = ''):
	""" Plot the trajectories and the peaks of the trajectories. """

	if axs is None:
		fig, axs = plt.subplots(1,1, figsize=(10,10))

	# Plot the paths of the agents gradually changing the color of the line for every point #
	for i in range(path.shape[0]):
		axs.plot(path[:,1], path[:,0], color=(i/path.shape[0], 0, 1-i/path.shape[0]))
	
	return axs


def plot_trajectory(ax, x, y, z=None, colormap = 'jet', num_of_points = None, linewidth = 1, k = 3, plot_waypoints=False, markersize = 0.5, alpha=1, zorder=1, s=0.0):

	# Remove consecuitve duplicates values of XYZ #
	if z is None:
		path = np.array([x,y]).T
	else:
		path = np.array([x,y,z]).T

	for i in range(path.shape[0]-1, 0, -1):
		if np.array_equal(path[i], path[i-1]):
			path = np.delete(path, i, axis=0)
	
	x = path[:,0]
	y = path[:,1]
	if z is not None:
		z = path[:,2]




	if z is None:
		tck, u = interpolate.splprep([x, y], s=s, k=k)
		x_i, y_i= interpolate.splev(np.linspace(0,1,num_of_points),tck)
		points = np.array([x_i,y_i]).T.reshape(-1,1,2)
		segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
		lc = LineCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth, alpha=alpha, zorder=zorder)
		lc.set_array(np.linspace(0,1,len(x_i)))
		ax.add_collection(lc)
		if plot_waypoints:
			ax.plot(x,y,'.', color = 'black', markersize = markersize, zorder=zorder+1)
	else:
		tck, u =interpolate.splprep([x, y, z], s=0.0)
		x_i, y_i, z_i= interpolate.splev(np.linspace(0,1,num_of_points), tck)
		points = np.array([x_i, y_i, z_i]).T.reshape(-1,1,3)
		segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
		lc = Line3DCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth)
		lc.set_array(np.linspace(0,1,len(x_i)))
		ax.add_collection(lc)
		ax.scatter(x,y,z,'k')
		if plot_waypoints:
			ax.plot(x,y,'kx')




def run_evaluation(path: str, agent, algorithm: str, reward_type: str, runs: int, n_agents: int, ground_truth_type: str, render = False):

	
	metrics = {'Algorithm': [], 
			'Reward type': [],  
			'Run': [], 
			'Step': [],
			'N_agents': [],
			'Ground Truth': [],
			'Mean distance': [],
			'Accumulated Reward': [],
			'$\Delta \mu$': [], 
			'$\Delta \sigma$': [], 
			'Total uncertainty': [],
			'Error $\mu$': [],
			'Max. Error in $\mu_{max}$': [], 
			'Mean Error in $\mu_{max}$': [],
			'SOP GLOBAL GP':[],
			'MSE GLOBAL GP':[],
			'R2 GLOBAL GP':[],}
	
	for i in range(n_agents): 
		metrics['Agent {} X '.format(i)] = []
		metrics['Agent {} Y'.format(i)] = []
		metrics['Agent {} reward'.format(i)] = []

	for run in trange(runs):

		#Increment the step counter #
		step = 0
		
		# Reset the environment #
		state = agent.env.reset()

		if render:
			agent.env.render()

		# Reset dones #
		done = {agent_id: False for agent_id in range(agent.env.number_of_agents)}

		# Reset modules
		for module in agent.nogoback_masking_modules.values():
			module.reset()

		# Update the metrics #
		metrics['Algorithm'].append(algorithm)
		metrics['Reward type'].append(reward_type)
		metrics['Run'].append(run)
		metrics['Step'].append(step)
		metrics['N_agents'].append(n_agents)
		metrics['Ground Truth'].append(ground_truth_type)
		metrics['Mean distance'].append(0)
		metrics['Total uncertainty'].append(agent.env.gp_coordinator.sigma_map[agent.env.gp_coordinator.X[:,0], agent.env.gp_coordinator.X[:,1]].mean())
		metrics['$\Delta \mu$'].append(0)
		metrics['$\Delta \sigma$'].append(0)
		metrics['Error $\mu$'].append(agent.env.get_error())
		metrics['Max. Error in $\mu_{max}$'].append(1)
		metrics['Mean Error in $\mu_{max}$'].append(1)

		peaks, vals = find_peaks(agent.env.gt.read())
		if peaks.shape[0] == 0:
			peaks, vals = find_peaks(agent.env.gt.read(), threshold=0.3)

		positions = agent.env.fleet.get_positions()
		for i in range(n_agents): 
			metrics['Agent {} X '.format(i)].append(positions[i,0])
			metrics['Agent {} Y'.format(i)].append(positions[i,1])
			metrics['Agent {} reward'.format(i)].append(0)

		metrics['Accumulated Reward'].append(0)
		
		acc_reward = 0

		while not all(done.values()):

			step += 1

			# Select the action using the current policy
			if not 	agent.masked_actions:
				actions = agent.select_action(state, deterministic=True)
			else:
				actions = agent.select_masked_action(states=state, positions=agent.env.fleet.get_positions(), deterministic=True)
				
			actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}

			# Process the agent step #
			next_state, reward, done, _ = agent.env.step(actions)

			if render:
				agent.env.render()

			acc_reward += sum(reward.values())

			# Update the state #
			state = next_state

			# Datos de estado
			metrics['Algorithm'].append(algorithm)
			metrics['Reward type'].append(reward_type)
			metrics['Run'].append(run)
			metrics['Step'].append(step)
			metrics['N_agents'].append(n_agents)
			metrics['Ground Truth'].append(ground_truth_type)
			metrics['Mean distance'].append(agent.env.fleet.get_distances().mean())

			# Datos de cambios en la incertidumbre y el mu
			changes_mu, changes_sigma = agent.env.gp_coordinator.get_changes()
			metrics['$\Delta \mu$'].append(changes_mu.sum())
			metrics['$\Delta \sigma$'].append(changes_sigma.sum())
			# Incertidumbre total aka entropía
			metrics['Total uncertainty'].append(agent.env.gp_coordinator.sigma_map[agent.env.gp_coordinator.X[:,0], agent.env.gp_coordinator.X[:,1]].mean())
			# Error en el mu
			metrics['Error $\mu$'].append(agent.env.get_error())
			# Error en el mu max
			peaks, vals = find_peaks(agent.env.gt.read())
			if peaks.shape[0] == 0:

				peaks = np.unravel_index(np.argmax(agent.env.gt.read()), agent.env.gt.read().shape) 
				vals = agent.env.gt.read()[peaks[0], peaks[1]]
				#peaks, vals = find_peaks(agent.env.gt.read(), threshold=0.8)
				estimated_vals = agent.env.gp_coordinator.mu_map[peaks[0], peaks[1]]
			else:
				estimated_vals = agent.env.gp_coordinator.mu_map[peaks[:,0], peaks[:,1]]
			error = np.abs(estimated_vals - vals)
			metrics['Max. Error in $\mu_{max}$'].append(np.max(error))
			metrics['Mean Error in $\mu_{max}$'].append(np.mean(error.mean()))

			positions = agent.env.fleet.get_positions()
			for i in range(n_agents): 
				metrics['Agent {} X '.format(i)].append(positions[i,0])
				metrics['Agent {} Y'.format(i)].append(positions[i,1])
				metrics['Agent {} reward'.format(i)].append(0)

			metrics['Accumulated Reward'].append(acc_reward)

		if render:
			plt.show()

		# Compute the final error using all the points in the map #
		gp_unique = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3)), n_restarts_optimizer=10)
		gp_unique.fit(agent.env.gp_coordinator.x, agent.env.gp_coordinator.y)
		mu_global = gp_unique.predict(agent.env.gp_coordinator.X)
		real_values = agent.env.gt.read()[agent.env.gp_coordinator.X[:,0], agent.env.gp_coordinator.X[:,1]]
		sop_GLOBAL_GP = np.abs(mu_global - real_values).sum()
		mse_GLOBAL_GP = mean_squared_error(mu_global, real_values)
		r2_GLOBAL_GP = r2_score(mu_global, real_values)

		# Add the final error to the metrics #
		metrics['SOP GLOBAL GP'].extend([sop_GLOBAL_GP] * (step + 1))
		metrics['MSE GLOBAL GP'].extend([mse_GLOBAL_GP] * (step + 1))
		metrics['R2 GLOBAL GP'].extend([r2_GLOBAL_GP] * (step + 1))


	df = pd.DataFrame(metrics)



	df.to_csv(path + '/{}_{}_{}_{}.csv'.format(algorithm, ground_truth_type, reward_type, n_agents))

