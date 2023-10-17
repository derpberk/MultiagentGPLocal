import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel as W
from tqdm import trange
from Evaluation.EvaluationUtils import find_peaks
import pandas as pd

class ParticleSwarmOptimizer:

	def __init__(self, n_agents, navigation_map, ground_truth, max_distance, initial_positions, parameters):
			   
		self.navigation_map = navigation_map
		self.ground_truth = ground_truth
		self.max_distance = max_distance
		self.GP = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(5.0, length_scale_bounds=(0.5, 100)) + W(0.001), n_restarts_optimizer=1, alpha=0.001)
		self.mu_map = np.zeros_like(navigation_map)
		self.sigma_map = np.zeros_like(navigation_map)
		self.visitable_positions = np.argwhere(navigation_map == 1)
		self.initial_positions = initial_positions
		self.n_agents = n_agents

		self.w, self.c1, self.c2, self.c3, self.c4 = parameters

		self.fig = None


	def render(self):

		if self.fig is None:

			plt.ion()
			self.fig, self.axs = plt.subplots(1, 2)

			self.axs[0].imshow(self.navigation_map, cmap='gray', alpha = 1-self.navigation_map, zorder = 10)
			self.axs[1].imshow(self.navigation_map, cmap='gray', alpha = 1-self.navigation_map, zorder = 10)
			self.axs[1].imshow(self.ground_truth.read(), cmap='viridis', zorder = 1)

			self.d1 = self.axs[0].imshow(self.mu_map, cmap='viridis', vmin=0, vmax=1, zorder=1)
			self.d2, = self.axs[0].plot(self.positions[:,1], self.positions[:,0], 'o', color='red', zorder=20)


		else:

			self.d1.set_data(self.mu_map)
			self.d2.set_data(self.positions[:,1], self.positions[:,0])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		plt.pause(0.1)


	def reset(self):

		self.mu_map = np.zeros_like(self.navigation_map)
		self.sigma_map = np.zeros_like(self.navigation_map)
		self.positions = self.initial_positions.copy()
		self.velocities = np.zeros_like(self.positions)
		self.distances = np.zeros(self.n_agents)

		self.fig = None

		self.ground_truth.reset()

		# Get the initial fitness
		self.values = self.ground_truth.read(self.positions)

		self.gp_values = np.atleast_2d(self.values).T
		self.gp_positions = self.positions.copy()

		self.best_positions = self.positions.copy()
		self.best_values = self.values.copy()
		self.best_global_position = self.positions[np.argmax(self.values)]
		self.best_global_value = np.max(self.values)

		# Initialize the GP
		self.GP.fit(self.gp_positions, self.gp_values)

		# Get the initial uncertainty
		self.mu, self.sigma = self.GP.predict(self.visitable_positions, return_std=True)
		self.mu_map[self.visitable_positions[:,0], self.visitable_positions[:,1]] = self.mu.reshape(-1)
		self.sigma_map[self.visitable_positions[:,0], self.visitable_positions[:,1]] = self.sigma.reshape(-1)

		self.done = False

	def optimize(self):
		

		# Update the velocities
		r1 = self.c1 * np.random.rand(*self.positions.shape)
		r2 = self.c2 * np.random.rand(*self.positions.shape)
		r3 = self.c3 * np.random.rand(*self.positions.shape)
		r4 = self.c4 * np.random.rand(*self.positions.shape)

		distances_to_best_global = self.best_global_position - self.positions
		distances_to_best_global = distances_to_best_global / (np.linalg.norm(distances_to_best_global, axis=1)[:,None] + 1e-8)

		distance_to_best_local = self.best_positions - self.positions
		distance_to_best_local = distance_to_best_local / (np.linalg.norm(distance_to_best_local, axis=1)[:,None]  + 1e-8)

		distance_to_highest_uncertainty = self.visitable_positions[np.argmax(self.sigma)] - self.positions
		distance_to_highest_uncertainty = distance_to_highest_uncertainty / (np.linalg.norm(distance_to_highest_uncertainty, axis=1)[:,None]  + 1e-8)

		distance_to_highest_mu = self.visitable_positions[np.argmax(self.mu)] - self.positions
		distance_to_highest_mu = distance_to_highest_mu / (np.linalg.norm(distance_to_highest_mu, axis=1)[:,None]  + 1e-8)

		self.velocities = self.w * self.velocities + r2 * distances_to_best_global + r1 * distance_to_best_local + r3 * distance_to_highest_uncertainty + r4 * distance_to_highest_mu

		# Update the positions
		self.velocities = self.clip_velocities(0.05*self.velocities)

		next_positions = self.positions + self.velocities

		# Move the agents to the closest valid position
		for i in range(len(next_positions)):
			next_valid_position = self._get_closest_valid_position(next_positions[i])
			self.distances[i] += np.linalg.norm(next_valid_position - self.positions[i])
			self.positions[i] = next_valid_position


		# Get the fitness of the next positions
		values = self.ground_truth.read(self.positions)

		# Store the new positions and values
		self.gp_values = np.concatenate((self.gp_values, np.atleast_2d(values).T), axis=0)
		self.gp_positions = np.concatenate((self.gp_positions, self.positions), axis=0)

		# Get the unique positions
		unique_positions, unique_indices = np.unique(self.gp_positions, axis=0, return_index=True)
		self.gp_positions = unique_positions
		self.gp_values = self.gp_values[unique_indices]

		# Update the best positions
		for i in range(len(values)):
			if self.values[i] > self.best_values[i]:
				self.best_values[i] = values[i]
				self.best_positions[i] = self.positions[i]
		
		# Update the best global position
		if np.max(self.values) > self.best_global_value:
			self.best_global_value = np.max(self.values)
			self.best_global_position = self.positions[np.argmax(self.values)]
		
		# Update the GP
		self.GP.fit(self.gp_positions, self.gp_values)

		# Get the uncertainty
		self.mu, self.sigma = self.GP.predict(self.visitable_positions, return_std=True)
		self.mu_map[self.visitable_positions[:,0], self.visitable_positions[:,1]] = self.mu.reshape(-1)
		self.sigma_map[self.visitable_positions[:,0], self.visitable_positions[:,1]] = self.sigma.reshape(-1)

		# Check if the algorithm is done
		if any(self.distances > self.max_distance):
			self.done = True


	def _get_closest_valid_position(self, position):
		distances = np.linalg.norm(self.visitable_positions - position, axis=1)
		closest_position = self.visitable_positions[np.argmin(distances)]
		return closest_position
	
	def clip_velocities(self, velocities):

		# Clip every velocity to have a minimum norm of 1
		for i in range(len(velocities)):
			norm = np.linalg.norm(velocities[i])
			if norm < 2:
				velocities[i] = 2 * velocities[i] / norm
		
		# Clip every velocity to have a maximum norm of 2

		for i in range(len(velocities)):
			norm = np.linalg.norm(velocities[i])
			if norm > 5:
				velocities[i] = 5 * velocities[i] / norm
			
		return velocities
	
	def get_error(self):
		return np.sum(np.abs(self.mu_map - self.ground_truth.read())) / self.ground_truth.read().sum()
	

def run_evaluation(path: str, agent, algorithm: str, runs: int, n_agents: int, ground_truth_type: str, render = False):

	metrics = {'Algorithm': [], 
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
			'Mean Error in $\mu_{max}$': []}
	
	for i in range(n_agents): 
		metrics['Agent {} X '.format(i)] = []
		metrics['Agent {} Y'.format(i)] = []
		metrics['Agent {} reward'.format(i)] = []

	for run in trange(runs):
		#Increment the step counter #
		step = 0
		
		# Reset the environment #
		agent.reset()

		if render:
			agent.render()

		# Reset dones #
		done = {agent_id: False for agent_id in range(n_agents)}

		# Update the metrics #
		metrics['Algorithm'].append(algorithm)
		#metrics['Reward type'].append(reward_type)
		metrics['Run'].append(run)
		metrics['Step'].append(step)
		metrics['N_agents'].append(n_agents)
		metrics['Ground Truth'].append(ground_truth_type)
		metrics['Mean distance'].append(0)
		U0 = agent.sigma_map.sum()
		metrics['Total uncertainty'].append(agent.sigma_map.sum() / U0)
		metrics['$\Delta \mu$'].append(0)
		metrics['$\Delta \sigma$'].append(0)
		metrics['Error $\mu$'].append(agent.get_error())
		metrics['Max. Error in $\mu_{max}$'].append(1)
		metrics['Mean Error in $\mu_{max}$'].append(1)
		peaks, vals = find_peaks(agent.ground_truth.read())
		positions = agent.positions
		for i in range(n_agents): 
			metrics['Agent {} X '.format(i)].append(positions[i,0])
			metrics['Agent {} Y'.format(i)].append(positions[i,1])
			metrics['Agent {} reward'.format(i)].append(0)

		metrics['Accumulated Reward'].append(0)
		
		acc_reward = 0

		while not agent.done:

			step += 1

			old_mu = agent.mu_map.copy()
			old_sigma = agent.sigma_map.copy()

			agent.optimize()

			changes_mu = np.abs(agent.mu_map - old_mu).sum()
			changes_sigma = np.abs(agent.sigma_map - old_sigma).sum()

			if render:
				agent.render()

			acc_reward += 0

			# Datos de estado
			metrics['Algorithm'].append(algorithm)
			#metrics['Reward type'].append(reward_type)
			metrics['Run'].append(run)
			metrics['Step'].append(step)
			metrics['N_agents'].append(n_agents)
			metrics['Ground Truth'].append(ground_truth_type)
			metrics['Mean distance'].append(agent.distances.mean())

			# Datos de cambios en la incertidumbre y el mu
			metrics['$\Delta \mu$'].append(changes_mu)
			metrics['$\Delta \sigma$'].append(changes_sigma)
			# Incertidumbre total aka entropía
			metrics['Total uncertainty'].append(agent.sigma_map.sum() / U0)
			# Error en el mu
			metrics['Error $\mu$'].append(agent.get_error())
			# Error en el mu max
			peaks, vals = find_peaks(agent.ground_truth.read())
			if len(peaks) == 0:
				metrics['Max. Error in $\mu_{max}$'].append(0)
				metrics['Mean Error in $\mu_{max}$'].append(0)
			else:
				estimated_vals = agent.mu_map[peaks[:,0], peaks[:,1]]
				error = np.abs(estimated_vals - vals)
				metrics['Max. Error in $\mu_{max}$'].append(error.max())
				metrics['Mean Error in $\mu_{max}$'].append(error.mean())

			positions = agent.positions
			for i in range(n_agents): 
				metrics['Agent {} X '.format(i)].append(positions[i,0])
				metrics['Agent {} Y'.format(i)].append(positions[i,1])
				metrics['Agent {} reward'.format(i)].append(0)

			metrics['Accumulated Reward'].append(acc_reward)


	df = pd.DataFrame(metrics)

	df.to_csv(path + '/{}_{}_{}.csv'.format(algorithm, ground_truth_type, n_agents))

	return df









		


