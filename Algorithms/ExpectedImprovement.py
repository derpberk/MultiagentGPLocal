import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from tqdm import trange
from Evaluation.EvaluationUtils import find_peaks, plot_path, plot_trajectory
import pandas as pd



class ExpectedImprovementMultiAgent:

	def __init__(self, navigation_map: np.ndarray, number_of_agents: int, max_movement_distance: float, tolerance: float, travel_distance: float):
		
		self.navigation_map = navigation_map
		self.visitable_positions = np.argwhere(self.navigation_map == 1)
		self.number_of_agents = number_of_agents
		self.max_movement_distance = max_movement_distance
		self.travel_distance = travel_distance
		self.tolerance = tolerance

		self.actions = {i: None for i in range(self.number_of_agents)}
		self.reached_point = np.ones((self.number_of_agents, )).astype(bool)
		self.objectives = np.empty((self.number_of_agents, 2))
		self.distance = np.zeros((self.number_of_agents, ))
		self.first = True

	def get_actions(self, positions: np.ndarray, mu_map: np.ndarray, sigma_map: np.ndarray):

		# Check if any agent has reached its point
		if self.first:
			for agent_id in range(self.number_of_agents):
				self.objectives[agent_id] = positions[agent_id]
			self.first = False


		for agent_id in range(self.number_of_agents):
			if np.linalg.norm(positions[agent_id] - self.objectives[agent_id]) <= self.tolerance or self.distance[agent_id] >= self.travel_distance:
				self.reached_point[agent_id] = True
				self.distance[agent_id] = 0

		if any(self.reached_point):

			# 2) If so, generate a new map for each agent
			new_maps = self.nearest_neighbour_map_generation(positions)

			# 3) Compute the expected improvement for each agent
			ei_map = self.expected_improvement(mu_map, sigma_map, xi=0.3, y_opt=0.0)

			# 4) Update the objective of every agent
			for agent_id in range(self.number_of_agents):
				if self.reached_point[agent_id]:
					self.update_objective(agent_id, ei_map, new_maps[agent_id], position=positions[agent_id])

		# 5) Compute the actions for each agent

		actions = self.compute_actions(positions)

		self.travel_distance += self.max_movement_distance

		return actions
	
	def compute_truncated_mask(self, position: np.ndarray, radius: float = 5):

		truncated_map = np.zeros_like(self.navigation_map)

		# 2.1) Compute the distance between each position and the agent
		distances = np.linalg.norm(self.visitable_positions - position, axis=1)

		# 2.2) Compute the mask for the positions that are within the radius
		mask = distances <= radius

		# 2.3) Compute the truncated expected improvement
		truncated_map[self.visitable_positions[mask, 0], self.visitable_positions[mask, 1]] = 1

		return truncated_map
	
	def compute_actions(self, positions: np.ndarray):

		# Compute the movement for each agent to reach its objective
		movements = [np.round(np.array([np.cos(angle), np.sin(angle)]) * self.max_movement_distance) for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
		actions = {}

		for agent_id in range(self.number_of_agents):
			next_positions = positions[agent_id] + movements
			collision_mask = np.array([self.check_collision(position) for position in next_positions])

			# Compute the next position that is nearest to the objective

			distances = np.linalg.norm(next_positions - self.objectives[agent_id], axis=1)
			distances[collision_mask] = np.inf

			action = np.argmin(distances)

			actions[agent_id] = action

		return actions
			
	def check_collision(self, position):

		if position[0] < 0 or position[0] >= self.navigation_map.shape[0] or position[1] < 0 or position[1] >= self.navigation_map.shape[1]:
			return True

		if self.navigation_map[int(position[0]), int(position[1])] == 0:
			return True

		return False

	def update_objective(self, agent_id, ei_map, mask_map, position):
		
		masked_ei_map = ei_map * mask_map * self.compute_truncated_mask(position, radius=7)

		# 1) Find the maximum value's position in the map
		max_value = np.array(np.unravel_index(np.argmax(masked_ei_map), masked_ei_map.shape)).astype(int)

		# 2) Update the objective of the agent
		self.objectives[agent_id] = max_value
		self.reached_point[agent_id] = False


	def expected_improvement(self, mu_map: np.ndarray, sigma_map: np.ndarray, y_opt: float = 0, xi: float = 0):
		""" Compute the expected improvement for each position in the map """

		flat_mu_map = mu_map.flatten()
		flat_sigma_map = sigma_map.flatten()
		
		values = np.zeros_like(flat_mu_map)
		mask = sigma_map.flatten() > 0
		improve = y_opt - xi - flat_mu_map[mask]
		scaled = improve / flat_sigma_map[mask]
		cdf = norm.cdf(scaled)
		pdf = norm.pdf(scaled)
		exploit = improve * cdf
		explore = flat_sigma_map[mask] * pdf
		values[mask] = exploit + explore

		ei_map = values.reshape(mu_map.shape)

		return ei_map

	def nearest_neighbour_map_generation(self, positions:np.ndarray):
		""" Given a set of positions, generate a map with the nearest neighbour of each position"""

		resulting_maps = np.zeros((len(positions), self.navigation_map.shape[0], self.navigation_map.shape[1]))

		# Fit the KNN classifier
		KNNC = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(positions, np.arange(1, len(positions)+1))

		classification = KNNC.predict(self.visitable_positions)

		for i in range(len(positions)):
			map_positions = self.visitable_positions[classification == i+1]
			resulting_maps[i, map_positions[:,0], map_positions[:,1]] = 1

		return resulting_maps * self.navigation_map


def run_evaluation(path: str, agent, env, algorithm: str, runs: int, n_agents: int, ground_truth_type: str, render = False):

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
		state = env.reset()

		if render:
			env.render()

		# Reset dones #
		done = {agent_id: False for agent_id in range(env.number_of_agents)}

		# Update the metrics #
		metrics['Algorithm'].append(algorithm)
		#metrics['Reward type'].append(reward_type)
		metrics['Run'].append(run)
		metrics['Step'].append(step)
		metrics['N_agents'].append(n_agents)
		metrics['Ground Truth'].append(ground_truth_type)
		metrics['Mean distance'].append(0)
		U0 = env.gp_coordinator.sigma_map.sum()
		metrics['Total uncertainty'].append(env.gp_coordinator.sigma_map.sum() / U0)
		metrics['$\Delta \mu$'].append(0)
		metrics['$\Delta \sigma$'].append(0)
		metrics['Error $\mu$'].append(env.get_error())
		metrics['Max. Error in $\mu_{max}$'].append(1)
		metrics['Mean Error in $\mu_{max}$'].append(1)
		peaks, vals = find_peaks(env.gt.read())
		positions = env.fleet.get_positions()
		for i in range(n_agents): 
			metrics['Agent {} X '.format(i)].append(positions[i,0])
			metrics['Agent {} Y'.format(i)].append(positions[i,1])
			metrics['Agent {} reward'.format(i)].append(0)

		metrics['Accumulated Reward'].append(0)
		
		acc_reward = 0

		while not all(done.values()):

			step += 1

			actions = agent.get_actions(env.fleet.get_positions(), env.gp_coordinator.mu_map, env.gp_coordinator.sigma_map)

			# Process the agent step #
			next_state, reward, done, _ = env.step(actions)

			if render:
				env.render()

			acc_reward += sum(reward.values())

			# Datos de estado
			metrics['Algorithm'].append(algorithm)
			#metrics['Reward type'].append(reward_type)
			metrics['Run'].append(run)
			metrics['Step'].append(step)
			metrics['N_agents'].append(n_agents)
			metrics['Ground Truth'].append(ground_truth_type)
			metrics['Mean distance'].append(env.fleet.get_distances().mean())

			# Datos de cambios en la incertidumbre y el mu
			changes_mu, changes_sigma = env.gp_coordinator.get_changes()
			metrics['$\Delta \mu$'].append(changes_mu.sum())
			metrics['$\Delta \sigma$'].append(changes_sigma.sum())
			# Incertidumbre total aka entropía
			metrics['Total uncertainty'].append(env.gp_coordinator.sigma_map.sum() / U0)
			# Error en el mu
			metrics['Error $\mu$'].append(env.get_error())
			# Error en el mu max
			peaks, vals = find_peaks(env.gt.read())
			estimated_vals = env.gp_coordinator.mu_map[peaks[:,0], peaks[:,1]]
			error = np.abs(estimated_vals - vals)
			metrics['Max. Error in $\mu_{max}$'].append(error.max())
			metrics['Mean Error in $\mu_{max}$'].append(error.mean())

			positions = env.fleet.get_positions()
			for i in range(n_agents): 
				metrics['Agent {} X '.format(i)].append(positions[i,0])
				metrics['Agent {} Y'.format(i)].append(positions[i,1])
				metrics['Agent {} reward'.format(i)].append(0)

			metrics['Accumulated Reward'].append(acc_reward)

		if render:
			plt.show()


	df = pd.DataFrame(metrics)

	df.to_csv(path + '/{}_{}_{}.csv'.format(algorithm, ground_truth_type, n_agents))

		

		
if __name__ == '__main__':

	import time
	from Algorithms.LawnMower import LawnMowerAgent
	from Algorithms.NRRA import WanderingAgent
	import matplotlib.pyplot as plt

	scenario_map = np.genfromtxt('Environment/Maps/example_map.csv')
	#scenario_map = np.ones((50,50))
	seed = 1564
	np.random.seed(seed)
	N = 3
	D = 7
	# Generate initial positions with squares of size 3 x 3 around positions
	center_initial_zones = np.array([[17,9], [22,8], [28,9]]) 
	# 9 positions in the sorrounding of the center
	area_initial_zones = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]])
	# Generate the initial positions with the sum of the center and the area
	fleet_initial_zones = np.array([area_initial_zones + center_initial_zones[i] for i in range(len(center_initial_zones))])
	env = MultiagentInformationGathering(
			scenario_map = scenario_map,
			number_of_agents = N,
			distance_between_locals = D,
			radius_of_locals = np.sqrt(2) * D / 2 if D < 25 else 1000,
			distance_budget = 100,
			distance_between_agents = 1,
			fleet_initial_zones=fleet_initial_zones,
			fleet_initial_positions=None,
			seed = 5,
			movement_length = 2,
			max_collisions = 5,
			ground_truth_type = 'shekel',
			local = True
	)

	# Create the agents
	agent = ExpectedImprovementMultiAgent(navigation_map = scenario_map, number_of_agents= 3, max_movement_distance = 3, tolerance=3, travel_distance=10000)

	# Initialize the metrics 
	done = {i:False for i in range(N)}
	t = 0

	runtime = 0

	while not any(list(done.values())):
		
		#action = {i: lawn_mower_agents[i].move(env.fleet.vehicles[i].position) for i in range(N)}

		action = agent.get_actions(env.fleet.get_positions(), env.gp_coordinator.mu_map, env.gp_coordinator.sigma_map)

		s, r, done, _ = env.step(action)

		print(r)

		env.render()

		t+=1