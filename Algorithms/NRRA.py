import numpy as np
import pandas as pd 
from tqdm import trange
from Evaluation.EvaluationUtils import find_peaks
import matplotlib.pyplot as plt


class WanderingAgent:

	def __init__(self, world: np.ndarray, movement_length: float, number_of_actions: int, consecutive_movements = None, seed = 0):
		
		self.world = world
		self.move_length = movement_length
		self.number_of_actions = number_of_actions
		self.consecutive_movements = consecutive_movements
		self.t = 0
		self.action = None
		self.seed = seed
		np.random.seed(self.seed)
	
	def move(self, actual_position):

		if self.action is None:
			self.action = self.select_action_without_collision(actual_position)
		
		# Compute if there is an obstacle or reached the border #
		OBS = self.check_collision(self.action, actual_position)

		if OBS:
			self.action = self.select_action_without_collision(actual_position)

		if self.consecutive_movements is not None:
			if self.t == self.consecutive_movements:
				self.action = self.select_action_without_collision(actual_position)
				self.t = 0

		self.t += 1
		return self.action
	
	
	def action_to_vector(self, action):
		""" Transform an action to a vector """

		vectors = np.array([[np.cos(2*np.pi*i/self.number_of_actions), np.sin(2*np.pi*i/self.number_of_actions)] for i in range(self.number_of_actions)])

		return np.round(vectors[action]).astype(int)
	
	def opposite_action(self, action):
		""" Compute the opposite action """
		return (action + self.number_of_actions//2) % self.number_of_actions
	
	def check_collision(self, action, actual_position):
		""" Check if the agent collides with an obstacle """
		new_position = actual_position + self.action_to_vector(action) * self.move_length
		new_position = np.ceil(new_position).astype(int)
		
		OBS = (new_position[0] < 0) or (new_position[0] >= self.world.shape[0]) or (new_position[1] < 0) or (new_position[1] >= self.world.shape[1])
		if not OBS:
			OBS = self.world[new_position[0], new_position[1]] == 0

		return OBS

	def select_action_without_collision(self, actual_position):
		""" Select an action without collision """
		action_caused_collision = [self.check_collision(action, actual_position) for action in range(self.number_of_actions)]

		# Select a random action without collision and that is not the oppositve previous action #
		if self.action is not None:
			opposite_action = self.opposite_action(self.action)
			action_caused_collision[opposite_action] = True
		action = np.random.choice(np.where(np.logical_not(action_caused_collision))[0])

		return action
	


def run_evaluation(path: str, env, algorithm: str, runs: int, n_agents: int, ground_truth_type: str, render = False):

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

	random_wandering_agents = [WanderingAgent(world = env.scenario_map, number_of_actions = 8, movement_length = 3, seed=0) for _ in range(n_agents)]

	for run in trange(runs):
		#Increment the step counter #
		step = 0
		
		# Reset the environment #
		env.reset()

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

			actions = {i: random_wandering_agents[i].move(env.fleet.vehicles[i].position) for i in range(n_agents)}

			# Process the agent step #
			_, reward, done, _ = env.step(actions)

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