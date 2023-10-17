import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from Evaluation.EvaluationUtils import find_peaks

class LawnMowerAgent:

	def __init__(self, world: np.ndarray, number_of_actions: int, movement_length: int, forward_direction: int, seed=0):

		""" Finite State Machine that represents a lawn mower agent. """
		self.world = world
		self.action = None
		self.number_of_actions = number_of_actions
		self.move_length = movement_length
		self.state = 'FORWARD'
		self.turn_count = 0
		self.initial_action = forward_direction
		self.seed = seed
		np.random.seed(seed)

	
	def compute_obstacles(self, position):
		# Compute if there is an obstacle or reached the border #
		OBS = position[0] < 0 or position[0] >= self.world.shape[0] or position[1] < 0 or position[1] >= self.world.shape[1]
		if not OBS:
			OBS = OBS or self.world[position[0], position[1]] == 0

		return OBS


	def move(self, actual_position):
		""" Compute the new state """

		# Compute the new position #
		new_position = actual_position + self.action_to_vector(self.state_to_action(self.state)) * self.move_length 
		# Compute if there is an obstacle or reached the border #
		OBS = self.compute_obstacles(new_position)

		if self.state == 'FORWARD':
			
			if not OBS:
				self.state = 'FORWARD'
			else:

				# Check if with the new direction there is an obstacle #
				new_position = actual_position + self.action_to_vector(self.state_to_action('TURN')) * self.move_length
				OBS = self.compute_obstacles(new_position)

				if not OBS:
					self.state = 'TURN'
				else:
					self.state = 'RECEED'

		elif self.state == 'RECEED':
			# Stay in receed state until there is no obstacle #

			# Check if with the new direction there is an obstacle #
			new_position = actual_position + self.action_to_vector(self.state_to_action('TURN')) * self.move_length
			OBS = self.compute_obstacles(new_position)
			if OBS:
				self.state = 'RECEED'
			else:
				self.state = 'TURN'

		elif self.state == 'TURN':

			if self.turn_count == 1 or OBS:
				self.state = 'REVERSE'
				self.turn_count = 0
			else:
				self.state = 'TURN'
				self.turn_count += 1

		elif self.state == 'REVERSE':

			if not OBS:
				self.state = 'REVERSE'
			else:

				# Check if with the new direction there is an obstacle #
				new_position = actual_position + self.action_to_vector(self.state_to_action('TURN2')) * self.move_length
				OBS = self.compute_obstacles(new_position)

				if not OBS:
					self.state = 'TURN2'
				else:
					self.state = 'RECEED2'

		elif self.state == 'RECEED2':
			# Stay in receed state until there is no obstacle #
			new_position = actual_position + self.action_to_vector(self.state_to_action('TURN2')) * self.move_length
			OBS = self.compute_obstacles(new_position)
			if OBS:
				self.state = 'RECEED2'
			else:
				self.state = 'TURN2'

		elif self.state == 'TURN2':
				
				if self.turn_count == 1 or OBS:
					self.state = 'FORWARD'
					self.turn_count = 0
				else:
					self.state = 'TURN2'
					self.turn_count += 1

		# Compute the new position #
		new_position = actual_position + self.action_to_vector(self.state_to_action(self.state)) * self.move_length 
		# Compute if there is an obstacle or reached the border #
		OBS = self.compute_obstacles(new_position)

		if OBS:
			self.initial_action = self.perpendicular_action(self.initial_action)
			self.state = 'FORWARD'
		
		return self.state_to_action(self.state)
	
	def state_to_action(self, state):

		if state == 'FORWARD':
			return self.initial_action
		elif state == 'TURN':
			return self.perpendicular_action(self.initial_action)
		elif state == 'REVERSE':
			return self.opposite_action(self.initial_action)
		elif state == 'TURN2':
			return self.perpendicular_action(self.initial_action)
		elif state == 'RECEED':
			return self.opposite_action(self.initial_action)
		elif state == 'RECEED2':
			return self.initial_action

	def action_to_vector(self, action):
		""" Transform an action to a vector """

		vectors = np.round(np.array([[np.cos(2*np.pi*i/self.number_of_actions), np.sin(2*np.pi*i/self.number_of_actions)] for i in range(self.number_of_actions)]))

		return vectors[action].astype(int)
	
	def perpendicular_action(self, action):
		""" Compute the perpendicular action """
		return (action - self.number_of_actions//4) % self.number_of_actions
	
	def opposite_action(self, action):
		""" Compute the opposite action """
		return (action + self.number_of_actions//2) % self.number_of_actions
	
	def reset(self, initial_action):
		""" Reset the state of the agent """
		self.state = 'FORWARD'
		self.initial_action = initial_action
		self.turn_count = 0


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
				
	initial_directions = np.random.choice([0,1,2,3,4], size=n_agents, replace=False)

	lawn_mower_agents = [LawnMowerAgent(world = env.scenario_map, number_of_actions = 8, movement_length = 3, forward_direction = initial_directions[i], seed=0) for i in range(n_agents)]

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

			actions = {i: lawn_mower_agents[i].move(env.fleet.vehicles[i].position) for i in range(n_agents)}

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



		
		

