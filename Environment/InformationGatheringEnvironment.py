import sys
sys.path.append('.')
import numpy as np
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_bloom, algae_colormap
from Environment.GroundTruthsModels.ShekelGroundTruth import shekel
from GPModel.GPmodel import LocalGaussianProcessCoordinator, GlobalGaussianProcessCoordinator
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
import gym
from scipy.spatial import distance_matrix
from Environment.Wrappers.time_stacking_wrapper import MultiAgentTimeStackingMemory
import matplotlib.pyplot as plt


class DiscreteVehicle:

	def __init__(self, initial_position, n_actions, movement_length, navigation_map):
		
		""" Initial positions of the drones """
		np.random.seed(0)
		self.initial_position = initial_position
		self.position = np.copy(initial_position)

		""" Initialize the waypoints """
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)

		""" Detection radius for the contmaination vision """
		self.navigation_map = navigation_map

		""" Reset other variables """
		self.distance = 0.0
		self.num_of_collisions = 0
		self.action_space = gym.spaces.Discrete(n_actions)
		self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
		self.movement_length = movement_length

		

	def move(self, action, valid=True):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		angle = self.angle_set[action]
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int)
		next_position = self.position + movement
		self.distance += np.linalg.norm(self.position - next_position)

		if self.check_collision(next_position) or not valid:
			collide = True
			self.num_of_collisions += 1
		else:
			collide = False
			self.position = next_position
			self.waypoints = np.vstack((self.waypoints, [self.position]))

		return collide

	def check_collision(self, next_position):

		if next_position[0] < 0 or next_position[0] >= self.navigation_map.shape[0] or next_position[1] < 0 or next_position[1] >= self.navigation_map.shape[1]:
			return True

		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
			return True  # There is a collision
		
		return False

	def reset(self, initial_position):
		""" Reset the agent - Position, detection mask, etc. """

		self.initial_position = initial_position
		self.position = np.copy(initial_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.distance = 0.0
		self.num_of_collisions = 0

	def check_action(self, action):
		""" Return True if the action leads to a collision """

		angle = self.angle_set[action]
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int)
		next_position = self.position + movement

		return self.check_collision(next_position)

	def move_to_position(self, goal_position):
		""" Move to the given position """

		assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
		self.distance += np.linalg.norm(goal_position - self.position)
		""" Update the position """
		self.position = goal_position


class DiscreteFleet:

	def __init__(self,
				 number_of_vehicles,
				 n_actions,
				 fleet_initial_positions,
				 movement_length,
				 navigation_map):

		""" Coordinator of the movements of the fleet. Coordinates the common model, the distance between drones, etc. """
		np.random.seed(0)
		self.number_of_vehicles = number_of_vehicles
		self.initial_positions = fleet_initial_positions
		self.n_actions = n_actions
		self.movement_length = movement_length

		""" Create the vehicles object array """
		self.vehicles = [DiscreteVehicle(initial_position=fleet_initial_positions[k],
										 n_actions=n_actions,
										 movement_length=movement_length,
										 navigation_map=navigation_map) for k in range(self.number_of_vehicles)]

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])

		# Reset model variables 
		self.measured_values = None
		self.measured_locations = None

		self.fleet_collisions = 0
		self.danger_of_isolation = None
		self.distance_between_agents = None


	@staticmethod
	def majority(arr: np.ndarray) -> bool:
		return arr.sum() >= len(arr) // 2

	def check_fleet_collision_within(self, veh_actions):
		""" Check if there is any collision between agents """
		
		new_positions = []

		for idx, veh_action in veh_actions.items():

			angle = self.vehicles[idx].angle_set[veh_action]
			movement = np.round(np.array([self.vehicles[idx].movement_length * np.cos(angle), self.vehicles[idx].movement_length * np.sin(angle)])).astype(int)
			new_positions.append(list(self.vehicles[idx].position + movement))

		_, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0)

		# True if repeated #
		not_collision_within = counts[inverse_index] == 1

		return not_collision_within

	def move(self, fleet_actions):

		# Check if there are collisions between vehicles #
		self_colliding_mask = self.check_fleet_collision_within(fleet_actions)
		# Process the fleet actions and move the vehicles #
		collision_array = {k: self.vehicles[k].move(fleet_actions[k], valid=valid) for k, valid in zip(list(fleet_actions.keys()), self_colliding_mask)}
		# Update vector with agent positions #
		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		# Sum up the collisions for termination #
		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])


		return collision_array

	def measure(self, gt_field):

		"""
		Take a measurement in the given N positions
		:param gt_field:
		:return: An numpy array with dims (N,2)
		"""
		positions = np.array([self.vehicles[k].position for k in range(self.number_of_vehicles)])

		values = []
		for pos in positions:
			values.append([gt_field[int(pos[0]), int(pos[1])]])

		if self.measured_locations is None:
			self.measured_locations = positions
			self.measured_values = values
		else:
			self.measured_locations = np.vstack((self.measured_locations, positions))
			self.measured_values = np.vstack((self.measured_values, values))

		return self.measured_values, self.measured_locations

	def reset(self, initial_positions=None):
		""" Reset the fleet """

		if initial_positions is None:
			initial_positions = self.initial_positions

		for k in range(self.number_of_vehicles):
			self.vehicles[k].reset(initial_position=initial_positions[k])

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		self.measured_values = None
		self.measured_locations = None
		self.fleet_collisions = 0
		
	def get_distances(self):
		return np.array([self.vehicles[k].distance for k in range(self.number_of_vehicles)])

	def check_collisions(self, test_actions):
		""" Array of bools (True if collision) """
		return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)]

	def move_fleet_to_positions(self, goal_list):
		""" Move the fleet to the given positions.
		 All goal positions must ve valid. """

		goal_list = np.atleast_2d(goal_list)

		for k in range(self.number_of_vehicles):
			self.vehicles[k].move_to_position(goal_position=goal_list[k])

	def get_distance_matrix(self):
		return distance_matrix(self.agent_positions, self.agent_positions)

	def get_positions(self):

		return np.asarray([veh.position for veh in self.vehicles])
	

class MultiagentInformationGathering:
	
	def __init__(self,
				 scenario_map: np.ndarray,
				 number_of_agents: int,
				 distance_between_locals: float,
				 radius_of_locals: float,
				 distance_budget: float,
				 distance_between_agents: float,
				 fleet_initial_positions = None,
				 fleet_initial_zones = None,
				 seed: int = 0,
				 movement_length: int = 1,
				 max_collisions: int = 1,
				 ground_truth_type: str = 'algae_blooms',
				 frame_stacking = 0,
				 state_index_stacking = (0,1,2,3,4),
				 local = True
				 ):
		
		""" 
		
		:param scenario_map: A numpy array with the scenario map. 1 is a valid position, 0 is an invalid position.
		:param number_of_agents: Number of agents in the fleet
		:param distance_between_locals: Distance between local GPs
		:param radius_of_locals: Radius of the local GPs
		:param distance_budget: Distance budget for the fleet
		:param distance_between_agents: Distance between agents
		:param fleet_initial_positions: Initial positions of the fleet. If None, random positions are chosen.
		:param seed: Seed for the random number generator
		:param movement_length: Length of every movement of the agents
		:param max_collisions: Maximum number of collisions allowed
		:param ground_truth_type: Type of ground truth. 'algae_blooms' or 'water_quality'
		:param frame_stacking: Number of frames to stack
		:param state_index_stacking: Indexes of the state to stack
		:param local: If True, the GP method is local
		
		"""
		
		# Set the seed
		self.seed = seed
		np.random.seed(seed)
		
		# Set the variables
		self.scenario_map = scenario_map
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T
		self.number_of_agents = number_of_agents
		self.distance_budget = distance_budget
		self.distance_between_agents = distance_between_agents
		self.fleet_initial_positions = fleet_initial_positions
		self.movement_length = movement_length
		self.max_collisions = max_collisions
		self.ground_truth_type = ground_truth_type
		self.fleet_initial_zones = fleet_initial_zones

		self.max_steps = self.distance_budget // self.movement_length
		
		# Initial positions
		if fleet_initial_positions is None and fleet_initial_zones is None:
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), number_of_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]
		elif self.fleet_initial_zones is not None:
				self.random_inititial_positions = True
				# Obtain the initial positions as random valid positions inside the zones
				self.initial_positions = np.asarray([region[np.random.randint(0, len(region))] for region in self.fleet_initial_zones])
		else:
			self.initial_positions = fleet_initial_positions
   
		# Create the fleets 
		self.fleet = DiscreteFleet(number_of_vehicles=self.number_of_agents,
								   n_actions=8,
								   fleet_initial_positions=self.initial_positions,
								   movement_length=movement_length,
								   navigation_map=self.scenario_map)

		# Ground truth selection
		if ground_truth_type == 'shekel':
			self.gt = shekel(self.scenario_map, max_number_of_peaks=4, is_bounded=True, seed=self.seed)
		elif ground_truth_type == 'algae_bloom':
			self.gt = algae_bloom(self.scenario_map, seed=self.seed)
		else:
			raise NotImplementedError("This Benchmark is not implemented. Choose one that is.")


		# Set the observation space
		if frame_stacking != 0:
			self.frame_stacking = MultiAgentTimeStackingMemory(n_agents = self.number_of_agents,
			 													n_timesteps = frame_stacking - 1, 
																state_indexes = state_index_stacking, 
																n_channels = 5)
			self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5 + len(state_index_stacking)*(frame_stacking - 1), *self.scenario_map.shape), dtype=np.float32)

		else:
			self.frame_stacking = None
			self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, *self.scenario_map.shape), dtype=np.float32)

		self.state_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, *self.scenario_map.shape), dtype=np.float32)
		self.action_space = gym.spaces.Discrete(8)

		
		# Create a 2D grid of points with a distance of D between them
		self.distance_between_locals = distance_between_locals
		x = np.arange(self.distance_between_locals//2, self.scenario_map.shape[0] - self.distance_between_locals//2, self.distance_between_locals)
		y = np.arange(self.distance_between_locals//2, self.scenario_map.shape[1] - self.distance_between_locals//2, self.distance_between_locals)
		X, Y = np.meshgrid(x, y)
		gp_positions = np.vstack((X.flatten(), Y.flatten())).T
  
		# Select the points that are are in 1 in the map
		gp_positions = gp_positions[self.scenario_map[gp_positions[:,0].astype(int), gp_positions[:,1].astype(int)] == 1]
		self.radius_of_locals = radius_of_locals
  		# Create the GP coordinator	
		self.local = local
		if self.local:
			self.gp_coordinator = LocalGaussianProcessCoordinator(gp_positions = gp_positions,
								kernel = C(1.0)*RBF(length_scale=5.0, length_scale_bounds=(0.5, 10.0)) + W(noise_level=1e-5, noise_level_bounds=(1e-5, 1e-5)),
								scenario_map = self.scenario_map,
								n_restarts_optimizer=0,
								alpha=1e-5,
								distance_threshold=radius_of_locals)
		else:
			self.gp_coordinator = GlobalGaussianProcessCoordinator(
								kernel = C(1.0)*RBF(length_scale=5.0, length_scale_bounds=(0.5, 10.0)) + W(noise_level=1e-5, noise_level_bounds=(1e-5, 1e-5)),
								scenario_map = self.scenario_map,
								n_restarts_optimizer=0,
								alpha=1e-5)
		

		self.fig = None
									

	def reset(self):
		""" Reset the variables of the environment """

		self.steps = 0

		# Reset the ground truth #
		self.gt.reset()

		# Initial positions
		if self.fleet_initial_positions is None and self.fleet_initial_zones is None:
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), self.number_of_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]
		elif self.fleet_initial_zones is not None:
				self.random_inititial_positions = True
				# Obtain the initial positions as random valid positions inside the zones
				self.initial_positions = np.asarray([region[np.random.randint(0, len(region))] for region in self.fleet_initial_zones])
		else:
			self.initial_positions = self.fleet_initial_positions

		self.fleet.reset(initial_positions=self.initial_positions)

		# Reset the GP coordinator #
		self.gp_coordinator.reset()

		# Take measurements #
		self.measurements = self.gt.read(self.fleet.agent_positions).reshape(-1,1)	

		# Update the GP coordinator #
		self.gp_coordinator.update(self.fleet.agent_positions, self.measurements)

		# Update the state of the agents #
		self.update_state()

		if self.fig is not None:
			plt.close(self.fig)
			self.fig = None

		# Return the state #
		return self.state if self.frame_stacking is None else self.frame_stacking.process(self.state)
	
	def step(self, actions):
		""" Take a step in the environment """

		self.steps += 1

		# Process action movement only for active agents #
		collision_mask = self.fleet.move(actions)
		
		# Collision mask to list 
		collision_mask = np.array(list(collision_mask.values()))

		if any(collision_mask):
			print('COLLISION!')

		# Take measurements #
		self.measurements = self.gt.read(self.fleet.agent_positions).reshape(-1,1)

		# Update the GP coordinator in those places without collision #
		self.gp_coordinator.update(self.fleet.agent_positions[np.logical_not(collision_mask)], self.measurements[np.logical_not(collision_mask)])

		# Compute the reward #
		reward = self.compute_reward(collisions=collision_mask)

		# Update the state of the agents #
		self.update_state()

		# Check if the episode is done #
		done = self.check_done()
		

		# Return the state #
		return self.state if self.frame_stacking is None else self.frame_stacking.process(self.state), reward, done, {}

	def update_state(self):
		""" Update the state of the environment """

		state = {}

		# State 0 -> Mu map
		mu_map = self.gp_coordinator.mu_map
		# State 1 -> Sigma map
		sigma_map = self.gp_coordinator.sigma_map
		# State 2 -> Agent 

		# Create fleet position #
		fleet_position_map = np.zeros_like(self.scenario_map)
		fleet_position_map[self.fleet.agent_positions[:,0], self.fleet.agent_positions[:,1]] = 1.0

		# State 3 and 4
		for i in range(self.number_of_agents):
			
			agent_observation_of_fleet = fleet_position_map.copy()
			agent_observation_of_fleet[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 0.0

			agent_observation_of_position = np.zeros_like(self.scenario_map)
			agent_observation_of_position[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 1.0
			
			state[i] = np.concatenate((
				np.clip(mu_map[np.newaxis], 0, 1),
				sigma_map[np.newaxis],
				agent_observation_of_fleet[np.newaxis],
				agent_observation_of_position[np.newaxis],
				self.scenario_map[np.newaxis].copy()
			))

		self.state = state

	def compute_reward(self, collisions):
		""" Compute the reward of the environment """

		# 1) obtain the index of the local GP of every agent
		agent_local_gp_index = [self.gp_coordinator.get_local_gp_indexes(pos) for pos in self.fleet.agent_positions]

		# 2) obtain the changes in those local_gps
		changes = [np.sum(self.gp_coordinator.get_local_gp_changes(indexes)) for indexes in agent_local_gp_index]

		# 3) Compute the distance between agents
		distance_between_agents = distance_matrix(self.fleet.agent_positions, self.fleet.agent_positions) # Compute the distance matrix
		distance_between_agents[distance_between_agents <= 1] = 1.0 # Clip the min to 1.0 
		distance_between_agents[distance_between_agents > self.radius_of_locals] = np.inf # If the distance is greater than the radius of the local gp, set it to infinity
		np.fill_diagonal(distance_between_agents, 1.0) # Set the diagonal to 1.0 to simplify the computation of the redundancy
		distance_between_agents = 1.0 / distance_between_agents # Compute the inverse of the distance
		redundancy = np.sum(distance_between_agents, axis=1) # Compute the redundancy of each agent

		# 4) Compute the reward
		if self.local:
			reward = {agent_id: changes[agent_id] / redundancy[agent_id] for agent_id in range(self.number_of_agents)}
		else:
			reward = {agent_id: changes for agent_id in range(self.number_of_agents)}

		# 5) Add a penalty for collisions
		for agent_id in range(self.number_of_agents):
			if collisions[agent_id] == 1:
				reward[agent_id] = -1

		return reward

	def check_done(self):

		# Check if the episode is done #
		done = {agent_id: False for agent_id in range(self.number_of_agents)}

		# Check if the episode is done #
		"""
		if self.fleet.fleet_collisions > self.max_collisions or any(self.fleet.get_distances() >= self.distance_budget):
			done = {agent_id: True for agent_id in range(self.number_of_agents)}
		"""

		if self.fleet.fleet_collisions > self.max_collisions or self.steps >= self.max_steps:
			done = {agent_id: True for agent_id in range(self.number_of_agents)}


		return done
		
	def render(self):

		import matplotlib.pyplot as plt


		if self.fig is None:

			self.fig, self.axs = plt.subplots(1, 6, figsize=(15,5))
			
			# Print the Mu
			# Create a background for unknown places #
			unknown = np.zeros_like(self.scenario_map) + 0.25
			unknown[1::2, ::2] = 0.5
			unknown[::2, 1::2] = 0.5
			self.axs[1].imshow(unknown, cmap='gray', vmin=0, vmax=1)
			self.im0 = self.axs[1].imshow(self.state[0][0], cmap = algae_colormap, vmin=0, vmax=1)
			self.axs[1].set_title(r'GP $\mu$ ')
			# Add the params to the plot
			params = self.gp_coordinator.get_kernel_params()
			# Plot the text in ever gp_position #
			self.text = []
			for i in range(len(params)):
				self.text.append(self.axs[1].text(self.gp_coordinator.gp_positions[i,1], self.gp_coordinator.gp_positions[i,0], '{:.1f}'.format(float(params[i])), color='black', fontsize=8))
			
	
			# Print the Std
			self.im1 = self.axs[2].imshow(self.state[0][1], cmap = 'gray_r')
			self.axs[2].set_title(r'GP $\sigma$ ')
			# Print the Fleet
			self.im2 = self.axs[3].imshow(self.state[0][2], cmap = 'gray', vmin=0, vmax=1)
			self.axs[3].set_title(r'Fleet')
			# Print the Agent
			self.im3 = self.axs[4].imshow(self.state[0][3], cmap = 'gray', vmin=0, vmax=1)
			self.axs[4].set_title(r'Agent')
			# Print the Map
			self.im5 = self.axs[5].imshow(self.state[0][4], cmap = 'gray', vmin=0, vmax=1)
			self.axs[5].set_title(r'Map')
			# Print the gts
			self.axs[0].imshow(self.scenario_map * self.gt.read(), cmap = algae_colormap, vmin=0, vmax=1, interpolation='bilinear')
			try:
				self.axs[0].plot(self.gp_coordinator.gp_positions[:,1], self.gp_coordinator.gp_positions[:,0], 'k.', markersize=5)
			except:
				pass
			
			self.axs[0].set_title(r'Ground Truth')
			self.im4 = self.axs[0].plot(self.gp_coordinator.x[:,1], self.gp_coordinator.x[:,0], 'gx', markersize=5)
			# Scatterplot with the position of the agents. Every agent position is in a different color
			colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'olive']
			self.impos = self.axs[5].plot(self.fleet.agent_positions[:,1], self.fleet.agent_positions[:,0], 'o', c=np.linspace(0, 1, self.number_of_agents), markersize=5)

		else:
			
			self.im0.set_data(self.state[0][0])
			self.im1.set_data(self.state[0][1])
			self.im2.set_data(self.state[0][2])
			self.im3.set_data(self.state[0][3])
			self.im5.set_data(self.state[0][4])
			# Set the new positions of the agents
			self.im4[0].set_xdata(self.gp_coordinator.x[:,1])
			self.im4[0].set_ydata(self.gp_coordinator.x[:,0])
			params = self.gp_coordinator.get_kernel_params()
			for i in range(len(params)):
				self.text[i].set_text("{:.1f}".format(params[i]))
			self.impos[0].set_data(self.fleet.agent_positions[:,1], self.fleet.agent_positions[:,0])
			

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

		plt.draw()

		plt.pause(0.01)

	def get_error(self):
		""" Compute the MSE error """
		
		# Compute the error #
		error = np.sum((self.gt.read() - self.gp_coordinator.mu_map)**2)

		return error
		
		
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
				radius_of_locals = D*2/3,
				distance_budget = 170,
				distance_between_agents = 1,
				fleet_initial_zones=fleet_initial_zones,
				fleet_initial_positions=None,
				seed = seed,
				movement_length = 2,
				max_collisions = 5,
				ground_truth_type = 'algae_bloom',
				local = True
	)

	lawn_mower_agents = [LawnMowerAgent(world = scenario_map, number_of_actions = 8, movement_length = 2, forward_direction = np.random.choice([1,3,5]), seed=seed) for _ in range(N)]
	random_wandering_agents = [WanderingAgent(world = scenario_map, number_of_actions = 8, movement_length = 2, seed=seed) for _ in range(N)]

	np.random.seed(seed)
	env.reset()

	done = {i:False for i in range(N)}
	R = []
	ERROR = []
	UNCERTAINTY = []

	t = 0

	runtime = 0
	while not any(list(done.values())):
		
		action = {i: lawn_mower_agents[i].move(env.fleet.vehicles[i].position) for i in range(N)}

		t0 = time.time()
		s, r, done, _ = env.step(action)
		t1 = time.time()

		runtime += t1-t0
		env.render()

		R.append(np.sum(list(r.values())))
		ERROR.append(env.get_error())
		UNCERTAINTY.append(env.gp_coordinator.sigma_map.sum())

		print(r)
		t+=1

	R = np.array(R)
	Racc = np.cumsum(R)
	
	env.render()
	plt.show()

	print('Total runtime: ', runtime)

	# Plot the accumulated reward and the error, in the same plot different y axis
	fig, ax1 = plt.subplots()
	ax1.plot(Racc, 'b-')
	ax1.set_xlabel('Time')
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('Accumulated Reward', color='b')
	ax1.tick_params('y', colors='b')

	ax2 = ax1.twinx()
	ax2.plot(ERROR, 'r-')
	ax2.set_ylabel('Error', color='r')
	ax2.set_ylim([0, ERROR[0]+5])
	ax2.tick_params('y', colors='r')

	plt.grid()
	plt.show()

	# Scatter plot with x = Reward and y = Error
	plt.scatter(Racc, ERROR)
	plt.xlabel('Reward')
	plt.ylabel('Error')
	plt.grid()
	plt.show()

	# Plot the accumulated reward and the error, in the same plot different y axis
	fig, ax1 = plt.subplots()
	ax1.plot(Racc, 'b-')
	ax1.set_xlabel('Time')
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('Accumulated Reward', color='b')
	ax1.tick_params('y', colors='b')

	ax2 = ax1.twinx()
	ax2.plot(UNCERTAINTY, 'r-')
	ax2.set_ylabel('Uncertainty', color='r')
	ax2.set_ylim([0, UNCERTAINTY[0]+5])
	ax2.tick_params('y', colors='r')

	plt.grid()
	plt.show()

	# Scatter plot with x = Reward and y = Error
	plt.scatter(Racc, UNCERTAINTY)
	plt.xlabel('Reward')
	plt.ylabel('Uncertainty')
	plt.grid()
	plt.show()


