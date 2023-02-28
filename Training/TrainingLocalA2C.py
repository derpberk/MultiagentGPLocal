import sys
sys.path.append('.')
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.DRL.Agent.AdvanteActorCritic import A2CAgent
import numpy as np

scenario_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')

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
			distance_budget = 120,
			distance_between_agents = 1,
			fleet_initial_zones=fleet_initial_zones,
			fleet_initial_positions=None,
			seed = 0,
			movement_length = 2,
			max_collisions = 5,
			ground_truth_type = 'algae_bloom',
			local = True
)

agent = A2CAgent(env = env,
				n_steps = None,
				learning_rate = 1e-4,
				gamma = 0.99,
				logdir = 'runs/A2C',
				log_name = 'A2C',
				save_every = 1000,
				device = 'cuda:1'
)

agent.train(10000)
