import sys
sys.path.append('.')

from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, required=True)
parser.add_argument('--R', type=str, required=True)
parser.add_argument('--GT', type=str, required=True)
parser.add_argument('--GPU', type=int, required=True)
args = parser.parse_args()

scenario_map = np.genfromtxt('Environment/Maps/example_map.csv')


N = args.N
reward = args.R
GT = args.GT
devn = args.GPU

print(N,reward,GT)

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
			radius_of_locals = np.sqrt(2) * D / 2,
			distance_budget = 100,
			distance_between_agents = 1,
			fleet_initial_zones=fleet_initial_zones,
			fleet_initial_positions=None,
			seed = 0,
			movement_length = 2,
			max_collisions = 5,
			ground_truth_type = GT,
			local = True,
			reward_type=reward
)

agent = MultiAgentDuelingDQNAgent(env = env,
			memory_size = 500_000,
			batch_size = 64,
			target_update = 1000,
			soft_update = True,
			tau = 0.001,
			epsilon_values = [1.0, 0.05],
			epsilon_interval = [0.0, 0.5],
			learning_starts = 100,
			gamma = 0.99,
			lr = 1e-4,
			# NN parameters
			number_of_features = 512,
			logdir=f'runs/DuelingDQN_{GT}_{reward}_{N}_vehicles',
			log_name="DQL",
			save_every=1000,
			train_every=10,
			masked_actions= True,
			device=f'cuda:{devn}',
			seed = 0,
			eval_every = 200,
			eval_episodes = 50,)

agent.train(10000)
