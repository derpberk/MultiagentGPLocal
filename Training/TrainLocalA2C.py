import sys
sys.path.append('.')

from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.DRL.Agent.AdvantageActorCritic import A2Cagent
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

agent = A2Cagent(env = env,
	n_steps = 50,
	actor_learning_rate = 0.0001,
	critic_learning_rate = 0.001,
	gamma = 0.99,
	lambda_gae = 0.95,
	ent_coef = 0.01,
	device = f'cuda:{devn}',
	logdir=f'runs/A2C_{GT}_{reward}_{N}_vehicles',
	save_every=1000,
	eval_every=200,
    n_eval_episodes=50)
								

agent.train(10000)
