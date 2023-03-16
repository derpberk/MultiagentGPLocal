import sys
sys.path.append('.')
import numpy as np
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
import argparse
from tqdm import trange
from EvaluationUtils import run_evaluation


""" Create the parser to parse N_episodes, N_agents, and the path to the model """
parser = argparse.ArgumentParser()
parser.add_argument('--N_episodes', type=int, default=2, help='Number of episodes to evaluate the policy')
parser.add_argument('--N_agents', type=int, default=3, help='Number of agents in the environment')
parser.add_argument('--path', type=str, default='Evaluation/FinalPolicy_algae.pth', help='Path to the model')
parser.add_argument('--reward', type=str, default='changes_mu', help='Type of the reward', choices=['changes_mu', 'changes_sigma'])
parser.add_argument('--gt', type=str, default='algae_bloom', help='Ground truth str')


N_EPISODES = parser.parse_args().N_episodes
N_AGENTS = parser.parse_args().N_agents
PATH = parser.parse_args().path
reward_type = parser.parse_args().reward
ground_truth_type = parser.parse_args().gt
scenario_map = np.genfromtxt('Environment/Maps/example_map.csv')


N = N_AGENTS
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
			seed = 42,
			movement_length = 2,
			max_collisions = 5,
			ground_truth_type = ground_truth_type,
			local = True,
			reward_type = reward_type
)

agent = MultiAgentDuelingDQNAgent(env = env,
			memory_size = 1_000,
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
			logdir='runs/None',
			log_name="DQL",
			save_every=1000,
			train_every=10,
			masked_actions= True,
			device='cuda:0',
			seed = 0,
			eval_every = 200,
			eval_episodes = 50,)


agent.load_model(PATH)


run_evaluation(path='Evaluation/EvaluationRuns/', 
				agent=agent,
				algorithm='DDQN',
				reward_type=reward_type,
				ground_truth_type=ground_truth_type,
				runs=N_EPISODES,
				n_agents=N_AGENTS,
				render=False)



