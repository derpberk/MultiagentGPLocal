import sys
sys.path.append('.')
import numpy as np
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
import argparse
from tqdm import trange
from EvaluationUtils import run_evaluation


scenario_map = np.genfromtxt('Environment/Maps/example_map.csv')
D = 7
# Generate initial positions with squares of size 3 x 3 around positions
center_initial_zones = np.array([[17,9], [22,8], [28,9]]) 
# 9 positions in the sorrounding of the center
area_initial_zones = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]])
# Generate the initial positions with the sum of the center and the area
fleet_initial_zones = np.array([area_initial_zones + center_initial_zones[i] for i in range(len(center_initial_zones))])

N_agents = 3
N_EPISODES = 1
reward_type = 'changes_sigma'
ground_truth_type = 'algae_bloom'

PATH = f'runs/DuelingDQN_{ground_truth_type}_{reward_type}_{N_agents}_vehicles/FinalPolicy.pth'

env = MultiagentInformationGathering(
			scenario_map = scenario_map,
			number_of_agents = N_agents,
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

run_evaluation(path='Evaluation/PathsAndDemostrations/', 
				agent=agent,
				algorithm='Dueling DDQN',
				reward_type=reward_type,
				ground_truth_type=ground_truth_type,
				runs=N_EPISODES,
				n_agents=N_agents,
				render=False)


np.save(f'Evaluation/PathsAndDemostrations/gt_{reward_type}_{ground_truth_type}_seed_42.npy', agent.env.gt.read())
np.save(f'Evaluation/PathsAndDemostrations/mu_{reward_type}_{ground_truth_type}_seed_42.npy', agent.env.gp_coordinator.mu_map)

points = np.asarray([veh.waypoints for veh in agent.env.fleet.vehicles])
np.save(f'Evaluation/PathsAndDemostrations/points_{reward_type}_{ground_truth_type}_seed_42.npy', points)


print("Done")