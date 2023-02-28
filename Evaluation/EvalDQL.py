import sys
sys.path.append('.')
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
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
			distance_budget = 170,
			distance_between_agents = 5,
			fleet_initial_zones=fleet_initial_zones,
			fleet_initial_positions=None,
			seed = 0,
			movement_length = 2,
			max_collisions = 5,
			ground_truth_type = 'algae_bloom',
			local = True
)

agent = MultiAgentDuelingDQNAgent(env = env,
			memory_size = 100_000,
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
			logdir='runs/DuelingDQN',
			log_name="DQL",
			save_every=None,
			train_every=15,
			masked_actions= True,
			device='cuda:0',
			seed = 0,
			eval_every = 200,
			eval_episodes = 20)

agent.load_model('Evaluation/BestPolicy.pth')

res = agent.evaluate_env(5, render = True)

print(res)


