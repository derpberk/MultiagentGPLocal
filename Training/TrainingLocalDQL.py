import sys
sys.path.append('.')
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

scenario_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')


N = 3
D = 7
initial_positions = np.array([[16,8], [20,6], [24,4], [28,9]])[:N]

env = MultiagentInformationGathering(
            scenario_map = scenario_map,
            number_of_agents = N,
            distance_between_locals = D,
            radius_of_locals = np.sqrt(2)*D/2,
            distance_budget = 150,
            distance_between_agents = 1,
            fleet_initial_positions = initial_positions,
            seed = 0,
            movement_length = 2,
            max_collisions = 500,
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

agent.train(10000)
