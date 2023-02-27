import sys
sys.path.append('.')
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.DRL.Agent.AdvanteActorCritic import A2CAgent
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

agent = A2CAgent(env = env,
				n_steps = None,
				learning_rate = 1e-4,
				gamma = 0.99,
				logdir = 'runs/A2C',
				log_name = 'A2C',
				save_every = 1000,
				device = 'cuda:0'
)

agent.load_model('runs/A2C/best_model_at_episode_793.pth')

res = agent.evaluate(5, render = True)

print(res)
