import sys
sys.path.append('.')
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

scenario_map = np.genfromtxt('Environment/Maps/example_map.csv')


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
			radius_of_locals = np.sqrt(2) * D / 2,
			distance_budget = 100,
			distance_between_agents = 1,
			fleet_initial_zones=fleet_initial_zones,
			fleet_initial_positions=None,
			seed = 42,
			movement_length = 2,
			max_collisions = 5,
			ground_truth_type = 'shekel',
			local = True
)

agent = MultiAgentDuelingDQNAgent(env = env,
			memory_size = 10_000,
			batch_size = 64,
			target_update = 1000,
            number_of_features=512,
            masked_actions=True,
            device='cuda:0')

agent.load_model(r'.\runs\DuelingDQN_shekel_changes_sigma_3_vehicles\FinalPolicy.pth')

res = agent.evaluate_env(1, render = False)

agent.env.render()

print(res)


