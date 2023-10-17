
import sys
sys.path.append('.')
import numpy as np
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
import argparse
from tqdm import trange
from Algorithms.NRRA import run_evaluation




scenario_map = np.genfromtxt('Environment/Maps/example_map.csv')
D = 7
# Generate initial positions with squares of size 3 x 3 around positions
center_initial_zones = np.array([[17,9], [22,8], [28,9]]) 
# 9 positions in the sorrounding of the center
area_initial_zones = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]])
# Generate the initial positions with the sum of the center and the area
fleet_initial_zones = np.array([area_initial_zones + center_initial_zones[i] for i in range(len(center_initial_zones))])


for gt in ['algae_bloom', 'shekel']:
    

        for N_agents in [3,2,1]:


            N_EPISODES = 300
            reward_type = 'changes_mu'
            ground_truth_type = gt
            
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
                        max_collisions = 10000,
                        ground_truth_type = ground_truth_type,
                        local = True,
                        reward_type = reward_type,
                        collitions_within = False,
            )

            run_evaluation(path=f'Evaluation/EvaluationRuns_REV1/', 
                            env=env, 
                            algorithm = 'RandomWandering',
                            runs = N_EPISODES, 
                            n_agents = N_agents, 
                            ground_truth_type = ground_truth_type, 
                            render = False)





