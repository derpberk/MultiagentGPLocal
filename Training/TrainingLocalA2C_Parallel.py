import sys
sys.path.append('.')
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.DRL.Agent.ParallelAdvantageActorCritic import AsyncronousActorCritic
import numpy as np



if __name__ == '__main__':

    scenario_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')


    N = 3
    D = 7
    initial_positions = np.array([[16,8], [20,6], [24,4], [28,9]])[:N]

    envs = [MultiagentInformationGathering(
                scenario_map = scenario_map,
                number_of_agents = N,
                distance_between_locals = D,
                radius_of_locals = np.sqrt(2)*D/2,
                distance_budget = 150,
                distance_between_agents = 1,
                fleet_initial_positions = initial_positions,
                seed = i,
                movement_length = 2,
                max_collisions = 500,
                ground_truth_type = 'algae_bloom',
                local = True) 
    for i in range(3)
    ]

    agent = AsyncronousActorCritic(envs = envs,
                                n_steps = 75,
                                actor_learning_rate = 0.0001,
                                critic_learning_rate = 0.0001,
                                gamma = 0.99,
                                lambda_gae = 0.95,
                                ent_coef = 0.01,
                                device = 'cpu')
                            

    agent.train(10000)
