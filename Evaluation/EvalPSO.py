
import sys
sys.path.append('.')
import numpy as np
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
import argparse
from tqdm import trange
from Algorithms.ParticleSwarm import ParticleSwarmOptimizer, run_evaluation
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_bloom
from Environment.GroundTruthsModels.ShekelGroundTruth import shekel

scenario_map = np.genfromtxt('Environment/Maps/example_map.csv')
D = 7
# Generate initial positions with squares of size 3 x 3 around positions
center_initial_zones = np.array([[17,9], [22,8], [28,9]]) 
seed = 0

for gt_name in ['algae_bloom', 'shekel']:

	if gt_name == 'shekel':
			ground_truth = shekel(scenario_map, max_number_of_peaks=4, is_bounded=True, seed=seed)
	elif gt_name == 'algae_bloom':
			ground_truth = algae_bloom(scenario_map, seed=seed)
	
	for N_agents in range(1, 4):


		N_EPISODES = 20
		reward_type = 'changes_mu'
		ground_truth_type = gt_name

		agent = ParticleSwarmOptimizer(n_agents = N_agents, 
				 navigation_map = scenario_map, 
				 ground_truth = ground_truth, 
				 max_distance=100, 
				 initial_positions= center_initial_zones[:N_agents], 
				 parameters=(0.5, 0.5, 0.5, 0.5, 0.5))

		run_evaluation(path=f'Evaluation/EvaluationRuns/', 
						agent=agent, 
						algorithm = 'ParticleSwarm',
						runs = N_EPISODES, 
						n_agents = N_agents, 
						ground_truth_type = ground_truth_type, 
						render = False)