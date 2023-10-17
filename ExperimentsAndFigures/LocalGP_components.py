import sys
sys.path.append('.')
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.NRRA import WanderingAgent
from Algorithms.LawnMower import LawnMowerAgent
import numpy as np
import matplotlib.pyplot as plt
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap, background_colormap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import trange
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor 


scenario_map = np.genfromtxt('Environment/Maps/example_map.csv')
N = 3
D = 7
# Generate initial positions with squares of size 3 x 3 around positions
center_initial_zones = np.array([[17,9], [22,8], [28,9]]) 
# 9 positions in the sorrounding of the center
area_initial_zones = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]])
# Generate the initial positions with the sum of the center and the area
fleet_initial_zones = np.array([area_initial_zones + center_initial_zones[i] for i in range(len(center_initial_zones))])


RUN = 0
COMPUTE = False
SAVE = True
rows = []
ALGORITHMS = ['NRRA', 'LawnMower']
RUNS = 50



if COMPUTE: 

	for method, distance_between_locals, radius_of_locals in zip(['Local GP', 'Global GP', 'Decision Tree', 'KNN'], [D,30,30,30],[np.sqrt(2) * D / 2, 10000, 10000,10000]):
		
		if method == "KNN":
			model = KNeighborsRegressor(n_neighbors=5)
		elif method == "Decision Tree":
			model = DecisionTreeRegressor(max_depth=10)

		for algorithm in ALGORITHMS:

			np.random.seed(0)
			
			env = MultiagentInformationGathering(
					scenario_map = scenario_map,
					number_of_agents = N,
					distance_between_locals = distance_between_locals,
					radius_of_locals = radius_of_locals,
					distance_budget = 100,
					distance_between_agents = 1,
					fleet_initial_zones=fleet_initial_zones,
					fleet_initial_positions=None,
					seed = 0,
					movement_length = 2,
					max_collisions = 50000,
					ground_truth_type = 'algae_bloom',
					local = True
				)

			for RUN in trange(RUNS):

				env.reset()

				if algorithm == 'NRRA':
					# Wandering
					agents = [WanderingAgent(world = scenario_map, number_of_actions = 8, movement_length = 2, seed=100) for _ in range(N)]
				elif algorithm == 'LawnMower':
					# Lawn Mower
					agents =  [LawnMowerAgent(world = scenario_map, number_of_actions = 8, movement_length = 3, forward_direction = np.random.choice([1,2,3,4,5,6,7]), seed=100) for _ in range(N)]

				done = False
				Racc = 0
				STEP = 0

				while not done:

					STEP += 1

					t0 = time.time()
					action = {i: agents[i].move(env.fleet.get_positions()[i]) for i in range(N)}
					_, reward, done, _ = env.step(action)
					t1 = time.time()
					done = all(done.values())

					Racc += np.sum(list(reward.values()))

					# Plot the results

					if method not in ["Decision Tree", "KNN"]:
						rows.append([algorithm, RUN, STEP, Racc, env.get_error(), method, t1-t0, env.gt.read().flatten(), env.gp_coordinator.mu_map.flatten(), env.gp_coordinator.sigma_map.flatten(), env.fleet.get_positions().flatten()])
					else:
						t0 = time.time()
						Y_map = np.zeros_like(scenario_map)
						model.fit(env.gp_coordinator.x, env.gp_coordinator.y.flatten())
						Y_pred = model.predict(env.gp_coordinator.X)
						Y_map[env.gp_coordinator.X[:,0], env.gp_coordinator.X[:,1]] = Y_pred
						error = np.sum(np.abs((env.gt.read() - Y_map)))
						t1 = time.time()
						rows.append([algorithm, RUN, STEP, Racc, error, method, t1-t0, env.gt.read().flatten(), env.gp_coordinator.mu_map.flatten(), env.gp_coordinator.sigma_map.flatten(), env.fleet.get_positions().flatten()])


	df = pd.DataFrame(rows, columns=['Algorithm', 'Run', 'Step', 'Rewards', 'Error', 'GP method', 'Execution time', ' GT', 'Mu', 'Sigma', 'Agent positions'])

	if SAVE:
		df.to_pickle('ExperimentsAndFigures/LocalVSGlobal.pkl')
else:
	df = pd.read_pickle('ExperimentsAndFigures/LocalVSGlobal.pkl')

sns.set_theme(style="darkgrid")

df["Accumulated Execution time"] = df.groupby(["Algorithm", "Run", "GP method"])["Execution time"].cumsum()


sns.lineplot(x="Step", y="Accumulated Execution time", hue="GP method", data=df, lw=2)
plt.ylabel('Accumulated Execution time (s)')

plt.show()

# Plot linear regression
#sns.regplot(x="Step", y="Accumulated Execution time", data=df[df['GP method'] == 'Local GP'], scatter=False, color='black', label='Linear Regression', line_kws={'linestyle':'--', 'alpha':0.5, 'linewidth':1})
# Plot cubic regression
#sns.regplot(x="Step", y="Accumulated Execution time", data=df[df['GP method'] == 'Global GP'], order=3, scatter=False, color='black', label='Cubic regression', line_kws={'linestyle':'-.', 'alpha':0.5, 'linewidth': 1})

plt.legend(loc='upper right')
plt.show()

