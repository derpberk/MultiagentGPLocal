import sys
sys.path.append('.')
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.NRRA import WanderingAgent
from Algorithms.LawnMower import LawnMowerAgent
import numpy as np
import matplotlib.pyplot as plt
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap, background_colormap

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
			distance_budget = 150,
			distance_between_agents = 1,
			fleet_initial_zones=fleet_initial_zones,
			fleet_initial_positions=None,
			seed = 0,
			movement_length = 2,
			max_collisions = 50000,
			ground_truth_type = 'algae_bloom',
			local = True
)


env.reset()

# Wandering
random_wandering_agents = [WanderingAgent(world = scenario_map, number_of_actions = 8, movement_length = 2, seed=100) for _ in range(N)]
done = False
rewards = []

while not done:

	action = {i: random_wandering_agents[i].move(env.fleet.get_positions()[i]) for i in range(N)}
	_, reward, done, _ = env.step(action)
	rewards.append(np.sum(list(reward.values())))

	done = all(done.values())


# Show the gt and the mu_map #
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

back = scenario_map.copy()
ax[0].imshow(back, cmap=background_colormap, zorder=5, interpolation='nearest', alpha=1-back)
ax[0].imshow(env.gt.read(), cmap=algae_colormap, interpolation='bilinear')
ax[0].set_title('Ground Truth')
params = env.gp_coordinator.get_kernel_params()
for i in range(len(params)):
	ax[0].text(env.gp_coordinator.gp_positions[i,1], env.gp_coordinator.gp_positions[i,0], '{:.1f}'.format(float(params[i])), fontsize=params[i]/10 * 4 + 7, zorder=10, ha='center', va='center')
	# Plot a circle around the GP #
	ax[0].add_patch(plt.Circle((env.gp_coordinator.gp_positions[i,1], env.gp_coordinator.gp_positions[i,0]), radius=np.round(env.gp_coordinator.distance_threshold)+1, fill=True, zorder=10, facecolor='white', edgecolor='blue', alpha=0.1))


ax[1].imshow(back, cmap=background_colormap, zorder=5, alpha=1-back)
ax[1].set_title(r'GP $\mu(x)$')


# Set a colorbar #
cbar = fig.colorbar(ax[1].imshow(env.gp_coordinator.mu_map, cmap=algae_colormap), ax=ax[1])
cbar.set_label('Contamination level')

# plot the uncertainty #
ax[2].imshow(back, cmap=background_colormap, zorder=5, interpolation='nearest', alpha=1-back)
cbar = fig.colorbar(ax[2].imshow(env.gp_coordinator.sigma_map, cmap='gray'), ax=ax[2])
ax[2].set_title(r'GP $\sigma(x)$')
# Set a colorbar #
cbar.set_label('Predictive uncertainty')

# Plot the waypoints of every vehicle in the fleet in a different color #
colors = ['blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, vehicle in enumerate(env.fleet.vehicles):
	ax[1].plot(vehicle.waypoints[:, 1], vehicle.waypoints[:, 0], '.-', color=colors[i], label='Vehicle {}'.format(i))

# Set legent 
ax[1].legend()

plt.tight_layout()
plt.show()




