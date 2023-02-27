import sys
sys.path.append('.')
from GPModel.GPmodel import LocalGaussianProcessCoordinator
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel as W
import matplotlib.pyplot as plt
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_bloom
from Environment.GroundTruthsModels.ShekelGroundTruth import shekel
import numpy as np
import time
from tqdm import trange


# Create a search space 
navigation_map = np.ones((30, 30))
# Create a GT #
#gt = algae_bloom(navigation_map, dt=0.05)
gt = shekel(navigation_map, dt=0.05, seed=5456)
gt2 = algae_bloom(navigation_map, dt=0.1, seed=42136)
gt.reset()
gt2.reset()

for _ in range(50):
    gt.step()

gt_field = gt.read() + gt2.read()
gt_field = gt_field / np.max(gt_field)
# Obtain the valid position of the search space, i.e the positions that makes navigation_map == 1
valid_positions = np.argwhere(navigation_map == 1)
# Create a grid of N equidistant coordinated in navigation_map #
N = 5
gp_positions = np.array(np.meshgrid(np.linspace(N//4, navigation_map.shape[0]-N//4, N), np.linspace(N//4, navigation_map.shape[1]-5, N))).T.reshape(-1, 2)

# Create the GP coordinator #
kernel = RBF(length_scale=5, length_scale_bounds=(0.5, 10))
gp_coordinator = LocalGaussianProcessCoordinator(gp_positions, navigation_map, kernel, alpha=1e-5, n_restarts_optimizer=0, distance_threshold=10)

# Create a 1x2 figure to represent the mu and the std #
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].set_title('mu')
ax[1].set_title('std')
ax[2].set_title('GT')
mu_fig = ax[0].imshow(np.zeros(navigation_map.shape), vmin=0, vmax=1)
std_fig = ax[1].imshow(np.zeros(navigation_map.shape), vmin=0, vmax=1)
ax[2].imshow(gt_field, vmin=0, vmax=1)
sampled_points_fig = ax[0].scatter([], [], c='r', s=10)
sampled_points_fig2 = ax[2].scatter([], [], c='r', s=10)
ax[0].scatter(gp_positions[:, 1], gp_positions[:, 0], c='g', s=10)

# Draw circles around the initial positions #

# update the figure #
fig.canvas.draw()
plt.pause(0.1)
T = []
position = np.array([5, 5])
S = 2

def onclick(event):
    """ Key handler """

    key_pressed = event.key
    # Obtain the key pressed. Save only the AWSD #
    if key_pressed == '8':
        position[0] -= S
    elif key_pressed == '2':
        position[0] += S
    elif key_pressed == '4':
        position[1] -= S
    elif key_pressed == '6':
        position[1] += S
    elif key_pressed == '3':
        position[0] += S
        position[1] += S
    elif key_pressed == '7':
        position[0] -= S
        position[1] -= S
    elif key_pressed == '1':
        position[0] += S
        position[1] -= S
    elif key_pressed == '9':
        position[0] -= S
        position[1] += S
    else:
        return
    
    # Get the position of the click and transform to pixel position#
    new_position = position
    new_y = gt_field[int(new_position[0]), int(new_position[1])]
    
    t0 = time.time()
    mu, std = gp_coordinator.update(new_position, new_y)
    T.append(time.time() - t0)
    print('Time to update: ', T[-1])
    
    # Update the figure #
    mu_fig.set_data(mu)
    std_fig.set_data(std)
    # interchange the x and y coordinates to match the image coordinates #
    
    sampled_places = np.array([gp_coordinator.x[:, 1], gp_coordinator.x[:, 0]]).T
    sampled_points_fig.set_offsets(sampled_places)
    sampled_points_fig2.set_offsets(sampled_places)
    
    
    fig.canvas.draw()
    
def on_key():
    """ Key handler """
    T = []
    for t in trange(4*75):
        
        # Get the position of the click and transform to pixel position#
        new_position = np.random.normal(loc = gp_positions[0], scale = 5, size = 2)
        new_y = gt_field[int(new_position[0]), int(new_position[1])]

        
        t0 = time.time()
        mu, std = gp_coordinator.update(new_position, new_y)
        T.append(time.time() - t0)
        print('Time to update: ', T[-1])
        
        # Update the figure #
        mu_fig.set_data(mu)
        std_fig.set_data(std)
        # interchange the x and y coordinates to match the image coordinates #
        
        sampled_places = np.array([gp_coordinator.x[:, 1], gp_coordinator.x[:, 0]]).T
        sampled_points_fig.set_offsets(sampled_places)
        sampled_points_fig2.set_offsets(sampled_places)

        
        #fig.canvas.draw()
    return T
    
cid = fig.canvas.mpl_connect('key_press_event', onclick)
plt.show()


plt.figure()
plt.plot(np.arange(0,len(T)), T, 'r-', label='GP')
plt.legend()
plt.show()
print("Total time: ", np.sum(T))

"""
T_classic = on_key()
N = 5
gp_positions = np.array(np.meshgrid(np.linspace(N, navigation_map.shape[0]-N, N), np.linspace(N, navigation_map.shape[1]-N, N))).T.reshape(-1, 2)
#gp_positions = np.array([[25, 25]])

# Create the GP coordinator #
kernel = C(1.0, constant_value_bounds=(1,1)) * RBF(length_scale=5.0, length_scale_bounds=(0.5, 10))
gp_coordinator = LocalGaussianProcessCoordinator(gp_positions, valid_positions, kernel, alpha=1e-10, n_restarts_optimizer=0, map_shape=navigation_map.shape)
T_new = on_key()


plt.show()
plt.close()

plt.plot(np.arange(0,len(T_classic)), T_classic, 'r-', label='GP')
plt.plot(np.arange(0,len(T_new)), T_new, 'b-', label='GP_dist')
plt.grid()
plt.ylabel('Time (s)')
plt.show()

print("Total time classic: ", np.sum(T_classic))
print("Total time new: ", np.sum(T_new))



"""