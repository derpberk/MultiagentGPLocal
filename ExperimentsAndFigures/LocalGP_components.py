import sys
sys.path.append('.')
from Environment.InformationGatheringEnvironment import MultiagentInformationGathering
from Algorithms.NRRA import WanderingAgent
from Algorithms.LawnMower import LawnMowerAgent
import numpy as np
import matplotlib.pyplot as plt
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap, background_colormap


fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# Erase the box and the ticks of the axis

# Set the limits of the axis
ax.set_aspect('equal')

# Draw to circles given the centers in two np.arrays and the radius

def draw_circles(ax, centers, radius, color):
	for i in range(len(centers)):
		circle = plt.Circle((centers[i][0], centers[i][1]), radius, color='blue', fill=True, alpha=0.1, edgecolor='blue', linewidth=1)
		ax.add_artist(circle)
		
	# Plot the centers of the circles with a cross in blue
	ax.scatter(centers[:,0], centers[:,1], color='blue', marker='+', s=100, linewidth=2, label='GP centroids')


def draw_double_arrow(orig, dest, color, text, scale=0.01):
	# Draw the arrow
	ax.arrow(orig[0], orig[1], dest[0]-orig[0], dest[1]-orig[1], color=color, head_width=0.5*scale, head_length=0.5*scale, length_includes_head=True, linewidth=1.5)
	# Draw the arrow head
	ax.arrow(dest[0], dest[1], orig[0]-dest[0], orig[1]-dest[1], color=color, head_width=0.5*scale, head_length=0.5*scale, length_includes_head=True, linewidth=1.5)
	# Put text in the middle of the arrow
	ax.text((orig[0]+dest[0])/2, (orig[1]+dest[1])/2, text, color='black', fontsize=12, horizontalalignment='center', verticalalignment='top')

C = np.array([[1,0.5], [1.2,0.75]])
R = 0.25
	
draw_circles(ax, C, R, 'red')
draw_double_arrow(C[0], [C[0][0]+R,C[0][1]], 'blue', 'Radius \n of influence')

# Set the limits of the axisq
ax.set_xlim([0.5, 2])
ax.set_ylim([0.5, 2])

plt.show()