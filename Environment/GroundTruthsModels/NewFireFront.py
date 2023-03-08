import sys
sys.path.append('.')
import math
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import time


class WildFiresSimulator:

	def __init__(self, scenario_map: np.ndarray, seed: int = 0, max_init_fires: int = 2, initial_time = 80) -> None:
		
		
		self.scenario_map = scenario_map
		self.valid_cells = np.column_stack(np.where(scenario_map == 1))
		self.max_distance = 2
		self.ignition_factor = 0.01
		self.temperature_map = np.zeros_like(scenario_map)
		self.initial_fuel_map = np.full_like(scenario_map, 10)
		self.extinguished_map = np.zeros_like(scenario_map)
		self.ignite_map = np.ones_like(scenario_map)
		self.wind_angle = None
		self.wind_speed = None
		self.max_init_fires = max_init_fires

		self.fig = None

		# Gaussian convolution kernel 5x5
		self.kernel = np.array([[1, 4, 7, 4, 1],
								[4, 16, 26, 16, 4],
								[7, 26, 41, 26, 7],
								[4, 16, 26, 16, 4],
								[1, 4, 7, 4, 1]]) / 273
		
	

		self.burning_map = np.zeros_like(scenario_map)
		self.initial_time = initial_time


	def init_fire(self):
		

		cells = self.valid_cells[np.random.randint(0, len(self.valid_cells), np.random.randint(1, self.max_init_fires+1))]
		# Start the fire
		self.burning_map[cells[:,0], cells[:,1]] = 1

	def reset(self):

		if self.fig is not None:
			plt.close(self.fig)
			self.fig = None

		self.temperature_map = np.zeros_like(self.scenario_map)
		self.fuel_map = self.initial_fuel_map.copy()
		self.burning_map = np.zeros_like(self.scenario_map)
		self.extinguished_map = np.zeros_like(self.scenario_map)
		self.ignite_map = np.ones_like(self.scenario_map)

		self.set_wind()
		self.init_fire()

		self.temperature_map = convolve2d(self.burning_map, self.kernel, mode='same')
		# Gaussian convolution kernel

		if self.initial_time > 0:
			for _ in range(self.initial_time):
				self.step()
		

	def step(self):

		# Get positions of burning_map that are not yet extinguished
		burning_cells = np.column_stack(np.where(self.burning_map == 1))

		#Get every position surrounding and update tge 
		for cell in burning_cells:

			if self.fuel_map[cell[0],cell[1]] == 0:
				self.burning_map[cell[0],cell[1]] = 0
				self.ignite_map[cell[0],cell[1]] = 1
				self.extinguished_map[cell[0],cell[1]] = 1
				continue

			# Get the positions positions of the surrounding cells that are closer to the burning cell than the max distance
			X = np.array([x for x in range(cell[0] - self.max_distance, cell[0] + self.max_distance + 1) if x >= 0 and x < self.scenario_map.shape[0]])
			Y = np.array([y for y in range(cell[1] - self.max_distance, cell[1] + self.max_distance + 1) if y >= 0 and y < self.scenario_map.shape[1]])
			surrounding_cells = np.array(np.meshgrid(X, Y)).T.reshape(-1, 2)

			# Compute the distance to the burning cell
			dxdy = surrounding_cells - cell
			distances = np.linalg.norm(dxdy, axis=1)
			close_condition = np.logical_and(distances <= self.max_distance, distances > 0)
			surrounding_cells = surrounding_cells[close_condition]
			distances = distances[close_condition]
			dxdy = dxdy[close_condition]            
			wind_factor = np.sum(dxdy * self.wind, axis=1)
			
			new_p = np.clip(self.ignition_factor * (1 + wind_factor)/distances**2, 0, 1)
			# Update the ignition map
			self.ignite_map[surrounding_cells[:, 0], surrounding_cells[:, 1]] *= (1.0 - new_p)

		
			# Update the fuel map
			self.fuel_map[cell[0],cell[1]] -= 1

			# Update the extinguished map
			self.extinguished_map[cell[0],cell[1]] = 1

		# Ignite the new cells with the ignition map probabilities
		new_burning_cells = np.column_stack(np.where(np.random.rand(*self.scenario_map.shape) > self.ignite_map))
		self.burning_map[new_burning_cells[:, 0], new_burning_cells[:, 1]] = 1

		self.temperature_map = convolve2d(self.burning_map + self.extinguished_map * 0.25, self.kernel, mode='same')

	def set_wind(self) -> None:
		
		# Random wind angle and speed
		self.wind_angle = np.random.uniform(0, 2 * np.pi)
		self.wind_speed = np.random.uniform(0, 0.1)
		self.wind = np.array([np.cos(self.wind_angle), np.sin(self.wind_angle)]) * self.wind_speed


	def render(self, mode='human'):

		if self.fig is None:

			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(111)
			self.ax.set_title('WildFiresSimulator')
			self.ax.set_xlabel('X')
			self.ax.set_ylabel('Y')

			self.im = self.ax.imshow(self.temperature_map, cmap='hot', vmin=0, vmax=1, interpolation='nearest')

		else:
			self.im.set_data(self.temperature_map)

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		plt.pause(0.01)

	def time_travel(self, steps: int) -> None:

		for i in range(steps):
			self.step()

	def read(self, position=None):

		

		if position is None:
			return self.temperature_map
		else:
			position = np.asarray(position).astype(int)
			return self.temperature_map[position[:,0], position[:,1]]



if __name__ == '__main__':

	import time

	gt = WildFiresSimulator(np.ones((30,30)))

	gt.reset()

	for _ in range(10):
		t0 = time.time()
		gt.reset()
		print(time.time() - t0)
		gt.render()
		plt.show()

