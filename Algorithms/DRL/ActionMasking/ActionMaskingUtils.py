import numpy as np

class SafeActionMasking:

	def __init__(self, action_space_dim: int, movement_length: float) -> None:
		""" Safe Action Masking """

		self.navigation_map = None
		self.position = None
		self.angle_set = np.linspace(0, 2 * np.pi, action_space_dim, endpoint=False)
		self.movement_length = movement_length

	def update_state(self, position: np.ndarray, new_navigation_map: np.ndarray = None):
		""" Update the navigation map """

		if new_navigation_map is not None:
			self.navigation_map = new_navigation_map

		""" Update the position """
		self.position = position

	def mask_action(self, q_values: np.ndarray = None):

		if q_values is None:
			""" Random selection """
			q_values = np.random.rand(8)

		movements = np.asarray([np.round(np.array([np.cos(angle), np.sin(angle)])).astype(int) * self.movement_length for angle in self.angle_set])
		next_positions = self.position + movements

		action_mask = np.array([self.navigation_map[int(next_position[0]), int(next_position[1])] == 0 for next_position in next_positions]).astype(bool)

		q_values[action_mask] = -np.inf

		return q_values, np.argmax(q_values)

class NoGoBackMasking:

	def __init__(self) -> None:
		
		self.previous_action = None

	def mask_action(self, q_values: np.ndarray = None):

		if q_values is None:
			""" Random selection """
			q_values = np.random.rand(8)

		if self.previous_action is None:
			self.previous_action = np.argmax(q_values)
		else:
			
			return_action = (self.previous_action + len(q_values) // 2) % len(q_values)
			q_values[return_action] = -np.inf

		return q_values, np.argmax(q_values)

	def update_last_action(self, last_action):

		self.previous_action = last_action

	def reset(self):
		
		self.previous_action = None
	


		
		
