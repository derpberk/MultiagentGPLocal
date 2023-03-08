import numpy as np
import torch
from torch.nn.functional import softmax

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
			q_values = np.random.rand(len(self.angle_set))

		movements = np.asarray([np.round(np.array([np.cos(angle), np.sin(angle)])) * self.movement_length for angle in self.angle_set]).astype(int)
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
	
class ConsensusSafeActionMasking:
	""" The optimists decide first! """

	def __init__(self, navigation_map, action_space_dim: int, movement_length: float) -> None:
		
		self.movement_length = movement_length
		self.angle_set = np.linspace(0, 2 * np.pi, action_space_dim, endpoint=False)
		self.position = None
		self.fleet_map = np.zeros_like(navigation_map)


	def query_actions(self, q_values: np.ndarray, positions: np.ndarray):

		# 1) The largest q-value agent decides first
		# 2) If there are multiple agents with the same q-value, the agent is selected randomly
		# 3) Then, compute the next position of the agent and update the fleet map
		# 4) The next agent is selected based on the updated fleet map, etc
		
		self.fleet_map = np.ones_like(self.fleet_map)
		agents_order = np.argsort(q_values.max(axis=1))[::-1]
		final_actions = np.zeros(q_values.shape[0], dtype=int)

		for agent in agents_order:
			
			#Unpack the agent position
			agent_position = positions[agent]
			# Compute the impossible actions
			movements = np.asarray([np.round(np.array([np.cos(angle), np.sin(angle)])) * self.movement_length for angle in self.angle_set]).astype(int)
			next_positions = agent_position + movements
			action_mask = np.array([self.fleet_map[int(next_position[0]), int(next_position[1])] == 0 for next_position in next_positions]).astype(bool)
			# Censor the impossible actions in the Q-values
			q_values[agent][action_mask] = -np.inf

			# Select the action
			action = np.argmax(q_values[agent])

			# Update the fleet map
			next_position = next_positions[action]
			self.fleet_map[int(next_position[0]), int(next_position[1])] = 0

			# Store the action
			final_actions[agent] = action.copy()

		return {agent: final_actions[agent] for agent in range(q_values.shape[0])}


class ConsensusSafeActionDistributionMasking:
	""" The same as ConsensusSafeActionMasking, but the action is selected from the action distribution, conditiones on the action mask """

	def __init__(self, navigation_map, action_space_dim: int, movement_length: float) -> None:
		
		self.movement_length = movement_length
		self.angle_set = np.linspace(0, 2 * np.pi, action_space_dim, endpoint=False)
		self.position = None
		self.fleet_map = np.zeros_like(navigation_map)

	def query_actions_from_logits(self, logits: torch.Tensor, positions: np.ndarray, device, deterministic: bool = False):

		# 1) The largest q-value agent decides first
		# 2) If there are multiple agents with the same q-value, the agent is selected randomly
		# 3) Then, compute the next position of the agent and update the fleet map
		# 4) The next agent is selected based on the updated fleet map, etc
		
		self.fleet_map = np.ones_like(self.fleet_map)
		agents_order = np.argsort(logits.cpu().detach().numpy().max(axis=1))[::-1]
		final_actions = torch.zeros(logits.shape[0], dtype=int, device=device)
		action_log_probs = torch.zeros(logits.shape[0], dtype=float, device=device)
		entropy = torch.zeros(logits.shape[0], dtype=float, device=device)

		for agent in agents_order:
			
			#Unpack the agent position
			agent_position = positions[agent]
			# Compute the impossible actions
			movements = np.asarray([np.round(np.array([np.cos(angle), np.sin(angle)])) * self.movement_length for angle in self.angle_set]).astype(int)
			next_positions = agent_position + movements
			action_mask = np.array([self.fleet_map[int(next_position[0]), int(next_position[1])] == 0 for next_position in next_positions]).astype(bool)
			# Censor the impossible actions in the Q-values
			logits[agent][action_mask] = -torch.finfo(torch.float).max

			# Select the action
			action_probabilities = softmax(logits[agent], dim=0)
			action_distribution = torch.distributions.Categorical(probs=action_probabilities)
			if deterministic:
				action = action_distribution.mode
			else:
				action = action_distribution.sample()

			action_log_probs[agent] = action_distribution.log_prob(action)
			entropy[agent] = action_distribution.entropy().mean()
			

			# Update the fleet map
			next_position = next_positions[action]
			self.fleet_map[int(next_position[0]), int(next_position[1])] = 0

			# Store the action
			final_actions[agent] = action

		return final_actions, action_log_probs, entropy
