from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Algorithms.DRL.ReplayBuffers.ReplayBuffers import PrioritizedReplayBuffer, ReplayBuffer
from Algorithms.DRL.Networks.network import DuelingVisualNetwork, NoisyDuelingVisualNetwork, DistributionalVisualNetwork
import torch.nn.functional as F
from tqdm import trange
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import NoGoBackMasking, SafeActionMasking, ConsensusSafeActionMasking
import time
import json
import os

class MultiAgentDuelingDQNAgent:

	def __init__(
			self,
			env: gym.Env,
			memory_size: int,
			batch_size: int,
			target_update: int,
			soft_update: bool = False,
			tau: float = 0.0001,
			epsilon_values: List[float] = [1.0, 0.0],
			epsilon_interval: List[float] = [0.0, 1.0],
			learning_starts: int = 10,
			gamma: float = 0.99,
			lr: float = 1e-4,
			# PER parameters
			alpha: float = 0.2,
			beta: float = 0.6,
			prior_eps: float = 1e-6,
			# NN parameters
			number_of_features: int = 1024,
			noisy: bool = False,
			# Distributional parameters #
			distributional: bool = False,
			num_atoms: int = 51,
			v_interval: Tuple[float, float] = (0.0, 100.0),
			logdir=None,
			log_name="Experiment",
			save_every=None,
			train_every=1,
			masked_actions= False,
			device='cpu',
			seed = 0,
			eval_every = None,
			eval_episodes = 1000,
	):
		"""

		:param env: Environment to optimize
		:param memory_size: Size of the experience replay
		:param batch_size: Mini-batch size for SGD steps
		:param target_update: Number of episodes between updates of the target
		:param soft_update: Flag to activate the Polyak update of the target
		:param tau: Polyak update constant
		:param gamma: Discount Factor
		:param lr: Learning Rate
		:param alpha: Randomness of the sample in the PER
		:param beta: Bias compensating constant in the PER weights
		:param prior_eps: Minimal probability for every experience to be samples
		:param number_of_features: Number of features after the visual extractor
		:param logdir: Directory to save the tensorboard log
		:param log_name: Name of the tb log
		"""

		""" Logging parameters """
		np.random.seed(seed)
		self.logdir = logdir
		self.experiment_name = log_name
		self.writer = None
		self.save_every = save_every
		self.eval_every = eval_every
		self.eval_episodes = eval_episodes

		""" Observation space dimensions """
		obs_dim = env.observation_space.shape
		action_dim = env.action_space.n

		""" Agent embeds the environment """
		self.env = env
		self.batch_size = batch_size
		self.target_update = target_update
		self.soft_update = soft_update
		self.tau = tau
		self.gamma = gamma
		self.learning_rate = lr
		self.epsilon_values = epsilon_values
		self.epsilon_interval = epsilon_interval
		self.epsilon = self.epsilon_values[0]
		self.learning_starts = learning_starts
		self.noisy = noisy
		self.distributional = distributional
		self.v_interval = v_interval
		self.num_atoms = num_atoms
		self.train_every = train_every
		self.masked_actions = masked_actions

		""" Automatic selection of the device """
		self.device = device

		print("Selected device: ", self.device)

		""" Prioritized Experience Replay """
		self.beta = beta
		self.prior_eps = prior_eps
		self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha)

		""" Create the DQN and the DQN-Target (noisy if selected) """
		if self.noisy:
			self.dqn = NoisyDuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = NoisyDuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
		elif self.distributional:
			self.support = torch.linspace(self.v_interval[0], self.v_interval[1], self.num_atoms).to(self.device)
			self.dqn = DistributionalVisualNetwork(obs_dim, action_dim, number_of_features, num_atoms, self.support).to(self.device)
			self.dqn_target = DistributionalVisualNetwork(obs_dim, action_dim, number_of_features, num_atoms, self.support).to(self.device)
		else:
			self.dqn = DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)

		self.dqn_target.load_state_dict(self.dqn.state_dict())
		self.dqn_target.eval()

		""" Optimizer """
		self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

		""" Actual list of transitions """
		self.transition = list()

		""" Evaluation flag """
		self.is_eval = False

		""" Data for logging """
		self.episodic_reward = []
		self.episodic_loss = []
		self.episodic_length = []
		self.episode = 0

		# Sample new noisy parameters
		if self.noisy:
			self.dqn.reset_noise()
			self.dqn_target.reset_noise()

		# Masking utilities #
		self.safe_masking_module = SafeActionMasking(action_space_dim = action_dim, movement_length = self.env.movement_length)
		self.nogoback_masking_modules = {i: NoGoBackMasking() for i in range(self.env.number_of_agents)}
		self.consensus_masking_module = ConsensusSafeActionMasking(self.env.scenario_map, action_space_dim = action_dim, movement_length = self.env.movement_length)

	# TODO: Implement an annealed Learning Rate (see:
	#  https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)

	def predict_action(self, state: np.ndarray, deterministic: bool = False):

		"""Select an action from the input state. If deterministic, no noise is applied. """

		if self.epsilon > np.random.rand() and not self.noisy and not deterministic:
			selected_action = self.env.action_space.sample()

		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			selected_action = np.argmax(q_values)

		return selected_action
		
	def select_action(self, states: dict, deterministic: bool = False) -> dict:

		actions = {agent_id: self.predict_action(state, deterministic) for agent_id, state in states.items()}

		return actions

	def select_masked_action(self, states: dict, positions: np.ndarray, deterministic: bool = False):
		""" This is the core of the masking module. It selects an action for each agent, masked to avoid collisions and so"""
		
		#Initialize some things 
		actions = dict()
		q_values_agents = np.zeros((self.env.number_of_agents, self.env.action_space.n))

		for agent_id, state in states.items():
			""" First, we censor agent-to-env collisions """

			# Update the state of the safety module #
			self.safe_masking_module.update_state(position = positions[agent_id], new_navigation_map = self.env.scenario_map)

			if self.epsilon > np.random.rand() and not self.noisy and not deterministic:
				
				# Compute randomly the q_values but with censor #
				q_values, _ = self.safe_masking_module.mask_action(q_values = None)
				q_values, _ = self.nogoback_masking_modules[agent_id].mask_action(q_values = q_values)

			else:
				# Compute the q_values using the Policy but with censor #
				q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
				q_values, _ = self.safe_masking_module.mask_action(q_values = q_values.flatten())
				q_values, _ = self.nogoback_masking_modules[agent_id].mask_action(q_values = q_values)

			q_values_agents[agent_id] = q_values
			
		# Once we have the q_values for each agent, we compute the consensus q_values and we select the action for each agent #
		# IMPORTANT NOTE: CONSENSUS MASKING SHOULD BE APPLIED AT THE END. IF THE CONSENSUS IS BROKEN, ANYTHING COULD HAPPEND, EVEN COLLISIONS #
		actions = self.consensus_masking_module.query_actions(q_values = q_values_agents, positions = positions)

		# Update the last action for each agent #
		for agent_id, action in actions.items():
			self.nogoback_masking_modules[agent_id].update_last_action(action)

		return actions


	def step(self, action: dict) -> Tuple[np.ndarray, np.float64, bool]:
		"""Take an action and return the response of the env."""

		next_state, reward, done, _ = self.env.step(action)

		return next_state, reward, done

	def update_model(self) -> torch.Tensor:
		"""Update the model by gradient descent."""

		# PER needs beta to calculate weights
		samples = self.memory.sample_batch(self.beta)
		weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
		indices = samples["indices"]

		# PER: importance sampling before average
		elementwise_loss = self._compute_dqn_loss(samples)
		loss = torch.mean(elementwise_loss * weights)

		# Compute gradients and apply them
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# PER: update priorities
		loss_for_prior = elementwise_loss.detach().cpu().numpy()
		new_priorities = loss_for_prior + self.prior_eps
		self.memory.update_priorities(indices, new_priorities)

		# Sample new noisy distribution
		if self.noisy:
			self.dqn.reset_noise()
			self.dqn_target.reset_noise()

		return loss.item()

	@staticmethod
	def anneal_epsilon(p, p_init=0.1, p_fin=0.9, e_init=1.0, e_fin=0.0):

		if p < p_init:
			return e_init
		elif p > p_fin:
			return e_fin
		else:
			return (e_fin - e_init) / (p_fin - p_init) * (p - p_init) + 1.0

	@staticmethod
	def anneal_beta(p, p_init=0.1, p_fin=0.9, b_init=0.4, b_end=1.0):

		if p < p_init:
			return b_init
		elif p > p_fin:
			return b_end
		else:
			return (b_end - b_init) / (p_fin - p_init) * (p - p_init) + b_init

	def train(self, episodes):
		# TODO: Change for multiagent

		""" Train the agent. """

		# Optimization steps #
		steps = 0
		
		# Create train logger and save configs #
		if self.writer is None:
			assert not os.path.exists(self.logdir), "El directorio ya existe. He evitado que se sobrescriba"
			self.writer = SummaryWriter(log_dir=self.logdir, filename_suffix=self.experiment_name)
			self.write_experiment_config()

		# Agent in training mode #
		self.is_eval = False
		# Reset episode count #
		self.episode = 1
		# Reset metrics #
		episodic_reward_vector = []
		record = -np.inf

		for module in self.nogoback_masking_modules.values():
			module.reset()

		for episode in trange(1, int(episodes) + 1):

			done = {i:False for i in range(self.env.number_of_agents)}
			state = self.env.reset()
			score = 0
			length = 0
			losses = []

			# Initially sample noisy policy #
			if self.noisy:
				self.dqn.reset_noise()
				self.dqn_target.reset_noise()

			# PER: Increase beta temperature
			self.beta = self.anneal_beta(p=episode / episodes, p_init=0, p_fin=0.9, b_init=0.4, b_end=1.0)

			# Epsilon greedy annealing
			self.epsilon = self.anneal_epsilon(p=episode / episodes,
											   p_init=self.epsilon_interval[0],
											   p_fin=self.epsilon_interval[1],
											   e_init=self.epsilon_values[0],
											   e_fin=self.epsilon_values[1])

			# Run an episode #
			while not all(done.values()):

				# Inrease the played steps #
				steps += 1

				# Select the action using the current policy
				if not self.masked_actions:
					actions = self.select_action(state)
				else:
					actions = self.select_masked_action(states=state, positions=self.env.fleet.get_positions())
					
				actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}

				# Process the agent step #
				next_state, reward, done = self.step(actions)

				for agent_id in next_state.keys():

				# Store every observation for every agent
					self.transition = [state[agent_id],
									   actions[agent_id],
									   reward[agent_id],
									   next_state[agent_id],
									   done[agent_id],
									   {}]

					self.memory.store(*self.transition)

				# Update the state
				state = next_state
				# Accumulate indicators
				score += np.mean(list(reward.values()))  # The mean reward among the agents
				length += 1

				# if episode ends
				if all(done.values()):

					# Append loss metric #
					if losses:
						self.episodic_loss = np.mean(losses)

					# Compute average metrics #
					self.episodic_reward = score
					self.episodic_length = length
					episodic_reward_vector.append(self.episodic_reward)
					self.episode += 1

					# Log progress
					self.log_data()

				# If training is ready
				if len(self.memory) >= self.batch_size and episode >= self.learning_starts:

					# Update model parameters by backprop-bootstrapping #
					if steps % self.train_every == 0:

						loss = self.update_model()
						# Append loss #
						losses.append(loss)

					# Update target soft/hard #
					if self.soft_update:
						self._target_soft_update()
					elif episode % self.target_update == 0 and all(done.values()):
						self._target_hard_update()

			if self.save_every is not None:
				if episode % self.save_every == 0:
					self.save_model(name=f'Episode_{episode}_Policy.pth')

			if self.eval_every is not None:
				if episode % self.eval_every == 0:
					mean_reward, mean_length, mean_error = self.evaluate_env(self.eval_episodes)
					self.writer.add_scalar('test/accumulated_reward', mean_reward, self.episode)
					self.writer.add_scalar('test/accumulated_length', mean_length, self.episode)
					self.writer.add_scalar('test/mean_error', mean_error, self.episode)

					if mean_reward > record:
						print(f"New best policy with mean reward of {mean_reward}")
						print("Saving model in " + self.writer.log_dir)
						record = mean_reward
						try:
							self.save_model(name='BestPolicy.pth')
						except:
							print("Error saving model")

					
		# Save the final policy #
		self.save_model(name='FinalPolicy.pth')

	def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:

		"""Return dqn loss."""
		device = self.device  # for shortening the following lines
		state = torch.FloatTensor(samples["obs"]).to(device)
		next_state = torch.FloatTensor(samples["next_obs"]).to(device)
		action = torch.LongTensor(samples["acts"]).to(device)
		reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
		done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

		# G_t   = r + gamma * v(s_{t+1})  if state != Terminal
		#       = r                       otherwise

		if not self.distributional:

			action = action.reshape(-1, 1)
			curr_q_value = self.dqn(state).gather(1, action)
			done_mask = 1 - done

			with torch.no_grad():
				next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0]
				target = (reward + self.gamma * next_q_value * done_mask).to(self.device)

			# calculate element-wise dqn loss
			elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")

		else:
			# Distributional Q-Learning - Here is where the fun begins #
			delta_z = float(self.v_interval[1] - self.v_interval[0]) / (self.num_atoms - 1)

			with torch.no_grad():

				# max_a = argmax_a' Q'(s',a')
				next_action = self.dqn_target(next_state).argmax(1)
				# V'(s', max_a)
				next_dist = self.dqn_target.dist(next_state)
				next_dist = next_dist[range(self.batch_size), next_action]

				# Compute the target distribution by adding the
				t_z = reward + (1 - done) * self.gamma * self.support
				t_z = t_z.clamp(min=self.v_interval[0], max=self.v_interval[1])
				b = (t_z - self.v_interval[0]) / delta_z
				lower_bound = b.floor().long()
				upper_bound = b.ceil().long()

				offset = (torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size
					).long()
					.unsqueeze(1)
					.expand(self.batch_size, self.num_atoms)
					.to(self.device)
				)

				proj_dist = torch.zeros(next_dist.size(), device=self.device)
				proj_dist.view(-1).index_add_(
					0, (lower_bound + offset).view(-1), (next_dist * (upper_bound.float() - b)).view(-1)
				)
				proj_dist.view(-1).index_add_(
					0, (upper_bound + offset).view(-1), (next_dist * (b - lower_bound.float())).view(-1)
				)

			dist = self.dqn.dist(state)
			log_p = torch.log(dist[range(self.batch_size), action])

			elementwise_loss = -(proj_dist * log_p).sum(1)

		return elementwise_loss

	def _target_hard_update(self):
		"""Hard update: target <- local."""
		print(f"Hard update performed at episode {self.episode}!")
		self.dqn_target.load_state_dict(self.dqn.state_dict())

	def _target_soft_update(self):
		"""Soft update: target_{t+1} <- local * tau + target_{t} * (1-tau)."""
		for target_param, local_param in zip(self.dqn_target.parameters(), self.dqn_target.parameters()):
			target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

	def log_data(self):

		if self.episodic_loss:
			self.writer.add_scalar('train/loss', self.episodic_loss, self.episode)

		self.writer.add_scalar('train/epsilon', self.epsilon, self.episode)
		self.writer.add_scalar('train/beta', self.beta, self.episode)

		self.writer.add_scalar('train/accumulated_reward', self.episodic_reward, self.episode)
		self.writer.add_scalar('train/accumulated_length', self.episodic_length, self.episode)

		self.writer.flush()

	def load_model(self, path_to_file):

		self.dqn.load_state_dict(torch.load(path_to_file, map_location=self.device))

	def save_model(self, name='experiment.pth'):

		torch.save(self.dqn.state_dict(), self.writer.log_dir + '/' + name)

	def evaluate_env(self, eval_episodes, render=False):
		""" Evaluate the agent on the environment for a given number of episodes with a deterministic policy """

		self.dqn.eval()
		total_reward = 0
		total_length = 0
		total_error = 0

		for _ in trange(eval_episodes):

			# Reset the environment #
			state = self.env.reset()
			if render:
				self.env.render()
			done = {agent_id: False for agent_id in range(self.env.number_of_agents)}

			for module in self.nogoback_masking_modules.values():
				module.reset()
			

			while not all(done.values()):

				total_length += 1

				# Select the action using the current policy
				if not self.masked_actions:
					actions = self.select_action(state, deterministic=True)
				else:
					actions = self.select_masked_action(states=state, positions=self.env.fleet.get_positions(), deterministic=True)
					
				actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}

				# Process the agent step #
				next_state, reward, done = self.step(actions)


				if render:
					self.env.render()
					time.sleep(0.25)

				# Update the state #
				state = next_state
				
				total_reward += np.sum(list(reward.values()))

			total_error += self.env.get_error()

		self.dqn.train()

		# Return the average reward, average length

		return total_reward / eval_episodes, total_length / eval_episodes, total_error / eval_episodes

	def write_experiment_config(self):
		""" Write experiment and environment variables in a json file """

		self.experiment_config = {
			"save_every": self.save_every,
			"eval_every": self.eval_every,
			"eval_episodes": self.eval_episodes,
			"batch_size": self.batch_size,
			"gamma": self.gamma,
			"tau": self.tau,
			"lr": self.learning_rate,
			"epsilon": self.epsilon,
			"epsilon_values": self.epsilon_values,
			"epsilon_interval": self.epsilon_interval,
			"beta": self.beta,
			"num_atoms": self.num_atoms,
			"masked_actions": self.masked_actions,
		}

		with open(self.writer.log_dir + '/experiment_config.json', 'w') as f:

			json.dump(self.experiment_config, f, indent=4)





