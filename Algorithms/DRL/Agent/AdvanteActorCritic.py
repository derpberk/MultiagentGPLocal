import torch as th
import torch.nn as nn
from Algorithms.DRL.Networks.actorscritic import ActorCriticNetwork
from torch.utils.tensorboard import SummaryWriter
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import NoGoBackMasking, SafeActionMasking
from collections import deque
import numpy as np
from tqdm import trange


class MultiagentRolloutMemory:

	def __init__(self, n_agents: int, gamma: float):
		""" This object will store the multiagent rollouts for an episode """

		# Store number of agents
		self.gamma = gamma
		self.n_agents = n_agents
		self.timestep = {agent_id: 0 for agent_id in range(self.n_agents)}
		# Store state of the agents #
		self.dones = {agent_id: False for agent_id in range(self.n_agents)}
		self.rewards = {agent_id: [] for agent_id in range(self.n_agents)}
		self.acummulated_rewards = {agent_id: 0 for agent_id in range(self.n_agents)}
		self.Q_values = {agent_id: [] for agent_id in range(self.n_agents)}
		self.actions = {agent_id: [] for agent_id in range(self.n_agents)}
		self.states = {agent_id: [] for agent_id in range(self.n_agents)}
		self.next_states = {agent_id: [] for agent_id in range(self.n_agents)}
		self.log_probs = {agent_id: [] for agent_id in range(self.n_agents)}
		self.values = {agent_id: [] for agent_id in range(self.n_agents)}
		self.entropies = {agent_id: 0 for agent_id in range(self.n_agents)}

	def store(self, agent_id: int, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool, log_prob: float, value: float, entropy: float):
		""" Store the experience of the agent """

		self.states[agent_id].append(state)
		self.actions[agent_id].append(action)
		self.rewards[agent_id].append(reward)
		self.acummulated_rewards[agent_id] += reward
		self.next_states[agent_id].append(next_state)
		self.dones[agent_id] = done
		self.log_probs[agent_id].append(log_prob)
		self.values[agent_id].append(value)
		self.entropies[agent_id] += entropy
		# Increment timestep #
		self.timestep[agent_id] += 1

		if done:
			# Compute the Q discounted values #
			Q_value = 0
			for reward in self.rewards[agent_id][::-1]:
				Q_value = reward + self.gamma * Q_value
				self.Q_values[agent_id].insert(0, Q_value)

	def get(self):
		""" Get the stored experience of the agent """
		return self.states, self.actions, self.rewards, self.acummulated_rewards, self.next_states, self.dones, self.log_probs, self.values, self.entropies, self.Q_values
	
	def clear(self):

		self.timestep = {agent_id: 0 for agent_id in range(self.n_agents)}
		# Store state of the agents #
		self.dones = {agent_id: False for agent_id in range(self.n_agents)}
		self.rewards = {agent_id: [] for agent_id in range(self.n_agents)}
		self.acummulated_rewards = {agent_id: 0 for agent_id in range(self.n_agents)}
		self.Q_values = {agent_id: [] for agent_id in range(self.n_agents)}
		self.actions = {agent_id: [] for agent_id in range(self.n_agents)}
		self.states = {agent_id: [] for agent_id in range(self.n_agents)}
		self.next_states = {agent_id: [] for agent_id in range(self.n_agents)}
		self.log_probs = {agent_id: [] for agent_id in range(self.n_agents)}
		self.values = {agent_id: [] for agent_id in range(self.n_agents)}
		self.entropies = {agent_id: 0 for agent_id in range(self.n_agents)}

class A2CAgent:

	def __init__(self,
	      		env,
				n_steps,
				learning_rate,
				gamma,
				logdir,
				log_name,
				save_every,
				device = 'cuda:0'):

		""" Log directory and name for tensorboard logging"""
		self.logdir = logdir
		self.experiment_name = log_name
		self.writer = None
		self.save_every = save_every
		""" Save learning parameters """
		self.env = env
		self.number_of_agents = self.env.number_of_agents
		self.n_steps = n_steps
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.n_steps = n_steps

		self.device = th.device(device)

		""" Create the Actor and Critic networks """
		self.network = ActorCriticNetwork(self.env.observation_space.shape, self.env.action_space.n).to(self.device)

		""" Optimizer """
		self.optimizer = th.optim.Adam(self.network.parameters(), lr=self.learning_rate)

		self.multiagent_rollout_memory = MultiagentRolloutMemory(self.env.number_of_agents, self.gamma)
		
		# Masking utilities #
		self.safe_masking_module = SafeActionMasking(action_space_dim = self.env.action_space.n, movement_length = self.env.movement_length)
		self.nogoback_masking_modules = {i: NoGoBackMasking() for i in range(self.env.number_of_agents)}
		

	def sample_rollout(self):
		""" Play an episode and store the experiences, rewards, etc in the multiagent_rollout_memory """

		# Clear the rollout memory #
		self.multiagent_rollout_memory.clear()

		# Reset environment #
		states = self.env.reset()
		# Reset done #
		dones = {agent_id: False for agent_id in range(self.env.number_of_agents)}
		steps = 0
		active_agents = list(range(self.env.number_of_agents))

		for module in self.nogoback_masking_modules.values():
			module.reset()

		while not all(dones.values()):

			# Stack values of states into a batch of obervations #
			batch_states = th.stack([th.FloatTensor(states[agent_id]) for agent_id in active_agents]).to(self.device)

			# Get the positions #
			positions = self.env.fleet.get_positions()

			masks = np.zeros((batch_states.shape[0], self.env.action_space.n))
			for i, agent_id in enumerate(active_agents):
				# Update the navigation map #
				self.safe_masking_module.update_state(positions[agent_id], self.env.scenario_map)
				action_mask, _ = self.safe_masking_module.mask_action(np.ones((self.env.action_space.n, )).astype(float))
				# Update the nogoback mask #
				self.nogoback_masking_modules[agent_id].mask_action(action_mask)
				# Update the mask #
				masks[i] = action_mask

			masks = th.FloatTensor(masks).to(self.device).clamp_min(0.0)

			# Get batches actions from the network #
			batch_actions, batch_log_probs, batch_values, batch_entropies = self.network.get_action(batch_states, mask = masks)

			# Update the nogoback mask #
			for i, agent_id in enumerate(active_agents):
				self.nogoback_masking_modules[agent_id].update_last_action(batch_actions[i].detach().cpu().item())

			# Transform batch_actions into a dictionary of actions if the agent is not done#
			actions = {agent_id: batch_actions[i].detach().cpu().item() for i, agent_id in enumerate(active_agents)}

			# Take actions in the environment #
			next_states, rewards, dones, _ = self.env.step(actions)

			# Store the experience in the multiagent_rollout_memory #
			for i, agent_id in enumerate(active_agents):
				self.multiagent_rollout_memory.store(
					 agent_id, 
					 states[agent_id],
					 actions[agent_id], 
					 rewards[agent_id], 
					 next_states[agent_id], 
					 dones[agent_id], 
					 batch_log_probs[i], # This is a tensor
					 batch_values[i], # This is a tensor
					 batch_entropies[i] # This is a tensor
					 )
			
			# Update states #
			states = next_states

			# Truncate rollout if it reaches the maximum number of steps #
			if self.n_steps is not None:
				if steps >= self.n_steps and not (self.n_steps is None):
					break

			steps += 1

			for agent_id in range(self.env.number_of_agents):
				if dones[agent_id] and agent_id in active_agents:
					active_agents.remove(agent_id)

		return self.multiagent_rollout_memory.get()
	
	def update_policy(self, rollout):

		# Unpack rollout #
		states, actions, rewards, acummulated_rewards, next_states, dones, log_probs, values, entropies, Q_values = rollout

		# Calculate the advantage #
		actor_critics_loss = 0
		for i in range(self.env.number_of_agents):
			
			agent_values = th.FloatTensor(values[i]).to(self.device)
			agent_Q_values = th.FloatTensor(Q_values[i]).to(self.device)
			agent_log_probs = th.stack(log_probs[i])
			
			# Compute the advantage #
			advantage = agent_Q_values - agent_values
			# Compute the actor and critic losses #
			actor_loss = -(agent_log_probs * advantage.detach()).mean()
			critic_loss = 0.5 * advantage.pow(2).mean()
			actor_critics_loss += actor_loss + critic_loss + 0.0001 * entropies[i]

		actor_critics_loss /= self.env.number_of_agents

		# Update the network #
		self.optimizer.zero_grad()
		actor_critics_loss.backward()
		self.optimizer.step()

		return actor_loss.item()/self.env.number_of_agents, critic_loss.item()/self.env.number_of_agents
		
	def train(self, episodes):
		""" Play epsiodes and train the network using Actor Critic arquitecture """

		# Create train logger #
		if self.writer is None:
			self.writer = SummaryWriter(log_dir=self.logdir, filename_suffix=self.experiment_name)

		# Agent in training mode #
		self.is_eval = False
		# Reset metrics #
		record = -np.inf

		total_reward_list = []

		# Play ! #
		
		for episode in trange(1, int(episodes) + 1):
			# Sample a rollout from the environment #
			rollout = self.sample_rollout()
			# Update the network #
			actor_loss, critic_loss = self.update_policy(rollout)

			#Publish the metrics #
			self.publish_metrics(rollout, actor_loss, critic_loss, episode)

			# Save the model if the total reward is better than the previous one #
			total_reward_list.append(np.sum([rollout[3][i] for i in range(self.number_of_agents)]))
			mean_reward = np.mean(total_reward_list[-200:])
			if mean_reward > record and len(total_reward_list) > 200:
				record = mean_reward
				print(f"New best policy with mean reward of {mean_reward}")
				print("Saving model in " + self.writer.log_dir)
				self.save_model(f'best_model.pth')

			# Save the model every N episodes #
			if episode % self.save_every == 0:
				self.save_model(f'model_episode_{episode}.pth')

		self.save_model('final_model.pth')

	def save_model(self, name):
		""" Save the model in the path """
		th.save(self.network.state_dict(), self.writer.log_dir + '/' + name)

	def load_model(self, path):
		""" Load the model from the path """
		self.network.load_state_dict(th.load(path, map_location=self.device))

	def publish_metrics(self, rollout, actor_loss, critic_loss, episode):
		""" Write in the Tensorboard the total final reward, the length of the episode and the losses """

		states, actions, rewards, acummulated_rewards, next_states, dones, log_probs, values, entropies, Q_values = rollout

		length = np.max([len(rewards[i]) for i in range(self.number_of_agents)])
		entropy = np.mean([val.item() for val in entropies.values()])
		rewards = np.sum([acummulated_rewards[i] for i in range(self.number_of_agents)])

		self.writer.add_scalar('Train/Total reward', rewards, episode)
		self.writer.add_scalar('Train/Episode length', length, episode)
		self.writer.add_scalar('Train/Actor loss', actor_loss, episode)
		self.writer.add_scalar('Train/Critic loss', critic_loss, episode)
		self.writer.add_scalar('Train/Entropy', entropy, episode)

	def evaluate(self, episodes, render = False):
		""" Evaluate the agent in the environment """

		R = []
		lengths = []

		for episode in trange(1, int(episodes) + 1):

			acc_reward = 0
			steps = 0
			self.network.eval()
			# Reset environment #
			states = self.env.reset()
			# Reset done #
			dones = {agent_id: False for agent_id in range(self.env.number_of_agents)}

			active_agents = list(range(self.env.number_of_agents))

			for module in self.nogoback_masking_modules.values():
				module.reset()

			while not all(dones.values()):

				steps += 1

				# Stack values of states into a batch of obervations #
				batch_states = th.stack([th.FloatTensor(states[agent_id]) for agent_id in active_agents]).to(self.device)

				# Get the positions #
				positions = self.env.fleet.get_positions()

				masks = np.zeros((batch_states.shape[0], self.env.action_space.n))
				for i, agent_id in enumerate(active_agents):
					# Update the navigation map #
					self.safe_masking_module.update_state(positions[agent_id], self.env.scenario_map)
					action_mask, _ = self.safe_masking_module.mask_action(np.ones((self.env.action_space.n, )).astype(float))
					# Update the nogoback mask #
					self.nogoback_masking_modules[agent_id].mask_action(action_mask)
					# Update the mask #
					masks[i] = action_mask

				masks = th.FloatTensor(masks).to(self.device).clamp_min(0.0)

				# Get batches actions from the network #
				batch_actions, _, _, _ = self.network.get_action(batch_states, mask = masks, deterministic=False)

				# Update the nogoback mask #
				for i, agent_id in enumerate(active_agents):
					self.nogoback_masking_modules[agent_id].update_last_action(batch_actions[i].detach().cpu().item())

				# Transform batch_actions into a dictionary of actions if the agent is not done#
				actions = {agent_id: batch_actions[i].detach().cpu().item() for i, agent_id in enumerate(active_agents)}

				# Take actions in the environment #
				next_states, rewards, dones, _ = self.env.step(actions)

				if render:
					self.env.render()

				states = next_states

				for agent_id in range(self.env.number_of_agents):
					if dones[agent_id] and agent_id in active_agents:
						active_agents.remove(agent_id)

				acc_reward += np.sum(list(rewards.values()))

			R.append(acc_reward)
			lengths.append(steps)

		print(f"Mean reward: {np.mean(R)}")
		print(f"Mean length: {np.mean(lengths)}")

		return np.mean(R), np.mean(lengths)

			