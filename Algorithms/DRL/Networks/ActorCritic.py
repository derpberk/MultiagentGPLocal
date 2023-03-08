import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from Algorithms.DRL.Networks.FeatureExtractors import FeatureExtractor
import numpy as np
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import NoGoBackMasking, SafeActionMasking, ConsensusSafeActionDistributionMasking

class A2C(nn.Module):
	"""
	(Synchronous) Advantage Actor-Critic agent class

	Args:
		n_features: The number of features of the input state.
		n_actions: The number of actions the agent can take.
		device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
				for this code CPU is totally fine).
		critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
		actor_lr: The learning rate for the actor network.
		n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
	"""

	def __init__(
		self,
		obs_dim,
		n_features: int,
		n_actions: int,
		device: torch.device,
		critic_lr: float,
		actor_lr: float,
		n_agents: int,
		conditioned_actions = False,
		environment_model_info = None
	) -> None:
		
		"""Initializes the actor and critic networks and their respective optimizers."""
		super().__init__()
		self.device = device
		self.n_agents = n_agents

		self.critic = nn.Sequential(
			FeatureExtractor(obs_dim, n_features),
			nn.Linear(n_features, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 1),  # estimate V(s)
		).to(self.device)

		self.actor = nn.Sequential(
			FeatureExtractor(obs_dim, n_features),
			nn.Linear(n_features, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, n_actions),
		).to(self.device)


		# define optimizers for actor and critic
		self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
		self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)

		self.conditioned_actions = conditioned_actions
		if self.conditioned_actions:
			self.environment_model_info = environment_model_info
			self.consensus_module = ConsensusSafeActionDistributionMasking(self.environment_model_info['scenario_map'], action_space_dim = n_actions, movement_length = self.environment_model_info['movement_length'])


	def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Forward pass of the networks.

		Args:
			x: A batched vector of states.

		Returns:
			state_values: A tensor with the state values, with shape [n_envs,].
			action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
		"""
		x = torch.Tensor(x).to(self.device)
		state_values = self.critic(x)  # shape: [n_envs,]
		action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
		return (state_values, action_logits_vec)

	def select_action(self, x, action_mask = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Returns a tuple of the chosen actions and the log-probs of those actions.

		Args:
			x: A batched vector of states.

		Returns:
			actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
			action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
			state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
		"""

		state_values, action_logits = self.forward(x)
		action_probabilites = F.softmax(action_logits, dim=1)
		if action_mask is not None:
			action_probabilites = (action_probabilites * action_mask).clip(min=1e-10)


		action_pd = torch.distributions.Categorical(probs=action_probabilites)  # implicitly uses softmax
		actions = action_pd.sample()
		action_log_probs = action_pd.log_prob(actions)
		entropy = action_pd.entropy()

		return (actions, action_log_probs, state_values, entropy)
	
	def select_consensual_action(self, x, action_mask: np.ndarray, scenario_map: np.ndarray, positions: np.ndarray, deterministic = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

		""" Process the action mask to be used by the consensus module """

		# Forward the observations through the network
		state_values, action_logits = self.forward(x)
		# Process the action mask
		if action_mask is not None:
			action_logits[action_mask] = -torch.finfo(torch.float).max

		# Get the consensus action mask
		actions, action_log_probs, entropy = self.consensus_module.query_actions_from_logits(action_logits, positions, device=self.device, deterministic=deterministic)

		return (actions, action_log_probs, state_values, entropy)



	def get_losses(
		self,
		rewards: torch.Tensor,
		action_log_probs: torch.Tensor,
		value_preds: torch.Tensor,
		entropy: torch.Tensor,
		masks: torch.Tensor,
		gamma: float,
		lam: float,
		ent_coef: float,
		device: torch.device,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
		using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

		Args:
			rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs, n_agents].
			action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
			value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
			masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
			gamma: The discount factor.
			lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
										  and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
										  because the estimates are generated by a Neural Net).
			device: The device to run the computations on (e.g. CPU or GPU).

		Returns:
			critic_loss: The critic loss for the minibatch.
			actor_loss: The actor loss for the minibatch.
		"""

		T = len(rewards)

		advantages = torch.zeros(T, self.n_agents, device=device)

		# compute the advantages using GAE
		gae = 0.0
		for t in reversed(range(T - 1)):
			td_error = (rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t])
			gae = td_error + gamma * lam * masks[t] * gae
			advantages[t] = gae

		# calculate the loss of the minibatch for actor and critic
		critic_loss = advantages.pow(2).mean()

		# give a bonus for higher entropy to encourage exploration
		actor_loss = (-(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.sum())

		return (critic_loss, actor_loss)

	def update_parameters(self, critic_loss: torch.Tensor, actor_loss: torch.Tensor) -> None:
		"""
		Updates the parameters of the actor and critic networks.

		Args:
			critic_loss: The critic loss.
			actor_loss: The actor loss.
		"""
		self.critic_optim.zero_grad()
		critic_loss.backward()
		self.critic_optim.step()

		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()
