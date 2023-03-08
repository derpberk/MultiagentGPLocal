from Algorithms.DRL.Networks.ActorCritic import A2C
import torch
import numpy as np
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import NoGoBackMasking, SafeActionMasking
from tqdm import trange
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter


class A2Cagent:
	
	def __init__(self,
		  		env,
				n_steps,
				actor_learning_rate,
				critic_learning_rate,
				gamma,
				lambda_gae,
				ent_coef,
				device,
				save_every = 100,
				logdir = 'runs',
				log_name = 'A2C',
				eval_every = 1000,
				n_eval_episodes = 10,
				):
		
		self.env = env
		self.n_steps = n_steps
		self.actor_learning_rate = actor_learning_rate
		self.critic_learning_rate = critic_learning_rate
		self.gamma = gamma
		self.lambda_gae = lambda_gae
		self.ent_coef = ent_coef

		""" Log directory and name for tensorboard logging"""
		self.logdir = logdir
		self.experiment_name = log_name
		self.writer = None
		self.save_every = save_every

		self.device = torch.device(device)
		self.n_agents = self.env.number_of_agents

		self.obs_dim = self.env.observation_space
		self.n_actions = self.env.action_space.n
		self.eval_every = eval_every
		self.n_eval_episodes = n_eval_episodes

		# Create the actor and critic networks and their respective optimizers
		self.A2C = A2C(obs_dim = self.obs_dim.shape, 
		 				n_features = 512, 
						n_actions = self.n_actions, 
						device = self.device, 
						critic_lr = self.critic_learning_rate, 
						actor_lr = self.actor_learning_rate,
						n_agents = self.n_agents,
						conditioned_actions = True,
						environment_model_info = dict(scenario_map = self.env.scenario_map, movement_length=self.env.movement_length))

		# Masking utilities #
		self.safe_masking_module = SafeActionMasking(action_space_dim = self.env.action_space.n, movement_length = self.env.movement_length)
		self.nogoback_masking_modules = {i: NoGoBackMasking() for i in range(self.n_agents)}

	def sample_rollout(self, n_steps, deterministic = False, render = False):
		""" Samples a rollout of n_steps in the environment and returns the experiences"""
		
		# Initialize the lists that will contain the experiences #
		ep_value_preds = torch.zeros(n_steps, self.n_agents, device=self.device)
		ep_rewards = torch.zeros(n_steps, self.n_agents, device=self.device)
		ep_action_log_probs = torch.zeros(n_steps, self.n_agents, device=self.device)
		ep_entropies = torch.zeros(n_steps, self.n_agents, device=self.device)

		masks = torch.zeros(n_steps, self.n_agents, device=self.device)  # This is a mask that indicates if the agent/episode is done or not - involved in return calculations #

		for module in self.nogoback_masking_modules.values():
			module.reset()

		# Play N steps in the environment #
		# Reset the environment if it is done #
		state = self.env.reset()

		if render:
			self.env.render()

		for step in range(n_steps):
			
			# Transform the state from a dictionary into a tensor. Every key is a different agent #
			state_tensor = torch.tensor(np.asarray([state[i] for i in range(self.n_agents)]), device=self.device, dtype=torch.float)

			# Update the action masks #
			action_masks = np.zeros((state_tensor.shape[0], self.env.action_space.n))

			for agent_id in range(self.n_agents):
				# Update the navigation map #
				self.safe_masking_module.update_state(self.env.fleet.get_positions()[agent_id], self.env.scenario_map)
				action_mask, _ = self.safe_masking_module.mask_action(np.ones((self.env.action_space.n, )).astype(float))
				# Update the nogoback mask #
				action_mask, _ = self.nogoback_masking_modules[agent_id].mask_action(action_mask)

				# Update the mask #
				action_masks[agent_id] = action_mask

			action_masks[action_masks == -np.inf] = 0.0
			action_masks = np.logical_not(action_masks.astype(bool))

			# Get the action log probabilities and the value prediction for the current state #
			actions, action_log_probs, state_value_preds, entropy = self.A2C.select_consensual_action(x = state_tensor, 
											action_mask=torch.BoolTensor(action_masks).to(device=self.device), 
											scenario_map=self.env.scenario_map,
											positions=self.env.fleet.get_positions(),
											deterministic=deterministic)

			action_dict = {i: actions[i].item() for i in range(self.n_agents)}

			for agent_id in range(self.n_agents):
				# Update the nogoback mask #
				self.nogoback_masking_modules[agent_id].update_last_action(action_dict[agent_id])

			# Perform the action in the environment #
			next_state, rewards, dones, _ = self.env.step(action_dict)

			if render:
				self.env.render()

			ep_value_preds[step] = torch.squeeze(state_value_preds)
			ep_rewards[step] = torch.tensor(np.array(list(rewards.values())), device=self.device)
			ep_action_log_probs[step] = action_log_probs
			ep_entropies[step] = entropy

			# add a mask (for the return calculation later);
			# for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
			masks[step] = torch.tensor([not done for done in dones.values()])

			# Update the state #
			state = next_state

		return ep_rewards, ep_action_log_probs, ep_value_preds, entropy, masks
	

	def train(self, episodes: int):
		""" Train for a number of episodes """

		# Create train logger #
		if self.writer is None:
			self.writer = SummaryWriter(log_dir=self.logdir, filename_suffix=self.experiment_name)

		best_reward = -np.inf

		for episode in trange(episodes):

			# Sample a batch of experiences in parallel using joblib #
			#batches = Parallel(n_jobs=self.n_envs)(delayed(self.sample_batch)(env_id) for env_id in range(self.n_envs))
			ep_rewards, ep_action_log_probs, ep_value_preds, entropy, masks = self.sample_rollout(self.n_steps)

			# compute the lossed from actor and critic networks #
			critic_loss, actor_loss = self.A2C.get_losses(
				ep_rewards,
				ep_action_log_probs,
				ep_value_preds,
				entropy,
				masks,
				self.gamma,
				self.lambda_gae,
				self.ent_coef,
				self.device,
			)

			# Update the actor and critic networks #
			self.A2C.update_parameters(critic_loss, actor_loss)

			# The length is the max number
			length = masks.sum(dim=0).max().item()

			# Publish the metrics #
			self.publish_metrics(ep_rewards.sum().item(), entropy.sum().item(), actor_loss.item(), critic_loss.item(), length, episode)

			# Save the model #
			if episode % self.save_every == 0:
				self.save_model("policy_episode_{}.pth".format(episode))

			if episode % self.eval_every == 0 and episode > 0:
				mean_reward = self.evaluate(self.n_eval_episodes)
				self.writer.add_scalar('Eval/Total reward', mean_reward, episode)

				if mean_reward > best_reward:
					best_reward = mean_reward
					try:
						self.save_model("best_policy_so_far.pth")
					except:
						print("Error saving the best policy")

	def publish_metrics(self, rewards, entropy, actor_loss, critic_loss, lengths, episode):
			""" Write in the Tensorboard the total final reward, the length of the episode and the losses """

			self.writer.add_scalar('Train/Total reward', rewards, episode)
			self.writer.add_scalar('Train/Actor loss', actor_loss, episode)
			self.writer.add_scalar('Train/Critic loss', critic_loss, episode)
			self.writer.add_scalar('Train/Entropy', entropy, episode)
			self.writer.add_scalar('Train/Episode length', lengths, episode)

	def load_model(self, path_to_file):

		self.A2C.load_state_dict(torch.load(path_to_file, map_location=self.device))

	def save_model(self, name='experiment.pth'):

		torch.save(self.A2C.state_dict(), self.writer.log_dir + '/' + name)

	def evaluate(self, eval_episodes: int, render: bool = False):
		""" Evaluate for a number of episodes """

		mean_reward = np.mean([np.sum(self.sample_rollout(self.n_steps, deterministic=True, render=render)[0].detach().cpu().numpy()) for _ in range(eval_episodes)])


		return mean_reward