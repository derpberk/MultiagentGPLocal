from Algorithms.DRL.Networks.ActorCritic import A2C
import torch
import numpy as np
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import NoGoBackMasking, SafeActionMasking
from tqdm import trange
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter


def outside_sample_batch(environment, n_steps, device, A2C):


	n_agents = environment.number_of_agents
	safe_masking_module = SafeActionMasking(action_space_dim = environment.action_space.n, movement_length = environment.movement_length)
	nogoback_masking_modules = {i: NoGoBackMasking() for i in range(n_agents)}
		
	# Initialize the lists that will contain the experiences #
	ep_value_preds = torch.zeros(n_steps, n_agents, device=device)
	ep_rewards = torch.zeros(n_steps, n_agents, device=device)
	ep_action_log_probs = torch.zeros(n_steps, n_agents, device=device)
	ep_entropies = torch.zeros(n_steps, n_agents, device=device)

	masks = torch.zeros(n_steps, n_agents, device=device)

	for module in nogoback_masking_modules.values():
		module.reset()

	# Play N steps in the environment #
	# Reset the environment if it is done #
	state = environment.reset()

	for step in range(n_steps):
		
		# Transform the state from a dictionary into a tensor. Every key is a different agent #
		state_tensor = torch.tensor(np.asarray([state[i] for i in range(n_agents)]), device=device, dtype=torch.float)

		# Update the action masks #
		action_masks = np.zeros((state_tensor.shape[0], environment.action_space.n))

		for agent_id in range(n_agents):
			# Update the navigation map #
			safe_masking_module.update_state(environment.fleet.get_positions()[agent_id], environment.scenario_map)
			action_mask, _ = safe_masking_module.mask_action(np.ones((environment.action_space.n, )).astype(float))
			# Update the nogoback mask #
			action_mask, _ = nogoback_masking_modules[agent_id].mask_action(action_mask)
			# Update the mask #
			action_masks[agent_id] = action_mask

		# Get the action log probabilities and the value prediction for the current state #
		actions, action_log_probs, state_value_preds, entropy = A2C.select_action(state_tensor, action_mask=torch.Tensor(action_masks).clamp_min(0.0).to(device=device))

		action_dict = {i: actions[i].item() for i in range(n_agents)}

		for agent_id in range(n_agents):
			# Update the nogoback mask #
			nogoback_masking_modules[agent_id].update_last_action(action_dict[agent_id])

		# Perform the action in the environment #
		next_state, rewards, dones, _ = environment.step(action_dict)

		ep_value_preds[step] = torch.squeeze(state_value_preds)
		ep_rewards[step] = torch.tensor(np.array(list(rewards.values())), device=device)
		ep_action_log_probs[step] = action_log_probs
		ep_entropies[step] = entropy

		# add a mask (for the return calculation later);
		# for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
		masks[step] = torch.tensor([not done for done in dones.values()])

		# Update the state #
		state = next_state

	return ep_rewards, ep_action_log_probs, ep_value_preds, entropy, masks



class AsyncronousActorCritic:
	
	def __init__(self,
		  		envs,
				n_steps,
				actor_learning_rate,
				critic_learning_rate,
				gamma,
				lambda_gae,
				ent_coef,
				device,
				save_every = 100,
				logdir = 'runs',
				log_name = 'A2C_parallel'
				):
		
		self.envs = envs
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

		self.n_envs = len(envs)
		self.n_agents = envs[0].number_of_agents

		self.obs_dim = envs[0].observation_space
		self.n_actions = envs[0].action_space.n

		# Create the actor and critic networks and their respective optimizers
		self.A2C = A2C(obs_dim = self.obs_dim.shape, 
		 				n_features = 512, 
						n_actions = self.n_actions, 
						device = self.device, 
						critic_lr = self.critic_learning_rate, 
						actor_lr = self.actor_learning_rate, 
						n_envs = self.n_envs, 
						n_agents = self.n_agents)

		# Masking utilities #
		self.safe_masking_module = SafeActionMasking(action_space_dim = self.envs[0].action_space.n, movement_length = self.envs[0].movement_length)
		self.nogoback_masking_modules = {i: NoGoBackMasking() for i in range(self.n_agents)}

	
	def join_batches(self, batches):

		# Join the batches #
		ep_rewards, ep_action_log_probs, ep_value_preds, entropy, masks = zip(*batches)

		# Concatenate the batches #
		ep_rewards = torch.cat(ep_rewards, dim=1)
		ep_action_log_probs = torch.cat(ep_action_log_probs, dim=1)
		ep_value_preds = torch.cat(ep_value_preds, dim=1)
		entropy = torch.cat(entropy, dim=0).mean()
		masks = torch.cat(masks, dim=1)

		return ep_rewards, ep_action_log_probs, ep_value_preds, entropy, masks
	

	def train(self, episodes: int):
		""" Train for a number of episodes """

		# Create train logger #
		if self.writer is None:
			self.writer = SummaryWriter(log_dir=self.logdir, filename_suffix=self.experiment_name)

		for episode in trange(episodes):

			# Sample a batch of experiences in parallel using joblib #
			#batches = Parallel(n_jobs=self.n_envs)(delayed(self.sample_batch)(env_id) for env_id in range(self.n_envs))
			#batches = [self.sample_batch(env_id) for env_id in range(self.n_envs)]
			batches = Parallel(n_jobs=self.n_envs)(delayed(outside_sample_batch)(self.envs[env_id], self.n_steps, self.device, self.A2C) for env_id in range(self.n_envs))

			# Join the batches #
			ep_rewards, ep_action_log_probs, ep_value_preds, entropy, masks = self.join_batches(batches)

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

			# Publish the metrics #
			self.publish_metrics(ep_rewards.sum(dim=0).mean().item(), entropy.item(), actor_loss.item(), critic_loss.item(), episode)

			# Save the model #
			if episode % self.save_every == 0:
				self.save_model("policy_episode_{}.pth".format(episode))

	def publish_metrics(self, rewards, entropy, actor_loss, critic_loss, episode):
			""" Write in the Tensorboard the total final reward, the length of the episode and the losses """

			self.writer.add_scalar('Train/Total reward', rewards, episode)
			self.writer.add_scalar('Train/Actor loss', actor_loss, episode)
			self.writer.add_scalar('Train/Critic loss', critic_loss, episode)
			self.writer.add_scalar('Train/Entropy', entropy, episode)

	def load_model(self, path_to_file):

		self.A2C.load_state_dict(torch.load(path_to_file, map_location=self.device))

	def save_model(self, name='experiment.pth'):

		torch.save(self.A2C.state_dict(), self.writer.log_dir + '/' + name)
