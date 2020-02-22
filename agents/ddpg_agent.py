import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import random

from config import Config
from utils import plot_final_results
from  memory import Memory


class DDPGAgent:
	def __init__(self, device, memory_size, env, gamma=0.99, tau=0.01):
		self.memory = Memory(memory_size)
		self.device = device
		self.gamma = gamma
		self.tau = tau
		self.env = env

		self.input_size = env.observation_space.shape[0]
		self.output_size = env.action_space.shape[0]

		self.actor = Actor(self.input_size, self.output_size).to(self.device)
		self.critic = Critic(self.input_size, self.output_size).to(self.device)

		self.actor_target = Actor(self.input_size, self.output_size).to(self.device)
		self.critic_target = Critic(self.input_size, self.output_size).to(self.device)

		self.noise = OUNoise(env.action_space)

		for target_param, param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(param)

		for target_param, param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(param)

		self.optimizer_actor = torch.optim.Adam(self.actor.parameters())
		self.optimizer_critic = torch.optim.Adam(self.critic.parameters())
		
		
	def get_action(self, state, eps=0.1, step=0):
		state = torch.tensor(state).float().unsqueeze(0).to(self.device)
		action = self.actor(state)
		action = action.squeeze().cpu().detach().numpy()
		return self.noise.get_action(action, step)

	def compute_loss(self, batch):     
		states, actions, rewards, next_states, dones = batch
		self.actor.train()
		self.critic.train()

		states = torch.tensor(states).to(self.device)
		actions = torch.tensor(actions).float().to(self.device)
		rewards = torch.tensor(rewards).to(self.device)
		next_states = torch.tensor(next_states).to(self.device)
		dones = torch.tensor(dones).float().to(self.device)

		# Critic Loss
		curr_q_value = self.critic(states, actions)
		next_action = self.actor(next_states)
		next_q_value = self.critic_target(states, next_action)

		target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
		critic_loss = F.mse_loss(curr_q_value, target_q_value.detach())

		# Actor Loss
		actor_loss = -self.critic(states, self.actor(states)).mean()

		return critic_loss, actor_loss

	def update(self, batch_size, ep):
		batch = self.memory.sample(batch_size)
		critic_loss, actor_loss = self.compute_loss(batch)

		self.optimizer_actor.zero_grad()
		actor_loss.backward()
		self.optimizer_actor.step()

		self.optimizer_critic.zero_grad()
		critic_loss.backward()
		self.optimizer_critic.step()
		
		if ep % 2 == 0:
			for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
				target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

			for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
				target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


	def save_model(self, file_name = None):
		if file_name is None:
			file_name = f"model_ddpg_{datetime.datetime.now()}.pt"
		torch.save(self.actor.state_dict(), file_name)

	def load_model(self, file_name):
		self.actor.load_state_dict(torch.load(file_name))



class Actor(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.input_size = input_size
		self.output_size = output_size
		
		self.fc = nn.Sequential(
			nn.Linear(self.input_size, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, self.output_size)
		)

	def forward(self, state):
		return self.fc(state)


class Critic(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.input_size = input_size
		self.output_size = output_size

		self.fc = nn.Sequential(
			nn.Linear(self.input_size + self.output_size, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		)

	def forward(self, state, action):
		input = torch.cat((state, action), 1)
		return F.tanh(self.fc(input))


class OUNoise():
	def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
		self.mu           = mu
		self.theta        = theta
		self.sigma        = max_sigma
		self.max_sigma    = max_sigma
		self.min_sigma    = min_sigma
		self.decay_period = decay_period
		self.action_dim   = action_space.shape[0]
		self.low          = action_space.low
		self.high         = action_space.high
		self.reset()
		
	def reset(self):
		self.state = np.ones(self.action_dim) * self.mu
		
	def evolve_state(self):
		x  = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
		self.state = x + dx
		return self.state
	
	def get_action(self, action, t=0):
		ou_state = self.evolve_state()
		self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
		return np.clip(action + ou_state, self.low, self.high)