import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import random

from config import Config
from  memory import Memory
from utils import plot_final_results

class DoubleDQNAgent:
	def __init__(self, device, memory_size, env, gamma=0.99, tau=0.01):
		self.memory = Memory(memory_size)
		self.device = device
		self.gamma = gamma
		self.tau = tau
		self.env = env

		self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
		self.target_model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
		
		for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
			target_param.data.copy_(param)

		self.optimizer = torch.optim.Adam(self.model.parameters())
		
		
	def get_action(self, state, eps=0.1, step=0):
		if random.random() < eps:
			return self.env.action_space.sample()

		state = torch.tensor(state).float().unsqueeze(0).to(self.device)
		q_vals = self.model(state)
		action = np.argmax(q_vals.cpu().detach().numpy())
		return action

	def compute_loss(self, batch):     
		states, actions, rewards, next_states, dones = batch
		self.model.train()

		states = torch.tensor(states).to(self.device)
		actions = torch.tensor(actions).to(self.device)
		rewards = torch.tensor(rewards).to(self.device)
		next_states = torch.tensor(next_states).to(self.device)
		dones = torch.tensor(dones).float().to(self.device)

		actions = actions.view(actions.size(0), 1)

		curr_q_val = self.model(states).gather(1, actions).squeeze()
		next_q_val = self.target_model(next_states)

		max_next_q_val = torch.max(next_q_val, 1)[0]
		# max_next_q_val = max_next_q_val.view(max_next_q_val.size(0), 1)

		target_q = rewards + (1 - dones) * self.gamma * max_next_q_val
		
		loss = F.mse_loss(curr_q_val, target_q.detach())
		return loss

	def update(self, batch_size, ep):
		batch = self.memory.sample(batch_size)
		loss = self.compute_loss(batch)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		if ep % 2 == 0:
			for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
				target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

	def save_model(self, file_name = None):
		if file_name is None:
			file_name = f"./saved_models/model_double_dqn.pt"
		torch.save(self.model.state_dict(), file_name)

	def load_model(self, file_name):
		self.model.load_state_dict(torch.load(file_name))

class DQN(nn.Module):
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