import numpy as np
import argparse
from argparse import Namespace
from collections import OrderedDict
import time
import os
import copy
import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import gym

from config import Config
from agents.double_dqn_agent import DoubleDQNAgent
from agents.ddpg_agent import DDPGAgent
from  memory import Memory
from utils import plot_final_results


def get_environment(mode="discret"):
	if mode == "discret":
		return gym.make("LunarLander-v2")
	else:
		return gym.make("LunarLanderContinuous-v2")


def train(train_config):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	env = get_environment(train_config.mode)
	agent = None

	if train_config.mode == "discret":
		agent = DoubleDQNAgent(device, train_config.replay_memory_size, env)
	else:
		agent = DDPGAgent(device, train_config.replay_memory_size, env)
	

	episode_rewards = []
	episode_loss = []
	evaluation_rewards = []


	for episode in range(train_config.episodes):
		state = env.reset()
		episode_reward = 0

		for step in range(train_config.max_steps):
			action = agent.get_action(state, eps=0.1, step=step)

			next_state, reward, done, _ = env.step(action)
			agent.memory.push(state, action, reward, next_state, done)
			episode_reward += reward

			if len(agent.memory) > train_config.batch_size:
				agent.update(train_config.batch_size, episode)

			state = next_state

			if done:
				break

		# End of episode
		episode_rewards.append(episode_reward)
		print("[Training] Episode " + str(episode) + ": " + str(episode_reward))

		if episode % 5 == 0:
			curr_rewards = evaluate(agent, env, 1, True)
			evaluation_rewards.append(sum(curr_rewards) / len(curr_rewards))


		if hasattr(agent, "noise"):
			agent.noise.reset()


	plot_final_results({
		"Rewards": episode_rewards,
		"Evaluation": evaluation_rewards
	})


def evaluate(agent, env, num_episodes = 1, render = False, num_steps = 50000):
	episode_rewards = []

	with torch.no_grad():
		for episode in range(num_episodes):
			state = env.reset()
			episode_reward = 0

			for step in range(num_steps):
				action = agent.get_action(state)

				if render == True:
					env.render()

				next_state, reward, done, _ = env.step(action)
				episode_reward += reward
				state = next_state

				if done:
					break
			
			print(f"[Evaluation] episode reward: {episode_reward}")

			# End of episode
			episode_rewards.append(episode_reward)

	return episode_rewards


def main(args: Namespace) -> None:
	train_config = None
	if Config.use_config == True:
		train_config = Config
	else:
		train_config = args

		agent = train(train_config)
		
		if train_config.save == True:
			agent.save_model()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, default="continuous", help="mode of env")
	parser.add_argument("-ep", "--episodes", type=int, default=1000, help="number of episodes")
	parser.add_argument("-mem", "--replay-memory-size", type=int, default=10000, help="replay memory size")
	parser.add_argument("--max-steps", type=int, default=1000, help="max steps for episode")
	parser.add_argument("--batch-size", type=int, default=32, help="model optimization batch sizes")
	parser.add_argument("--save", type=bool, default=False, help="save the trained model")
	args = parser.parse_args()
	main(args)