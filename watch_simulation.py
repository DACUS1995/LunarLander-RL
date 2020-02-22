import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse

from config import Config
from agents.dqn_agent  import DQNAgent
from  memory import Memory
from utils import plot_final_results


def get_environment():
	pass


def get_agent(device, env):
	return DQNAgent(device, 0, env)


def run_simulation(max_steps, num_episodes = 1):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	env = get_environment()

	agent = get_agent(device, env)

	with torch.no_grad():
		for episode in range(num_episodes):
			state = env.reset()
			episode_reward = 0

			for step in range(max_steps):
				action = agent.get_action(state)

				env.render()
				next_state, reward, done, _ = env.step(action)
				episode_reward += reward
				state = next_state

				if done:
					break

			# End of episode
			episode_rewards.append(episode_reward)

	return episode_rewards



def main(args):
	run_simulation(
		max_steps=args.max_steps, 
		num_episodes=args.episodes
	)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-ep", "--episodes", type=int, default=100, help="number of episodes")
	parser.add_argument("--max-steps", type=int, default=1000, help="max steps for episode")
	args = parser.parse_args()
	main(args)