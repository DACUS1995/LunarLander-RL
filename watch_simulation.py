import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import gym

from config import Config
from agents.double_dqn_agent import DoubleDQNAgent
from agents.ddpg_agent import DDPGAgent
from  memory import Memory
from utils import plot_final_results


def get_environment(mode="discrete"):
	if mode == "discrete":
		return gym.make("LunarLander-v2")
	else:
		return gym.make("LunarLanderContinuous-v2")

def get_agent(device, env, mode="discrete"):
	if mode == "discrete":
		agent = DoubleDQNAgent(device, 0, env)
		agent.load_model("./saved_models/model_double_dqn.pt")
		return agent
	else:
		agent = DDPGAgent(device, 0, env)
		agent.load_model("./saved_models/model_ddpg.pt")
		return agent


def run_simulation(max_steps, num_episodes = 1, mode="discrete"):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	env = get_environment(mode)
	agent = get_agent(device, env, mode)

	with torch.no_grad():
		for episode in range(num_episodes):
			state = env.reset()
			episode_reward = 0

			for step in range(max_steps):
				action = agent.get_action(state, 0)

				env.render()
				next_state, reward, done, _ = env.step(action)
				episode_reward += reward
				state = next_state

				if done:
					break

			# End of episode
			# episode_rewards.append(episode_reward)

	return



def main(args):
	run_simulation(
		max_steps=args.max_steps, 
		num_episodes=args.episodes,
		mode=args.mode
	)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-ep", "--episodes", type=int, default=100, help="number of episodes")
	parser.add_argument("--max-steps", type=int, default=1000, help="max steps for episode")
	parser.add_argument("--mode", type=str, default="discrete", help="environment mode")
	args = parser.parse_args()
	main(args)