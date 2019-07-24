import json
import sys
import numpy as np
import platform

from environment import *
from agents import *

def play(env, agents, num_episodes=1, verbose=True):

    winner = -1

    print()
    print('Playing:')

    for episode in range(num_episodes):

        timestep = env.new_episode()
        stats = np.zeros(len(agents))
        game_over = False
        step = 0

        for i in range(env.first_agent, len(agents)):
        	timestep = env.timestep(i)
        	action = agents[i].act(timestep.observation, timestep.reward)
        	env.choose_action(action)
        	game_over = env.is_game_over
        step += 1
        while not game_over:

        	for i in range(first_agent, len(agents)):
        		if verbose:
        			print("-------------------------------")
        			print(f"Player {i} hand: {agents[i].hand}")
        		actions = agent.act(timestep.observation, timestep.reward)
        		action = env.choose_action(actions)
        		if verbose:
        			print(f"Player {i} played {action}")
        		timestep = env.timestep()
        		game_over = env.is_game_over
        	step += 1
           
        winner = env.get_winner()

        stats[winner] += 1

        print(f"******* Episode {episode+1} / {num_episodes} | Timesteps {step} | Player {winner} won")
        print(summary.format(episode + 1, num_episodes, env.stats.timesteps_survived, env.stats.fruits_eaten))

    print()
    print("Results:")
    print(f"Player 0: {stats[0]} | Player 1: {stats[1]} | Player 2: {stats[2]} | Player 3: {stats[3]}")


if platform.system() == "Windows":
	print("cls")
else:
	print("clear")

agents = []
agents.append(RandomAgent())
agents.append(RandomAgent())
agents.append(RandomAgent())
agents.append(RandomAgent())
play(Environment(agents), agents)