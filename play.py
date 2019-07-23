import json
import sys
import numpy as np
import platform

from environment import *
from agents import *

def play(agents, env, verbose=False):
	winner = env.get_winner()
	turn = 0
	passed_count = 0
	first_tile = [-1, -1]
	first_agent = 0
	first_tile_pos = 0
	turn += 1
	for i in range(first_agent, len(agents)):
		print("-------------------------------")
		print(f"Table: {env.table}")
		print(f"Player {i} hand: {agents[i].hand}")
		played_tile = agents[i].act(env.get_observation())
		if len(played_tile) > 0:
			print(f"Player {i} played {played_tile}")
			if(played_tile[1] == 0):
				env.table.insert(0, played_tile[0])
			elif(played_tile[1] == 1):
				env.table.append(played_tile[0])
			else:
				print("Something went wrong")
		else:
			passed_count += 1
			print(f"Player {i} passed")
		winner = env.get_winner()
		if winner != -1:
			break
	while(winner == -1):
		print(env.get_observation_nn(0))
		if winner != -1:
			break
		turn += 1
		if passed_count == len(agents):
			winner = env.get_winner_2()
			break
		passed_count = 0
		for i in range(len(agents)):
			print("-------------------------------")
			print(f"Table: {env.table}")
			print(f"Player {i} hand: {agents[i].hand}")
			played_tile = agents[i].act(env.get_observation())
			if len(played_tile) > 0:
				print(f"Player {i} played {played_tile}")
				if(played_tile[1] == 0):
					env.table.insert(0, played_tile[0])
				elif(played_tile[1] == 1):
					env.table.append(played_tile[0])
				else:
					print("Something went wrong")
			else:
				passed_count += 1
				print(f"Player {i} passed")
			winner = env.get_winner()
			if winner != -1:
				break
	if winner == -1:
		print()
		print("Tie!")
	else:
		print()
		print(f"Player {winner} won!")


def play(env, agent, num_episodes=1, verbose=True):

    winner = -1

    print()
    print('Playing:')

    for episode in range(num_episodes):

        timestep = env.new_episode()
        agent.begin_episode()
        game_over = False
        step = 0

        for i in range(first_agent, len(agents)):
        	if verbose:
				print("-------------------------------")
				print(f"Table: {env.table}")
				print(f"Player {i} hand: {agents[i].hand}")
			played_tile = agents[i].act(env.get_observation())
			if len(played_tile) > 0:
				if verbose:
					print(f"Player {i} played {played_tile}")
				if(played_tile[1] == 0):
					env.table.insert(0, played_tile[0])
				elif(played_tile[1] == 1):
					env.table.append(played_tile[0])
				else:
					if verbose:
						print("Something went wrong")
			else:
				if verbose:
					print(f"Player {i} passed")
			winner = env.get_winner()
			if winner != -1:
				break
		step += 1
        while not game_over:

        	for i in range(first_agent, len(agents)):
	        	if verbose:
					print("-------------------------------")
					print(f"Table: {env.table}")
					print(f"Player {i} hand: {agents[i].hand}")
				action = agent.act(timestep.observation, timestep.reward)
            	env.choose_action(action)
            	timestep = env.timestep()
            	game_over = timestep.is_episode_end


				played_tile = agents[i].act(env.get_observation())
				if len(played_tile) > 0:
					if verbose:
						print(f"Player {i} played {played_tile}")
					if(played_tile[1] == 0):
						env.table.insert(0, played_tile[0])
					elif(played_tile[1] == 1):
						env.table.append(played_tile[0])
					else:
						if verbose:
							print("Something went wrong")
				else:
					if verbose:
						print(f"Player {i} passed")
				winner = env.get_winner()
				if winner != -1:
					break
            step += 1
            

        fruit_stats.append(env.stats.fruits_eaten)

        summary = '******* Episode {:3d} / {:3d} | Timesteps {:4d} | Fruits {:2d}'
        print(summary.format(episode + 1, num_episodes, env.stats.timesteps_survived, env.stats.fruits_eaten))

    print()
    print(f"Player {winner} won!")


if platform.system() == "Windows":
	print("cls")
else:
	print("clear")