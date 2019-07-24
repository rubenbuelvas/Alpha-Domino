import json
import sys
import numpy as np
import platform
import os

from environment import *
from agents import *

def load_model(filename):
    """ Load a pre-trained agent model. """
    from keras.models import load_model
    return load_model(filename)


def play(env, agents, num_episodes=1, verbose=True):

    winner = -1

    print()
    print('Playing:')

    for episode in range(num_episodes):

        stats = np.zeros(len(agents))
        game_over = False
        step = 0
        timestep = env.new_episode()

        for i in range(env.first_agent + 1, len(agents)):
            if verbose:
                print("-------------------------------")
                print(f"Player {i} hand: {agents[i].hand}")
                print(f"Table {env.table}")
            actions = agents[i].act(timestep.observation, timestep.reward)
            action = env.choose_action(actions, i)
            if verbose:
                print(f"Player {i} played {action}")
            timestep = env.timestep(i)
            game_over = env.is_game_over
            if game_over:
            	break
        step += 1
        while not game_over:

            for i in range(len(agents)):
                if verbose:
                    print("-------------------------------")
                    print(f"Player {i} hand: {agents[i].hand}")
                    print(f"Table {env.table}")
                actions = agents[i].act(timestep.observation, timestep.reward)
                action = env.choose_action(actions, i)
                if verbose:
                    print(f"Player {i} played {action}")
                timestep = env.timestep(i)
                game_over = env.is_game_over
                winner = env.winner
                if game_over:
                	break
            step += 1

        if winner == -1:
            print()
            print(f"******* Episode {episode+1} / {num_episodes} | Timesteps {step} | Tie")
        else:
            stats[winner] += 1
            print()
            print(f"******* Episode {episode+1} / {num_episodes} | Timesteps {step} | Player {winner} won")

    print()
    print("Results:")
    print(f"Player 0: {int(stats[0])} | Player 1: {int(stats[1])} | Player 2: {int(stats[2])} | Player 3: {int(stats[3])}")


if platform.system() == "Windows":
    os.system("cls")
else:
    os.system("clear")

agents = []
agents.append(DeepQNetworkAgent(model=load_model("dqn-final.model")))
agents.append(RandomAgent())
agents.append(RandomAgent())
agents.append(RandomAgent())
play(Environment(agents), agents)