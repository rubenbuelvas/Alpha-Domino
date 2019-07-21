import numpy as np
import random

class Environment():
	
	def __init__(self, agents, n_players=4, tiles_per_player=5):
		self.n_players = n_players
		self.agents = agents
		self.pile = generate_tiles()
		self.hand_sizes = []
		for agent in agents:
			for i in range(tiles_per_player):
				agent.hand.append(self.pile.pop())
				self.hand_sizes.append(len(agent.hand))
		self.table = []

	def recalculate_hand_sizes(self):
		for i in range(self.n_players):
			self.hand_sizes[i] = len(self.agents[i].hand)

	def get_observation(self):
		return -1

	def get_winner(self):
		self.recalculate_hand_sizes()
		winner = -1
		for i in range(len(self.hand_sizes)):
			if self.hand_sizes[i] == 0:
				winner = i
		return winner


class RandomAgent():
	def __init__(self):
		self.hand = []

	def act(self, observation):
		tile = self.hand.pop(random.randrange(len(self.hand)))
		return tile


def generate_tiles(max_value=6):
	tiles = []
	for i in range(max_value+1):
		for j in range(i+1):
			tiles.append([i, j])
	random.shuffle(tiles)
	return tiles


def turn_tile(tile):
	new_tile = [tile[1], tile[0]]
	return new_tile

	
def play(agents, env, verbose=False):
	winner = env.get_winner()
	turn = 0

	first_tile = [-1, -1]
	first_agent = 0
	first_tile_pos = 0
	for i in range(len(agents)):
		for j in range(len(agents[i].hand)):
			if agents[i].hand[j][0] > first_tile[0] and agents[i].hand[j][0] == agents[i].hand[j][1]:
				first_tile = agents[i].hand[j]
				first_agent = i
				first_tile_pos = j
	if first_tile != [-1, -1]:
		pass

	while(winner == -1):
		turn += 1
		for agent in agents:
			played_tile = agent.act(env.get_observation())
			winner = env.get_winner()
			if winner != -1:
				break

	print(f"Player {winner} won!")
		

agents = []
for i in range(4):
	agents.append(RandomAgent())

env = Environment(agents)

play(agents, env, verbose=True)