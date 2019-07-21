import numpy as np
import random

class Environment():
	
	def __init__(self, agents, n_players=4, tiles_per_player=7):
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
		observation = []
		observation.append(self.table)
		observation.append(self.hand_sizes)
		return observation

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
		play = 0
		for i in range(10):
			print(len(self.hand))
			pos = random.randrange(len(self.hand))
			tile = self.hand.pop(pos)
			play = playable_tile(observation[0], tile)
			if play != -1:
				break
		else:
			play = 0
		return play


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


def playable_tile(table, tile):
	playable = -1
	t_left = table[0]
	t_right = table[-1]
	if t_left[0] == tile[0]:
		playable = [tile, 0]
	elif t_left[0] == tile[1]:
		playable = [turn_tile(tile), 0]
	elif t_right[1] == tile[0]: 
		playable = [turn_tile(tile), 1]
	elif t_right[1] == tile[1]:
		playable = [tile, 1]
	return playable


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
		env.table.append(agents[first_agent].hand.pop(first_tile_pos))
		turn += 1
		for i in range(first_agent, len(agents)):
			played_tile = agents[i].act(env.get_observation())
			winner = env.get_winner()
			if winner != -1:
				break
	else:
		played_tile = pile.pop()
		pass
	while(winner == -1):
		print(env.table)
		turn += 1
		for agent in agents:
			played_tile = agent.act(env.get_observation())
			if played_tile != 0:
				if(played_tile[1] == 0):
					env.table.insert(0, played_tile[0])
				elif(played_tile[1] == 1):
					env.table.append(played_tile[0])
				else:
					print("Something went wrong")
			winner = env.get_winner()
			if winner != -1:
				break

	print(f"Player {winner} won!")
		

agents = []
for i in range(4):
	agents.append(RandomAgent())

env = Environment(agents)

play(agents, env, verbose=True)