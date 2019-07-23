import numpy as np

from agents import *

class Environment():
	
	def __init__(self, agents, n_players=4, tiles_per_player=7):
		self.rewards = []
		self.tiles_per_player = tiles_per_player
		self.hand_sizes = []
		self.n_players = n_players
		self.agents = agents
		self.pile = generate_tiles()
		self.tiles_ids = {():0}
		self.is_game_over = False
		for agent in agents:
			for i in range(tiles_per_player):
				agent.hand.append(self.pile.pop())
				self.hand_sizes.append(len(agent.hand))
		self.table = []

	def new_episode(self):
		self.is_game_over = False
		self.pile = generate_tiles()
		for agent in agents:
			for i in range(tiles_per_player):
				agent.hand.append(self.pile.pop())
				self.hand_sizes.append(len(agent.hand))
		self.table = []

	def recalculate_hand_sizes(self):
		for i in range(self.n_players):
			self.hand_sizes[i] = len(self.agents[i].hand)

	def get_observation(self, agent_id):
		observation = []
		observation.append(self.table)
		self.recalculate_hand_sizes()
		observation.append(self.hand_sizes)
		return observation

	def get_observation_nn(self, agent_id):
		observation = []
		observation.append(len(self.agents[agent_id].hand))
		for i in range(self.tiles_per_player):
			if i < len(self.agents[agent_id].hand):
				observation.append(tiles_ids[tuple(self.agents[agent_id].hand[i])])
			else:
				observation.append(-1)
		for i in range(len(self.hand_sizes)):
			if agent_id != i:
				observation.append(self.hand_sizes[i])
		for i in range(len(self.agents)):
			if agent_id != i:
				observation.append(tiles_ids[tuple(self.agents[i].last_tile_played)])
				observation.append(str(self.agents[i].last_pos_played))
		for i in range(28):
			if i < len(self.table):
				observation.append(tiles_ids[tuple(self.table[i])])
			else:
				observation.append(-1)
		return observation


	def get_winner(self):
		winner = -1
		for i in range(len(agents)):
			if len(agents[i].hand) == 0:
				winner = i
		if winner != -1:
			self.is_game_over = True
		return winner

	def get_winner_2(self):
		print("Overtime")
		winner = -1
		min_ = self.tiles_per_player
		for i in range(len(agents)):
			if len(agents[i].hand) < min_:
				min_ = len(agents[i].hand)
				winner = i
			elif len(agents[i].hand) == min_:
				winner = -1
				break
		if winner != -1:
			self.is_game_over = True
		return winner

	def generate_tiles(self, max_value=6):
		tiles = []
		a=0
		for i in range(max_value+1):
			for j in range(i+1):
				my_tile = (i,j)
				tiles.append([i, j])
				tiles_ids[my_tile] = a
				tiles_ids[tuple(turn_tile(my_tile))] = a
				a += 1
		print(tiles_ids)
		random.shuffle(tiles)
		return tiles


	def turn_tile(self, tile):
		new_tile = [copy.copy(tile[1]), copy.copy(tile[0])]
		return new_tile


	def playable_tile(self, tile):
		playable = []
		t_left = self.table[0]
		t_right = self.table[-1]
		if t_left[0] == tile[1]:
			playable = [tile, 0]
		elif t_left[0] == tile[0]:
			playable = [turn_tile(tile), 0]
		elif t_right[1] == tile[1]: 
			playable = [turn_tile(tile), 1]
		elif t_right[1] == tile[0]:
			playable = [tile, 1]
		return playable