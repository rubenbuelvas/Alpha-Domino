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
		self.tiles_ids = {[-1, -1]:0}
		self.ids_tiles = {0: [-1, -1]}
		self.is_game_over = False
		self.first_agent = -1
		self.turns_passed = 0
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
		for agent in agents:
			tmp = []
			for i in range(tiles_per_player):
				tmp.append(tiles_ids[agent.hand[i]])
			tmp.sort()
			for i in range(tiles_per_player):
				agent.hand[i] = ids_tiles[tmp[i]]
		self.table = []
		first_tile = [-1, -1]
		first_tile_pos = 0
		for i in range(len(self.agents)):
			for j in range(len(self.agents[i].hand)):
				if self.agents[i].hand[j][0] > first_tile[0] and self.agents[i].hand[j][0] == self.agents[i].hand[j][1]:
					first_tile = self.agents[i].hand[j]
					self.first_agent = i
					first_tile_pos = j
		self.table.append(self.agents[first_agent].hand.pop(first_tile_pos))

	def recalculate_hand_sizes(self):
		for i in range(self.n_players):
			self.hand_sizes[i] = len(self.agents[i].hand)

	def get_observation(self, agent_id):
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

	def observation_shape(self):
		return len(self.get_observation())

	def get_winner(self):
		winner = -1
		for i in range(len(agents)):
			if len(agents[i].hand) == 0:
				winner = i
		if winner != -1:
			self.is_game_over = True
		return winner

	def get_winner_overtime(self):
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
				ids_tiles[a] = my_tile
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

	def choose_action(self, action, agent_id):
		self.current_action = action
        for i in action

	def timestep(self, agent_id):

        self.timestep_index += 1
        reward = 0

        if self.snake.peek_next_move() == self.fruit:
            self.snake.grow()
            self.generate_fruit()
            old_tail = None
            reward += self.rewards['ate_fruit'] * self.snake.length
            self.stats.fruits_eaten += 1

        # If not, just move forward.
        else:
            self.snake.move()
            reward += self.rewards['timestep']

        self.field.update_snake_footprint(old_head, old_tail, self.snake.head)

        # Hit a wall or own body?
        if not self.is_alive():
            if self.has_hit_wall():
                self.stats.termination_reason = 'hit_wall'
            if self.has_hit_own_body():
                self.stats.termination_reason = 'hit_own_body'

            self.field[self.snake.head] = CellType.SNAKE_HEAD
            self.is_game_over = True
            reward = self.rewards['died']


        result = TimestepResult(
            observation=self.get_observation_nn(),
            reward=reward,
            is_game_over=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result


def play(agents, env, verbose=False):
	winner = env.get_winner()
	turn = 0
	passed_count = 0
	first_tile = [-1, -1]
	first_agent = 0
	first_tile_pos = 0
	for i in range(len(agents)):
		for j in range(len(agents[i].hand)):
			if env.agents[i].hand[j][0] > first_tile[0] and env.agents[i].hand[j][0] == agents[i].hand[j][1]:
				first_tile = agents[i].hand[j]
				first_agent = i
				first_tile_pos = j
	env.table.append(agents[first_agent].hand.pop(first_tile_pos))
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