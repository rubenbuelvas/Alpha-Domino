import numpy as np
import random
import copy

from agents import *

class Environment():
    
    def __init__(self, agents, n_players=4, tiles_per_player=7):
        self.rewards = []
        self.tiles_per_player = tiles_per_player
        self.hand_sizes = []
        self.n_players = n_players
        self.agents = agents
        self.tiles_ids = {(-1, -1): 0}
        self.ids_tiles = {0: (-1, -1)}
        self.pile = self.generate_tiles()
        self.is_game_over = False
        self.first_agent = -1
        self.turns_passed = 0
        self.current_action = [[-1, -1], -1]
        self.timestep_index = 0
        for agent in agents:
            for i in range(tiles_per_player):
                agent.hand.append(self.pile[0])
                agent.hand_ids.append(self.tiles_ids[tuple(self.pile.pop())])
                self.hand_sizes.append(len(agent.hand))
        self.table = []

    def new_episode(self):
        self.is_game_over = False
        self.pile = self.generate_tiles()
        for agent in self.agents:
            for i in range(self.tiles_per_player):
                agent.hand.append(self.pile.pop())
                self.hand_sizes.append(len(agent.hand))
        for agent in self.agents:
            tmp = []
            for i in range(self.tiles_per_player):
                tmp.append(self.tiles_ids[tuple(agent.hand[i])])
            tmp.sort()
            for i in range(self.tiles_per_player):
                agent.hand[i] = self.ids_tiles[tmp[i]]
        self.table = []
        first_tile = [-1, -1]
        first_tile_pos = 0
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i].hand)):
                if self.agents[i].hand[j][0] > first_tile[0] and self.agents[i].hand[j][0] == self.agents[i].hand[j][1]:
                    first_tile = self.agents[i].hand[j]
                    self.first_agent = i
                    first_tile_pos = j
        self.table.append(self.agents[self.first_agent].hand.pop(first_tile_pos))
        result = TimestepResult(
            observation=self.get_observation(0),
            reward=0,
            is_game_over=self.is_game_over
        )
        return result

    def recalculate_hand_sizes(self):
        for i in range(self.n_players):
            self.hand_sizes[i] = len(self.agents[i].hand)

    def get_observation(self, agent_id):
        observation = []
        observation.append(len(self.agents[agent_id].hand))
        for i in range(self.tiles_per_player):
            if i < len(self.agents[agent_id].hand):
                observation.append(self.tiles_ids[tuple(self.agents[agent_id].hand[i])])
            else:
                observation.append(-1)
        for i in range(len(self.hand_sizes)):
            if agent_id != i:
                observation.append(self.hand_sizes[i])
        for i in range(len(self.agents)):
            if agent_id != i:
                observation.append(self.tiles_ids[tuple(self.agents[i].last_tile_played)])
                observation.append(str(self.agents[i].last_pos_played))
        observation.append(len(self.table))
        observation.append(self.tiles_ids[tuple(self.table[0])])
        observation.append(self.tiles_ids[tuple(self.table[-1])])
        for i in range(28):
            if i < len(self.table):
                observation.append(self.tiles_ids[tuple(self.table[i])])
            else:
                observation.append(-1)
        return observation

    def observation_shape(self):
        return len(self.get_observation())

    def get_winner_zero(self):
        winner = -1
        for i in range(len(self.agents)):
            if len(self.agents[i].hand) == 0:
                winner = i
        if winner != -1:
            self.is_game_over = True
        return winner

    def get_winner(self):
        self.is_game_over = True
        winner = -1
        min_ = self.tiles_per_player
        for i in range(len(self.agents)):
            if len(self.agents[i].hand) < min_:
                min_ = len(self.agents[i].hand)
                winner = i
            elif len(self.agents[i].hand) == min_:
                winner = -1
                break
        return winner

    def turn_tile(self, tile):
        new_tile = [copy.copy(tile[1]), copy.copy(tile[0])]
        return new_tile

    def generate_tiles(self, max_value=6):
        tiles = []
        a=0
        for i in range(max_value+1):
            for j in range(i+1):
                my_tile = (i,j)
                tiles.append([i, j])
                self.ids_tiles[a] = my_tile
                self.tiles_ids[my_tile] = a
                self.tiles_ids[tuple(self.turn_tile(my_tile))] = a
                a += 1
        random.shuffle(tiles)
        return tiles

    def get_play(self, tile):
        play = [[-1, -1], -1]
        if tile == [-1, -1]:
            return play
        t_left = self.table[0]
        t_right = self.table[-1]
        if t_left[0] == tile[1]:
            play = [tile, 0]
        elif t_left[0] == tile[0]:
            play = [self.turn_tile(tile), 0]
        elif t_right[1] == tile[1]: 
            play = [self.turn_tile(tile), 1]
        elif t_right[1] == tile[0]:
            play = [tile, 1]
        return play

    def choose_action(self, actions, agent_id):
        current_action = [[-1, -1], -1]
        for i in range(len(actions)):
            tile_id = np.argmax(np.array(actions))
            tile = self.ids_tiles[tile_id]
            play = self.get_play(tile)
            if play != [[-1, -1], -1] and tile in self.agents[agent_id].hand:
                current_action = play
                break
        self.current_action = current_action
        self.agents[agent_id].last_tile_played = copy.deepcopy(current_action[0])
        self.agents[agent_id].last_pos_played = copy.copy(current_action[1])
        return current_action
        
    def timestep(self, agent_id):

        self.timestep_index += 1

        reward = 0
        if self.current_action != [[-1, -1], -1]:
            self.turns_passed = 0
            if self.current_action[1] == 0:
                self.table.insert(0, self.current_action[0])
            elif self.current_action[1] == 1:
                self.table.append(self.current_action[0]) 
            reward += 5
        else:
            self.turns_passed += 1

        print(self.turns_passed)

        reward += self.tiles_per_player - len(self.agents[agent_id].hand)

        for i in range(len(self.agents)):
            if i != agent_id:
                if self.agents[agent_id].last_pos_played == -1:
                    reward += 15
        
        if self.get_winner_zero() == agent_id:
            reward += 1000

        print(len(self.agents))
        if self.turns_passed == len(self.agents):
            if self.get_winner() == agent_id:
                reward += 800

        result = TimestepResult(
            observation=self.get_observation(agent_id),
            reward=reward,
            is_game_over=self.is_game_over
        )
        #self.record_timestep_stats(result)
        return result


class TimestepResult():

    def __init__(self, observation, reward, is_game_over):
        self.observation = observation
        self.reward = reward
        self.is_game_over = is_game_over

    def __str__(self):
        return f"obs : {self.observation}\nr : {self.reward}\n end : {self.is_game_over}"