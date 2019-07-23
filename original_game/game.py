import numpy as np
import random
import copy

class Environment():
	
	def __init__(self, agents, n_players=4, tiles_per_player=7):
		self.tiles_per_player = tiles_per_player
		self.hand_sizes = []
		self.n_players = n_players
		self.agents = agents
		self.pile = generate_tiles()
		for agent in agents:
			for i in range(tiles_per_player):
				agent.hand.append(self.pile.pop())
				self.hand_sizes.append(len(agent.hand))
		self.table = []

	def new_episode(self):
		self.pile = generate_tiles()
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
		return winner


class RandomAgent():
	def __init__(self):
		self.hand = []
		self.last_tile_played = []
		self.last_pos_played = -1

	def act(self, observation):
		play = []
		for i in range(10):
			pos = random.randrange(len(self.hand))
			tile = self.hand[pos]
			play = playable_tile(observation[0], tile)
			if play != []:
				self.hand.pop(pos)
				break
		else:
			play = []

		if play == []:
			last_tile_played = -1
			last_pos_played = -1
		else:	
			last_tile_played = play[0]
			last_pos_played = play[1]
		return play


class DeepQNetworkAgent(AgentBase):
    """ Represents a Snake agent powered by DQN with experience replay. """

    def __init__(self, model, num_last_frames=4, memory_size=1000):
        """
        Create a new DQN-based agent.
        
        Args:
            model: a compiled DQN model.
            num_last_frames (int): the number of last frames the agent will consider.
            memory_size (int): memory size limit for experience replay (-1 for unlimited). 
        """
        assert model.input_shape[1] == num_last_frames, 'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model.output_shape) == 2, 'Model output shape should be (num_samples, num_actions)'

        self.model = model
        self.num_last_frames = num_last_frames
        self.memory = ExperienceReplay((num_last_frames,) + model.input_shape[-2:], model.output_shape[-1], memory_size)
        self.frames = None

    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.frames = None

    def get_last_frames(self, observation):
        """
        Get the pixels of the last `num_last_frames` observations, the current frame being the last.
        
        Args:
            observation: observation at the current timestep. 

        Returns:
            Observations for the last `num_last_frames` frames.
        """
        frame = observation
        if self.frames is None:
            self.frames = collections.deque([frame] * self.num_last_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.expand_dims(self.frames, 0)

    def train(self, env, num_episodes=1000, batch_size=50, discount_factor=0.9, checkpoint_freq=None,
              exploration_range=(1.0, 0.1), exploration_phase_size=0.5):
        """
        Train the agent to perform well in the given Snake environment.
        
        Args:
            env:
                an instance of Snake environment.
            num_episodes (int):
                the number of episodes to run during the training.
            batch_size (int):
                the size of the learning sample for experience replay.
            discount_factor (float):
                discount factor (gamma) for computing the value function.
            checkpoint_freq (int):
                the number of episodes after which a new model checkpoint will be created.
            exploration_range (tuple):
                a (max, min) range specifying how the exploration rate should decay over time. 
            exploration_phase_size (float):
                the percentage of the training process at which
                the exploration rate should reach its minimum.
        """

        # Calculate the constant exploration decay speed for each episode.
        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (num_episodes * exploration_phase_size))
        exploration_rate = max_exploration_rate

        for episode in range(num_episodes):
            # Reset the environment for the new episode.
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss = 0.0

            # Observe the initial state.
            state = self.get_last_frames(timestep.observation)

            while not game_over:
                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action = np.random.randint(env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    q = self.model.predict(state)
                    action = np.argmax(q[0])

                # Act on the environment.
                env.choose_action(action)
                timestep = env.timestep()

                # Remember a new piece of experience.
                reward = timestep.reward
                state_next = self.get_last_frames(timestep.observation)
                game_over = timestep.is_episode_end
                experience_item = [state, action, reward, state_next, game_over]
                self.memory.remember(*experience_item)
                state = state_next

                # Sample a random batch from experience.
                batch = self.memory.get_batch(
                    model=self.model,
                    batch_size=batch_size,
                    discount_factor=discount_factor
                )
                # Learn on the batch.
                if batch:
                    inputs, targets = batch
                    loss += float(self.model.train_on_batch(inputs, targets))

            if checkpoint_freq and (episode % checkpoint_freq) == 0:
                self.model.save(f'dqn-{episode:08d}.model')

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            summary = 'Episode {:5d}/{:5d} | Loss {:8.4f} | Exploration {:.2f} | ' + \
                      'Fruits {:2d} | Timesteps {:4d} | Total Reward {:4d}'
            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))

        self.model.save('dqn-final.model')

    def act(self, observation, reward):
        """
        Choose the next action to take.
        
        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.

        Returns:
            The index of the action to take next.
        """
        state = self.get_last_frames(observation)
        q = self.model.predict(state)[0]
        return np.argmax(q)



tiles_ids = {():0}
def generate_tiles(max_value=6):
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


def turn_tile(tile):
	new_tile = [copy.copy(tile[1]), copy.copy(tile[0])]
	return new_tile


def playable_tile(table, tile):
	playable = []
	t_left = table[0]
	t_right = table[-1]
	if t_left[0] == tile[1]:
		playable = [tile, 0]
	elif t_left[0] == tile[0]:
		playable = [turn_tile(tile), 0]
	elif t_right[1] == tile[1]: 
		playable = [turn_tile(tile), 1]
	elif t_right[1] == tile[0]:
		playable = [tile, 1]
	return playable


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
		

agents = []
for i in range(4):
	agents.append(RandomAgent())

for i in range(1):
	env = Environment(agents)
	play(agents, env, verbose=True)
