import numpy as np
import random
import collections

import json
import sys

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from environment import *

class RandomAgent():

    def __init__(self, ):
        self.hand = []
        self.hand_ids = []
        self.last_tile_played = [-1, -1]
        self.last_pos_played = -1

    def act(self, observation, reward):
        actions = np.zeros(29)
        for i in range(len(actions)):
            if i in self.hand_ids:
                actions[i] = random.uniform(0.5, 1)
        return actions

class HumanAgent():

    def __init__(self):
        self.hand = []

    def act():
        pass


class DeepQNetworkAgent():

    def __init__(self, model, num_last_frames=0, memory_size=-1):
        """
        Create a new DQN-based agent.
        
        Args:
            model: a compiled DQN model.
            num_last_frames (int): the number of last frames the agent will consider.
            memory_size (int): memory size limit for experience replay (-1 for unlimited). 
        """
        assert model.input_shape[1] == num_last_frames, 'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model.output_shape) == 2, 'Model output shape should be (num_samples, num_actions)'
        self.hand = []
        self.hand_ids = []
        self.last_tile_played = [-1, -1]
        self.last_pos_played = -1
        self.model = model
        self.num_last_frames = num_last_frames
        self.memory = ExperienceReplay((num_last_frames,) + model.input_shape[-2:], model.output_shape[-1], memory_size)
        #self.frames = None

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
        return none
        #return np.expand_dims(self.frames, 0)

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
            self.timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss = 0.0

            # Observe the initial state.
            state = self.timestep.observation


            stats = np.zeros(len(env.agents))
            game_over = False
            step = 0
            timestep = env.new_episode()

            for i in range(env.first_agent + 1, len(env.agents)):
                if i == 0:
                    if np.random.random() < exploration_rate:
                        # Explore: take a random action.
                        actions = np.zeros(29)
                        for i in range(len(actions)):
                            if i in self.hand_ids:
                                actions[i] = random.uniform(0.5, 1)
                        action = actions
                    else:
                        # Exploit: take the best known action for this state.
                        q = self.model.predict(state)
                        action = np.argmax(q[0])

                    # Act on the environment.
                    env.choose_action(action, 0)
                    timestep = env.timestep(0)

                    # Remember a new piece of experience.
                    reward = timestep.reward
                    state_next = self.timestep.observation
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
                else:

                    actions = env.agents[i].act(timestep.observation, timestep.reward)
                    action = env.choose_action(actions, i)

                    timestep = env.timestep(i)
                    game_over = env.is_game_over
                    if game_over:
                        break
            step += 1
            while not game_over:

                for i in range(len(env.agents)):
                    if i == 0:
                        if np.random.random() < exploration_rate:
                            # Explore: take a random action.
                            actions = np.zeros(29)
                            for i in range(len(actions)):
                                if i in self.hand_ids:
                                    actions[i] = random.uniform(0.5, 1)
                            action = actions
                        else:
                            # Exploit: take the best known action for this state.
                            q = self.model.predict(state)
                            action = np.argmax(q[0])

                        # Act on the environment.
                        env.choose_action(action, 0)
                        timestep = env.timestep(0)

                        # Remember a new piece of experience.
                        reward = timestep.reward
                        state_next = self.timestep.observation
                        game_over = timestep.is_game_over
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
                    else:

                        actions = agents[i].act(timestep.observation, timestep.reward)
                        action = env.choose_action(actions, i)

                        timestep = env.timestep(i)
                        game_over = env.is_game_over
                        if game_over:
                            break

            if checkpoint_freq and (episode % checkpoint_freq) == 0:
                self.model.save(f'dqn-{episode:08d}.model')

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay
            """
            summary = 'Episode {:5d}/{:5d} | Loss {:8.4f} | Exploration {:.2f} | ' + \
                      'Timesteps {:4d} | Total Reward {:4d}'
            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate, env.stats.timesteps, env.stats.sum_episode_rewards,
            ))
            """

        self.model.save('dqn-final.model')

    def act(self, observation, reward):
        state = self.observation
        q = self.model.predict(state)[0]
        return q


class ExperienceReplay():
    """ Represents the experience replay memory that can be randomly sampled. """

    def __init__(self, input_shape, num_actions, memory_size=100):
        """
        Create a new instance of experience replay memory.
        
        Args:
            input_shape: the shape of the agent state.
            num_actions: the number of actions allowed in the environment.
            memory_size: memory size limit (-1 for unlimited).
        """
        self.memory = collections.deque()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.memory_size = memory_size

    def reset(self):
        """ Erase the experience replay memory. """
        self.memory = collections.deque()

    def remember(self, state, action, reward, state_next, is_episode_end):
        """
        Store a new piece of experience into the replay memory.
        
        Args:
            state: state observed at the previous step.
            action: action taken at the previous step.
            reward: reward received at the beginning of the current step.
            state_next: state observed at the current step. 
            is_episode_end: whether the episode has ended with the current step.
        """
        memory_item = np.concatenate([
            state,
            np.array(action).flatten(),
            np.array(reward).flatten(),
            state_next,
            1 * np.array(is_episode_end).flatten()
        ])
        self.memory.append(memory_item)
        if 0 < self.memory_size < len(self.memory):
            self.memory.popleft()

    def get_batch(self, model, batch_size, discount_factor=0.9):
        """ Sample a batch from experience replay. """

        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions = experience[:, input_dim]
        rewards = experience[:, input_dim + 1]
        states_next = experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = experience[:, 2 * input_dim + 2]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions = np.cast['int'](actions)
        rewards = rewards.repeat(self.num_actions).reshape((batch_size, self.num_actions))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions).reshape((batch_size, self.num_actions))

        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = model.predict(X)
        Q_next = np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions))

        delta = np.zeros((batch_size, self.num_actions))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        return states, targets
