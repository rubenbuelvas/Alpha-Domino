import numpy as np
import random
import collections
import statistics 
import json
import sys

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from environment import *

class RandomAgent(object):

    def __init__(self):
        self.hand = []
        self.hand_ids = []
        self.last_tile_played = (-1, -1)
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


class DeepQNetworkAgent(object):

    def __init__(self, model, num_actions, epsilon=1.8, batch_size=32, input_dims=48, alpha=0.0085, gamma=0.99, epsilon_dec=0.99, epsilon_end=0.01, memory_size=1000000, f_name="dqn_model.model"):
        self.action_space = [i for i in range(num_actions)]
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.model_file = f_name

        self.memory = ReplayBuffer(memory_size, input_dims, num_actions)
        self.q_eval = model

        self.hand = []
        self.hand_ids = []
        self.last_tile_played = (-1, -1)
        self.last_pos_played = -1

    def remember(self, observation, action, reward, new_observation, done):
        self.memory.store_transition(observation, action, reward, new_observation, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        observation, action, reward, new_observation, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int32)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict((observation))
        q_next = self.q_eval.predict((np.array(new_observation)))

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done

        t = self.q_eval.fit(observation, q_target, verbose=0)
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end 

    def train(self, env, num_episodes=3000):
        scores = []
        try:
            self.q_eval.load_model(self.model_file)
        except:
            pass
        for episode in range(num_episodes):
            env.new_episode()
            game_over = False
            loss = 0.0
            stats = np.zeros(len(env.agents))
            game_over = False
            step = 0
            score = 0
            timestep = env.new_episode()
            observation = timestep.observation

            for i in range(env.first_agent + 1, len(env.agents)):
                if i == 0:
                    if np.random.random() < self.epsilon:
                        actions = np.zeros(29)
                        for i in range(len(actions)):
                            if i in self.hand_ids:
                                actions[i] = random.uniform(0.5, 1)
                    else:
                        q = self.q_eval.predict(observation)
                        actions = q[0]

                    observation_ = timestep.observation
                    reward = timestep.reward
                    score += reward
                    self.remember(observation, env.tiles_ids[env.current_action[0]], reward, observation_, env.is_game_over)
                    timestep = env.timestep(0)
                    observation = observation_
                    self.learn()
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
                        if np.random.random() < self.epsilon:
                            actions = np.zeros(29)
                            for i in range(len(actions)):
                                if i in self.hand_ids:
                                    actions[i] = random.uniform(0.5, 1)
                        else:
                            q = self.q_eval.predict(observation)
                            actions = q[0]

                    observation_ = timestep.observation
                    reward = timestep.reward
                    score += reward
                    self.remember(observation, env.tiles_ids[env.current_action[0]], reward, observation_, env.is_game_over)
                    timestep = env.timestep(0)
                    observation = observation_
                    self.learn()
                else:
                    actions = env.agents[i].act(timestep.observation, timestep.reward)
                    action = env.choose_action(actions, i)

                    timestep = env.timestep(i)
                    game_over = env.is_game_over
                    if game_over:
                        break

            scores.append(score)


            if episode % 100 == 0:
                print()
                print(f"Episode {episode} / {num_episodes} | Reward = {statistics.mean(scores)}")
                self.q_eval.save(self.model_file)
                scores = []



    def act(self, observation, reward):
        observation = observation[np.newaxis, :]
        if np.random.random() < self.epsilon:
            actions = np.zeros(29)
            for i in range(len(actions)):
                if i in self.hand_ids:
                    actions[i] = random.uniform(0.5, 1)
        else:
            actions = self.q_eval.predict(observation)
        return actions



class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, num_actions):
        self.mem_cntr = 0
        self.memory_size = max_size
        self.observation_memory = np.zeros((self.memory_size, input_shape))
        self.new_observation_memory = np.zeros((self.memory_size, input_shape))
        dtype = np.int32
        self.action_memory = np.zeros((self.memory_size, num_actions), dtype=dtype)
        self.reward_memory = np.zeros((self.memory_size, num_actions))
        self.terminal_memory = np.zeros((self.memory_size), dtype=np.float32)
        
    def store_transition(self, observation, action, reward, observation_, done):
        index = self.mem_cntr % self.memory_size
        self.observation_memory[index] = observation
        self.new_observation_memory = observation_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.memory_size)
        batch = np.random.choice(max_mem, batch_size)
        observations = self.observation_memory[batch]
        observations_ = np.array(self.new_observation_memory)[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return observations, actions, rewards, observations_, terminal