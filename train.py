#!/usr/bin/env python3

""" Front-end script for training a Snake agent. """

import json
import sys

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from agents import *
from environment import *


def create_dqn_model(env, num_last_frames):
    """
    Build a new DQN model to be used for training.
    
    Args:
        env: an instance of Snake environment. 
        num_last_frames: the number of last frames the agent considers as state.

    Returns:
        A compiled DQN model.
    """

    model = Sequential()

    # Convolutions.
    model.add(Conv1D(
        16,
        kernel_size=(1),
        strides=(1),
        data_format='channels_first',
        input_shape=(env.observation_shape(0), 1)
    ))
    model.add(Activation('relu'))
    model.add(Conv1D(
        32,
        kernel_size=(1),
        strides=(1),
        data_format='channels_first'
    ))
    model.add(Activation('relu'))

    # Dense layers.
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(29))

    model.summary()
    model.compile(RMSprop(), 'MSE')

    return model


num_episodes=3000
agents = []

agents.append(RandomAgent())
agents.append(RandomAgent())
agents.append(RandomAgent())
agents.append(RandomAgent())

env = Environment(agents)

model = create_dqn_model(env, num_last_frames=0)

env.agents[0] = DeepQNetworkAgent(
    model=model,
    memory_size=-1,
    num_last_frames=model.input_shape[1]
)

env.agents[0].train(
    env,
    batch_size=64,
    num_episodes=num_episodes,
    checkpoint_freq=num_episodes // 10,
    discount_factor=0.95
)
