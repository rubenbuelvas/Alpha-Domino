#!/usr/bin/env python3

""" Front-end script for training a Snake agent. """

import json
import sys

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from agents import *
from environment import *


def create_dqn_model(env, lr, num_actions, input_dims):

    model = Sequential()

    model.add(Dense(256, input_shape=(input_dims,)))
    model.add(Activation("relu"))
   
    model.add(Dense(256))
    model.add(Activation("relu"))
    
    model.add(Dense(num_actions))
    model.add(Activation("relu"))

    model.summary()
    model.compile(optimizer=Adam(lr=lr), loss="MSE")

    return model


num_episodes=30000
agents = []

agents.append(RandomAgent())
agents.append(RandomAgent())
agents.append(RandomAgent())
agents.append(RandomAgent())

env = Environment(agents)

model = create_dqn_model(env, 0.001, env.observation_shape(0), 48)

env.agents[0] = DeepQNetworkAgent(model, num_actions=env.observation_shape(0))
env.agents[0].train(env, num_episodes=1000)
