# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:14:05 2020

@author: gaspe
"""
import gym
import random
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

env = gym.make('CartPole-v1')


def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
  
model = agent(env.observation_space.shape[0], env.action_space.n)

from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy
policy = EpsGreedyQPolicy()
sarsa = SARSAAgent(model = model, policy = policy, nb_actions = env.action_space.n)
sarsa.compile('adam', metrics = ['mse'])

sarsa.fit(env, nb_steps = 50000, visualize = False, verbose = 1)


scores = sarsa.test(env, nb_episodes = 100, visualize= False)
print('Average score over 100 test games:{}'.format(np.mean(scores.history['episode_reward'])))


sarsa.save_weights('sarsa_weights.h5f', overwrite=True)

_ = sarsa.test(env, nb_episodes = 2, visualize= True)


env.close()
