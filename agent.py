# A script to demonstrate training an RL agent with my parallelized Evolution Strategies implementation - Adam Hung

import gymnasium as gym
from NESP import NESP
import time
import numpy as np

# Try me with your favorite benchmark or custom environment! Just remember to tune your hyperparameters.
env_id = "Ant-v4"
env_kwargs = {'render_mode': 'rgb_array','reset_noise_scale': 0}

def train(its, n_envs):
    envs = [gym.make(env_id, **env_kwargs) for _ in range(n_envs)]
    model = NESP(envs,
               n_envs = n_envs,
               lamb = 100, 
               learning_rate = 0.002,
               stddev = 0.01,
               #rseed = 1
               #state_dict_path = "ant_net.pt",
               ) 
    model.learn(iterations = its, evaluation_episodes = 1, stop_training_at = 1000)

def test(eps):
    env = gym.make(env_id, render_mode="human")
    model = NESP(envs = [env]) 
    model.test(eps, path = "ant_net.pt")
        
if __name__ == '__main__':
    #run these separately
    train(1000, 5)
    #test(10)