import gym
import numpy as np

class RewardMemory(gym.Wrapper):
    # environment remembers the state so that the PolicyWrapper can use it.
    # starts with the initial state
    # updates every time a step is taken
    def __init__(self, env, seed=0):
        super().__init__(env)
        self.env = env
        
        # score (episode) and reward (time step) memory, so that we can print
        self.score_memory = 0
        self.scores_memory = [np.nan]#[self.score_memory]
        self.finish_memory = [np.nan]#[True]
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.score_memory += reward
        if done:
            self.scores_memory.append(self.score_memory)
            self.score_memory = 0
            self.finish_memory.append(info['finish'])
        return next_state, reward, done, info
    
def wrap(env, seed=0):
    wrapped_env = RewardMemory(env, seed)
    return wrapped_env