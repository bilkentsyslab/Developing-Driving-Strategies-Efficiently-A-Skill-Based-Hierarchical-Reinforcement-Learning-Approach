from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from traffic_env_multi_disc import TrafficEnv as TrafficEnvv

class TrafficEnv(TrafficEnvv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

# init
    def __init__(self, version):
        super(TrafficEnv, self).__init__(version)
        self.randomness = False
        low  = np.array([-1, -0.1])
        high = np.array([ 1, 1.1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float16)
        self.spec.id = 1

# taking a step
    def step(self, action):
        if self.randomness:
            action[0] += np.random.uniform(-0.20,0.20)
            action[1] += np.random.uniform(-0.15,0.15)
        return step_imported(self, action)

if __name__ == "__main__":
    version = 1
    env = TrafficEnv(version)
    obs = env.reset()
    while True:
        env.render()
        if env.l == 1:
            act = np.array([0.0, 1])
            #act = np.array([-1, 0])
        else:
            act = np.array([0, 0])
        obs, rewards, dones, info = env.step(act)
        if dones: break
    env.close()