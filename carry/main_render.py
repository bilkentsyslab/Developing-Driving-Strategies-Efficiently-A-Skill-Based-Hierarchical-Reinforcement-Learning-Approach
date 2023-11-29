from stable_baselines3 import DQN
import cv2

import sys, os
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "/environments")
sys.path.append(cwd + "/DIAYNPyTorch")
from environments.traffic_env_multi import TrafficEnv as ContTrafficEnv
from environments.traffic_env_multi_disc import TrafficEnv as DiscTrafficEnv
from DIAYNPyTorch.load_policy_network import load_policy_network
from wrapper2 import add_policy_network 

n_skills = 10
file = 1
name = 'tests/test_'
loc = name + str(file) + '/'

# DQN
"""
env = DiscTrafficEnv(version=1)
model = DQN.load(loc+"baseline.zip", env=env)

vec_env = env
obs = vec_env.reset()
speed_up_rate = 1
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
    cv2.waitKey(0)
    if dones: break
    #key = int(cv2.waitKey()) - 49

    #if key == -50:
    #    key = 5
    #action = np.array([key])
    
    #obss = obs[0]
    #keys = ['self speed', 'lane 0', 'lane 1', 'is mergable', 'left tail dist', 'left tail speed', 'left head dist', 'left head speed', \
    #'same tail dist', 'same tail speed', 'same head dist', 'same head speed']
    #for i in range(len(keys)):
    #    print(keys[i]+"\t\t"+str(obss[i]))
    #print(rewards)
    #print(action)
    #print()
"""  
    


# DIAYN 
initial_env = ContTrafficEnv(version=1) 
n_skills = 10
n_hidden = 64
path = f"{cwd}/Checkpoints/params.pth"
policy_network = load_policy_network(initial_env, n_skills, n_hidden, path) 
modified_env = add_policy_network(initial_env, policy_network, n_skills, k=4)
env = modified_env

model = DQN.load(loc+"hrl16.zip", env=env)

vec_env = env
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    action = 1
    obs, rewards, dones, info = vec_env.step(action)
    if dones: break
    vec_env.render()
    cv2.waitKey(0)
sys.exit()
"""



# DQN
env = DiscTrafficEnv()
model = DQN.load(loc+"baseline.zip", env=env)

vec_env = model.get_env()
obs = vec_env.reset()
speed_up_rate = 1
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    if dones: break
    vec_env.render()

# DIAYN 1
initial_env = ContTrafficEnv() #ContinuousCartPoleEnv()
#n_skills = 10
n_hidden = 64
policy_network = load_policy_network(initial_env, n_skills, n_hidden) # put params.pth in /diayn_and_dqn/
modified_env = add_policy_network(initial_env, policy_network, n_skills, k=1)
env = modified_env

model = DQN.load(loc+"hrl1.zip", env=env)

vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    if dones: break
    vec_env.render()

# DIAYN 4
initial_env = ContTrafficEnv() #ContinuousCartPoleEnv()
#n_skills = 10
n_hidden = 64
policy_network = load_policy_network(initial_env, n_skills, n_hidden) # put params.pth in /diayn_and_dqn/
modified_env = add_policy_network(initial_env, policy_network, n_skills, k=4)
env = modified_env

model = DQN.load(loc+"hrl4.zip", env=env)

vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    if dones: break
    vec_env.render()

# DIAYN 8
initial_env = ContTrafficEnv() #ContinuousCartPoleEnv()
#n_skills = 10
n_hidden = 64
policy_network = load_policy_network(initial_env, n_skills, n_hidden) # put params.pth in /diayn_and_dqn/
modified_env = add_policy_network(initial_env, policy_network, n_skills, k=8)
env = modified_env

model = DQN.load(loc+"hrl8.zip", env=env)

vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    if dones: break
    vec_env.render()

# DIAYN 12
initial_env = ContTrafficEnv() #ContinuousCartPoleEnv()
#n_skills = 10
n_hidden = 64
policy_network = load_policy_network(initial_env, n_skills, n_hidden) # put params.pth in /diayn_and_dqn/
modified_env = add_policy_network(initial_env, policy_network, n_skills, k=12)
env = modified_env

model = DQN.load(loc+"hrl12.zip", env=env)

vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    if dones: break
    vec_env.render()



#"""