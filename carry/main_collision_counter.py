import gym

from stable_baselines3 import DQN

from traffic_env_multi import TrafficEnv as ContTrafficEnv
from traffic_env_multi_disc import TrafficEnv as DiscTrafficEnv
from load_policy_network import load_policy_network
from wrapper2 import add_policy_network 

episode_count = 100

n_skills = 8
file = 6
name = 'tests'
#name = 'results'
loc = name + str(file) + '/'

# baseline
baseline_collision_counter = 0
baseline_finish_counter = 0
baseline_max_length_counter = 0
baseline_episode_counter = 0
env = DiscTrafficEnv()
model = DQN.load(loc+"baseline.zip", env=env)
print(model.exploration_rate)
model.exploration_rate = 0

vec_env = model.get_env()
for _ in range(episode_count):
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if info[0]['collision'] == True: baseline_collision_counter += 1
        elif info[0]['finish'] == True: baseline_finish_counter += 1
        elif info[0]['max_length'] == True: baseline_max_length_counter += 1
        if dones: 
            baseline_episode_counter += 1
            break
"""
# hrl1
hrl1_collision_counter = 0
hrl1_finish_counter = 0
hrl1_max_length_counter = 0
hrl1_episode_counter = 0
initial_env = ContTrafficEnv() #ContinuousCartPoleEnv()
n_skills = 50
n_hidden = 64
policy_network = load_policy_network(initial_env, n_skills, n_hidden) # put params.pth in /diayn_and_dqn/
modified_env = add_policy_network(initial_env, policy_network, n_skills, k=1)
env = modified_env

model = DQN.load("hrl1.zip", env=env)

vec_env = model.get_env()
for _ in range(episode_count):
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if info[0]['collision'] == True: hrl1_collision_counter += 1
        elif info[0]['finish'] == True: hrl1_finish_counter += 1
        elif info[0]['max_length'] == True: hrl1_max_length_counter += 1
        if dones: 
            hrl1_episode_counter += 1
            break

# hrl4
hrl4_collision_counter = 0
hrl4_finish_counter = 0
hrl4_max_length_counter = 0
hrl4_episode_counter = 0
initial_env = ContTrafficEnv() #ContinuousCartPoleEnv()
n_skills = 50
n_hidden = 64
policy_network = load_policy_network(initial_env, n_skills, n_hidden) # put params.pth in /diayn_and_dqn/
modified_env = add_policy_network(initial_env, policy_network, n_skills, k=4)
env = modified_env

model = DQN.load("hrl4.zip", env=env)

vec_env = model.get_env()
for _ in range(episode_count):
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if info[0]['collision'] == True: hrl4_collision_counter += 1
        elif info[0]['finish'] == True: hrl4_finish_counter += 1
        elif info[0]['max_length'] == True: hrl4_max_length_counter += 1
        if dones: 
            hrl4_episode_counter += 1
            break


print("-------------------- Collision-Finish-Max / Episode")
"""
print(f"Baseline : {baseline_collision_counter}-{baseline_finish_counter}-{baseline_max_length_counter}/{baseline_episode_counter}")
#print(f"HRL 1    : {hrl1_collision_counter}-{hrl1_finish_counter}-{hrl1_max_length_counter}/{hrl1_episode_counter}")
#print(f"HRL 4    : {hrl4_collision_counter}-{hrl4_finish_counter}-{hrl4_max_length_counter}/{hrl4_episode_counter}")