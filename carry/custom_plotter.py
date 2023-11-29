import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

import sys, os
os.chdir('C:/Users/syslab123/Desktop/USD')
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "/environments")
sys.path.append(cwd + "/carry")
from environments.constants import *
from carry.callback_collision_rate_while_training import FinishPercentageCallback

style = "seaborn-v0_8-paper"
plt.style.use(style)


def train_and_plot(training_env, test_env, env_name, k, seed=0):
    """n_timesteps: !!float 1.2e5
  policy: 'MlpPolicy'
  learning_rate: !!float 4e-3
  batch_size: 128
  buffer_size: 10000
  learning_starts: 1000
  gamma: 0.98
  target_update_interval: 600
  train_freq: 16
  gradient_steps: 8
  exploration_fraction: 0.2
  exploration_final_eps: 0.07
  policy_kwargs: "dict(net_arch=[256, 256])
    """
    if (k == 1) and (env_name[-1] != 'e'):
        prolonger = 0.8
    elif (k == 1) and (env_name[-1] == 'e'):
        prolonger = 1
    elif k == 4:
        prolonger = 1.6
    elif k == 8:
        prolonger = 2.0
    else:
        prolonger = 1.85
    n_timesteps = int(int(500000/k)*prolonger) # target_update_interval=16, net_arch=64, gamma=0.977
    learning_starts = int(int((n_timesteps/10)))
    training_env.seed(seed)
    """
    target_update_interval=int(160)
    train_freq = int(16)
    buffer_size = int(n_timesteps)
    model = DQN(policy='MlpPolicy', env=training_env, learning_rate=9*10**(-4), batch_size=512, buffer_size=buffer_size, learning_starts=learning_starts,\
        gamma=0.98, target_update_interval=target_update_interval, train_freq=train_freq, gradient_steps=8, exploration_fraction=0.3/prolonger,\
            exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[64, 64, 64]), exploration_initial_eps=1.0, device=device) 
    """
    model = DQN('MlpPolicy', 
            training_env, 
            verbose=0,
            tensorboard_log="./logs/tensorboard",
            buffer_size=100000,
            batch_size=128,
            learning_starts=learning_starts,
            target_update_interval=256,
            train_freq=32,
            gradient_steps=8,
            exploration_fraction=0.3,
            exploration_final_eps=0.07,
            policy_kwargs=dict(net_arch=[128, 128, 64]),
            learning_rate=0.001,
            gamma=0.98,
            device=device
    ) 
    #model.learn(total_timesteps=n_timesteps, progress_bar=True)
    #model = DQN("MlpPolicy", training_env, verbose=0, exploration_fraction=exp_fraction, 
    #            exploration_initial_eps=start, exploration_final_eps=end, learning_starts=learning_starts, learning_rate=learning_rate)
    callback = FinishPercentageCallback(training_env, test_env, env_name, model, train_how_long=400, runs_per_test=10, test_count=100)
    model.learn(total_timesteps=n_timesteps, callback=callback, progress_bar=True)
    #model.learn(total_timesteps=n_timesteps, progress_bar=True)
    
    # save
    model.save(f"{env_name}")
    np.save(f"{env_name}_reward", np.array(training_env.scores_memory))
    
    y_axis = np.array(training_env.scores_memory)
    x_axis = np.arange(y_axis.size)+1
    
    """
    plt.figure(0, figsize=(9,6))
    plt.clf()
    plt.plot(x_axis, y_axis)
    plt.title(env_name)
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=20, tight=True)
    plt.show()
    """