import torch, gym
import numpy as np
from tqdm import tqdm

from Brain import SACAgent
from Common import Play, Logger, get_params

import sys, os
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "/environments")
sys.path.append(cwd + "/atrenv")
from environments.traffic_env_multi import TrafficEnv  # uses regular enviornment
#from atrenv.atrgym import TrafficEnv  # uses act to reason environment




def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

if __name__ == "__main__":
    for version in [1]:
        # TODO: we added these params, move them to config
        discreteness = False
        randomness = False
        use_traffic_env = True
        train_freq = 16
        gradient_steps = 4

        params = get_params()
        
        # TODO: move these initalisations to config
        params['do_train'] = True
        params['train_from_scratch'] = True
        params['env_name'] = "trafficenv___"
        params['n_skills'] = 10
        params["max_n_episodes"] = 5000
        
        # TODO: is this part necessary? can it be moved to config
        if use_traffic_env:
            test_env = TrafficEnv(version)
            test_env.randomness = randomness # TODO: can we make this sth like eval mode that is supported by gym?
        else:
            test_env = gym.make(params["env_name"])
        test_env = TrafficEnv(version) # gym.make(env_n)
        n_states = test_env.observation_space.shape[0]
        n_actions = test_env.action_space.shape[0]
        action_bounds = [torch.tensor(test_env.action_space.low, device='cpu'), torch.tensor(test_env.action_space.high, device='cpu')]

        params.update({"n_states": n_states,
                    "n_actions": n_actions,
                    "action_bounds": action_bounds})

        test_env.close()

        if use_traffic_env:
            env = TrafficEnv(version)
            env.randomness = randomness
        else:
            env = gym.make(params["env_name"])

        p_z = np.full(params["n_skills"], 1 / params["n_skills"])
        agent = SACAgent(p_z=p_z, **params)
        logger = Logger(agent, **params)

        if params["do_train"]:
            if not params["train_from_scratch"]:
                episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
                agent.hard_update_target_network()
                min_episode = episode
                np.random.set_state(np_rng_state)
                env.np_random.set_state(env_rng_states[0])
                env.observation_space.np_random.set_state(env_rng_states[1])
                env.action_space.np_random.set_state(env_rng_states[2])
                agent.set_rng_states(torch_rng_state, random_rng_state)
                print("Keep training from previous run.")

            else:
                min_episode = 0
                last_logq_zs = 0
                """
                np.random.seed(params["seed"])
                env.seed(params["seed"])
                env.observation_space.seed(params["seed"])
                env.action_space.seed(params["seed"])
                """
                print("Training from scratch.")

            logger.on()
            counter = 0
            for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
                z = np.random.choice(params["n_skills"], p=p_z)
                state = env.reset()
                state = concat_state_latent(state, z, params["n_skills"])
                episode_reward = 0
                logq_zses = []
                max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
                for step in range(1, 1 + max_n_steps):
                    counter += 1

                    action = agent.choose_action(state)
                    next_state, reward, done, _ = env.step(action)
                    next_state = concat_state_latent(next_state, z, params["n_skills"])
                    agent.store(state, z, done, action, next_state)
                    if (counter % train_freq) == 0:
                        for i in range(gradient_steps):
                            logq_zs = agent.train()
                            if logq_zs is None:
                                logq_zses.append(last_logq_zs)
                            else:
                                logq_zses.append(logq_zs)
                    episode_reward += reward
                    state = next_state
                    if done:
                        break

                logger.log(episode,
                        episode_reward,
                        z,
                        sum(logq_zses) / len(logq_zses) if len(logq_zses) > 0 else 0,
                        step,
                        0,
                        0,
                        0,
                        0,
                        *agent.get_rng_states(),
                        )

                if (episode % 500) == 0:
                    env.randomness = False
                    player = Play(env, agent, n_skills=params["n_skills"])
                    player.evaluate(dir=f"{episode}/")
                    env.randomness = randomness

        else:
            logger.load_weights()
            player = Play(env, agent, n_skills=params["n_skills"])
            player.evaluate()
