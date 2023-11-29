import gym
import numpy as np
import torch as T

# same as wrapper 1, but does it in 1 step

select_mode = True
selected_skills = [i for i in range(10)]

class PolicyWrapper(gym.Wrapper):
    # environment remembers the state so that the PolicyWrapper can use it.
    # starts with the initial state
    # updates every time a step is taken
    def __init__(self, env, policy_network, n_skills, k, seed=0):
        super().__init__(env)
        self.env = env
        self.k = k
        
        # read previous state from the memory
        self.env.seed(seed)
        self.state_memory = self.env.reset()
        
        # policy network takes 1 hot skill and state
        self.policy_network = policy_network
        
        # number of skills so we can do 1 hot
        self.n_skills = n_skills
        
        # score (episode) and reward (time step) memory, so that we can print
        self.score_memory = 0
        self.scores_memory = [np.nan]#[self.score_memory]
        self.finish_memory = [np.nan]#[True]
    
    def step(self, psuedo_action):
        
        # DQN used to select actions. Now it selects skills.
        if select_mode:
            action = selected_skills[psuedo_action]
        selected_skill = np.array([action]).reshape((-1,))
        
        # one hot encoding
        one_hot_skill = np.zeros(self.n_skills)
        one_hot_skill[selected_skill] = 1.0
        
        with T.no_grad():
            # repet the skill for k steps:
            total_reward = 0
            for _ in range(self.k):
                # what will be given to the policy network as input
                policy_network_input = T.Tensor(np.append(self.state_memory, one_hot_skill)).to(self.policy_network.device)
            
                # determine the real action
                action = self.policy_network.sample_or_likelihood(policy_network_input)[0].detach().cpu().numpy()#[0]
            
                # the usual stuff. Make sure that you take this step in self.env, not in self.
                # otherwise, you will have endless recursion.
                next_state, reward, done, info = self.env.step(action)
                self.score_memory += reward
                total_reward += reward
                
                # memory update for the next skill-action
                self.state_memory = next_state
                
                if done:
                    self.scores_memory.append(self.score_memory)
                    self.score_memory = 0
                    self.finish_memory.append(info['finish'])
                    return next_state, total_reward, done, info
        
        return next_state, total_reward, done, info

class ActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env, n_skills):
        super().__init__(env)
        # adjust the new pseudo action space
        n_actions = n_skills
        if select_mode:
            n_actions = len(selected_skills)
        self.action_space = gym.spaces.Discrete(n_actions)
    
    def action(self, act):
        return act
    
def add_policy_network(initial_env, policy_network, n_skills, k=4, seed=0):
    action_space_wrapped_env = ActionSpaceWrapper(initial_env, n_skills)
    policy_wrapped_env = PolicyWrapper(action_space_wrapped_env, policy_network, n_skills, k, seed=seed)
    return policy_wrapped_env