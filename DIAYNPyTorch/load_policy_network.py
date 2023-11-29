from policy_network import PolicyNetwork
import torch as T

def load_policy_network(env, n_skills, n_hidden, path):
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    print(n_states, n_actions, action_bounds)
    
    # create
    n_states = n_states + n_skills # input of the policy network
    n_actions = n_actions # output of the policy network
    action_bounds = action_bounds
    n_hidden_filters = n_hidden
    policy_network = PolicyNetwork(n_states, n_actions, action_bounds, n_hidden_filters).to("cpu")
    
    # import and load:
    checkpoint = T.load(path)
    policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
    
    return policy_network