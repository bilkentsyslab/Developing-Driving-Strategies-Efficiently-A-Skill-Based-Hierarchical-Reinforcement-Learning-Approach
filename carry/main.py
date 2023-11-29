from custom_plotter import train_and_plot
from wrapper2 import add_policy_network 
from wrapper3 import wrap 

import sys, os
os.chdir('C:/Users/syslab123/Desktop/USD')
cwd = os.getcwd()
sys.path.append(cwd)

print(cwd)
sys.path.append(cwd + "/environments")
sys.path.append(cwd + "/DIAYNPyTorch")

from DIAYNPyTorch.load_policy_network import load_policy_network
from environments.traffic_env_multi import TrafficEnv as ContTrafficEnv
from environments.traffic_env_multi_disc import TrafficEnv as DiscTrafficEnv

which_ones = [16] # [4,8,16]

def main(file_no, version, skill_name='params'):
    file_name = f"tests/test_{file_no}"
    if not os.path.exists(file_name):
        os.mkdir(file_name)

    # BASELINE

    env_name = f"{file_name}/baseline"
    training_env = DiscTrafficEnv(version)
    test_env = DiscTrafficEnv(version)
    
    train_and_plot(wrap(training_env), wrap(test_env), env_name, k=1)

    for k in which_ones:
        env_name = f"{file_name}/hrl{k}"
        initial_training_env = ContTrafficEnv(version)
        initial_test_env = ContTrafficEnv(version)

        path = f"{cwd}/Checkpoints/{skill_name}.pth"
        n_hidden = 64
        n_skills = 10
        policy_network = load_policy_network(initial_training_env, n_skills, n_hidden, path)
        training_env = add_policy_network(initial_training_env, policy_network, n_skills, k=k)
        test_env = add_policy_network(initial_test_env, policy_network, n_skills, k=k)
        
        train_and_plot(training_env, test_env, env_name, k=k)
    

if __name__ == "__main__":
    if not os.path.exists("tests"):
        os.mkdir("tests")
    skill_name = "params_no_randomness"
    for file in range(21, 31):#[1,5,6,7,8,9,10]:
        main(file, version=1, skill_name=skill_name)
    skill_name = "params_with_randomness"
    for file in range(31, 41):#[1,5,6,7,8,9,10]:
        main(file, version=1, skill_name=skill_name)














#with cProfile.Profile() as pr:
#    pass
#stats = pstats.Stats(pr)
#stats.sort_stats(pstats.SortKey.TIME)
#stats.print_stats()
#stats.dump_stats(filename='profiler_output_3.prof')