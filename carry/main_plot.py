import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
os.chdir('C:/Users/syslab123/Desktop/USD')

style = "seaborn-v0_8-paper"
plt.style.use(style)
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

n_skills = 10
file = 45
name = 'tests'
#name = 'results'
loc = name + str(file) + '/'

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def smooth_same(y, box_pts):
    box_pts = int(np.ceil(int(y.shape[0]*box_pts/100)))
    box = np.ones(box_pts)/box_pts
    checker = np.ones(y.shape[0])
    checker = np.convolve(checker, box, 'same')
    checker = 1/checker
    y_smooth = np.convolve(y, box, mode='same')*checker
    return y_smooth


def plotter(which_ones=[0,1,4,8,16], smooths = [100, 100, 10, 10], reduce_smoothing_wrt_k=True):
    legend = []
    plt.figure(0, figsize=(16, 12))
    smooths_memory = np.array(smooths).astype(int)
    for i in which_ones:
        if i == 0: 
            i = 'baseline'
            legend.append('Baseline')
            smooths = smooths_memory
        else:
            if reduce_smoothing_wrt_k:
                smooths = (smooths_memory/i).astype(int)
                smooths[2:] = smooths_memory[2:]
                smooths[smooths==0] = 1
            else:
                smooths = smooths_memory
            legend.append(f"HRL {i} steps")
            i = f"hrl{i}"
        times = np.load(f"{loc}{i}_training_times.npy")
        reward = smooth_same(np.load(f"{loc}{i}_training_episode_reward.npy"), smooths[0])
        finish_rate = smooth_same(np.load(f"{loc}{i}_training_episode_finish_rate.npy"), smooths[1])
        if False:
            print(i)
            print(times.shape)
            print(reward.shape)
            print(finish_rate.shape)

        plt.subplot(221)#plt.figure(0, figsize=(9,6))
        plt.plot(times, reward, linewidth=5)
        plt.subplot(222)#plt.figure(1, figsize=(9,6))
        plt.plot(times, finish_rate, linewidth=5)

        try:
            times = np.load(f"{loc}{i}_test_times.npy")
            reward = smooth_same(np.load(f"{loc}{i}_test_episode_reward.npy"), smooths[2])
            finish_rate = smooth_same(np.load(f"{loc}{i}_test_episode_finish_rate.npy"), smooths[3])

            plt.subplot(223)#plt.figure(2, figsize=(9,6))
            plt.plot(times, reward, linewidth=5)
            plt.subplot(224)#plt.figure(3, figsize=(9,6))
            plt.plot(times, finish_rate, linewidth=5)
        except: 
            print("!")
    # figure 0 - train reward
    plt.subplot(221)#plt.figure(0)
    plt.legend(legend)
    plt.xlabel("Time [sec]")
    plt.ylabel("Reward")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("train reward")

    # figure 1 - train finish rate
    plt.subplot(222)#plt.figure(1)
    plt.legend(legend)
    plt.xlabel("Time [sec]")
    plt.ylabel("Finish rate [%]")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("train finish rate")

    # figure 2 - test reward
    plt.subplot(223)#plt.figure(2)
    plt.legend(legend)
    plt.xlabel("Time [sec]")
    plt.ylabel("Reward")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("test reward")

    # figure 3 - test finish rate
    plt.subplot(224)#plt.figure(3)
    plt.legend(legend)
    plt.xlabel("Time [sec]")
    plt.ylabel("Finish rate [%]")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("test finish rate")

    plt.tight_layout()
    plt.show()

def std_plotter(paths, which_ones=[0,1,4,8,16], smooths = [100, 100, 10, 10], reduce_smoothing_wrt_k=True):
    legend = []
    plt.figure(0, figsize=(16, 6))
    smooths_memory = np.array(smooths).astype(int)
    data_dict = dict()
    min_len_train = dict()
    min_len_test = dict()
    for loc in paths:
        for i in which_ones:
            if i == 0: 
                i = 'baseline'
                legend.append('Baseline')
                smooths = smooths_memory
            else:
                if reduce_smoothing_wrt_k:
                    smooths = (smooths_memory/i).astype(int)
                    smooths[2:] = smooths_memory[2:]
                    smooths[smooths==0] = 1
                else:
                    smooths = smooths_memory
                legend.append(f"HRL {i} steps")
                i = f"hrl{i}"
            if i in data_dict.keys():
                pass
            else:
                data_dict[i] = dict()
                data_dict[i]['train_times'] = list()
                data_dict[i]['train_reward'] = list()
                data_dict[i]['train_finish_rate'] = list()
                data_dict[i]['test_times'] = list()
                data_dict[i]['test_reward'] = list()
                data_dict[i]['test_finish_rate'] = list()
                min_len_train[i] = None 
                min_len_test[i] = None

            data_dict[i]['train_times'].append(np.load(f"{loc}{i}_training_times.npy"))
            data_dict[i]['train_reward'].append(smooth_same(np.load(f"{loc}{i}_training_episode_reward.npy"), smooths[0]))
            data_dict[i]['train_finish_rate'].append(smooth_same(np.load(f"{loc}{i}_training_episode_finish_rate.npy"), smooths[1]))
            if min_len_train[i] is None:
                min_len_train[i] = len(data_dict[i]['train_times'][-1])
            else:
                min_len_train[i] = min(min_len_train[i], len(data_dict[i]['train_times'][-1]))

            try:
                data_dict[i]['test_times'].append(np.load(f"{loc}{i}_test_times.npy"))
                data_dict[i]['test_reward'].append(smooth_same(np.load(f"{loc}{i}_test_episode_reward.npy"), smooths[2]))
                data_dict[i]['test_finish_rate'].append(smooth_same(np.load(f"{loc}{i}_test_episode_finish_rate.npy"), smooths[3]))
                if min_len_test[i] is None:
                    min_len_test[i] = len(data_dict[i]['test_times'][-1])
                else:
                    min_len_test[i] = min(min_len_test[i], len(data_dict[i]['test_times'][-1]))
            except: 
                print("!")
    
    # apply minimum len so that they are all of the same length
    for i in data_dict.keys():
        for j in range(len(data_dict[i]['train_times'])):
            data_dict[i]['train_times'][j] = data_dict[i]['train_times'][j][:min_len_train[i]]
            data_dict[i]['train_reward'][j] = data_dict[i]['train_reward'][j][:min_len_train[i]]
            data_dict[i]['train_finish_rate'][j] = data_dict[i]['train_finish_rate'][j][:min_len_train[i]]
            data_dict[i]['test_times'][j] = data_dict[i]['test_times'][j][:min_len_test[i]]
            data_dict[i]['test_reward'][j] = data_dict[i]['test_reward'][j][:min_len_test[i]]
            data_dict[i]['test_finish_rate'][j] = data_dict[i]['test_finish_rate'][j][:min_len_test[i]]

    # calculate mean and std
    mean_dict = dict()
    std_dict = dict()
    for i in data_dict.keys():
        mean_dict[i] = dict()
        std_dict[i] = dict()

        mean_dict[i]['train_times'] = np.mean(np.array(data_dict[i]['train_times']), axis=0)
        mean_dict[i]['train_reward'] = np.mean(np.array(data_dict[i]['train_reward']), axis=0)
        mean_dict[i]['train_finish_rate'] = np.mean(np.array(data_dict[i]['train_finish_rate']), axis=0)
        mean_dict[i]['test_times'] = np.mean(np.array(data_dict[i]['test_times']), axis=0)
        mean_dict[i]['test_reward'] = np.mean(np.array(data_dict[i]['test_reward']), axis=0)
        mean_dict[i]['test_finish_rate'] = np.mean(np.array(data_dict[i]['test_finish_rate']), axis=0)

        std_dict[i]['train_times'] = np.std(np.array(data_dict[i]['train_times']), axis=0)
        std_dict[i]['train_reward'] = np.std(np.array(data_dict[i]['train_reward']), axis=0)
        std_dict[i]['train_finish_rate'] = np.std(np.array(data_dict[i]['train_finish_rate']), axis=0)
        std_dict[i]['test_times'] = np.std(np.array(data_dict[i]['test_times']), axis=0)
        std_dict[i]['test_reward'] = np.std(np.array(data_dict[i]['test_reward']), axis=0)
        std_dict[i]['test_finish_rate'] = np.std(np.array(data_dict[i]['test_finish_rate']), axis=0)



    for index, i in enumerate(data_dict.keys()):

        plt.subplot(221)#plt.figure(0, figsize=(9,6))
        error_in_the_mean = std_dict[i]['train_reward']/np.sqrt(len(paths))
        plt.plot(mean_dict[i]['train_times'], mean_dict[i]['train_reward'], linewidth=5, label=legend[index])
        plt.fill_between(mean_dict[i]['train_times'], mean_dict[i]['train_reward']-error_in_the_mean, mean_dict[i]['train_reward']+error_in_the_mean, alpha=0.5)
        
        plt.subplot(222)#plt.figure(1, figsize=(9,6))
        error_in_the_mean = std_dict[i]['train_finish_rate']/np.sqrt(len(paths))
        plt.plot(mean_dict[i]['train_times'], 100*mean_dict[i]['train_finish_rate'], linewidth=5, label=legend[index])
        plt.fill_between(mean_dict[i]['train_times'], 100*mean_dict[i]['train_finish_rate']-100*error_in_the_mean, 100*mean_dict[i]['train_finish_rate']+100*error_in_the_mean, alpha=0.5)

        plt.subplot(223)#plt.figure(2, figsize=(9,6))
        error_in_the_mean = std_dict[i]['test_reward']/np.sqrt(len(paths))
        plt.plot(mean_dict[i]['test_times'], mean_dict[i]['test_reward'], linewidth=5, label=legend[index])
        plt.fill_between(mean_dict[i]['test_times'], mean_dict[i]['test_reward']-error_in_the_mean, mean_dict[i]['test_reward']+error_in_the_mean, alpha=0.5)
        
        plt.subplot(224)#plt.figure(3, figsize=(9,6))
        error_in_the_mean = std_dict[i]['test_finish_rate']/np.sqrt(len(paths))
        plt.plot(mean_dict[i]['test_times'], 100*mean_dict[i]['test_finish_rate'], linewidth=5, label=legend[index])
        plt.fill_between(mean_dict[i]['test_times'], 100*mean_dict[i]['test_finish_rate']-100*error_in_the_mean, 100*mean_dict[i]['test_finish_rate']+100*error_in_the_mean, alpha=0.5)

    # figure 0 - train reward
    plt.subplot(221)#plt.figure(0)
    #plt.legend(legend)
    plt.legend()
    plt.xlabel("Time [sec]")
    plt.ylabel("Reward")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("Train Reward")

    plt.ylim(-1400, -100)

    # figure 1 - train finish rate
    plt.subplot(222)#plt.figure(1)
    #plt.legend(legend)
    plt.ylim(0, 100)
    plt.legend()
    plt.xlabel("Time [sec]")
    plt.ylabel("Finish rate [%]")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("Train Finish Rate")

    # figure 2 - test reward
    plt.subplot(223)#plt.figure(2)
    plt.ylim(-1400, -100)
    #plt.legend(legend)
    plt.legend()
    plt.xlabel("Time [sec]")
    plt.ylabel("Reward")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("Test Reward")

    # figure 3 - test finish rate
    plt.subplot(224)#plt.figure(3)
    plt.ylim(0, 100)
    #plt.legend(legend)
    plt.legend()
    plt.xlabel("Time [sec]")
    plt.ylabel("Finish rate [%]")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("Test Finish Rate")

    plt.tight_layout()
    plt.show()


def std_test_plotter(paths, which_ones=[0,1,4,8,16], smooths = [100, 100, 10, 10], reduce_smoothing_wrt_k=True):
    legend = []
    plt.figure(0, figsize=(16, 6))
    smooths_memory = np.array(smooths).astype(int)
    data_dict = dict()
    min_len_train = dict()
    min_len_test = dict()
    for loc in paths:
        for i in which_ones:
            if i == 0: 
                i = 'baseline'
                legend.append('Baseline')
                smooths = smooths_memory
            else:
                if reduce_smoothing_wrt_k:
                    smooths = (smooths_memory/i).astype(int)
                    smooths[2:] = smooths_memory[2:]
                    smooths[smooths==0] = 1
                else:
                    smooths = smooths_memory
                legend.append(f"HRL {i} steps")
                i = f"hrl{i}"
            if i in data_dict.keys():
                pass
            else:
                data_dict[i] = dict()
                data_dict[i]['train_times'] = list()
                data_dict[i]['train_reward'] = list()
                data_dict[i]['train_finish_rate'] = list()
                data_dict[i]['test_times'] = list()
                data_dict[i]['test_reward'] = list()
                data_dict[i]['test_finish_rate'] = list()
                min_len_train[i] = None 
                min_len_test[i] = None

            data_dict[i]['train_times'].append(np.load(f"{loc}{i}_training_times.npy"))
            data_dict[i]['train_reward'].append(smooth_same(np.load(f"{loc}{i}_training_episode_reward.npy"), smooths[0]))
            data_dict[i]['train_finish_rate'].append(smooth_same(np.load(f"{loc}{i}_training_episode_finish_rate.npy"), smooths[1]))
            if min_len_train[i] is None:
                min_len_train[i] = len(data_dict[i]['train_times'][-1])
            else:
                min_len_train[i] = min(min_len_train[i], len(data_dict[i]['train_times'][-1]))

            try:
                data_dict[i]['test_times'].append(np.load(f"{loc}{i}_test_times.npy"))
                data_dict[i]['test_reward'].append(smooth_same(np.load(f"{loc}{i}_test_episode_reward.npy"), smooths[2]))
                data_dict[i]['test_finish_rate'].append(smooth_same(np.load(f"{loc}{i}_test_episode_finish_rate.npy"), smooths[3]))
                if min_len_test[i] is None:
                    min_len_test[i] = len(data_dict[i]['test_times'][-1])
                else:
                    min_len_test[i] = min(min_len_test[i], len(data_dict[i]['test_times'][-1]))
            except: 
                print("!")
    
    # apply minimum len so that they are all of the same length
    for i in data_dict.keys():
        for j in range(len(data_dict[i]['train_times'])):
            data_dict[i]['train_times'][j] = data_dict[i]['train_times'][j][:min_len_train[i]]
            data_dict[i]['train_reward'][j] = data_dict[i]['train_reward'][j][:min_len_train[i]]
            data_dict[i]['train_finish_rate'][j] = data_dict[i]['train_finish_rate'][j][:min_len_train[i]]
            data_dict[i]['test_times'][j] = data_dict[i]['test_times'][j][:min_len_test[i]]
            data_dict[i]['test_reward'][j] = data_dict[i]['test_reward'][j][:min_len_test[i]]
            data_dict[i]['test_finish_rate'][j] = data_dict[i]['test_finish_rate'][j][:min_len_test[i]]

    # calculate mean and std
    mean_dict = dict()
    std_dict = dict()
    for i in data_dict.keys():
        mean_dict[i] = dict()
        std_dict[i] = dict()

        mean_dict[i]['train_times'] = np.mean(np.array(data_dict[i]['train_times']), axis=0)
        mean_dict[i]['train_reward'] = np.mean(np.array(data_dict[i]['train_reward']), axis=0)
        mean_dict[i]['train_finish_rate'] = np.mean(np.array(data_dict[i]['train_finish_rate']), axis=0)
        mean_dict[i]['test_times'] = np.mean(np.array(data_dict[i]['test_times']), axis=0)
        mean_dict[i]['test_reward'] = np.mean(np.array(data_dict[i]['test_reward']), axis=0)
        mean_dict[i]['test_finish_rate'] = np.mean(np.array(data_dict[i]['test_finish_rate']), axis=0)

        std_dict[i]['train_times'] = np.std(np.array(data_dict[i]['train_times']), axis=0)
        std_dict[i]['train_reward'] = np.std(np.array(data_dict[i]['train_reward']), axis=0)
        std_dict[i]['train_finish_rate'] = np.std(np.array(data_dict[i]['train_finish_rate']), axis=0)
        std_dict[i]['test_times'] = np.std(np.array(data_dict[i]['test_times']), axis=0)
        std_dict[i]['test_reward'] = np.std(np.array(data_dict[i]['test_reward']), axis=0)
        std_dict[i]['test_finish_rate'] = np.std(np.array(data_dict[i]['test_finish_rate']), axis=0)



    for index, i in enumerate(data_dict.keys()):
        plt.subplot(121)#plt.figure(2, figsize=(9,6))
        error_in_the_mean = std_dict[i]['test_reward']/np.sqrt(len(paths))
        plt.plot(mean_dict[i]['test_times'], mean_dict[i]['test_reward'], linewidth=5, label=legend[index])
        plt.fill_between(mean_dict[i]['test_times'], mean_dict[i]['test_reward']-error_in_the_mean, mean_dict[i]['test_reward']+error_in_the_mean, alpha=0.5)
        
        plt.subplot(122)#plt.figure(3, figsize=(9,6))
        error_in_the_mean = std_dict[i]['test_finish_rate']/np.sqrt(len(paths))
        plt.plot(mean_dict[i]['test_times'], 100*mean_dict[i]['test_finish_rate'], linewidth=5, label=legend[index])
        plt.fill_between(mean_dict[i]['test_times'], 100*mean_dict[i]['test_finish_rate']-100*error_in_the_mean, 100*mean_dict[i]['test_finish_rate']+100*error_in_the_mean, alpha=0.5)

    # figure 2 - test reward
    plt.subplot(121)#plt.figure(2)
    plt.ylim(-1200, -100)
    #plt.legend(legend)
    plt.legend()
    plt.xlabel("Time [sec]")
    plt.ylabel("Reward")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("Test Reward")

    # figure 3 - test finish rate
    plt.subplot(122)#plt.figure(3)
    plt.ylim(0, 100)
    #plt.legend(legend)
    plt.legend()
    plt.xlabel("Time [sec]")
    plt.ylabel("Finish rate [%]")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title("Test Finish Rate")

    plt.tight_layout()
    plt.show()
    
    return mean_dict

#paths = [f"tests"+ '/' for file in range(1, 2) for version in [1]]
paths = [f"tests/test_{file_no}/" for file_no in range(31,41)]
#which_ones=[0,16]
which_ones=[0,16]
#which_ones=[0,16]
#loc = paths[4]
#plotter(which_ones=which_ones, smooths = [5, 5, 5, 5], reduce_smoothing_wrt_k=False)
#std_plotter(paths, which_ones=which_ones, smooths = [5, 5, 5, 5], reduce_smoothing_wrt_k=False)
mean_dict = std_test_plotter(paths, which_ones=which_ones, smooths = [(30/4), (30/4), (30/4), (30/4)], reduce_smoothing_wrt_k=False)

def advantage_plot(mean_dict, skill_train_time):
    baseline_times = mean_dict['baseline']['test_times']
    baseline_finish_rates = mean_dict['baseline']['test_finish_rate']
    hrl16_times = mean_dict['hrl16']['test_times']
    hrl16_finish_rates = mean_dict['hrl16']['test_finish_rate']
    
    baseline_max_index = np.argmax(baseline_finish_rates)
    baseline_max_time = baseline_times[baseline_max_index]
    baseline_max_val = baseline_finish_rates[baseline_max_index]
    
    hrl16_earliest_index = hrl16_finish_rates > baseline_max_val
    for i in range(len(hrl16_earliest_index)):
        if hrl16_earliest_index[i] == True: 
            hrl16_earliest_index = i
            break
    
    hrl16_earliest_time = hrl16_times[hrl16_earliest_index]
    hrl16_earliest_val = hrl16_finish_rates[hrl16_earliest_index]
    time_difference = baseline_max_time - hrl16_earliest_time
    
    x_axis = np.arange(1,16)
    baseline_n_time = x_axis*baseline_max_time
    hrl16_n_time = x_axis*hrl16_earliest_time + skill_train_time
    
    plt.figure(2)
    plt.plot(x_axis, baseline_n_time)
    plt.plot(x_axis, hrl16_n_time)
    plt.legend(["Baseline", "HRL 16 Steps"])
    plt.xlabel("The Number Runs")
    plt.ylabel("Time [s]")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.title(f"For {round(baseline_max_val*100,1)}% or Higher Finish Rate")
    
    #plt.xlim((x_axis.min(),x_axis.max()))

    plt.tight_layout()
    plt.show()
    
    
    print(baseline_max_val, baseline_max_time)
    print(hrl16_earliest_val, hrl16_earliest_time)
    print(time_difference)

advantage_plot(mean_dict, 41*60+6)
