import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

from stable_baselines3.common.callbacks import BaseCallback


class FinishPercentageCallback(BaseCallback):
    def __init__(self, training_env, test_environment, env_name, model, verbose=0, train_how_long=200, runs_per_test=10, test_count=1000):
        super(FinishPercentageCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm

        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]

        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int

        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]

        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]


        #self.check_frequency = check_frequency
        #self.check_times = check_times
        self.training_environment = training_env
        self.test_environment = test_environment
        self.env_name = env_name
        self.the_model = model

        self.percentage_memory = [np.array([]),np.array([]),np.array([])]
        self.time_memory = np.array([0])
        self.start_time = time.time()

        # test stuff
        self.train_how_long = train_how_long
        self.runs_per_test = runs_per_test
        self.test_count = test_count

        #self.test_timestep_period = check_frequency
        #self.test_number_of_episodes = check_times

        # timers
        self.training_timer = time.time()
        self.training_times = [0]
        self.test_timer = time.time()
        self.test_times = [0]

        # memories
        self.training_episode_reward = []
        self.training_episode_finish_rate= []
        self.test_episode_reward = []
        self.test_episode_finish_rate = []

        # shift
        self.time_shift = 0

    def get_training_episode_reward(self):
        # run this at the end of the episode
        return self.training_environment.scores_memory[-1]
    def get_training_episode_finish_rate(self):
        return self.training_environment.finish_memory[-1]
    def get_test_episode_reward_and_finish_rate(self):
        finish_rate, reward = self.run_tests()
        return finish_rate, reward
    def run_tests(self):
        env = self.test_environment
        model = self.the_model

        finish_counter = 0
        total_reward = 0

        for _ in range(self.runs_per_test):
            obs = env.reset()
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                total_reward += rewards
                if info['finish'] == True: finish_counter += 1
                if dones: break
            
        return finish_counter/self.runs_per_test, total_reward/self.runs_per_test

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        how_long_has_it_been = time.time() - self.test_timer
        #if (self.num_timesteps % self.test_timestep_period) == 0:
        if how_long_has_it_been > self.train_how_long/self.test_count - self.time_shift:
            start_time = time.time()

            finish_rate, reward = self.get_test_episode_reward_and_finish_rate()
            self.test_episode_reward.append(reward)
            self.test_episode_finish_rate.append(finish_rate)
            
            end_time = time.time()
            duration = end_time-start_time
            self.training_timer = self.training_timer + duration
            self.test_timer = self.test_timer + duration

            self.test_times.append(time.time()-self.test_timer+self.test_times[-1])
            self.test_timer = time.time()

            self.time_shift = self.test_times[-1] - (len(self.test_times[1:]))*self.train_how_long/self.test_count
        
        if self.training_times[-1] > self.train_how_long:
            return False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.training_episode_reward.append(self.get_training_episode_reward())
        self.training_episode_finish_rate.append(self.get_training_episode_finish_rate())
        self.training_times.append(time.time()-self.training_timer+self.training_times[-1])
        self.training_timer = time.time()
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        if False:
            print(np.array(self.training_times[1:]))
            print(np.array(self.test_times[1:]))
            print(np.array(self.training_episode_reward))
            print(np.array(self.training_episode_finish_rate))
            print(np.array(self.test_episode_reward))
            print(np.array(self.test_episode_finish_rate))

        # save times
        np.save(f"{self.env_name}_training_times", np.array(self.training_times[1:]))
        np.save(f"{self.env_name}_test_times", np.array(self.test_times[1:]))

        # save data
        np.save(f"{self.env_name}_training_episode_reward", np.array(self.training_episode_reward))
        np.save(f"{self.env_name}_training_episode_finish_rate", np.array(self.training_episode_finish_rate))
        np.save(f"{self.env_name}_test_episode_reward", np.array(self.test_episode_reward))
        np.save(f"{self.env_name}_test_episode_finish_rate", np.array(self.test_episode_finish_rate))


