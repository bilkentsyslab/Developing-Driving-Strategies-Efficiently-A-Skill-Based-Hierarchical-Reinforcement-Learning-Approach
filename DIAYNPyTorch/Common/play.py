import cv2
import numpy as np
import os
import imageio

every_n_frames = 2

class Play:
    def __init__(self, env, agent, n_skills):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Vid/"):
            os.mkdir("Vid/")

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self, dir=""):
        for z in range(self.n_skills):
            f_count = 0
            #video_writer = cv2.VideoWriter(f"Vid/skill{z}" + ".avi", self.fourcc, 50.0, (250, 250))
            frames = []
            s = self.env.reset()
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0

            """
            xs = []
            for _ in range(len(self.env.cars)):
                xs.append([])
            for car_i, car in enumerate(self.env.cars):
                xs[car_i].append([car[0], 0])
            """
            for t in range(self.env.spec.max_episode_steps):
                action = self.agent.choose_action(s)
                s_, r, done, _ = self.env.step(action)
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                if done:
                    break

                s = s_
                #'''
                I = self.env.render(mode='rgb_array')
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                #I = cv2.resize(I, (250, 250))
                #video_writer.write(I)
                if f_count % every_n_frames == 0:
                    I *= 255  # or any coefficient
                    I = I.astype(np.uint8)
                    frames.append(I)
                #'''
                f_count += 1
                end_t = t + 1

            print(f"skill: {z}, episode reward:{episode_reward:.1f}")
            #video_writer.release()
            #'''
            if not os.path.exists(f"Vid/{dir}"):
                os.mkdir(f"Vid/{dir}")
            with imageio.get_writer(f"Vid/{dir}skill{z}" + ".gif", mode="I") as writer:
                for idx, frame in enumerate(frames):
                    writer.append_data(frame)
            #'''
        self.env.close()
        cv2.destroyAllWindows()
