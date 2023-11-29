import gym
from gym import spaces
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gym.utils.seeding
from constants import *

class Spec():
    def __init__(self):
        self.max_episode_steps = max_episode_steps # 200
        self.id = 0

class TrafficEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

# init
    def __init__(self, version):
        self.added_counter = 0
        self.last_action = [-1,-1]
        self.version = version
        super(TrafficEnv, self).__init__()
        self.dist_1 = max(min(np.random.normal(d_nomm, 1), d_nomm+3), d_nomm-3)
        self.dist_2 = max(min(np.random.normal(d_nomm, 1), d_nomm+3), d_nomm-3)
        
        # acceleration = [-1, 1], lane change prob = [-0.1, 1.1]
        low  = np.array([-1, -0.1])
        high = np.array([ 1, 1.1])
        self.action_space = spaces.Discrete(6)
        # self.action_space = spaces.Discrete(10)
      
        # speed = [0,1], one hot for two lanes, 
        # v and x for 6 cars tail and head on left same and right lanes
        # total 4 + 4*2 = 12 obs
        if self.version == 1:
            low  = np.zeros(12)
            high = np.ones(12)
        elif self.version == 2:
            low  = np.zeros(13)
            high = np.ones(13)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float16)
        
        self.t = 0
        self.init_agent()
        
        self.canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)
        
        self.spec = Spec()
        
        self.np_random = np.random
        
        self.change_x = -1
        self.change_x_2 = -1
        
        self.cars = []
        self.init_cars()
        self.step_counter = 0

        self.render_mode = "human"

# randomly spawn level-0 cars
    def init_cars(self):
        # fill onramp 1
        for location in onramp_locations:
            random_distance = np.random.uniform(*randomness)
            speedd = min(max(np.random.normal(v_nomm, 1), v_nomm-3), v_nomm+3)
            self.cars.append(np.array([location+random_distance, speedd, 0]))
            
        # fill onramp 2
        if self.version == 2:
            for location in onramp_locations_2:
                random_distance = np.random.uniform(*randomness)
                speedd = min(max(np.random.normal(v_nomm, 1), v_nomm-3), v_nomm+3)
                self.cars.append(np.array([location+random_distance, speedd, 1]))

        # merge and represented as car
        if self.version == 1:
            self.cars.append(np.array([merge_end+2*half_width, 0, 1]))
        elif self.version == 2: 
            self.cars.append(np.array([merge_end+2*half_width, 0, 2]))        

# level-0 car brain (acc, speed, lane)
    def update_cars(self):
        min_1 = 9999999
        min_2 = min_1
        max_1 = 0
        max_2 = max_1
        max_1_index = None
        max_2_index = max_1_index
        for i in range(len(self.cars)-1): # -1 to exlude the merge
            # find the car in front
            front_dist = 99999999999     
            front_ind = -1  
            front_speed = 999999
            for j in range(len(self.cars)):
                if (i != j) and (self.cars[i][2] == self.cars[j][2]):
                    rel_dist = (self.cars[j][0] - self.cars[i][0]) - 2*half_width
                    if (rel_dist >= 0):
                        if (rel_dist <= front_dist):
                            front_ind = j
                            front_dist = rel_dist
                            front_speed = self.cars[j][1] - self.cars[i][1]

            # agent
            if self.l == self.cars[i][2]:
                rel_dist = (self.x - self.cars[i][0]) - 2*half_width
                if (rel_dist >= 0):
                    if (rel_dist <= front_dist):
                        front_ind = -2
                        front_dist = rel_dist
                        front_speed = self.v - self.cars[i][1]

            #tc = -front_dist/front_speed
            speed = self.cars[i][1]

            # ------------------------------- level0 start
            TTChd = 3
            TTCd = 5
            d_close = d_closs

            FCv = front_speed
            FCd = front_dist
            if FCv > 0:
                TTC = TTCd + eps
            else:
                FCv = max(-FCv, eps)
                TTC = FCd/FCv
            
            # decide action
            if TTC <= TTChd or FCd <= d_close:
                dv = max(-(np.random.exponential(0.75) + 2), -4.5)#hard_dec
            elif TTC <= TTCd:
                dv = max(-(np.random.exponential(0.75) + 0.25), -2)#dec
            elif speed < v_nomm:
                dv = min(np.random.exponential(0.75) + 0.25, 3)#acc
            else:
                dv = min(max(np.random.laplace(0, 0.1), -0.25), 0.25)#maintain
            # ------------------------------- level0 end

            # execute action
            v = self.cars[i][1]
            self.cars[i][1] = max(min(v + dt*dv, v_maxx), v_minn)
            self.cars[i][0] += v*dt + dv*half_dtsqr
            if self.cars[i][0] > merge_end-distance_until_merge_threshold:
                pass#self.cars[i][2] = 0
            
            # ready
            if self.cars[i][2] == 0:
                min_1 = min(self.cars[i][0], min_1)
                max_1 = max(self.cars[i][0], max_1)
                if max_1 == self.cars[i][0]: max_1_index = i
            elif self.cars[i][2] == 1:
                min_2 = min(self.cars[i][0], min_2)
                max_2 = max(self.cars[i][0], max_2)
                if max_2 == self.cars[i][0]: max_2_index = i

            
        # add new car, delete old ones
        if (self.version == 1) or (self.version == 2):
            if min_1 > self.dist_1:
                self.dist_1 = max(min(np.random.normal(d_nomm, 1), d_nomm+3), d_nomm-3)
                speedd = min(max(np.random.normal(v_nomm, 1), v_nomm-3), v_nomm+3)
                self.cars.insert(0, np.array([0, speedd, 0]))
            if max_1 > end + half_width :
                self.cars.pop(max_1_index)
        if self.version == 2:
            if min_2 > self.dist_1:
                self.dist_1 = max(min(np.random.normal(d_nomm, 1), d_nomm+3), d_nomm-3)
                speedd = min(max(np.random.normal(v_nomm, 1), v_nomm-3), v_nomm+3)
                self.cars.insert(0, np.array([0, speedd, 1]))
            if max_2 > end + half_width :
                self.cars.pop(max_2_index)

# randomly spawn level-1 car
    def init_agent(self):
        self.x = 0
        self.v = agent_inital_speed#random.uniform(min_v, max_v)
        if self.version == 2: 
            self.l = 2
        elif self.version == 1: 
            self.l = 1

# whether there is a collision or not     
    def collides(self):
        for i in range(len(self.cars)):
            if self.l == self.cars[i][2]: # if on same lane
                if abs(self.x - self.cars[i][0]) < 2*half_width:
                    return True
        return False
    
    def too_close(self, threshold=1):
        for i in range(len(self.cars)):
            if self.l == self.cars[i][2]: # if on same lane
                if abs(self.x - self.cars[i][0]) < 2*half_width+1:
                    return True
        return False

# drawing stuff
    def draw_car(self, car):
        bias = 5
        total = (canvas_h-2*bias)//2 if self.version == 1 else (canvas_h-2*bias)//3
        l1 = 0*total + bias #canvas_h // 15
        l2 = 1*total + bias #canvas_h // 2
        l3 = 2*total + bias #canvas_h - l1
        l4 = 3*total + bias
        
        x = car[0]
        l = car[2]
        r_x = int((x/end)*canvas_w)
        r_y = 0
        if l == 0:
            r_y = int(l1 + (l2-l1)/2)
        if l == 1:
            r_y = int(l2 + (l3-l2)/2)
        if l == 2:
            r_y = int(l3 + (l4-l3)/2)
        r_w = int(((half_width*2)/end)*canvas_w*0.5)
        r_h = int((l2-l1)*0.7*0.5)
        r_c = (255/255,42/255,42/255)
        cv2.rectangle(self.canvas, (r_x-r_w,r_y-r_h), (r_x+r_w,r_y+r_h), r_c, -1)

# drawing stuff
    def draw_elements_on_canvas(self):
        self.canvas[:,:,:] = 1.0
        bias = 5
        total = (canvas_h-2*bias)//2 if self.version == 1 else (canvas_h-2*bias)//3
        l1 = 0*total + bias #canvas_h // 15
        l2 = 1*total + bias #canvas_h // 2
        l3 = 2*total + bias #canvas_h - l1
        l4 = 3*total + bias
        c = (0, 0, 0)
        merge_start_c = int(((merge_start-half_width)/end)*canvas_w)
        merge_end_c = int(((merge_end+half_width)/end)*canvas_w)
        if self.version == 1:
            cv2.line(self.canvas, (0,l1), (canvas_w,l1), c, 2)
            cv2.line(self.canvas, (0,l2), (merge_start_c,l2), c, 2)
            cv2.line(self.canvas, (merge_end_c,l2), (canvas_w,l2), c, 2)
            cv2.line(self.canvas, (0,l3), (merge_end_c,l3), c, 2)
            cv2.line(self.canvas, (merge_end_c,l2), (merge_end_c,l3), c, 2)
        elif self.version == 2:
            cv2.line(self.canvas, (0,l1), (canvas_w,l1), c, 2)
            cv2.line(self.canvas, (0,l3), (merge_start_c,l3), c, 2)
            cv2.line(self.canvas, (merge_end_c,l3), (canvas_w,l3), c, 2)
            cv2.line(self.canvas, (0,l4), (merge_end_c,l4), c, 2)
            cv2.line(self.canvas, (merge_end_c,l3), (merge_end_c,l4), c, 2)
        
        r_x = int((self.x/end)*canvas_w)
        r_y = 0
        if self.l == 0:
            r_y = int(l1 + (l2-l1)/2)
        if self.l == 1:
            r_y = int(l2 + (l3-l2)/2)
        if self.l == 2:
            r_y = int(l3 + (l4-l3)/2)
        r_w = int(((half_width*2)/end)*canvas_w*0.5)
        r_h = int((l2-l1)*0.7*0.5)
        r_c = (1/255,1/255,255/255)
        cv2.rectangle(self.canvas, (r_x-r_w,r_y-r_h), (r_x+r_w,r_y+r_h), r_c, -1)
        
        for i in range(len(self.cars)-1):
            self.draw_car(self.cars[i])
            
        if self.change_x > 0:
            if self.version == 1:
                x = int((self.change_x/end)*canvas_w)
                cv2.line(self.canvas, (x,l1), (x,l3), r_c, 2)
            elif self.version == 2:
                x = int((self.change_x/end)*canvas_w)
                cv2.line(self.canvas, (x,l1), (x,l4), r_c, 2)
        if self.change_x_2 > 0:
            x = int((self.change_x_2/end)*canvas_w)
            cv2.line(self.canvas, (x,l1), (x,l4), r_c, 2)

        # print the last action on bottom right
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (1300, 50) 
        fontScale = 0.5
        color = (0, 0, 0) 
        thickness = 1
        if self.last_action[0] < 0:
            text =  f'%.2f, %.2f'%(self.last_action[0],self.last_action[1])
        elif self.last_action[1] < 0: 
            text =  f' %.2f,%.2f'%(self.last_action[0],self.last_action[1])
        elif self.last_action[0] < 0 and self.last_action[1] < 0:
            text =  f'%.2f,%.2f'%(self.last_action[0],self.last_action[1])
        else:
            text =  f' %.2f, %.2f'%(self.last_action[0],self.last_action[1])
        text += f", %.2f " % self.t
        
        image = cv2.putText(self.canvas, text, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 

# without boxing
    def observe_lane_without_boxing(self, lane):
        return observe_lane_without_boxing_imported(self, lane)

# observing level0s by level1 in a specific lane, with boxing
    def observe_lane(self, lane):
        return observe_lane_imported(self, lane)

# observing level0s by level1
    def observe_cars(self):
        obs = []
        
        # cars on left
        obs += self.observe_lane(self.l - 1)
        # cars on same
        obs += self.observe_lane(self.l)
        # cars on right
        #obs += self.observe_lane(self.l + 1)
        
        return obs
    
# observation
    def observe(self):
        return observe_imported(self)

# calculate reward
    def calculate_reward(self, action, info):
        return calculate_reward_imported(self, action, info)

# action
    def action(self, action):
        hard_dec = max(-(np.random.exponential(0.75) + 2), -4.5)
        dec = max(-(np.random.exponential(0.75) + 0.25), -2)
        maintain = min(max(np.random.laplace(0, 0.1), -0.25), 0.25)
        acc = min(np.random.exponential(0.75) + 0.25, 3)
        hard_acc = max((np.random.exponential(0.75) + 2), 4.5)
        if action == 0:
            p_dl = 0
            dv = hard_dec
        elif action == 1:
            p_dl = 0
            dv = dec
        elif action == 2:
            p_dl = 0
            dv = maintain
        elif action == 3:
            p_dl = 0
            dv = acc
        elif action == 4:
            p_dl = 0
            dv = hard_acc
        elif action == 5:
            p_dl = 1
            dv = maintain
        
        return [dv, p_dl]

# taking a step
    def step(self, action):
        action = self.action(action)
        return step_imported(self, action)
            
# reset the environment
    def reset(self):
        self.t = 0
        self.change_x = -1
        self.change_x_2 = -1
        self.init_agent()
        self.cars = []
        self.init_cars()
        self.step_counter = 0

        self.info = {}
        self.info['collision'] = False
        self.info['finish'] = False
        self.info['max_length'] = False
        self.info['too_close'] = False
        
        return self.observe()
        
# render
    def render(self, mode = "human"):
        self.draw_elements_on_canvas()
        
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(50)
        
        elif mode == "rgb_array":
            return self.canvas
    
# close the environemnt
    def close(self):
        cv2.destroyAllWindows()

def main():
    np.set_printoptions(suppress=True)
    for version in [1]:
        env = TrafficEnv(version)
        max_episode_steps = env.spec.max_episode_steps
        obs = env.reset()
        for i in range(max_episode_steps):
            print(obs.shape)
            env.render()
            #print(i, int(max_episode_steps//2), int(max_episode_steps-max_episode_steps//3))
            if (i == int(max_episode_steps//2)) or (i==int(max_episode_steps-max_episode_steps//3)):
                act = np.array([5])
                print("lane change attempt!")
            else:
                act = np.array([2])
            obs, rewards, dones, info = env.step(act)
            if dones: 
                print(info)
                cv2.waitKey()
                break
        env.close()

# run
if __name__ == "__main__":
    main()

    