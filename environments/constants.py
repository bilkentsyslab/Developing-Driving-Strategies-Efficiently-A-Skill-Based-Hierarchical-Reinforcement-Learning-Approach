import numpy as np
import random

# values
v_maxx = 29.16
v_nomm = 9
v_minn = v_nomm / 3
d_close = d_closs = 7.73/2 #7.73
d_nomm = 23.28 #16.35
d_farr = 30
eps = 0.1

# box count (n+1)
bin_count = 10

# episode time (200)
max_episode_steps = 1000

# rewards weights
w_velocity = 1
w_headway_distance = 1*w_velocity
w_not_merging = 1*w_velocity
w_collision = 1000*w_velocity

reward_sum= 1#(w_collision + w_headway_distance + w_velocity + w_not_merging)/10
w_headway_distance /= reward_sum
w_velocity /= reward_sum
w_collision /= reward_sum
w_not_merging /= reward_sum

# speed boundaries for the agent, randomly chosen in between
min_v = v_minn
max_v = v_maxx
cars_speed_random_range = [-1, 1]
agent_inital_speed = v_nomm
agent_initial_lane = 1

# agent max observation distance
max_dist = d_farr

# acc. boundaries
max_dv = 4.5
min_dv = -4.5

# lane count
l_count = 2

# time interval for simulation
dt = 0.1
half_dtsqr = 0.5*(dt**2)

# merge boundaries
merge_start = 65
merge_end = 213
end = 263

# car width
half_width = 2.5

# how many cars
distance_between = 8.63

# how many cars
# onramp:
disttt = 0
onramp_locations = list()
onramp_locations_2 = list()
offramp_locations = list()
while disttt < end:
    onramp_locations.append(disttt)
    disttt += max(min(np.random.normal(d_nomm, 1), d_nomm+3), d_nomm-3)
disttt = 0
while disttt < end:
    onramp_locations_2.append(disttt)
    disttt += max(min(np.random.normal(d_nomm, 1), d_nomm+3), d_nomm-3)
distance_until_merge_threshold = 50
randomness = [-distance_between, 0]

# paint size
canvas_h = 60
canvas_w = 1500

# device
device = 'cpu'

def boxing(val, min_val, max_val, n):
    val -= min_val
    new_max = max_val-min_val
    m = new_max/n
    box = (val//m)/bin_count # 0 to 1
    return box

def onehot(i, n):
    res = [0 for k in range(n)]
    res[i] = 1
    return res

def calculate_reward_imported(self, action, info):
    """
    if info['max_length'] or info['collision']:
        return -100
    elif self.l == 1:
        return -1
    else:
        return 0
    #"""
    
    # collision
    collision = -1 if self.collides() else 0
    
    # headway distance
    d_nominal = d_nomm
    d_far = d_farr
    relative_distance = self.observe_lane_without_boxing(self.l)
    if relative_distance <= d_close:
        headway_distance = -1
    elif d_close < relative_distance and relative_distance <= d_nominal:
        headway_distance = 0#(relative_distance-d_nominal)/(d_nominal-d_close)
    else:
        headway_distance = 0
    
    # velocity
    v_nom = v_nomm
    v_max = v_maxx
    v = self.v
    velocity = (v-v_nom)/v_nom if v <= v_nom else (v_nom-v)/(v_max-v_nom)
    
    """
    # effort
    normal = 0.25
    hard = 2
    dv_abs = abs(action[1])
    if dv_abs <= normal:
        effort = -0.25
        dv_class = 'normal'
    elif dv_abs > normal and dv_abs <= hard:
        effort = -1
        dv_class = 'hard'
    else:
        effort = 0
        dv_class = 'no'
    """
    
    # not_merging
    if self.version == 1:
        not_merging = -1 if (self.l == 1) else 0
    elif self.version == 2:
        if self.l == 0: not_merging = 0
        elif self.l == 1: not_merging = -0.5
        elif self.l == 2: not_merging = -1


    """
    # stop
    de = merge_end - self.x
    if self.l == 1:
        relative_distance_back , relative_distance_front = np.array(self.observe_lane_without_boxing(0))[[0,3]]
        if relative_distance_front >= d_close and relative_distance_back <= 1.5*d_far:  #action[1]
            stop = -1
        elif de < d_far:
            stop = -0.05
        else:
            stop = 0
    else:
        if dv_class != 'hard' and relative_distance >= d_far and de >= d_far:
            stop = -1
        else:
            stop = 0
    """

    R = collision*w_collision + headway_distance*w_headway_distance + velocity*w_velocity + not_merging*w_not_merging
    #R += effort*w_effort + stop*w_stop
    return R

# without boxing
def observe_lane_without_boxing_imported(self, lane):
    cars = []
    for i in range(len(self.cars)):
        if self.cars[i][2] == lane:
            cars.append(self.cars[i])
    
    head = max_dist + 2* half_width
    
    for i in range(len(cars)):
        if cars[i][0] < self.x:
            continue
        else:
            dist = cars[i][0] - self.x
            if dist < head:
                head = dist
        
    head -= 2* half_width
                
    return head

# observing level0s by level1 in a specific lane, with boxing
def observe_lane_imported(self, lane):
    cars = []
    for i in range(len(self.cars)):
        if self.cars[i][2] == lane:
            cars.append(self.cars[i])
    
    tail = [max_dist+2*half_width, 
            0]
    head = [max_dist+2*half_width, 
            0]
    
    for i in range(len(cars)):
        if cars[i][0] < self.x:
            dist = self.x - cars[i][0] 
            if dist < tail[0]:
                tail[0] = dist
                tail[1] = cars[i][1] - self.v
        else:
            dist = cars[i][0] - self.x
            if dist < head[0]:
                head[0] = dist
                head[1] = self.v - cars[i][1]
    
    tail[0]-= 2*half_width
    tail[0] = max(0, tail[0])
    head[0]-= 2*half_width
    head[0] = max(0, head[0])

    # cap and normalize
    """
    tail[0] = min(max_dist, tail[0]) / max_dist
    tail[1] /= max_v - min_v
    head[0] = min(max_dist, head[0]) / max_dist
    head[1] /= max_v - min_v
    """
    return tail + head

def observe_imported(self):
    if self.version == 1: l_count = 2
    elif self.version == 2: l_count = 3
    obs_v = self.v
    obs_l = onehot(self.l, l_count)
    
    obs_cars = self.observe_cars() # this
    obs_merge = [1 if self.x > (merge_start + half_width) else 0]
    obs = np.array([obs_v] + obs_l + obs_merge + obs_cars, dtype=np.float16)
    # obs_cars = tail dist, tail speed, head dist, head speed
    # left first, then right

    # normalize
    boundaries = np.array([
        [min_v, max_v], # ego velocity
        [0,1], # is on left lane 
        [0,1], # is on right lane
        [0,1], # can merge
        [0,max_dist], # some relative dist?
        [min_v-max_v, max_v-min_v], # some relative vel?
        [0,max_dist], # some relative dist?
        [min_v-max_v, max_v-min_v], # some relative vel?
        [0,max_dist], # some relative dist?
        [min_v-max_v, max_v-min_v], # some relative vel?
        [0,max_dist], # some relative dist?
        [min_v-max_v, max_v-min_v], # some relative vel?
    ])
    #obs = (obs - boundaries[:,0])/(boundaries[:,1] - boundaries[:,0])
    #obs *= 10

    # boxing
    if False:
        obs = np.rint(obs * bin_count)/bin_count
    
    return obs

def step_imported(self, action):
    self.step_counter += 1
    self.last_action = action

    dv = action[0] #+ max(min(np.random.normal(0, 1), 1), -1)
    p_dl = action[1]
    
    if dv > 0:
        dv = dv * max_dv
    else:
        dv = - dv * min_dv
    self.x = self.x + self.v*dt + dv*half_dtsqr
    self.v = max(min(self.v + dv*dt, v_maxx), v_minn)
    if self.v > max_v:
        self.v = max_v
    elif self.v < min_v:
        self.v = min_v
    
    self.t = self.t + dt
    r = random.uniform(0,1)
    if (r < p_dl) and (self.l != 0) and (self.x > (merge_start+half_width)):
        self.l = self.l - 1
        if self.change_x < 0:
            self.change_x = self.x
        elif self.change_x_2 < 0:
             self.change_x_2 = self.x
        
    self.update_cars()
        
    done = False
    #if self.l == l_count - 1 and self.x > merge_end:
    #    done = True
    if self.collides():# or (self.l == 1 and self.x > merge_end):
        done = True
        self.info['collision'] = True
    if self.x > end:
        done = True
        self.info['finish'] = True
    if self.step_counter == self.spec.max_episode_steps:
        done = True
        self.info['max_length'] = True
    if self.too_close():
        self.info['too_close'] = True
        
    reward = self.calculate_reward(action, self.info)

    return self.observe(), reward, done, self.info.copy()