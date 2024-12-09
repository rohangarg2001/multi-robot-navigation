import numpy as np
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def update(self, v, w, dt, sigma_model):
        self.x += v * np.cos(self.theta) * dt 
        self.y += v * np.sin(self.theta) * dt 
        self.theta += w * dt 
        # self.theta = self.normalize_angle(self.theta)

class Environment:
    def __init__(self, n_robots, state):
        self.n_robots = n_robots
        self.deltaT = 0.1  # time steps
        self.robots = []

        # Initialize robots using the given state vector `s`
        for i in range(self.n_robots):
            # Each robot's state is given by 3 consecutive values in the vector `s`
            x = state[3*i]      # x position of robot i
            y = state[3*i + 1]  # y position of robot i
            theta = state[3*i + 2]  # orientation (theta) of robot i
            self.robots.append(Robot(x, y, theta))

        self.std_dev_uwb = 0.01  # STD for the UWB sensor
        self.std_compass = 0.05  # STD for the compass
        self.sigma_dyn = 0.05    # STD for the dynamics
        self.reachd_goal_ = False

    def step(self, actions):
        for robot, (v, w) in zip(self.robots, actions):
            robot.update(v, w, self.deltaT, self.sigma_dyn)

    def get_states(self):
        """
        returns a (3N,1) array 
        """
        states = np.array([np.array([robot.x, robot.y, robot.theta]) for robot in self.robots])
        return states.flatten().reshape(-1,1)
    
    def get_observations(self):
        """returns the observation vector of the entire n-robot system. The observation vector is a (3n,1) vector.
        The layout is that obs = [[dist(i,k) for all k != i], [heading angles]]  (ith robot observation)
        """
        observation = np.zeros((self.n_robots, self.n_robots - 1))
        for i in range(self.n_robots):
            # we are calculating the observation matrix {z_{i}}
            z_i = []
            for j in range(self.n_robots):
                if(i!=j):
                    noise = np.random.normal(0,self.std_dev_uwb )
                    z_i.append(np.linalg.norm((self.robots[j].x - self.robots[i].x, self.robots[j].y - self.robots[i].y)) + noise)
            observation[i,:] = z_i
        observation = list(observation.flatten())
        for j in range(self.n_robots):
            compass_reading = self.robots[j].theta + np.random.normal(0,self.std_compass) 
            observation.append(compass_reading)
        return np.array(observation)


