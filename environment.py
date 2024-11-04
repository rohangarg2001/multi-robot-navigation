import numpy as np
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def update(self, v, w, dt):
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += w * dt
        self.theta = self.normalize_angle(self.theta)

class Environment:
    def __init__(self, n_robots):
        self.n_robots = n_robots
        self.robots = [Robot(np.random.uniform(3, 10), np.random.uniform(1, 3), np.random.uniform(-np.pi, np.pi)) for _ in range(n_robots)]
        self.std_dev_uwb = 0.01                     ## STD for the UWB sensor
        self.std_compass = 0.05                     ## STD for the compass

    def step(self, actions, dt):
        for robot, (v, w) in zip(self.robots, actions):
            robot.update(v, w, dt)

    def get_states(self):
        return [(robot.x, robot.y, robot.theta) for robot in self.robots]
    
    def get_observations(self):
        """returns the observation vector of the entire n-robot system
        """
        observation = np.zeros((self.n_robots, self.n_robots))
        for i in range(self.n_robots):
            # we are calculating the observation matrix {z_{i}}
            z_i = []
            for j in range(self.n_robots):
                if(i!=j):
                    noise = np.random.randn() * self.std_dev_uwb 
                    z_i.append(np.linalg.norm((self.robots[j].x - self.robots[i].x, self.robots[j].y - self.robots[i].y)) + noise)
            compass_reading = self.robots[i].theta + np.random.randn() * self.std_compass
            z_i.append(compass_reading)
            observation[i,:] = z_i
        return observation

def visualize_robots(states):
    plt.figure()
    for x, y, theta in states:
        plt.plot(x, y, 'bo')
        plt.arrow(x, y, 0.5 * np.cos(theta), 0.5 * np.sin(theta), head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Positions')
    plt.grid()
    plt.show()

# Example usage
if __name__ == "__main__":
    env = Environment(n_robots=5)
    actions = [(1.0, 0.1) for _ in range(5)]  # Example actions for each robot
    dt = 0.1  # Time step
    # visualize_robots(env.get_states())
    for _ in range(3):  # Simulate 100 steps
        env.step(actions, dt)
        states = env.get_states()
    visualize_robots(env.get_states())
    print(env.get_observations())
                       