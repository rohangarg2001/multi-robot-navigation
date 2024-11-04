import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def update(self, v, w, dt):
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += w * dt

class Environment:
    def __init__(self, n_robots):
        self.robots = [Robot(np.random.uniform(0, 10), np.random.uniform(0, 10), 
                           np.random.uniform(0, 2*np.pi)) for _ in range(n_robots)]

    def step(self, actions, dt):
        for robot, (v, w) in zip(self.robots, actions):
            robot.update(v, w, dt)

    def get_states(self):
        return [(robot.x, robot.y, robot.theta) for robot in self.robots]

class RobotAnimation:
    def __init__(self, environment, actions, dt):
        self.env = environment
        self.actions = actions
        self.dt = dt
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True)
        
        # Initialize empty plots for each robot
        self.robot_plots = []
        self.arrow_plots = []
        states = self.env.get_states()
        for x, y, theta in states:
            robot, = self.ax.plot([], [], 'bo')
            arrow = self.ax.arrow(x, y, 0, 0, head_width=0.2, head_length=0.2, 
                                fc='blue', ec='blue')
            self.robot_plots.append(robot)
            self.arrow_plots.append(arrow)

    def init(self):
        return self.robot_plots + self.arrow_plots

    def animate(self, frame):
        self.env.step(self.actions, self.dt)
        states = self.env.get_states()
        
        for i, ((x, y, theta), robot, arrow) in enumerate(zip(states, 
                                                            self.robot_plots, 
                                                            self.arrow_plots)):
            # Update robot position
            robot.set_data([x], [y])
            
            # Remove old arrow and create new one
            arrow.remove()
            self.arrow_plots[i] = self.ax.arrow(x, y, 
                                              0.5 * np.cos(theta), 
                                              0.5 * np.sin(theta),
                                              head_width=0.2, 
                                              head_length=0.2, 
                                              fc='blue', 
                                              ec='blue')
        
        return self.robot_plots + self.arrow_plots

def run_animation(n_robots=5, n_frames=200):
    env = Environment(n_robots)
    actions = [(1.0, 0.1) for _ in range(n_robots)]  # Example actions
    dt = 0.1
    
    anim = RobotAnimation(env, actions, dt)
    animation = FuncAnimation(anim.fig, anim.animate, init_func=anim.init,
                            frames=n_frames, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    run_animation()