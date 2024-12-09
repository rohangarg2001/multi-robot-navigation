import numpy as np
import matplotlib.pyplot as plt
import time
import random

class DiscreteEnvironment():
    def __init__(self, n_robots, grid_size, initial_position, goal_positions):
        self.n_robots = n_robots
        self.grid_size = grid_size
        self.goal_positions = goal_positions
        # self.robot_positions = np.random.randint(0, self.grid_size, (self.n_robots, 2))        
        self.robot_positions = initial_position
        # self.action_map = {
        #     0: (-1, 0),   # North
        #     1: (-1, 1),   # Northeast
        #     2: (0, 1),    # East
        #     3: (1, 1),    # Southeast
        #     4: (1, 0),    # South
        #     5: (1, -1),   # Southwest
        #     6: (0, -1),   # West
        #     7: (-1, -1)   # Northwest
        # }
        self.action_map = {
            0: (-1, 0),   # North
            1: (0, 1),    # East
            2: (1, 0),    # South
            3: (0, -1),   # West
            4: (0, 0)
        }

        self.transition_probaility = {
            'intended' : 0.7,
            'random' : 0.15,
            'remain' : 0.15
        }
        # self.action_space = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]        
        self.action_space = [(-1, 0), (0, 1), (1, 0), (0, -1)] 
        self.cardinal_action_space = len(self.action_space)
        self.all_action_combinations = np.array(np.meshgrid(*[range(self.cardinal_action_space)] * self.n_robots)).T.reshape(-1, self.n_robots)
        self.gamma = 0.9

    def set_positon(self,position):
        self.robot_positions = position
    
    def reset(self):
        self.robot_positions = [np.random.randint(0, self.n_robots, size=2) for _ in range(self.n_robots)]

    def calculate_reward(self) -> float:
        reward = 0
        for i in range(self.n_robots):
            current_position = self.robot_positions[i]
            goal_position = self.goal_positions[i]

            distance = -np.sum(np.abs(current_position - goal_position))
            reward += distance
        return reward
    
    def calculate_U(self, s) -> float:
        reward = 0
        for i in range(self.n_robots):
            current_position = s[i]
            goal_position = self.goal_positions[i]

            distance = -np.sum(np.linalg.norm(current_position - goal_position) ** 2)
            reward += distance
        return reward
    
    def step(self, actions):
        new_positions = self.robot_positions.copy()
        for i, action in enumerate(actions):
            p = random.random()
            if(p < self.transition_probaility['intended']):
                dx, dy = self.action_map[action]

                new_x = new_positions[i][0] + dx
                new_y = new_positions[i][1] + dy
                if (0 <= new_x < self.grid_size) and (0 <= new_y < self.grid_size):
                    new_positions[i] = [new_x, new_y]
            elif(p < self.transition_probaility['intended'] + self.transition_probaility['random']):
                dx, dy = self.action_map[np.random.randint(0,len(self.action_space))]
                new_x = new_positions[i][0] + dx
                new_y = new_positions[i][1] + dy
                if (0 <= new_x < self.grid_size) and (0 <= new_y < self.grid_size):
                    new_positions[i] = [new_x, new_y]
            else:
                self.robot_positions = new_positions
        self.robot_positions = new_positions
        return self.robot_positions
    
    def check_goal(self):
        return np.all(self.robot_positions == self.goal_positions)
    
    def visualize_state(self, title="Current State"):
        plt.figure(figsize=(8, 8))
        colors = plt.cm.get_cmap('hsv', self.n_robots)
        for i in range(self.n_robots):
            plt.plot(self.robot_positions[i][1], self.robot_positions[i][0], 'o', color=colors(i / self.n_robots), markersize=10)
            plt.text(self.robot_positions[i][1], self.robot_positions[i][0], f'R{i}', fontsize=12, ha='right')
            plt.plot(self.goal_positions[i][1], self.goal_positions[i][0], '*', color=colors(i / self.n_robots), markersize=10)
            plt.text(self.goal_positions[i][1], self.goal_positions[i][0], f'G{i}', fontsize=12, ha='right')
        plt.grid(True)
        plt.xlim(-1, self.grid_size)
        plt.ylim(-1, self.grid_size)
        plt.title(title)
        plt.show()
        plt.close()

    def direct_to_goal(self):
        robot_positions , goal_positions = self.robot_positions, self.goal_positions
        actions = []
        for i in range(self.n_robots): 
            position_diff_x, position_diff_y = robot_positions[i][0] - goal_positions[i][0], robot_positions[i][1] - goal_positions[i][1]

            if(position_diff_x > 0):
                actions.append(0)
            elif(position_diff_x < 0):
                actions.append(2)
            else:
                if(position_diff_y > 0):
                    actions.append(3)
                elif(position_diff_y < 0):
                    actions.append(1)
                else:
                    actions.append(4)
        return actions

    def random_rollout(self):
        actions = [np.random.randint(0,self.cardinal_action_space) for _ in range(self.n_robots)]
        self.step(actions)
    
    def visualize_random_rollout(self, num_steps, visual_delay = 0.3):
        self.visualize_state()
        for i in range(num_steps):
            self.random_rollout()
            self.visualize_state()
            time.sleep(visual_delay)