import numpy as np
import pandas as pd
import math
from environment import Environment, Robot
from EKF import StateEstimator
from rhoPOMDP import PLAN
import matplotlib.pyplot as plt
import numpy as np

def compute_initial_actions(Env, target_goals):
    """
    Compute the initial actions for all robots to move toward their respective goals.
    Each robot's control action is computed as (v, w) based on its current position and heading.
    
    Args:
    Env: The environment object containing robot states.
    target_goals: Array of goal positions for all robots [xg1, yg1, xg2, yg2, ...].
    
    Returns:
    actions: Array of shape (2*num_robots, 1) containing (v, w) for each robot.
    """
    num_robots = Env.n_robots
    actions = np.zeros((2 * num_robots, 1))
    states = Env.get_states()  # Get current states of the robots [x1, y1, theta1, ..., xn, yn, thetan]

    for i in range(num_robots):
        # Extract current state and goal for the ith robot
        x, y, theta = states[3 * i: 3 * i + 3]
        xg, yg = target_goals[2 * i: 2 * i + 2]
        
        # Compute the error in position
        dx = xg - x
        dy = yg - y
        
        # Compute the desired heading angle and angular error
        desired_theta = np.arctan2(dy, dx)
        angle_error = desired_theta - theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Normalize angle
        
        # Compute control actions (v, w) using a simple proportional controller
        v = 0.2 * np.sqrt(dx**2 + dy**2)  # Proportional to distance to the goal
        w = 0.5 * angle_error             # Proportional to angle error

        # Assign actions to the actions array
        actions[2 * i] = v
        actions[2 * i + 1] = w
    
    return actions

def main():
    Env = Environment(3)
    EKF = StateEstimator(Env)

    target_goals = np.array([2, 3, 4, 5, 6, 7])  # Example target goals for 3 robots
    initial_actions = compute_initial_actions(Env, target_goals)  # Compute initial actions
    prev_action = initial_actions
    # Storing the values for results and plots
    i = 0
    Env_readings = []
    Pred_readings = []
    EKF_readings = []
    Cov = []

    while i<1:
        
        s_t_1 = EKF.s_t_1     
        print('The action input to system', prev_action)
        Env.step(prev_action)

        #Storing the readings
        Env_readings.append(Env.get_states())

        #Get sensor reading and get EKF estimate
        sensor_readings = Env.get_observations()
        cur_es_state, s_pred, P = EKF.estimate(sensor_readings, s_t_1, prev_action.flatten())

        #Storing the readings
        Pred_readings.append(s_pred)
        EKF_readings.append(cur_es_state)
        Cov.append(P[1][1])

        '''
        The mean and covariance of the state is passed into the POMDP problem.
        '''
        action_next = PLAN(cur_es_state,P)
        prev_action = action_next
        i+=1
    
    # while i<10000:
        
    #     s_t_1 = EKF.s_t_1    
    #     # prev_action = Controller.prev_actions() 
    #     prev_action = np.array([(0.10, 0.1), (0.10, 0.1),(0.10, 0.1) ])
    #     # prev_action = np.array([(1.0, 0.1), (1.0, 0.1)])
    #     Env.step(prev_action)
    #     Env_readings.append(Env.get_states())
        
    #     sensor_readings = Env.get_observations()
    #     cur_es_state, s_pred = EKF.estimate(sensor_readings, s_t_1, prev_action.flatten())
    #     Pred_readings.append(s_pred)
    #     EKF_readings.append(cur_es_state)

    #     # action_next = Controller.decision(cur_es_state)
    #     action_next = np.array([(0.10, 0.1), (0.10, 0.1),(0.10, 0.1)])
    #     # action_next = np.array([(1.0, 0.1), (1.0, 0.1)])
    #     i+=1
    
    plt.plot(Cov)
    plt.show()
    
    results(Env_readings, EKF_readings, Pred_readings)


def results(states, ekf, pred):
    env = np.array(states)
    ekf_r = np.array(ekf)
    pred = np.array(pred)
    

    num_plots = env.shape[1]
    rows, cols = 3, 3  # Set rows and columns for 2x3 layout
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))  # Adjust figure size as needed

    for i in range(num_plots):
        row, col = divmod(i, cols)
        axes[row, col].plot(env[:, i, 0], label='env')
        axes[row, col].plot(pred[:, i, 0], label='pred')
        axes[row, col].plot(ekf_r[:, i, 0], label='ekf')
        axes[row, col].set_xlabel('X')
        axes[row, col].set_ylabel('Y')
        axes[row, col].grid()
        axes[row, col].legend()
        axes[row, col].set_title(f'Subplot {i+1}')
    
    # Hide any unused subplots if num_plots < rows * cols
    for j in range(num_plots, rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

