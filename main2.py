import numpy as np
import pandas as pd
import math
from environment import Environment, Robot
from EKF import StateEstimator
from rhoPOMDP import PLAN, compute_actions, plot_tree, HistoryTree
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    state = np.array([0, 1, 0, 0.2, 1, 0, 0.3, 1, 0]) 
    Env = Environment(3,state)
    EKF = StateEstimator(Env)
    history_tree = HistoryTree()  # Initialize the history tree

    target_goals = np.array([7, 5, 5, 7, 2, 7])   # Example target goals for 3 robots
    initial_actions = compute_actions(Env, target_goals)  # Compute initial actions
    #Adding a child node for first initial action
    action_node = history_tree.root.add_child(action=initial_actions, is_observation_node=False)


    prev_action = initial_actions.reshape(-1,2)
    # Storing the values for results and plots
    i = 0
    Env_readings = []
    Pred_readings = []
    EKF_readings = []
    Cov = []
    start_time = time.time()

    # while i<1000:
        
    #     s_t_1 = EKF.s_t_1     
    #     #Storing the readings
    #     Env_readings.append(Env.get_states())

    #     #Get sensor reading and get EKF estimate
    #     sensor_readings = Env.get_observations()
    #     # Add the initial observation as a child of the first action node
    #     observation_node = action_node.add_child(observation=sensor_readings, is_observation_node=True)
    #     observation_node.parent = None  # Detach from the parent
    #     history_tree.root = observation_node
    #     # plot_tree(history_tree)

        
    #     Env.step(prev_action)

    #     cur_es_state, s_pred, P = EKF.estimate(sensor_readings, s_t_1, prev_action.flatten())

    #     #Storing the readings
    #     Pred_readings.append(s_pred)
    #     EKF_readings.append(cur_es_state)
    #     Cov.append(P[1][1])

    #     '''
    #     The mean and covariance of the state is passed into the POMDP problem.
    #     '''
    #     # direct go planner
    #     # action_next = compute_actions(Env, target_goals)  
    #     action_next, updated_tree = PLAN(cur_es_state, P, history_tree)

    #     print('This is the action to be taken:', action_next)
    #     # Add the chosen action as a child of the current observation node
    #     action_node = observation_node.add_child(action=action_next, is_observation_node=False)


    #     # Resetting values for next time step
    #     history_tree = prune_to_action_subtree(updated_tree, action_next)
    #     action_node = history_tree.root
    #     prev_action = action_next.reshape(-1,2)
    #     # plot_tree(history_tree,i)
    #     i+=1
    
    while i < 1000:
        s_t_1 = EKF.s_t_1
        # Storing the readings
        Env_readings.append(Env.get_states())

        # Get sensor reading and EKF estimate
        sensor_readings = Env.get_observations()

        if i >= 1:
            # Find the closest observation child node to the current sensor reading
            min_distance = float('inf')
            closest_node = None
            for child in action_node.children:
                if child.is_observation_node and child.observation is not None:
                    distance = np.linalg.norm(np.array(child.observation) - np.array(sensor_readings))
                    if distance < min_distance:
                        min_distance = distance
                        closest_node = child

            # Replace the closest node's observation with the current sensor reading
            if closest_node is not None:
                closest_node.observation = sensor_readings

            # Detach this node and make it the root
            closest_node.parent = None
            history_tree.root = closest_node
            observation_node = closest_node
        else:
            # Add the initial observation as a child of the first action node
            observation_node = action_node.add_child(observation=sensor_readings, is_observation_node=True)
            observation_node.parent = None  # Detach from the parent
            history_tree.root = observation_node

        # Step the environment
        Env.step(prev_action)

        # Get EKF estimate
        cur_es_state, s_pred, P = EKF.estimate(sensor_readings, s_t_1, prev_action.flatten())

        # Storing the readings
        Pred_readings.append(s_pred)
        EKF_readings.append(cur_es_state)
        Cov.append(P[1][1])

        # Pass mean and covariance of the state into the POMDP planner
        action_next, updated_tree = PLAN(cur_es_state, P, history_tree)

        print('This is the action to be taken:', action_next)

        # Add the chosen action as a child of the current observation node
        action_node = observation_node.add_child(action=action_next, is_observation_node=False)

        # Resetting values for next time step
        history_tree = prune_to_action_subtree(updated_tree, action_next)
        action_node = history_tree.root
        prev_action = action_next.reshape(-1, 2)
        i += 1

    
    end_time = time.time()
    print(f'Total Time taken : {end_time-start_time}')
    results(Env_readings, EKF_readings, Pred_readings)


def prune_to_action_subtree(tree, target_action):
    for child in tree.root.children:
        if np.array_equal(child.action, target_action):
            child.parent = None  # Detach the parent
            tree.root = child    # Set the child as the new root
            return tree
    # If no matching action is found, return an empty tree with a fresh root
    return HistoryTree()

def results(states, ekf, pred):
    env = np.array(states)
    ekf_r = np.array(ekf)
    pred = np.array(pred)
    

    num_plots = env.shape[1]
    rows, cols = 3, 3  # Set rows and columns for 2x3 layout
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))  # Adjust figure size as needed

    for i in range(num_plots):
        row, col = divmod(i, cols)
        # axes[row, col].plot(pred[:, i, 0], label='pred')
        axes[row, col].plot(env[:, i, 0], label='env')
        axes[row, col].plot(ekf_r[:, i, 0], label='ekf')

        axes[row, col].plot(abs(env[:, i, 0]-ekf_r[:, i, 0]), label='Error')
        axes[row, col].set_xlabel('time')
        if col==0:
            axes[row, col].set_ylabel('X')
        elif col==1:
            axes[row, col].set_ylabel('Y')
        elif col==2:
            axes[row, col].set_ylabel('Heading Angle')
        axes[row, col].grid()
        axes[row, col].legend()
        if row == 0:
            axes[row, col].set_title(f'Robot {1}')
        if row == 1:
            axes[row, col].set_title(f'Robot {2}')
        if row == 2:
            axes[row, col].set_title(f'Robot {3}')
    
    # Hide any unused subplots if num_plots < rows * cols
    for j in range(num_plots, rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

