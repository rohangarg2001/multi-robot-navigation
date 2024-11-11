import numpy as np
import pandas as pd
import math
from environment import Environment, Robot
from EKF import StateEstimator
from rhoPOMDP import SeqDec
import matplotlib.pyplot as plt

def results(states, ekf, pred):
    env = np.array(states)
    ekf_r = np.array(ekf)
    pred = np.array(pred)
    print(env.shape, ekf_r.shape)

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

def main():
    Env = Environment(3)
    EKF = StateEstimator(Env)
    # Controller = SeqDec()

    i = 0
    Env_readings = []
    Pred_readings = []
    EKF_readings = []
    while i<100:
        
        s_t_1 = EKF.s_t_1    
        # prev_action = Controller.prev_actions() 
        prev_action = np.array([(1.0, 0.1), (1.0, 0.1),(1.0, 0.1) ])
        # prev_action = np.array([(1.0, 0.1), (1.0, 0.1)])
        Env.step(prev_action)
        Env_readings.append(Env.get_states())
        
        sensor_readings = Env.get_observations()
        cur_es_state, s_pred = EKF.estimate(sensor_readings, s_t_1, prev_action.flatten())
        Pred_readings.append(s_pred)
        EKF_readings.append(cur_es_state)

        # action_next = Controller.decision(cur_es_state)
        action_next = np.array([(1.0, 0.1), (1.0, 0.1),(1.0, 0.1)])
        # action_next = np.array([(1.0, 0.1), (1.0, 0.1)])
        i+=1
    
    results(Env_readings, EKF_readings, Pred_readings)



if __name__ == '__main__':
    main()

