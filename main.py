import numpy as np
import pandas as pd
import math
from environment import Environment, Robot
from EKF import StateEstimator
from rhoPOMDP import SeqDec
import matplotlib.pyplot as plt
# def results(states, ekf, pred):
#     env = np.array(states)
#     ekf_r = np.array(ekf)
#     pred = np.array(pred)
#     print(env.shape, ekf_r.shape)

#     num_plots = env.shape[1]
#     rows, cols = num_plots//3, 3  # Set rows and columns for 2x3 layout
#     fig, axes = plt.subplots(rows, cols, figsize=(15, 10))  # Adjust figure size as needed
#     for i in range(num_plots):
#         if(i % 3 == 0):
#             row, col = divmod(i, cols)
#             axes[row, col].plot(env[:, i, 0], label='env')
#             axes[row, col].plot(pred[:, i, 0], label='pred')
#             axes[row, col].plot(ekf_r[:, i, 0], label='ekf')
#             axes[row, col].set_xlabel('time_step')
#             axes[row, col].set_ylabel('X')
#             axes[row, col].grid()
#             axes[row, col].legend()
#             axes[row, col].set_title('robot X: ' + str(i//3))
#         if(i % 3 == 1):
#             row, col = divmod(i, cols)
#             axes[row, col].plot(env[:, i, 0], label='env')
#             axes[row, col].plot(pred[:, i, 0], label='pred')
#             axes[row, col].plot(ekf_r[:, i, 0], label='ekf')
#             axes[row, col].set_xlabel('time_step')
#             axes[row, col].set_ylabel('Y')
#             axes[row, col].grid()
#             axes[row, col].legend()
#             axes[row, col].set_title('robot Y: ' + str(i//3))
#         if(i % 3 == 2):
#             row, col = divmod(i, cols)
#             axes[row, col].plot(env[:, i, 0], label='env')
#             axes[row, col].plot(pred[:, i, 0], label='pred')
#             axes[row, col].plot(ekf_r[:, i, 0], label='ekf')
#             axes[row, col].set_xlabel('time_step')
#             axes[row, col].set_ylabel('theta')
#             axes[row, col].grid()
#             axes[row, col].legend()
#             axes[row, col].set_title('robot theta: ' + str(i//3))

#     # Hide any unused subplots if num_plots < rows * cols
#     for j in range(num_plots, rows * cols):
#         fig.delaxes(axes.flatten()[j])

#     plt.tight_layout()
#     plt.show()
def results(states, ekf):
    env = np.array(states)
    ekf_r = np.array(ekf)

    num_plots = env.shape[1]
    rows, cols = num_plots//3, 3  # Set rows and columns for 2x3 layout
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))  # Adjust figure size as needed
    for i in range(num_plots):
        if(i % 3 == 0):
            row, col = divmod(i, cols)
            axes[row, col].plot(env[:, i, 0], label='env')
            axes[row, col].plot(ekf_r[:, i, 0], label='ekf')
            axes[row, col].set_xlabel('time_step')
            axes[row, col].set_ylabel('X')
            axes[row, col].grid()
            axes[row, col].legend()
            axes[row, col].set_title('robot X: ' + str(i//3))
        if(i % 3 == 1):
            row, col = divmod(i, cols)
            axes[row, col].plot(env[:, i, 0], label='env')
            axes[row, col].plot(ekf_r[:, i, 0], label='ekf')
            axes[row, col].set_xlabel('time_step')
            axes[row, col].set_ylabel('Y')
            axes[row, col].grid()
            axes[row, col].legend()
            axes[row, col].set_title('robot Y: ' + str(i//3))
        if(i % 3 == 2):
            row, col = divmod(i, cols)
            axes[row, col].plot(env[:, i, 0], label='env')
            axes[row, col].plot(ekf_r[:, i, 0], label='ekf')
            axes[row, col].set_xlabel('time_step')
            axes[row, col].set_ylabel('theta')
            axes[row, col].grid()
            axes[row, col].legend()
            axes[row, col].set_title('robot theta: ' + str(i//3))

    # Hide any unused subplots if num_plots < rows * cols
    for j in range(num_plots, rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()

# def main():
#     Env = Environment(n_robots=6)
#     EKF = StateEstimator(Env)
#     # Controller = SeqDec()

#     i = 0
#     Env_readings = []
#     Pred_readings = []
#     EKF_readings = []
#     while i<1000:
        
#         s_t_1 = EKF.s_t_1    
#         # prev_action = Controller.prev_actions() 
#         prev_action = np.array([(0.5, 0.1), (0.5, 0.1),(0.4, 0.1), (0.4, 0.1),(0.4, 0.1), (0.5, 0.1)])
#         # prev_action = np.array([(1.0, 0.1), (1.0, 0.1)])
#         Env.step(prev_action)
#         Env_readings.append(Env.get_states())
        
#         sensor_readings = Env.get_observations()
#         cur_es_state, s_pred = EKF.estimate(sensor_readings, s_t_1, prev_action.flatten())
#         Pred_readings.append(s_pred)
#         EKF_readings.append(cur_es_state)

#         # action_next = Controller.decision(cur_es_state)
#         action_next = np.array([(0.5, 0.1), (0.5, 0.1),(0.4, 0.1), (0.4, 0.1),(0.4, 0.1)])
#         # action_next = np.array([(1.0, 0.1), (1.0, 0.1)])
#         i+=1
#     results(Env_readings, EKF_readings, Pred_readings)

def main():
    environment = Environment(n_robots=5)
    ekf_environment = Environment(n_robots=5)
    ekf_environment.set_states(environment.get_states())
    EKF = StateEstimator(ekf_environment)
    iterations = 1000
    # actions = np.array([(0.0, 0.03), (0.0, 0.02), (0.0, 0.02), (0.0, 0.02), (0.0, 0.02)])
    actions = np.array([(0.04, 0.03), (0.04, 0.02), (0.04, 0.02), (0.011, 0.02), (0.01, 0.02)])

    P = EKF.P_t_1
    true_state = []
    ekf_estimate = []
    for i in range(iterations):
        environment.step(actions)                           # environment moved to t+1
        true_state.append(environment.get_states())
        observation = environment.get_observations()        # gives noisy observation at (t+1) 
        EKF.EKF_env.step(actions)                           # the we now have the prior on t+1
        P = P + EKF.Q                                       # covariance update
        H = EKF.compute_H_t(EKF.EKF_env.get_states())
        m1 = np.linalg.inv(EKF.R + H @ P @ H.T)   
        K = P @ H.T @ m1  
        y_tilda_t = observation.reshape(-1,1) - EKF.h_function(EKF.EKF_env.get_states())   
        EKF.EKF_env.set_states(EKF.EKF_env.get_states() + (K @ y_tilda_t).flatten().reshape(-1,1))
        t1 = (EKF.I - K @ H)
        # self.P_t = t1 @ P_pred_t @ (t1.T) + K_t @ self.R @ K_t.T
        P = t1 @ P
        ekf_estimate.append(EKF.EKF_env.get_states())
    results(true_state, ekf_estimate)
if __name__ == '__main__':
    main()

