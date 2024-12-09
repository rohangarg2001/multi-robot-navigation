import numpy as np
import pandas as pd
import math
from environment import Environment, Robot

class StateEstimator():
    def __init__(self, Env:Environment):
        '''
        Use following vales from environment file
        '''
        self.n_robots = Env.n_robots               # N : Number of rovers
        self.std_dev_uwb = Env.n_robots            # UWB covariance
        self.std_compass = Env.std_compass         # Compass Covariance
        self.sigma_dyn = Env.sigma_dyn             # Dynamics Covariance
        self.deltaT = Env.deltaT

        self.I = np.eye(3*self.n_robots)           # [3N, 3N]
        self.R = self.noise_covar_()               # [N^2, N^2]
        self.Q = (1e-02)*(self.sigma_dyn**2) * np.eye(3*self.n_robots)
        self.P_t_1 = (1e-04)*np.eye(3*self.n_robots)        # [3N, 3N] t=0
        self.s_t_1 = Env.get_states()               # [3N, 1] t=0

    def noise_covar_(self):
        R = np.eye(self.n_robots*self.n_robots)
        '''
        first N(N-1) elements uwb and last N elemets compass heading angle
        '''
        for i in range(R.shape[0]):
            if i < self.n_robots*(self.n_robots -1):
                R[i,i] = self.std_dev_uwb**2
            else:
                R[i,i] = self.std_compass**2
        return R


    def compute_F_t(self):
        """
        Compute the state transition Jacobian F_t for n unicyclic robots.
        """
        F_t = np.zeros((3 * self.n_robots, 3 * self.n_robots))  # (3n x 3n) matrix

        for i in range(self.n_robots):
            idx = 3 * i
            
            # Populate the 3x3 block for the i-th robot
            F_t[idx, idx] = 1  # ∂x / ∂x
            F_t[idx, idx + 1] = 0  # ∂x / ∂y
            F_t[idx, idx + 2] = -self.prev_actions[2*i] * np.sin(self.s_t_1[idx+2]) * self.deltaT  # ∂x / ∂theta

            F_t[idx + 1, idx] = 0  # ∂y / ∂x
            F_t[idx + 1, idx + 1] = 1  # ∂y / ∂y
            F_t[idx + 1, idx + 2] = self.prev_actions[2*i] * np.cos(self.s_t_1[idx+2]) * self.deltaT  # ∂y / ∂theta

            F_t[idx + 2, idx] = 0  # ∂theta / ∂x
            F_t[idx + 2, idx + 1] = 0  # ∂theta / ∂y
            F_t[idx + 2, idx + 2] = 1  # ∂theta / ∂theta

        return F_t

    def compute_B_t(self):
        """
        Compute the state transition B_t for n unicyclic robots.
        """
        B_t = np.zeros((3 * self.n_robots, 2 * self.n_robots))  # (3n x 2n) matrix

        for i in range(self.n_robots):
            idx = 3 * i
            idy = 2 * i
            
            # Populate the 3x2 block for the i-th robot
            B_t[idx, idy] = np.cos(self.s_t_1[idx+2]) * self.deltaT
            B_t[idx, idy + 1] = 0 

            B_t[idx + 1, idy] = np.sin(self.s_t_1[idx+2]) * self.deltaT  
            B_t[idx + 1, idy + 1] = 0  

            B_t[idx + 2, idy] = 0  
            B_t[idx + 2, idy + 1] = self.deltaT  
            

        return B_t

    def h_function(self, s_pred_t):
        '''
        Measurement of format(d,d,d....psi,psi,psi...)
        '''
        h_dist = []
        h_heading = np.ones((self.n_robots , 1))

        for i in range(self.n_robots):
            x_i, y_i = s_pred_t[3*i], s_pred_t[3*i+1]
            for j in range(self.n_robots):
                if i!=j:
                    x_j, y_j = s_pred_t[3*j], s_pred_t[3*j+1]
                    distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                    h_dist.append(distance)

        h_dist = np.array(h_dist)
        
        for i in range(self.n_robots):
            h_heading[i] = s_pred_t[3*i+2]
        

        h = np.vstack((h_dist, h_heading))            
        return h

    def compute_H_t(self, s_pred_t):
        """
        Compute the Measurement Jacobian H_t for n unicyclic robots at predicted position
        z_t = [d12,d13,..dNN-1,theta1,..thetaN]
        x_t = [x1,y1,theta1,....]
        """
        H_t_dist = np.zeros((self.n_robots * (self.n_robots-1), 3 * self.n_robots))  # (n^2-n x 3n) matrix
        H_t_head = np.zeros((self.n_robots , 3 * self.n_robots))  # (n x 3n) matrix

        for i in range(self.n_robots):
            H_t_head[i,2+3*i] = 1
        
        for i in range(self.n_robots):
            for j in range(self.n_robots):
                if i!=j:
                    x_i, y_i = s_pred_t[3*i], s_pred_t[3*i+1]
                    x_j, y_j = s_pred_t[3*j], s_pred_t[3*j+1]
                    dij = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                   
                    H_t_dist[(self.n_robots -1) * i + j, 3*i]     = ( x_i - x_j )/dij
                    H_t_dist[(self.n_robots -1) * i + j, 3*i + 1] = ( y_i - y_j )/dij
                    H_t_dist[(self.n_robots -1) * i + j, 3*j]     = ( x_j - x_i )/dij
                    H_t_dist[(self.n_robots -1) * i + j, 3*j + 1] = ( x_j - x_i )/dij


        H_t = np.vstack((H_t_dist, H_t_head))

        return H_t

    def estimate(self, sensor_readings, s_t_1, prev_action):
        self.sensor_readings = sensor_readings                  # [N^2, 1] - [d1,d1,...psi,psi,psi...]
        self.s_t_1 = s_t_1                                      # [3N, 1]  - [x,y,psi,x,y,psi,...]
        self.prev_actions = prev_action                         # [2N, 1]  - [v,w,v,w,...]

        #Predict step
        B_t = self.compute_B_t()
        F_t = self.compute_F_t()
        s_pred_t, P_pred_t = self.predict(B_t, F_t)

        #Update Step
        h_t_s= self.h_function(s_pred_t)
        H_t = self.compute_H_t(s_pred_t)
        self.update(s_pred_t, P_pred_t, h_t_s, H_t)

        return self.s_t, s_pred_t, self.P_t
    
    def predict(self, B_t, F_t):
        '''
        In this way, will P get updated?
        '''
        s_pred_t = self.s_t_1 + np.matmul(B_t, self.prev_actions.reshape(-1,1))     # Dynamics Update
        P_pred_t = F_t @ self.P_t_1 @ F_t.T + self.Q 
        
        return s_pred_t, P_pred_t


    def update(self, s_pred_t, P_pred_t, h_t_s, H_t):
        '''
        After fusing sensor reading and dynamics predictions self.s_t is updated
        Later make the curresnt estimate as prev estimate
        '''
        z_t = (self.sensor_readings).reshape(-1,1)
        y_tilda_t = z_t - h_t_s  #Innovation
        m1 = np.linalg.inv(self.R + H_t @ P_pred_t @ H_t.T)

        K_t = P_pred_t @ H_t.T @ m1 #Kalman Gain matrix
        self.s_t = s_pred_t + K_t @ y_tilda_t

        t1 = (self.I - K_t @ H_t)
        self.P_t = t1 @ P_pred_t @ (t1.T) + K_t @ self.R @ K_t.T
        
  
        ####################################################################
        #incrementing time by 1 step
        self.s_t_1 = self.s_t        
        self.P_t_1 = self.P_t  
        ####################################################################

    



