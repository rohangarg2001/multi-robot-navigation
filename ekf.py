import numpy as np

# I was here

class MultiRobotEKF:
    def __init__(self, num_robots, initial_states, process_noise_cov, measurement_noise_cov):
        self.num_robots = num_robots
        self.state_dim = 3 * num_robots  # [x, y, theta] for each robot
        self.state = np.array(initial_states).reshape(self.state_dim, 1)
        self.P = np.eye(self.state_dim)  # Initial covariance matrix
        self.Q = process_noise_cov  # Process noise
        self.R = measurement_noise_cov  # Measurement noise

    def predict(self, control_inputs, dt):
        """EKF Prediction step for unicycle model."""
        F = np.eye(self.state_dim)  # State transition Jacobian

        for i in range(self.num_robots):
            idx = i * 3
            x, y, theta = self.state[idx:idx+3].flatten()
            v, omega = control_inputs[i]
            
            # Predict the next state
            self.state[idx] += v * np.cos(theta) * dt
            self.state[idx + 1] += v * np.sin(theta) * dt
            self.state[idx + 2] += omega * dt

            # Jacobian entries for x, y, and theta
            F[idx, idx + 2] = -v * np.sin(theta) * dt
            F[idx + 1, idx + 2] = v * np.cos(theta) * dt

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurements):
        """EKF Update step using UWB and compass measurements."""
        H = np.zeros((len(measurements), self.state_dim))  # Measurement Jacobian
        z = np.array(measurements).reshape(-1, 1)  # Measurement vector
        h = []  # Expected measurement values
        
        # Compass measurements
        for i in range(self.num_robots):
            idx = i * 3 + 2
            H[i, idx] = 1  # Only theta is measured by compass
            h.append(self.state[idx, 0])  # Expected heading

        # UWB inter-ranging distances
        k = self.num_robots
        for i in range(self.num_robots - 1):
            for j in range(i + 1, self.num_robots):
                idx_i, idx_j = i * 3, j * 3
                x_i, y_i = self.state[idx_i:idx_i+2].flatten()
                x_j, y_j = self.state[idx_j:idx_j+2].flatten()

                dist = np.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)
                h.append(dist)

                # Partial derivatives for UWB measurements
                H[k, idx_i] = (x_i - x_j) / dist
                H[k, idx_i + 1] = (y_i - y_j) / dist
                H[k, idx_j] = (x_j - x_i) / dist
                H[k, idx_j + 1] = (y_j - y_i) / dist
                k += 1

        # Kalman Gain
        h = np.array(h).reshape(-1, 1)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state += K @ (z - h)
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def get_state(self):
        return self.state.reshape(self.num_robots, 3)  # Return [x, y, theta] for each robot



if __name__ == '__main__':
    mu = MultiRobotEKF()
