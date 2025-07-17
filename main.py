import numpy as np
import matplotlib.pyplot as plt

# F - State transition matrix
# B - Control input matrix (optional)
# H - Observation matrix
# Q - Process noise covariance matrix
# R - Measurement noise covariance matrix
# P - Initial estimate error covariance matrix
# x0 - Initial state estimate vector.

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Missing State Transition Matrix F and Observation Matrix H")

        self.n = F.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def run_kf(ax, title, Q = None):
    dt = 1.0 / 60

    state_transition_matrix = np.array([
        [1, dt, 0],
        [0, 1, dt],
        [0, 0, 1]
    ])
    observation_matrix = np.array([[1, 0, 0]])

    if Q is None:
        process_noise_covariance_matrix = np.array([
            [0.05, 0.05, 0.0],
            [0.05, 0.05, 0.0],
            [0.0,  0.0,  0.0]
        ])
    else:
        process_noise_covariance_matrix = Q

    measurement_noise_covariance_matrix = np.array([[0.5]])

    x = np.linspace(-10, 10, 100)
    noisy_observations = - (x**3 + 3*x**2 + 2*x - 2)  + np.random.normal(0, 30, 100)

    kf = KalmanFilter(F = state_transition_matrix, H = observation_matrix, Q = process_noise_covariance_matrix, R = measurement_noise_covariance_matrix)
    
    filtered_results = []

    for observation in noisy_observations:
        filtered_results.append(np.dot(observation_matrix,  kf.predict())[0])
        kf.update(observation)
    
    ax.plot(noisy_observations, label='Noisy Observations')
    ax.plot(np.array(filtered_results), label='Kalman Filter Filtered Results')
    ax.set_title(title)
    ax.legend()

if __name__ == '__main__':
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    run_kf(axes[0], "Optimal KF")

    new_Q = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
    run_kf(axes[1], "Larger Value in Q", Q = new_Q)

    plt.tight_layout()
    plt.show()