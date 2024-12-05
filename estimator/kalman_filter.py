import numpy as np
from numpy.linalg import inv

import numpy as np
from numpy.linalg import inv

class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q=None, R=None):
        self.f = f  # 非線形状態遷移関数
        self.h = h  # 非線形観測関数
        self.F_jacobian = F_jacobian  # 状態遷移関数のヤコビアン
        self.H_jacobian = H_jacobian  # 観測関数のヤコビアン
        self.Q = Q if Q is not None else np.diag([0.01, 0.01, 0.01, 0.01])  # プロセスノイズ共分散行列
        self.R = R if R is not None else np.diag([0.1, 0.1, 0.01, 0.01])  # 観測ノイズ共分散行列
        self.P = None  # 誤差共分散行列
        self.state_estimate = None  # 状態推定

    def initialize(self, x0, P0):
        self.state_estimate = x0
        self.P = P0

    def predict(self, u):
        F = self.F_jacobian(self.f, self.state_estimate, u)
        self.state_estimate = self.f(self.state_estimate,u)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, y):
        H = self.H_jacobian(self.state_estimate)
        S = H @ self.P @ H.T + self.R
        Kf = self.P @ H.T @ inv(S)
        self.state_estimate += Kf @ (y - self.h(self.state_estimate))
        self.P = (np.eye(len(self.P)) - Kf @ H) @ self.P
        
    def get_state_estimate(self):
        return self.state_estimate