import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv

class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q=None, R=None):
        self.f = f  # 非線形状態遷移関数: f(x,t,dt,u)
        self.h = h  # 非線形観測関数: h(x)
        self.F_jacobian = F_jacobian  # 状態遷移関数のヤコビアン: F_jacobian(x,u)
        self.H_jacobian = H_jacobian  # 観測関数のヤコビアン: H_jacobian(x)
        self.Q = Q if Q is not None else np.diag([0.01, 0.01, 0.01, 0.01])  # プロセスノイズ共分散行列
        self.R = R if R is not None else np.diag([0.1, 0.1, 0.01, 0.01])  # 観測ノイズ共分散行列
        self.P = None  # 誤差共分散行列
        self.state_estimate = None  # 状態推定値

    def initialize(self, x0, P0):
        """
        EKFの初期化：初期状態と初期誤差共分散を設定
        """
        self.state_estimate = x0
        self.P = P0

    def predict(self, u, t, dt):
        """
        予測ステップ：状態遷移モデルで次の状態を予測し、共分散行列を更新
        """
        F = self.F_jacobian(self.state_estimate, u)
        self.state_estimate = self.f(self.state_estimate, t, dt, u)
        # print("predicted state_estimate", self.state_estimate)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, y):
        """
        更新ステップ：観測値yを用いて状態と共分散行列を更新
        """
        H = self.H_jacobian(self.state_estimate)
        S = H @ self.P @ H.T + self.R
        Kf = self.P @ H.T @ pinv(S)
        # イノベーション
        innovation = y - self.h(self.state_estimate)
        # print("innovation", innovation)
        # print("S", S)
        self.state_estimate = self.state_estimate + Kf @ innovation
        self.P = (np.eye(len(self.P)) - Kf @ H) @ self.P

    def get_state_estimate(self):
        """
        現在の状態推定値を取得
        """
        return self.state_estimate

    def get_covariance(self):
        """
        現在の誤差共分散行列を取得
        """
        return self.P

    def set_state_and_cov(self, x, P):
        """
        状態と共分散行列を外部からセットする。
        観測遅延を考慮する際、過去の状態に戻るために使用。
        """
        self.state_estimate = x
        self.P = P
