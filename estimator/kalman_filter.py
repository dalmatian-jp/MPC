import numpy as np
from numpy.linalg import inv

import numpy as np
from numpy.linalg import inv

class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q=None, R=None):
        """
        拡張カルマンフィルタの初期化
        :param f: 状態遷移関数 f(x, u)
        :param h: 観測関数 h(x)
        :param F_jacobian: 状態遷移関数のヤコビアン（線形化のための行列）
        :param H_jacobian: 観測関数のヤコビアン（線形化のための行列）
        :param Q: プロセスノイズ共分散行列 (デフォルト: diag(0.01, 0.01, 0.01, 0.01))
        :param R: 観測ノイズ共分散行列 (デフォルト: diag(0.1, 0.1, 0.01, 0.01))
        """
        self.f = f  # 非線形状態遷移関数
        self.h = h  # 非線形観測関数
        self.F_jacobian = F_jacobian  # 状態遷移関数のヤコビアン
        self.H_jacobian = H_jacobian  # 観測関数のヤコビアン
        self.Q = Q if Q is not None else np.diag([0.01, 0.01, 0.01, 0.01])  # プロセスノイズ共分散行列
        self.R = R if R is not None else np.diag([0.1, 0.1, 0.01, 0.01])  # 観測ノイズ共分散行列
        self.P = None  # 誤差共分散行列
        self.state_estimate = None  # 状態推定

    def initialize(self, x0, P0):
        """
        フィルタの初期化
        :param x0: 状態推定の初期値
        :param P0: 誤差共分散の初期値
        """
        self.state_estimate = x0
        self.P = P0

    def predict(self, u):
        """
        予測ステップ
        :param u: 制御入力
        """
        # 状態遷移関数のヤコビアン（線形化）を計算
        F = self.F_jacobian(self.state_estimate, u)
        # 状態推定を更新
        self.state_estimate = self.f(self.state_estimate, u)
        # 誤差共分散を更新
        self.P = F @ self.P @ F.T + self.Q

    def update(self, y):
        """
        更新ステップ
        :param y: 観測値
        """
        # 観測関数のヤコビアンを計算
        H = self.H_jacobian(self.state_estimate)
        # カルマンゲインを計算
        S = H @ self.P @ H.T + self.R
        Kf = self.P @ H.T @ inv(S)
        # 状態推定を更新
        self.state_estimate += Kf @ (y - self.h(self.state_estimate))
        # 誤差共分散を更新
        self.P = (np.eye(len(self.P)) - Kf @ H) @ self.P

    def get_state_estimate(self):
        """
        現在の状態推定を取得
        :return: 状態推定値
        """
        return self.state_estimate

