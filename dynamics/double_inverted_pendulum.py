import numpy as np

from dynamics.base import Dynamics

G = 9.8  # acceleration due to gravity, in m/s^2

class DoubleInvertedPendulumDynamics(Dynamics):
    def __init__(self, L1, L2, l1, l2, M1, M2, I1, I2, c1, c2, use_linearlized_dynamics):
        self.L1 = L1
        self.L2 = L2
        self.l1 = l1
        self.l2 = l2
        self.M1 = M1
        self.M2 = M2
        self.I1 = I1
        self.I2 = I2
        self.c1 = c1
        self.c2 = c2
        self.alpha1 = I1 + M1 * l1**2 + M2 * L1**2
        self.alpha2 = I2 + M2 * l2**2
        self.alpha3 = M2 * L1 * l2
        self.alpha4 = (M1 * l1 + M2 * L1) * G
        self.alpha5 = M2 * l2 * G

        self.params = np.array(
            [
                self.alpha1,
                self.alpha2,
                self.alpha3,
                self.alpha4,
                self.alpha5,
                c1,
                c2,
            ]
        )

        self.use_linearlized_dynamics = use_linearlized_dynamics
        self.A, self.B, self.C, self.D = self.create_state_space()

    def create_state_space(self):
        denominator = self.alpha1 * self.alpha2 - self.alpha3**2

        A10 = self.alpha2 * self.alpha4 / denominator
        A11 = -(self.alpha2 * self.c1 + self.alpha2 * self.c2 + self.alpha3 * self.c2) / denominator
        A12 = -self.alpha3 * self.alpha5 / denominator
        A13 = self.c2 * (self.alpha2 + self.alpha3) / denominator

        A30 = -self.alpha3 * self.alpha4 / denominator
        A31 = (self.alpha1 * self.c2 + self.alpha3 * self.c1 + self.alpha3 * self.c2) / denominator
        A32 = self.alpha1 * self.alpha5 / denominator
        A33 = -self.c2 * (self.alpha1 + self.alpha3) / denominator

        B00 = self.alpha2 / denominator
        B30 = -self.alpha3 / denominator

        A = np.zeros((4, 4))
        A[0, 1] = 1
        A[2, 3] = 1
        A[1, 0] = A10
        A[1, 1] = A11
        A[1, 2] = A12
        A[1, 3] = A13
        A[3, 0] = A30
        A[3, 1] = A31
        A[3, 2] = A32
        A[3, 3] = A33

        B = np.zeros((4, 1))
        B[1, 0] = B00
        B[3, 0] = B30

        C = np.eye(4)
        D = np.zeros((4, 1))
        return A, B, C, D

    def update_state(self, state, t, u):
        return (
            self.update_state_with_liniarized_dynamics(state, t, u)
            if self.use_linearlized_dynamics
            else self.update_state_with_nonlinear_dynamics(state, t, u)
        )

    def update_state_with_liniarized_dynamics(self, state, t, u):
        state_dot = self.A @ state + self.B @ u
        return state_dot

    def update_state_with_nonlinear_dynamics(self, state, t, u):
        
        theta1, theta1_dot, theta2, theta2_dot = state

        theta12 = theta1 - theta2
        cos_theta12 = np.cos(theta12)
        sin_theta12 = np.sin(theta12)
        denominator = self.alpha1 * self.alpha2 - self.alpha3**2 * cos_theta12**2

        theta1_ddot = (
            -self.alpha2 * self.alpha3 * sin_theta12 * theta2_dot**2
            + self.alpha2 * self.alpha4 * np.sin(theta1)
            - self.alpha2 * self.c1 * theta1_dot
            - self.alpha2 * self.c2 * theta1_dot
            + self.alpha2 * self.c2 * theta2_dot
            + self.alpha2 * u
            - self.alpha3**2 * sin_theta12 * cos_theta12 * theta1_dot**2
            - self.alpha3 * self.alpha5 * np.sin(theta2) * cos_theta12
            - self.alpha3 * self.c2 * cos_theta12 * theta1_dot
            + self.alpha3 * self.c2 * cos_theta12 * theta2_dot
        ) / denominator

        theta2_ddot = (
            self.alpha1 * self.alpha3 * sin_theta12 * theta1_dot**2
            + self.alpha1 * self.alpha5 * np.sin(theta2)
            + self.alpha1 * self.c2 * theta1_dot
            - self.alpha1 * self.c2 * theta2_dot
            + self.alpha3**2 * sin_theta12 * cos_theta12 * theta2_dot**2
            - self.alpha3 * self.alpha4 * np.sin(theta1) * cos_theta12
            + self.alpha3 * self.c1 * cos_theta12 * theta1_dot
            + self.alpha3 * self.c2 * cos_theta12 * theta1_dot
            - self.alpha3 * self.c2 * cos_theta12 * theta2_dot
            - self.alpha3 * u * cos_theta12
        ) / denominator

        dxdt = np.zeros_like(state)
        dxdt[0] = theta1_dot
        dxdt[1] = theta1_ddot
        dxdt[2] = theta2_dot
        dxdt[3] = theta2_ddot
        return dxdt
    
    def calculate_com(self, state):
    # 状態変数: theta1（リンク1の角度）, theta2（リンク2の角度）
        theta1 = state[0]
        theta2 = state[2]

        # リンク1の質量中心の位置
        x_com1 = self.l1 * np.sin(theta1)
        y_com1 = self.l1 * np.cos(theta1)

        # リンク2の質量中心の位置
        x_com2 = self.L1 * np.sin(theta1) + self.l2 * np.sin(theta2)
        y_com2 = self.L1 * np.cos(theta1) + self.l2 * np.cos(theta2)

        # システム全体の重心位置 (COM)
        x_com = (self.M1 * x_com1 + self.M2 * x_com2) / (self.M1 + self.M2)
        y_com = (self.M1 * y_com1 + self.M2 * y_com2) / (self.M1 + self.M2)

        return x_com, y_com

        
    def calculate_cop(self, vertical_force, ankle_torque):
        # COPの計算（鉛直反力と足首トルクを基に計算）
        if vertical_force == 0:
            # 鉛直力がゼロの場合、COPは計算できないので無効値を返す
            return float('nan')
        cop_x = ankle_torque / vertical_force  # x方向のCOPのみを計算
        return cop_x
    
    def calculate_grf(self, state, state_dot):
        # COMの位置を取得
        x_com, y_com = self.calculate_com(state)

        # COMの加速度を計算（速度の変化率）
        y_com_dot = -self.L1 * state_dot[1] * np.sin(state[0]) - self.l2 * state_dot[3] * np.sin(state[2])

        # COMの加速度に基づき地面反力を計算
        Fy = (self.M1 + self.M2) * (y_com_dot + G)  # 鉛直方向のGRF (重力を含む)

        return Fy