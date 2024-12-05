import numpy as np
import sympy as sp
from dynamics.base import Dynamics
import casadi as ca

G = 9.8  # acceleration due to gravity, in m/s^2

class DoubleInvertedPendulumDynamics(Dynamics):
    def __init__(self, state, L0, L1, L2, l1, l2, M1, M2, I1, I2, c1, c2,phi0, use_linearlized_dynamics):
        self.L0 = L0 # リンク0の長さ
        self.L1 = L1 # リンク1の長さ
        self.L2 = L2 # リンク2の長さ
        self.l1 = l1 # リンク1の質量中心までの距離
        self.l2 = l2 # リンク2の質量中心までの距離
        self.M1 = M1 # リンク1の質量
        self.M2 = M2 # リンク2の質量
        self.I1 = I1 # リンク1の慣性モーメント 
        self.I2 = I2 # リンク2の慣性モーメント
        self.c1 = c1 # リンク1の粘性摩擦係数
        self.c2 = c2 # リンク2の粘性摩擦係数
        self.phi0 = phi0 # リンク0の角度
        self.use_linearlized_dynamics = use_linearlized_dynamics
        self.alpha1 = 1/4 * L1**2 * M1 + I1
        self.alpha2 = 1/4 * L2**2 * M2 + I2
        self.alpha3 = 1/2 * L1 * L2 * M2 * np.cos(state[1])
        self.alpha4 = L1**2 * M2


    def create_state_space(self):
        A = np.zeros((4, 4))
        B = np.zeros((4, 2))
        C = np.eye(4)
        D = np.zeros((4, 2))
        return A, B, C, D
    
    def update_state(self, state, t, u):
        return self.update_state_with_nonlinear_dynamics(state, t, u)
    
    def update_state_with_nonlinear_dynamics(self, state, t, u):
        # 非線形な動的モデルを用いた更新
        L1 = self.L1
        L2 = self.L2
        m1 = self.M1
        m2 = self.M2
        g = G
        theta1, theta2,theta1_dot, theta2_dot = state
        
        sin_theta1 = np.sin(theta1)
        sin_theta2 = np.sin(theta2)
        sin_theta12 = np.sin(theta1 + theta2)


        det_M = self.alpha2 * (self.alpha1 + self.alpha4) - self.alpha3**2
        M_inv = np.array([
            [ self.alpha2 / det_M, -(self.alpha2 + self.alpha3) / det_M],
            [-(self.alpha2 + self.alpha3) / det_M, (self.alpha1 + self.alpha2 + 2 * self.alpha3 + self.alpha4) / det_M]
        ])
        
        f_1 = (
            u[0] + 1/2 * L1 * L2 * m2 * sin_theta2 * theta2_dot ** 2 -
            L1 * L2 * m2 * sin_theta2 * theta1_dot * theta2_dot +
            1/2 * g * L2 * m2 * sin_theta12 +
            1/2 * g * L1 * m1 * sin_theta1 -
            g * L1 * m2 * sin_theta1
        )

        f_2 = (
            u[1] - 1/2 * L1 *theta1_dot ** 2 * sin_theta2 +
            1/2 * L2 * m2 * g * sin_theta12
        )

        f = np.array([
            f_1,
            f_2
        ])
        ddot_theta = M_inv @ f

        dxdt = np.zeros_like(state)
        dxdt[0] = theta1_dot
        dxdt[1] = theta2_dot
        dxdt[2:] = ddot_theta  # theta1_ddot, theta2_ddot を一度に更新

        return dxdt

    def calculate_com(self, state):
    # 状態変数: theta1（リンク1の角度）, theta2（リンク2の角度）
        theta1 = state[0]
        theta2 = state[1]

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
        cop = ankle_torque / vertical_force  # x方向のCOPのみを計算
        return cop
    
    def calculate_grf(self, state, state_dot):
        """
        地面反力（GRF）を計算する関数
        :param state: リンクの角度などの状態 [phi0, phi1, phi2]
        :param state_dot: 状態の時間微分（角速度・角加速度） [phi0_dot, phi1_dot, phi2_dot]
        :return: 鉛直方向の地面反力 F_g_v
        """
        # モデルパラメータ
        m1 = self.M1  # リンク1の質量
        m2 = self.M2  # リンク2の質量
        L0 = self.L0  # リンク0（基準）の長さ
        L1 = self.L1  # リンク1の長さ
        L2 = self.L2  # リンク2の長さ
        k1 = self.l1  # 質量中心までの距離（リンク1）
        k2 = self.l2  # 質量中心までの距離（リンク2）

        # 状態変数の分解
        phi0=self.phi0
        phi1 = state[0]  # 各リンクの角度
        phi2 = state[2]
        phi1_dot = state_dot[0]
        phi2_dot = state_dot[2]


        # 鉛直方向の地面反力の計算
        F_g_v = (
            (m1 * G * (L0 * k1) * np.cos(phi1 - phi0)) +
            (m2 * G * (L1 * k2) * np.cos(phi2 + phi1 - phi0)) -
            (m2 * (L2 + k2) * np.cos(phi2 + phi1 - phi0) * (phi2_dot + phi1_dot)) +
            (m1 * k1 * np.cos(phi1 - phi0) * phi1_dot)
        )
        return F_g_v
    
    def calculate_ut(self, state, state_dot):
        """
        u_t（トルク）を計算する関数
        :param phi0: リンク0の角度
        :param phi1: リンク1の角度
        :param phi2: リンク2の角度
        :param phi1_ddot: リンク1の角加速度
        :param phi2_ddot: リンク2の角加速度
        :return: u_t (トルク)
        """
        # モデルパラメータ
        m1 = self.M1  # リンク1の質量
        m2 = self.M2  # リンク2の質量
        k1 = self.l1  # 質量中心までの距離（リンク1）
        k2 = self.l2  # 質量中心までの距離（リンク2）
        L0 = self.L0  # リンク0の長さ
        L1 = self.L1  # リンク1の長さ
        phi0=self.phi0
        phi1 = state[0]  # 各リンクの角度
        phi2 = state[1]
        phi1_ddot = state_dot[2]
        phi2_ddot = state_dot[3]

        # 各項を計算
        term1 = -(m1 * k1**2 + m2 * L1**2 + m2 * k2**2) * phi1_ddot
        term2 = (2 * m2 * L1 * k2 * np.cos(phi2)) * phi1_ddot
        term3 = ((m1 * L0 * k1 + m2 * L0 * L1) * np.cos(phi0) * np.cos(phi1 - phi0)) * phi1_ddot
        term4 = -(m2 * L0 * k2 * np.cos(phi0) * np.cos(phi2 + phi1 - phi0)) * phi1_ddot
        term5 = -(m2 * k2**2 - m2 * L1 * k2 * np.cos(phi2)) * phi2_ddot
        term6 = -(m2 * L0 * k2 * np.cos(phi0) * np.cos(phi2 + phi1 - phi0)) * phi2_ddot

        # u_t を計算
        u_t = term1 + term2 + term3 + term4 + term5 + term6

        return u_t

    def state_transition_function(self, state, u):
        return self.update_state_with_nonlinear_dynamics(state, 0, u)
   
    def observation_function(self, state):
        return state  # 観測関数としては直接的に状態を返す

    def observation_jacobian(self, state):
        # 観測関数としては直接的に状態を返すので、単位行列を返す
        return np.eye(4)
    