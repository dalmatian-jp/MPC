import numpy as np
import sympy as sp
from dynamics.base import Dynamics
import casadi as ca

G = 9.8  # acceleration due to gravity, in m/s^2

class DoubleInvertedPendulumDynamics(Dynamics):
    def __init__(self,L0, L1, L2, l1, l2, M1, M2, I1, I2, c1, c2,phi0, use_linearlized_dynamics):
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
        self.create_casadi_dynamics()


    def create_state_space(self):
        # モデルパラメータ
        m1 = self.M1
        m2 = self.M2
        L1 = self.L1
        L2 = self.L2
        I1 = self.I1
        I2 = self.I2
        g = G

        # 共通の分母
        denominator = (
            16.0 * I1 * I2 + 4.0 * I1 * L2**2 * m2 + 4.0 * I2 * L1**2 * m1 +
            16.0 * I2 * L1**2 * m2 + L1**2 * L2**2 * m1 * m2
        )

        # 行列の各要素を計算
        A = np.zeros((4, 4))

        # 固定値の要素
        A[0, 2] = 1
        A[1, 3] = 1

        # A[2, 0]
        A[2, 0] = (
            g * (
                8.0 * I1 * L2 * m2 - 8.0 * I2 * L1 * m1 +
                16.0 * I2 * L1 * m2 - 2.0 * L1**2 * L2 * m1 * m2 +
                16.0 * L1**2 * L2 * m2**2 - 2.0 * L1 * L2**2 * m1 * m2 +
                8.0 * L1 * L2**2 * m2**2
            )
        ) / denominator

        # A[2, 1]
        A[2, 1] = (
            L2 * g * m2 * (
                8.0 * I1 + 2.0 * L1**2 * m1 +
                8.0 * L1**2 * m2 + 4.0 * L1 * L2 * m2
            )
        ) / denominator

        # A[3, 0]
        A[3, 0] = (
            L1 * g * (
                8.0 * I2 * m1 - 16.0 * I2 * m2 +
                2.0 * L2**2 * m1 * m2 - 8.0 * L2**2 * m2**2
            )
        ) / denominator

        # A[3, 1]
        A[3, 1] = -(
            4.0 * L1 * L2**2 * g * m2**2
        ) / denominator


        B = np.zeros((4, 2))

        B[2, 0] = (
            -16.0 * I2 - 8.0 * L1 * L2 * m2 - 4.0 * L2**2 * m2
        ) / denominator

        B[2, 1] = (
            16.0 * I1 + 16.0 * I2 + 4.0 * L1**2 * m1 +
            16.0 * L1**2 * m2 + 16.0 * L1 * L2 * m2 +
            4.0 * L2**2 * m2
        ) / denominator

        B[3, 0] = (
            16.0 * I2 + 4.0 * L2**2 * m2
        ) / denominator

        B[3, 1] = (
            -16.0 * I2 - 8.0 * L1 * L2 * m2 - 4.0 * L2**2 * m2
        ) / denominator
        
        C = np.eye(4)
        D = np.zeros((4, 2))
        return A, B, C, D
    
    def update_state(self, state, t, u):
        return self.update_state_with_nonlinear_dynamics(state, t, u)
    
    def update_state_with_nonlinear_dynamics(self, state, t, u):
        # 非線形な動的モデルを用いた更新
        I1 = self.I1
        I2 = self.I2
        L1 = self.L1
        L2 = self.L2
        m1 = self.M1
        m2 = self.M2
        g = G
        tau_a = u[0]
        tau_h = u[1]
        theta1, theta2,theta1_dot, theta2_dot = state
        
        # 分母の定義
        denominator = (
            16.0 * I1 * I2 +
            4.0 * I1 * L2**2 * m2 +
            4.0 * I2 * L1**2 * m1 +
            16.0 * I2 * L1**2 * m2 +
            1.0 * L1**2 * L2**2 * m1 * m2 +
            4.0 * L1**2 * L2**2 * m2**2 * np.sin(theta2)**2
        )

        # 分子の定義 (theta1_ddot)
        numerator_theta1 = (
            16.0 * I2 * L1 * L2 * m2 * theta1_dot * theta2_dot * np.sin(theta2) +
            8.0 * I2 * L1 * L2 * m2 * theta2_dot**2 * np.sin(theta2) +
            8.0 * I2 * L1 * g * m1 * np.sin(theta1) -
            16.0 * I2 * L1 * g * m2 * np.sin(theta1) +
            8.0 * I2 * L1 * theta1_dot**2 * np.sin(theta2) +
            16.0 * I2 * tau_a -
            16.0 * I2 * tau_h +
            2.0 * L1**2 * L2 * m2 * theta1_dot**2 * np.sin(2 * theta2) +
            4.0 * L1 * L2**3 * m2**2 * theta1_dot * theta2_dot * np.sin(theta2) +
            2.0 * L1 * L2**3 * m2**2 * theta2_dot**2 * np.sin(theta2) +
            2.0 * L1 * L2**2 * g * m1 * m2 * np.sin(theta1) -
            6.0 * L1 * L2**2 * g * m2**2 * np.sin(theta1) -
            2.0 * L1 * L2**2 * g * m2**2 * np.sin(theta1 + 2 * theta2) +
            2.0 * L1 * L2**2 * m2 * theta1_dot**2 * np.sin(theta2) -
            8.0 * L1 * L2 * m2 * tau_h * np.cos(theta2) +
            4.0 * L2**2 * m2 * tau_a -
            4.0 * L2**2 * m2 * tau_h
        )

        # 分子の定義 (theta2_ddot)
        numerator_theta2 = (
            -8.0 * I1 * L1 * theta1_dot**2 * np.sin(theta2) +
            8.0 * I1 * L2 * g * m2 * np.sin(theta1 + theta2) +
            16.0 * I1 * tau_h -
            16.0 * I2 * L1 * L2 * m2 * theta1_dot * theta2_dot * np.sin(theta2) -
            8.0 * I2 * L1 * L2 * m2 * theta2_dot**2 * np.sin(theta2) -
            8.0 * I2 * L1 * g * m1 * np.sin(theta1) +
            16.0 * I2 * L1 * g * m2 * np.sin(theta1) -
            -8.0 * I2 * L1 * theta1_dot**2 * np.sin(theta2) -
            16.0 * I2 * tau_a +
            16.0 * I2 * tau_h -
            2.0 * L1**3 * m1 * theta1_dot**2 * np.sin(theta2) -
            8.0 * L1**3 * m2 * theta1_dot**2 * np.sin(theta2) -
            4.0 * L1**2 * L2**2 * m2**2 * theta1_dot * theta2_dot * np.sin(2 * theta2) -
            2.0 * L1**2 * L2**2 * m2**2 * theta2_dot**2 * np.sin(2 * theta2) -
            2.0 * L1**2 * L2 * g * m1 * m2 * np.sin(theta1 - theta2) +
            4.0 * L1**2 * L2 * g * m2**2 * np.sin(theta1 - theta2) +
            12.0 * L1**2 * L2 * g * m2**2 * np.sin(theta1 + theta2) -
            4.0 * L1**2 * L2 * m2 * theta1_dot**2 * np.sin(2 * theta2) +
            4.0 * L1**2 * m1 * tau_h +
            16.0 * L1**2 * m2 * tau_h -
            4.0 * L1 * L2**3 * m2**2 * theta1_dot * theta2_dot * np.sin(theta2) -
            2.0 * L1 * L2**3 * m2**2 * theta2_dot**2 * np.sin(theta2) -
            2.0 * L1 * L2**2 * g * m1 * m2 * np.sin(theta1) +
            6.0 * L1 * L2**2 * g * m2**2 * np.sin(theta1) +
            2.0 * L1 * L2**2 * g * m2**2 * np.sin(theta1 + 2 * theta2) -
            2.0 * L1 * L2**2 * m2 * theta1_dot**2 * np.sin(theta2) -
            8.0 * L1 * L2 * m2 * tau_a * np.cos(theta2) +
            16.0 * L1 * L2 * m2 * tau_h * np.cos(theta2) -
            4.0 * L2**2 * m2 * tau_a +
            4.0 * L2**2 * m2 * tau_h
        )

        # theta1_ddot の定義
        theta1_ddot = numerator_theta1 / denominator
        # theta2_ddot の定義
        theta2_ddot = numerator_theta2 / denominator

        dxdt = np.zeros_like(state)
        dxdt[0] = theta1_dot
        dxdt[1] = theta2_dot
        dxdt[2] = theta1_ddot
        dxdt[3] = theta2_ddot
        return dxdt

    def create_casadi_dynamics(self):
        """
        CasADi形式での非線形動的モデル (f(x, u)) の構築
        """
        # CasADiシンボリック変数の定義
        theta1 = ca.MX.sym('theta1')
        theta2 = ca.MX.sym('theta2')
        theta1_dot = ca.MX.sym('theta1_dot')
        theta2_dot = ca.MX.sym('theta2_dot')
        tau_a = ca.MX.sym('tau_a')
        tau_h = ca.MX.sym('tau_h')
        state = ca.vertcat(theta1, theta2, theta1_dot, theta2_dot)
        u = ca.vertcat(tau_a, tau_h)

        # モデルパラメータ
        I1, I2 = self.I1, self.I2
        L1, L2 = self.L1, self.L2
        m1, m2 = self.M1, self.M2
        g = G

        # 分母
        denominator = (
            16.0 * I1 * I2 +
            4.0 * I1 * L2**2 * m2 +
            4.0 * I2 * L1**2 * m1 +
            16.0 * I2 * L1**2 * m2 +
            1.0 * L1**2 * L2**2 * m1 * m2 +
            4.0 * L1**2 * L2**2 * m2**2 * ca.sin(theta2)**2
        )

        # 分子 (theta1_ddot)
        numerator_theta1 = (
            16.0 * I2 * L1 * L2 * m2 * theta1_dot * theta2_dot * ca.sin(theta2) +
            8.0 * I2 * L1 * L2 * m2 * theta2_dot**2 * ca.sin(theta2) +
            8.0 * I2 * L1 * g * m1 * ca.sin(theta1) -
            16.0 * I2 * L1 * g * m2 * ca.sin(theta1) +
            8.0 * I2 * L1 * theta1_dot**2 * ca.sin(theta2) +
            16.0 * I2 * tau_a -
            16.0 * I2 * tau_h +
            2.0 * L1**2 * L2 * m2 * theta1_dot**2 * ca.sin(2 * theta2) +
            4.0 * L1 * L2**3 * m2**2 * theta1_dot * theta2_dot * ca.sin(theta2) +
            2.0 * L1 * L2**3 * m2**2 * theta2_dot**2 * ca.sin(theta2) +
            2.0 * L1 * L2**2 * g * m1 * m2 * ca.sin(theta1) -
            6.0 * L1 * L2**2 * g * m2**2 * ca.sin(theta1) -
            2.0 * L1 * L2**2 * g * m2**2 * ca.sin(theta1 + 2 * theta2) +
            2.0 * L1 * L2**2 * m2 * theta1_dot**2 * ca.sin(theta2) -
            8.0 * L1 * L2 * m2 * tau_h * ca.cos(theta2) +
            4.0 * L2**2 * m2 * tau_a -
            4.0 * L2**2 * m2 * tau_h
        )

        # 分子 (theta2_ddot)
        numerator_theta2 = (
            -8.0 * I1 * L1 * theta1_dot**2 * ca.sin(theta2) +
            8.0 * I1 * L2 * g * m2 * ca.sin(theta1 + theta2) +
            16.0 * I1 * tau_h -
            16.0 * I2 * L1 * L2 * m2 * theta1_dot * theta2_dot * ca.sin(theta2) -
            8.0 * I2 * L1 * L2 * m2 * theta2_dot**2 * ca.sin(theta2) -
            8.0 * I2 * L1 * g * m1 * ca.sin(theta1) +
            16.0 * I2 * L1 * g * m2 * ca.sin(theta1) -
            -8.0 * I2 * L1 * theta1_dot**2 * ca.sin(theta2) -
            16.0 * I2 * tau_a +
            16.0 * I2 * tau_h -
            2.0 * L1**3 * m1 * theta1_dot**2 * ca.sin(theta2) -
            8.0 * L1**3 * m2 * theta1_dot**2 * ca.sin(theta2) -
            4.0 * L1**2 * L2**2 * m2**2 * theta1_dot * theta2_dot * ca.sin(2 * theta2) -
            2.0 * L1**2 * L2**2 * m2**2 * theta2_dot**2 * ca.sin(2 * theta2) -
            2.0 * L1**2 * L2 * g * m1 * m2 * ca.sin(theta1 - theta2) +
            4.0 * L1**2 * L2 * g * m2**2 * ca.sin(theta1 - theta2) +
            12.0 * L1**2 * L2 * g * m2**2 * ca.sin(theta1 + theta2) -
            4.0 * L1**2 * L2 * m2 * theta1_dot**2 * ca.sin(2 * theta2) +
            4.0 * L1**2 * m1 * tau_h +
            16.0 * L1**2 * m2 * tau_h -
            4.0 * L1 * L2**3 * m2**2 * theta1_dot * theta2_dot * ca.sin(theta2) -
            2.0 * L1 * L2**3 * m2**2 * theta2_dot**2 * ca.sin(theta2) -
            2.0 * L1 * L2**2 * g * m1 * m2 * ca.sin(theta1) +
            6.0 * L1 * L2**2 * g * m2**2 * ca.sin(theta1) +
            2.0 * L1 * L2**2 * g * m2**2 * ca.sin(theta1 + 2 * theta2) -
            2.0 * L1 * L2**2 * m2 * theta1_dot**2 * ca.sin(theta2) -
            8.0 * L1 * L2 * m2 * tau_a * ca.cos(theta2) +
            16.0 * L1 * L2 * m2 * tau_h * ca.cos(theta2) -
            4.0 * L2**2 * m2 * tau_a +
            4.0 * L2**2 * m2 * tau_h
        )

        # ddot の計算
        theta1_ddot = numerator_theta1 / denominator
        theta2_ddot = numerator_theta2 / denominator

        dxdt = ca.vertcat(theta1_dot, theta2_dot, theta1_ddot, theta2_ddot)

        # CasADi関数として登録
        self.f_func = ca.Function('f_func', [state, u], [dxdt])


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

    # 状態遷移関数 (f)
    def state_transition_function(self, state, u):
        return self.update_state_with_nonlinear_dynamics(state, 0, u)

    # 観測関数 (h)
    def observation_function(self, state):
        return state  # 観測関数としては直接的に状態を返す

    # 状態遷移関数のヤコビアン (F_jacobian)
    def state_transition_jacobian(self, state, u):
        """
        数値計算による状態遷移ヤコビアンの計算
        :param state: 状態ベクトル [theta1, theta2, theta1_dot, theta2_dot]
        :param u: 入力ベクトル [tau_a, tau_h]
        :return: ヤコビアン行列 (4x4)
        """
        # モデルパラメータ
        m1 = self.M1
        m2 = self.M2
        L1 = self.L1
        L2 = self.L2
        I1 = self.I1
        I2 = self.I2
        g = G
        # 共通の分母
        denominator = (
            16 * I1 * I2 + 4 * I1 * L2**2 * m2 + 4 * I2 * L1**2 * m1 +
            16 * I2 * L1**2 * m2 + L1**2 * L2**2 * m1 * m2
        )

        # 行列の各要素を定義
        A = np.zeros((4, 4))

        A[0, 2] = 1
        A[1, 3] = 1

        A[3, 0] = (
            g * (8.0 * I1 * L2 * m2 - 8.0 * I2 * L1 * m1 + 16.0 * I2 * L1 * m2 - 
                2.0 * L1**2 * L2 * m1 * m2 + 16.0 * L1**2 * L2 * m2**2 - 
                2.0 * L1 * L2**2 * m1 * m2 + 8.0 * L1 * L2**2 * m2**2) /
            denominator
        )

        A[3, 1] = (
            L2 * g * m2 * (8.0 * I1 + 2.0 * L1**2 * m1 + 8.0 * L1**2 * m2 + 4.0 * L1 * L2 * m2) /
            denominator
        )

        A[3, 2] = 0
        A[3, 3] = 0

        A[2, 0] = (
            L1 * g * (8.0 * I2 * m1 - 16.0 * I2 * m2 + 2.0 * L2**2 * m1 * m2 - 
                    8.0 * L2**2 * m2**2) /
            denominator
        )

        A[2, 1] = -(
            4.0 * L1 * L2**2 * g * m2**2 /
            denominator
        )

        A[2, 2] = 0
        A[2, 3] = 0

        return A



    # 観測関数のヤコビアン (H_jacobian)
    def observation_jacobian(self, state):
        # 観測関数としては直接的に状態を返すので、単位行列を返す
        return np.eye(4)
    
    # 数値微分によるヤコビアンの計算
    def numerical_jacobian(self, f, x, u, delta=1e-5):
        """
        数値微分によりヤコビアンを計算する関数
        :param f: 関数
        :param x: 入力ベクトル
        :param u: 制御入力ベクトル
        :param delta: 数値微分のための微小変化量
        :return: 数値微分によるヤコビアン
        """
        n = len(x)
        m = len(u)
        F = np.zeros((n, n))
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += delta
            x_minus[i] -= delta
            F[:, i] = (f(x_plus, u) - f(x_minus, u)) / (2 * delta)
        return F
    
    def verify_state_transition_jacobian(self, state, u, delta=1e-5):
            """
            状態遷移ヤコビアンを数値的に計算した結果と比較するための関数
            :param state: 状態ベクトル
            :param u: 制御入力ベクトル
            :param delta: 数値微分のための微小変化量
            :return: 数値ヤコビアンと解析ヤコビアンの差
            """
            # 状態遷移関数をラップする
            f = lambda x, u: self.update_state_with_nonlinear_dynamics(x, 0, u)
            
            # 数値ヤコビアンの計算
            F_numerical = self.numerical_jacobian(f, state, u, delta)
            
            # 解析的に計算されたヤコビアンの取得
            F_analytical = self.state_transition_jacobian(state, u)
            
            # 数値ヤコビアンと解析ヤコビアンの違いを出力
            print("Numerical Jacobian:")
            print(F_numerical)
            print("\nAnalytical Jacobian:")
            print(F_analytical)
            
            difference = np.abs(F_numerical - F_analytical)
            print("\nDifference:")
            print(difference)
            
            return difference

