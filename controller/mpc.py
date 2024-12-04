import casadi as ca
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from controller.base import Controller

G = 9.8 

class NonlinearMPCControllerCasADi(Controller):
    def __init__(self, dynamics, A, B, Q, R, N, dt, horizon_dt, integration_method="rk4"):
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
        self.N = N  # Prediction horizon
        self.A = A
        self.B = B
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.integration_method = integration_method
        self.prev_u = np.zeros(self.nu)
        self.horizon_dt = horizon_dt
        self.dt = dt
        self.last_update_time = None
        self.u = np.zeros(self.nu)

        self.alpha1 = dynamics.alpha1
        self.alpha2 = dynamics.alpha2
        self.alpha3 = dynamics.alpha3
        self.alpha4 = dynamics.alpha4
        self.c1 = dynamics.c1
        self.c2 = dynamics.c2

        # CasADi symbolic variables
        self.state_sym = ca.MX.sym("state", self.nx)
        self.u_sym = ca.MX.sym("u", self.nu)
        self.dxdt_sym = self.f(self.state_sym, 0, self.u_sym)
        self.f_func = ca.Function("f_func", [self.state_sym, self.u_sym], [self.dxdt_sym])

    def predict_state(self, x, u, dt):
        if self.integration_method == "euler":
            return self.euler_step(self.f_func, x, dt, u)
        elif self.integration_method == "rk4":
            return self.runge_kutta_step(self.f_func, x, dt, u)
        else:
            raise ValueError("Invalid integration method. Choose 'euler' or 'rk4'.")

    def euler_step(self, func, x, dt, u):
        return x + dt * func(x, u)

    def runge_kutta_step(self, func, x, dt, u):
        k1 = func(x, u)
        k2 = func(x + 0.5 * dt * k1, u)
        k3 = func(x + 0.5 * dt * k2, u)
        k4 = func(x + dt * k3, u)
        y = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    def cost_function(self, U, x0, ref):
        x = x0
        cost = 0
        for i in range(self.N):
            u = U[i * self.nu : (i + 1) * self.nu]
            x = self.predict_state(x, u, self.horizon_dt)
            cost += ca.mtimes([(x - ref).T, self.Q, (x - ref)]) + ca.mtimes([u.T, self.R, u])
        return cost

    def check_condition_number(self, matrix):
        cond_number = np.linalg.cond(matrix)
        if cond_number > 1e10:
            print(f"Warning: The condition number of the matrix is very high: {cond_number}")
        return cond_number

    def f(self, state, t, u):
        # パラメータの取得
        L1 = self.dynamics.L1
        L2 = self.dynamics.L2
        m1 = self.dynamics.M1
        m2 = self.dynamics.M2
        g = G  # 重力加速度

        # シンボリック変数
        theta1 = state[0]
        theta2 = state[1]
        theta1_dot = state[2]
        theta2_dot = state[3]

        # シンボリック演算
        sin_theta1 = ca.sin(theta1)
        sin_theta2 = ca.sin(theta2)
        sin_theta12 = ca.sin(theta1 + theta2)

        # 質量行列の逆行列
        det_M = self.alpha1 * (self.alpha2 + self.alpha4) - self.alpha3**2
        M_inv = ca.MX.zeros(2, 2)  # CasADiで2x2のゼロ行列を作成
        M_inv[0, 0] = (self.alpha2 + self.alpha4) / det_M
        M_inv[0, 1] = -self.alpha3 / det_M
        M_inv[1, 0] = -self.alpha3 / det_M
        M_inv[1, 1] = self.alpha1 / det_M


        # 動力学項
        f_1 = (
            u[0] + 1/2 * L1 * L2 * m2 * sin_theta2 * theta2_dot**2 -
            L1 * L2 * m2 * sin_theta2 * theta1_dot * theta2_dot +
            1/2 * g * L2 * m2 * sin_theta12 +
            1/2 * g * L1 * m1 * sin_theta1 -
            g * L1 * m2 * sin_theta1
        )

        f_2 = (
            u[1] - 1/2 * L1 * theta1_dot**2 * ca.sin(theta2) +
            1/2 * g * L2 * m2 * sin_theta12
        )

        f = ca.vertcat(f_1, f_2)

        # 角加速度
        ddot_theta = ca.mtimes(M_inv, f)

        # シンボリックな状態微分ベクトルを作成
        dxdt = ca.vertcat(theta1_dot, theta2_dot, ddot_theta[0], ddot_theta[1])

        return dxdt
    


    
    def state_transition_jacobian(self, state, u):
        """
        状態遷移ヤコビアンを計算
        """
        # f_func のシンボリック出力を取得
        f_sym = self.f_func(self.state_sym, self.u_sym)

        # 必要ならフラット化
        f_sym = ca.reshape(f_sym, -1, 1)

        # ヤコビアン計算
        F = ca.jacobian(f_sym, self.state_sym)

        # 数値評価用の関数を作成
        F_func = ca.Function('F_func', [self.state_sym, self.u_sym], [F])

        # 数値として評価し、フラットな NumPy 配列として返す
        F_numeric = np.array(F_func(state, u)).squeeze()

        return F_numeric

    
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
            f = lambda x, u: self.state_transition_jacobian(x, u)
           # デバッグ: f の出力形状を確認
            test_state = ca.MX.sym("test_state", 4)
            test_input = ca.MX.sym("test_input", 2)
            test_dxdt = f(test_state,test_input)
            print(f"dxdt shape: {test_dxdt.shape}")
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


    def control(self, state, desired_state, t):
        if self.last_update_time is None or np.round(t - self.last_update_time, 2) >= self.dt:
            self.last_update_time = t
            x0 = state
            ref = desired_state

            opti = ca.Opti()

            U = opti.variable(self.N * self.nu)  # 制御入力変数
            x = ca.MX(x0)
            cost = 0

            # トルク制限の上限と下限
            t_min_a = -20  # 足首トルクの最小値
            t_max_a = 20   # 足首トルクの最大値
            t_min_h = -40  # 股関節トルクの最小値
            t_max_h = 40   # 股関節トルクの最大値

            # コスト関数とシステムの状態更新
            for i in range(self.N):
                u = U[i * self.nu : (i + 1) * self.nu]
                x = self.predict_state(x, u, self.horizon_dt)
                cost += ca.mtimes([(x - ref).T, self.Q, (x - ref)]) + ca.mtimes([u.T, self.R, u])

                # 制御入力 u の次元が2次元の場合のトルク制約
                opti.subject_to(u[0] >= t_min_a)  # 足首のトルク制約
                opti.subject_to(u[0] <= t_max_a)
                opti.subject_to(u[1] >= t_min_h)  # 股関節のトルク制約
                opti.subject_to(u[1] <= t_max_h)

            opti.minimize(cost)

            opts = {"ipopt.print_level": 0, "print_time": 0}
            opti.solver("ipopt", opts)

            try:
                sol = opti.solve()
                stats = sol.stats()
                if stats["return_status"] == "Solve_Succeeded":
                    self.u = np.array(sol.value(U)[: self.nu]).flatten()
                    self.last_U = sol.value(U)
                    self.prev_u = self.u
                else:
                    print(f"Optimization failed with status: {stats['return_status']}")
                    self.u = self.prev_u
            except RuntimeError as e:
                print(f"Optimization failed: {e}")
                self.u = self.prev_u

        return self.u