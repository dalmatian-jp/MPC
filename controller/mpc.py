import casadi as ca
import numpy as np

from controller.base import Controller

G = 9.8 

class NonlinearMPCControllerCasADi(Controller):
    def __init__(self, dynamics,Q, R, N, dt, horizon_dt, integration_method="rk4"):
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
        self.N = N  # Prediction horizon
        self.nx = 4
        self.nu = 2
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
        self.dxdt_sym = self.f(self.state_sym, self.u_sym)
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
    
    def make_f(self):
        nx = 4
        nu = 2
        L1 = self.dynamics.L1
        L2 = self.dynamics.L2
        M1 = self.dynamics.M1
        M2 = self.dynamics.M2
        I1 = self.dynamics.I1
        I2 = self.dynamics.I2
        g = G
        states = ca.SX.sym('states', nx)
        ctrls = ca.SX.sym('ctrls', nu)

        q_a = states[0]
        q_h = states[1]
        dq_a = states[2]
        dq_h = states[3]

        tau_a = ctrls[0]
        tau_h = ctrls[1]

        sin_q_a = ca.sin(q_a)
        sin_q_h = ca.sin(q_h)
        cos_q_h = ca.cos(q_h)
        sin_q_a_h = ca.sin(q_a + q_h)

        # SX型で計算
        alpha1 = 1/4 * L1**2 * M1 + I1
        alpha2 = 1/4 * L2**2 * M2 + I2
        alpha3 = 1/2 * L1 * L2 * M2 * cos_q_h
        alpha4 = L1**2 * M2

        det_M = alpha2 * (alpha1 + alpha4) - alpha3**2

        # MX型からSX型に変更
        M_inv = ca.SX(2, 2)  # 2x2行列の作成
        M_inv[0, 0] = alpha2 / det_M
        M_inv[0, 1] = -(alpha2 + alpha3) / det_M
        M_inv[1, 0] = -(alpha2 + alpha3) / det_M
        M_inv[1, 1] = (alpha1 + alpha2 + 2 * alpha3 + alpha4) / det_M

        f_1 = (
            tau_a + 1/2 * L1 * L2 * M2 * sin_q_h * dq_h**2 -
            L1 * L2 * M2 * sin_q_h * dq_a * dq_h +
            1/2 * g * L2 * M2 * sin_q_a_h +
            1/2 * g * L1 * M1 * sin_q_a -
            g * L1 * M2 * sin_q_a
        )
        f_2 = (
            tau_h - 1/2 * L1 * dq_a**2 * sin_q_h +
            1/2 * L2 * M2 * g * sin_q_a_h
        )

        f = ca.vertcat(f_1, f_2)
        f = ca.mtimes(M_inv, f)

        state_dot = ca.vertcat(dq_a, dq_h, f)

        return ca.Function('f', [states, ctrls], [state_dot])
    
    
    def make_RK4(self, make_f, dt):
        nx = 4
        nu = 2

        states = ca.SX.sym('states',nx)
        ctrls = ca.SX.sym('ctrls',nu)

        k1 = make_f()(states, ctrls)
        k2 = make_f()(states + dt * 1/2 * k1, ctrls)
        k3 = make_f()(states + dt * 1/2 * k2, ctrls)
        k4 = make_f()(states + dt * k3, ctrls)

        states_next = states + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

        return ca.Function('RK4', [states, ctrls], [states_next],["x","u"],["x_next"])  
    

    def make_integrater(self,make_f, dt=0.01):
        nx = 4
        nu = 2
        states = ca.SX.sym('states',nx)
        ctrls = ca.SX.sym('ctrls',nu)

        f=make_f()
        ode = f(states,ctrls)
        dae = {'x':states,'p':ctrls,'ode':ode}

        I = ca.integrator('I','cvodes',dae,0,dt)
        return I
    
    def compute_stage_cost(self,x,u):
        x_ref = ca.DM([0, 0, 0, 0])
        u_ref = ca.DM([0, 0])
        Q = ca.diag([5000,8000,7000,8500])
        R = ca.diag([0.02,0.01])
        x_diff = x - x_ref
        u_diff = u - u_ref
        cost = (ca.dot(Q@x_diff,x_diff) + ca.dot(R@u_diff,u_diff))/2
        return cost

    def compute_terminal_cost(self,x):
        x_ref = ca.DM([0, 0, 0, 0])
        Q = ca.diag([5000,8000,7000,8500])
        x_diff = x - x_ref
        cost = ca.dot(Q@x_diff,x_diff)/2
        return cost
    
    def make_nlp(self,make_RK4, compute_stage_cost, compute_terminal_cost,make_f):
        nx = 4
        nu = 2
        N = 10
        dt = 0.01

        RK4 = make_RK4(make_f,0.01)
        U = [ca.SX.sym(f'u_{i}',nu) for i in range(N)]
        X = [ca.SX.sym(f'x_{i}',nx) for i in range(N+1)]
        G = []

        J = 0

        for i in range(N):
            J += compute_stage_cost(X[i],U[i])*dt
            eq = X[i+1] - RK4(X[i],U[i])
            G.append(eq)
        J += compute_terminal_cost(X[-1])

        option = {"print_time":False,
                "ipopt":{"max_iter":10,"print_level":0}}
        nlp = {"x":ca.vertcat(*X,*U),"f":J,"g":ca.vertcat(*G)}
        S = ca.nlpsol("S","ipopt",nlp,option)
        return S
    
    def compute_optimal_control(self,S,x_init,x0):
        nx = 4
        nu = 2
        N = 10
        
        print("in compute_optimal_control-----------------")
        x_init = x_init.full().ravel().tolist()
        print(f"x_init: {x_init}")

        # トルク限界（入力制約）
        t_min_a = -20
        t_max_a = 20
        t_min_h = -40
        t_max_h = 40


        # 角度制約
        theta1_min = -0.35
        theta1_max = 0.53
        theta2_min = -0.53
        theta2_max = 0.87
        theta1_dot_min = -ca.inf
        theta1_dot_max = ca.inf
        theta2_dot_min = -ca.inf
        theta2_dot_max = ca.inf

        # 状態変数と制御変数の制約設定
        lbx = x_init + [theta1_min] * N + [theta2_min] * N + [theta1_dot_min] * N + [theta2_dot_min] * N  # 状態変数の下限
        ubx = x_init + [theta1_max] * N + [theta2_max] * N + [theta1_dot_max] * N + [theta2_dot_max] * N  # 状態変数の上限
        lbx += [t_min_a] * N + [t_min_h] * N  # 制御入力の下限（足首と股関節トルク）
        ubx += [t_max_a] * N + [t_max_h] * N  # 制御入力の上限（足首と股関節トルク）


        lbg = [0]*nx*N
        ubg = [0]*nx*N

        res = S(lbx=lbx,ubx=ubx,lbg=lbg,ubg=ubg,x0=x0)

        offset = nx*(N+1)
        x0 = res["x"]
        u_opt = x0[offset:offset+nu]
        print(f"u_opt: {u_opt}")    
        # print(f"type u_opt: {type(u_opt)}")
        return u_opt,x0


    def f(self, state, u):
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
        
        det_M = self.alpha2 * (self.alpha1 + self.alpha4) - self.alpha3**2

        M_inv = ca.MX.zeros(2, 2)  # CasADiで2x2のゼロ行列を作成
        M_inv[0, 0] = self.alpha2 / det_M
        M_inv[0, 1] = -(self.alpha2 + self.alpha3) / det_M
        M_inv[1, 0] = -(self.alpha2 + self.alpha3) / det_M
        M_inv[1, 1] = (self.alpha1 + self.alpha2 + 2 * self.alpha3 + self.alpha4) / det_M


        # 動力学項
        f_1 = (
            u[0] + 1/2 * L1 * L2 * m2 * sin_theta2 * theta2_dot**2 +
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
    
    def create_numeric_f(self, state, u):
        # パラメータの取得
        L1 = self.dynamics.L1
        L2 = self.dynamics.L2
        m1 = self.dynamics.M1
        m2 = self.dynamics.M2
        g = G  # 重力加速度

        # 状態変数
        theta1 = state[0]
        theta2 = state[1]
        theta1_dot = state[2]
        theta2_dot = state[3]

        # 三角関数
        sin_theta1 = np.sin(theta1)
        sin_theta2 = np.sin(theta2)
        sin_theta12 = np.sin(theta1 + theta2)

        # 質量行列の逆行列の要素
        alpha1 = self.alpha1
        alpha2 = self.alpha2
        alpha3 = self.alpha3
        alpha4 = self.alpha4

        det_M = alpha2 * (alpha1 + alpha4) - alpha3**2

        M_inv = np.zeros((2, 2))
        M_inv[0, 0] = alpha2 / det_M
        M_inv[0, 1] = -(alpha2 + alpha3) / det_M
        M_inv[1, 0] = -(alpha2 + alpha3) / det_M
        M_inv[1, 1] = (alpha1 + alpha2 + 2 * alpha3 + alpha4) / det_M

        # 動力学項
        f_1 = (
            u[0] + 1/2 * L1 * L2 * m2 * sin_theta2 * theta2_dot**2 +
            L1 * L2 * m2 * sin_theta2 * theta1_dot * theta2_dot +
            1/2 * g * L2 * m2 * sin_theta12 +
            1/2 * g * L1 * m1 * sin_theta1 -
            g * L1 * m2 * sin_theta1
        )

        f_2 = (
            u[1] - 1/2 * L1 * theta1_dot**2 * sin_theta2 +
            1/2 * g * L2 * m2 * sin_theta12
        )

        f = np.array([f_1, f_2])

        # 角加速度
        ddot_theta = np.dot(M_inv, f)

        # 状態微分ベクトル
        dxdt = np.array([theta1_dot, theta2_dot, ddot_theta[0], ddot_theta[1]])

        return dxdt


    def predict_state_numeric(self, state, u, dt, method="rk4"):
        """
        数値的な方法で状態遷移を予測します。
        
        :param state: 現在の状態 (numpy 配列)
        :param u: 制御入力 (numpy 配列)
        :param dt: タイムステップ (float)
        :param method: 積分方法 ("euler" または "rk4")
        :return: 次の状態 (numpy 配列)
        """
        if method == "euler":
            # オイラー法
            dxdt = self.create_numeric_f(state, u)
            next_state = state + dt * dxdt
        elif method == "rk4":
            # ルンゲクッタ法 (RK4)
            k1 = self.create_numeric_f(state, u)
            k2 = self.create_numeric_f(state + 0.5 * dt * k1, u)
            k3 = self.create_numeric_f(state + 0.5 * dt * k2, u)
            k4 = self.create_numeric_f(state + dt * k3, u)
            next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError("Invalid method. Choose 'euler' or 'rk4'.")
        
        return next_state

    def state_transition_jacobian(self, state, u):
        """
        状態遷移ヤコビアンを計算
        :param state: 現在の状態（数値型）
        :param u: 制御入力（数値型）
        :return: ヤコビアン行列（NumPy配列）
        """
        # シンボリックにヤコビアンを計算
        F_sym = ca.jacobian(self.dxdt_sym, self.state_sym)

        # 数値評価用関数を作成
        F_func = ca.Function('F_func', [self.state_sym, self.u_sym], [F_sym])
        # print("Value before state:", state)
        # print("Value before u:", u)
        # 数値評価を実行 (state, u は数値型)
        F_numeric = F_func(state, u)
        # print("Value after conversion:", F_numeric)
        F_numpy = np.array(F_numeric)
        # print("Value after conversion:", F_numpy)       
        return F_numpy

    
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
            f = lambda x, u: self.create_numeric_f(x, u)
         
            # 数値ヤコビアンの計算
            F_numerical = self.numerical_jacobian(f, state, u, delta)
            
            # 解析的に計算されたヤコビアンの取得
            F_analytical = self.state_transition_jacobian(state, u)
            
            # 数値ヤコビアンと解析ヤコビアンの違いを出力
            # print("Numerical Jacobian:")
            # print(F_numerical)
            print("\nAnalytical Jacobian:")
            print(F_analytical)
            
            difference = np.abs(F_numerical - F_analytical)
            # print("\nDifference:")
            # print(difference)
            
            return difference


    def control(self, state, desired_state, t):
        # 一定のサンプリング時間を経過した場合のみ最適化問題を解く
        if self.last_update_time is None or (t - self.last_update_time) >= self.dt - 1e-9:
            self.last_update_time = t
            x0 = state
            ref = desired_state
            print(f"\n--- Time Step: {t:.3f} ---")
            print("Initial State (x0):", x0)
            print("Desired State (ref):", ref)

            opti = ca.Opti()

            # 制御入力変数 U を定義
            U = opti.variable(self.N * self.nu)
            x = ca.MX(x0)
            cost = 0

            # # 必要に応じて前回の解を初期値として設定（ウォームスタート）
            # if hasattr(self, 'last_U') and self.last_U is not None:
            #     opti.set_initial(U, self.last_U)
            # else:
            #     opti.set_initial(U, np.zeros(self.N * self.nu))

            # トルク限界（入力制約）
            t_min_a = -20
            t_max_a = 20
            t_min_h = -40
            t_max_h = 40

            # 角度制約
            theta1_min = -0.35
            theta1_max = 0.53
            theta2_min = -0.53
            theta2_max = 0.87

            # 予測ホライズン N ステップ分の状態遷移・コスト定義
            for i in range(self.N):
                u_i = U[i * self.nu : (i + 1) * self.nu]  # iステップ目の入力
                # 状態を1ステップ予測
                x = self.predict_state(x, u_i, self.horizon_dt)
                
                # num_x = self.predict_state_numeric(x0, u,self.horizon_dt)
                # print(f"Step {i + 1}/{self.N}: Predicted State (x): {num_x}")

                # コスト関数定義
                cost_step = ca.mtimes([(x - ref).T, self.Q, (x - ref)]) + ca.mtimes([u_i.T, self.R, u_i])
                cost += cost_step
                # 入力制約
                opti.subject_to(u_i[0] >= t_min_a)
                opti.subject_to(u_i[0] <= t_max_a)
                opti.subject_to(u_i[1] >= t_min_h)
                opti.subject_to(u_i[1] <= t_max_h)

                # 状態制約（角度制約）
                opti.subject_to(x[0] >= theta1_min)
                opti.subject_to(x[0] <= theta1_max)
                opti.subject_to(x[1] >= theta2_min)
                opti.subject_to(x[1] <= theta2_max)

            # コスト最小化
            opti.minimize(cost)

            # ソルバ設定（詳細は用途に応じて調整可能）
            opts = {"ipopt.print_level": 0, "print_time": 0}
            opti.solver("ipopt", opts)

            try:
                sol = opti.solve()
                stats = sol.stats()
                if stats["return_status"] == "Solve_Succeeded":
                    # 解が成功した場合、最初の入力を抽出
                    self.u = np.array(sol.value(U)[: self.nu]).flatten()
                    print("U", sol.value(U))
                    # 次回のウォームスタート用に全Uを保存
                    self.last_U = sol.value(U)
                    self.prev_u = self.u
                    print(f"Optimal Input (u): {self.u}")
                else:
                    # 解が失敗した場合は前回の入力を再利用
                    print(f"Solver failed with status: {stats['return_status']}")
                    self.u = self.prev_u
            except RuntimeError as e:
                # ソルバがエラーを投げた場合も前回の入力を再利用
                print(f"Optimization failed: {e}")
                self.u = self.prev_u

        print(f"Final Control Input (u): {self.u}\n")
        return self.u
