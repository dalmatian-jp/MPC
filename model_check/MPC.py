import casadi as ca
import numpy as np

from controller.base import Controller

G = 9.8 
class NonlinearMPCControllerCasADi(Controller):
    def make_f(self):
        nx = 4
        nu = 2
        L1 = 0.90
        L2 = 0.88
        M1=28.047
        M2=56.093
        I1=9.305
        I2=5.681
        g = 9.8
        r1 = 0.576
        r2 = 0.318
        alpha1 = I1 + I2 + M1 * r1**2 + M2 * (L1**2 + r2**2)
        alpha2 = M2 * L1 * r2
        alpha4 = I2 + M2 * r2**2

        states = ca.SX.sym('states', nx)
        ctrls = ca.SX.sym('ctrls', nu)

        theta1 = states[0]
        theta2 = states[1]
        theta1_dot = states[2]
        theta2_dot = states[3]

        tau_a = ctrls[0]
        tau_h = ctrls[1]

        sin_theta1 = ca.sin(theta1)
        sin_theta2 = ca.sin(theta2)
        cos_theta2 = ca.cos(theta2)
        sin_theta12 = ca.sin(theta1 + theta2)


        det_M = alpha1 *alpha4 - alpha4**2 - alpha2**2 * cos_theta2**2

        # MX型からSX型に変更
        M_inv = ca.SX(2, 2)  # 2x2行列の作成
        M_inv[0, 0] = alpha4 / det_M
        M_inv[0, 1] = -(alpha4 + alpha2*cos_theta2) / det_M
        M_inv[1, 0] = -(alpha4 + alpha2*cos_theta2) / det_M
        M_inv[1, 1] = (alpha1 + 2* alpha2* cos_theta2) / det_M

        f_1 = (
            tau_a+ alpha2 * sin_theta2 * theta2_dot * theta1_dot +
            alpha2 * sin_theta2 * (theta2_dot + theta1_dot) * theta2_dot +
            g * (M1 * r1 + M2 * L1) * sin_theta1 +
            g * M2 * r2 * sin_theta12 
        )
        f_2 = (
            tau_h - alpha2 * sin_theta2 * theta1_dot * theta2_dot +
            g * M2 * r2 * sin_theta12 
        )

        f = ca.vertcat(f_1, f_2)
        f = ca.mtimes(M_inv, f)

        state_dot = ca.vertcat(theta1_dot, theta2_dot, f)

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
        
        # print("in compute_optimal_control-----------------")
        x_init = x_init.full().ravel().tolist()
        # print(f"x_init: {x_init}")

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