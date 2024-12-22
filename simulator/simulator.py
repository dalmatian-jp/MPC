import numpy as np
import pandas as pd
import casadi as ca
from numpy.linalg import inv

class Simulator:
    def __init__(
        self,
        state_space,
        controller,
        observer,
        initial_state,
        desired_state,
        simulation_time=10,
        dt=0.01,
        dead_time=0.05,
        add_measurement_noise=False,
        use_quantize=False,
        encoder_resolution=72,
    ):
        self.state_space = state_space
        self.controller = controller
        self.observer = observer
        self.initial_state = initial_state
        self.desired_state = desired_state
        self.dt = dt
        self.observation_delay = dead_time
        self.R_obs = np.eye(state_space.C.shape[0]) * 0.000025
        self.add_measurement_noise = add_measurement_noise
        self.encoder_resolution = encoder_resolution
        self.u_buffer = []
        self.state = initial_state
        self.states = []
        self.estimated_states = []
        self.observed_states = []
        self.control_inputs = []
        self.delayed_inputs = []
        self.time = np.arange(0, simulation_time, dt)
        self.integral_errors = 0
        self.use_quantize = use_quantize
        self.success_time = None
        self.diff_history = []

        # EKFの初期化
        # print("Initial state: ", initial_state)
        self.P0 = np.eye(len(initial_state)) * 0.1  # 初期の誤差共分散行列
        self.observer.initialize(initial_state, self.P0)

    def verify_jacobian(self, state, u):
        difference = self.controller.verify_state_transition_jacobian(state, u)
        max_difference = np.max(difference)
        if max_difference > 1e-4:
            print(f"Warning: Large difference detected in Jacobian calculation! Max difference: {max_difference:.6f}")

    # def run(self):
    #     self.com_history = []
    #     self.cop_history = []
    #     self.observation_buffer = []
    #     previous_state = self.state
    #     steps_delay = int(self.observation_delay / self.dt)
    #     self.ekf_history = []
    #     self.input_history = []
    #     self.observed_states = []
    #     self.estimated_states = []
    #     self.states = []
    #     self.diff_history = []
    #     self.control_inputs = []

    #     # 初期入力は0で開始
    #     u_clip = np.zeros(2)
        
    #     for i, ti in enumerate(self.time):
    #         print("ti: ", ti)
    #         # 1. i>0の場合、前回計算したu_clipで実状態を更新
    #         if i > 0:
    #             next_state = self.runge_kutta_step(
    #                 self.state_space.dynamics.update_state,
    #                 self.state,
    #                 ti,
    #                 self.dt,
    #                 u_clip
    #             )
    #             self.state_dot = (next_state - previous_state) / self.dt
    #             previous_state = next_state
    #             self.state = self.normalize_state(next_state)
    #             self.states.append(self.state)

    #             # COM, COP計算
    #             com_x, com_y = self.state_space.dynamics.calculate_com(self.state)
    #             self.com_history.append((com_x, com_y))
    #             grf = self.state_space.dynamics.calculate_grf(self.state, self.state_dot)
    #             ut = self.state_space.dynamics.calculate_ut(self.state, self.state_dot)
    #             cop = self.state_space.dynamics.calculate_cop(grf, ut)
    #             self.cop_history.append((cop))

    #             # 差分履歴更新
    #             diff = np.abs(self.state[0] - self.desired_state[0]) + np.abs(self.state[2] - self.desired_state[2])
    #             self.diff_history.append(diff)
    #             success_length = int(1 / self.dt)
    #             if len(self.diff_history) > success_length:
    #                 is_success = np.all(np.array(self.diff_history[-success_length:]) < np.deg2rad(5))
    #                 if is_success and self.success_time is None:
    #                     self.success_time = ti
    #         else:
    #             # 初期状態を記録（最初はrunge_kutta_stepしない）
    #             self.states.append(self.state)

    #         # 2. EKFで予測ステップ
    #         self.observer.predict(u_clip, ti, self.dt)

    #         # 3. 観測値生成
    #         if self.add_measurement_noise:
    #             y_k = self.state + np.random.multivariate_normal(
    #                 np.zeros(self.state.shape), self.R_obs
    #             )
    #         else:
    #             y_k = self.state

    #         # 4. 観測値バッファに保存
    #         self.observation_buffer.append((ti, y_k))
    #         while len(self.observation_buffer) > 0 and self.observation_buffer[0][0] < ti - self.observation_delay:
    #             self.observation_buffer.pop(0)

    #         # 5. 遅延観測値取得
    #         delayed_observation = self.get_delayed_observation(ti)
    #         self.observed_states.append(delayed_observation)

    #         # 現在のEKF状態・共分散・入力を記録
    #         x_est_current = self.observer.get_state_estimate()
    #         P_est_current = self.observer.get_covariance()
    #         self.ekf_history.append((ti, x_est_current, P_est_current, u_clip))

    #         delayed_index = i - steps_delay

    #         # 6. 遅延観測が利用可能な場合、過去に戻ってupdate & predict
    #         if delayed_index >= 0 and delayed_observation is not None:
    #             t_past, x_past, P_past, _ = self.ekf_history[delayed_index]
    #             self.observer.set_state_and_cov(x_past, P_past)
    #             self.observer.update(delayed_observation)
    #             # 過去から現在まで再予測
    #             for back_step in range(delayed_index+1, i+1):
    #                 t_back, _, _, u_back = self.ekf_history[back_step]
    #                 # print("t_back: ", t_back)
    #                 self.observer.predict(u_back, t_back, self.dt)

    #         # 7. 更新後、現在の推定状態を取得
    #         current_est = self.observer.get_state_estimate()
    #         print("current_est: ", current_est) 
    #         self.estimated_states.append(current_est)

    #         # 8. 次ステップで使う入力を計算（更新後の最新状態推定に基づく）
    #         if ti < self.observation_delay:
    #             u_new = np.zeros(2)
    #         else:
    #             u_new = self.controller.control(current_est, self.desired_state, ti)
    #             print("u_new: ", u_new)

    #         u_new = np.clip(u_new, -500, 500)
    #         u_clip = u_new
    #         self.control_inputs.append(u_clip)

    #     self.save_to_csv()

    #     return (
    #         np.array(self.states),
    #         np.array(self.estimated_states),
    #         np.array(self.observed_states),
    #         np.array(self.control_inputs),
    #         np.array(self.diff_history),
    #         self.success_time,
    #     )

    def run(self):
        self.com_history = []
        self.cop_history = []
        self.observation_buffer = []
        previous_state = self.state
        steps_delay = int(self.observation_delay / self.dt)
        self.ekf_history = []
        self.input_history = []
        self.observed_states = []
        self.estimated_states = []
        self.states = []
        self.diff_history = []
        self.control_inputs = []
        nu = self.controller.nu
        nx = self.controller.nx
        N = 10
        total = nx*(N+1)+nu*N
        print(self.state)
        x_init = ca.DM(self.state)
        x0 = ca.DM.zeros(total)

        S = self.controller.make_nlp(self.controller.make_RK4, self.controller.compute_stage_cost, self.controller.compute_terminal_cost,self.controller.make_f)
        T = self.controller.make_integrater(self.controller.make_f)

        X = [x_init]
        print("X: ", X)
        U = []
        x_current = x_init
        u_clip = np.zeros(2)

        for i, ti in enumerate(self.time):
            if i > 0:
                next_state = self.runge_kutta_step(
                    self.state_space.dynamics.update_state,
                    self.state,
                    ti,
                    self.dt,
                    u_clip
                )
                self.state_dot = (next_state - previous_state) / self.dt
                previous_state = next_state
                self.state = self.normalize_state(next_state)
                self.states.append(self.state)

                # COM, COP計算
                com_x, com_y = self.state_space.dynamics.calculate_com(self.state)
                self.com_history.append((com_x, com_y))
                grf = self.state_space.dynamics.calculate_grf(self.state, self.state_dot)
                ut = self.state_space.dynamics.calculate_ut(self.state, self.state_dot)
                cop = self.state_space.dynamics.calculate_cop(grf, ut)
                self.cop_history.append((cop))

                # 差分履歴更新
                diff = np.abs(self.state[0] - self.desired_state[0]) + np.abs(self.state[2] - self.desired_state[2])
                self.diff_history.append(diff)
                success_length = int(1 / self.dt)
                if len(self.diff_history) > success_length:
                    is_success = np.all(np.array(self.diff_history[-success_length:]) < np.deg2rad(5))
                    if is_success and self.success_time is None:
                        self.success_time = ti
            else:
                # 初期状態を記録（最初はrunge_kutta_stepしない）
                self.states.append(self.state)

            # 2. EKFで予測ステップ
            self.observer.predict(u_clip, ti, self.dt)

            # 3. 観測値生成
            if self.add_measurement_noise:
                y_k = self.state + np.random.multivariate_normal(
                    np.zeros(self.state.shape), self.R_obs
                )
            else:
                y_k = self.state

            # 4. 観測値バッファに保存
            self.observation_buffer.append((ti, y_k))
            while len(self.observation_buffer) > 0 and self.observation_buffer[0][0] < ti - self.observation_delay:
                self.observation_buffer.pop(0)

            # 5. 遅延観測値取得
            delayed_observation = self.get_delayed_observation(ti)
            self.observed_states.append(delayed_observation)

            # 現在のEKF状態・共分散・入力を記録
            x_est_current = self.observer.get_state_estimate()
            P_est_current = self.observer.get_covariance()
            self.ekf_history.append((ti, x_est_current, P_est_current, u_clip))

            delayed_index = i - steps_delay

            # 6. 遅延観測が利用可能な場合、過去に戻ってupdate & predict
            if delayed_index >= 0 and delayed_observation is not None:
                t_past, x_past, P_past, _ = self.ekf_history[delayed_index]
                self.observer.set_state_and_cov(x_past, P_past)
                self.observer.update(delayed_observation)
                # 過去から現在まで再予測
                for back_step in range(delayed_index, i):
                    t_back, _, _, u_back = self.ekf_history[back_step]
                    # print("t_back: ", t_back)
                    self.observer.predict(u_back, t_back, self.dt)

            # 7. 更新後、現在の推定状態を取得
            current_est = self.observer.get_state_estimate()
            print("time: ", ti)
            self.estimated_states.append(current_est)

            # 8. 次ステップで使う入力を計算（更新後の最新状態推定に基づく）
            if ti < self.observation_delay:
                u_new = np.zeros(2)
            else:
                current_est = ca.DM(current_est)
                u_new,x0 = self.controller.compute_optimal_control(S,current_est,x0)
                u_new = np.array(u_new).reshape(2)

            u_new = np.clip(u_new, -500, 500)
            u_clip = u_new
            self.control_inputs.append(u_clip)

        print(X)
        print(U)


    # 新しく追加した関数: 遅延観測値を取得するためのヘルパー
    def get_delayed_observation(self, current_time):
        delayed_time = current_time - self.observation_delay
        for t, obs in self.observation_buffer:
            if np.isclose(t, delayed_time, atol=1e-3):  # 時刻が一致する観測値を探す
                # print("observed state: ", obs)
                return obs
        return np.zeros_like(self.state)  # データが見つからない場合はゼロを返す



    def save_to_csv(self):
    # シミュレーション結果をデータフレームにまとめる
        data = {
            'time': self.time,
            'theta1': [state[0] for state in self.states],  # theta1 (リンク1の角度)
            'theta2': [state[1] for state in self.states],  # theta1の角速度
            'theta1_dot': [state[2] for state in self.states],  # theta2 (リンク2の角度)
            'theta2_dot': [state[3] for state in self.states],  # theta2の角速度
            'estimated_theta1': [state[0] for state in self.estimated_states],  # 推定されたtheta1
            'estimated_theta2': [state[1] for state in self.estimated_states],  # 推定されたtheta1_dot
            'estimated_theta1_dot': [state[2] for state in self.estimated_states],  # 推定されたtheta2
            'estimated_theta2_dot': [state[3] for state in self.estimated_states],  # 推定されたtheta2_dot
            'observed_theta1': [state[0] for state in self.observed_states],  # 観測されたtheta1
            'observed_theta2': [state[1] for state in self.observed_states],  # 観測されたtheta2
            'control_inputs': [list(u) for u in self.control_inputs],  # 制御入力
            'delayed_inputs': [list(u) for u in self.delayed_inputs],  # 遅延した制御入力
            'diff_history': self.diff_history,  # 差分履歴
            'com_x': [com[0] for com in self.com_history],  # COMのx座標
            'com_y': [com[1] for com in self.com_history],  # COMのy座標
            'cop': self.cop_history, 
        }
        
        # データフレームに変換
        df = pd.DataFrame(data)
        
        # CSVファイルに保存
        df.to_csv('simulation_results.csv', index=False)
        print("シミュレーション結果が 'simulation_results.csv' に保存されました。")

    def normalize_state(self, state):
        state[0] = (state[0] + np.pi) % (2 * np.pi) - np.pi
        state[1] = (state[1] + np.pi) % (2 * np.pi) - np.pi
        return state

    def runge_kutta_step(self, func, x, t, dt, u):
        k1 = func(x, t, u)
        k2 = func(x + 0.5 * dt * k1, t + 0.5 * dt, u)
        k3 = func(x + 0.5 * dt * k2, t + 0.5 * dt, u)
        k4 = func(x + dt * k3, t + dt, u)
        y = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y
