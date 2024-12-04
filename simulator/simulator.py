import numpy as np
import pandas as pd
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
        self.dead_time = dead_time
        self.R_obs = np.eye(state_space.C.shape[0]) * 0.001
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
        self.P0 = np.eye(len(initial_state)) * 0.1  # 初期の誤差共分散行列
        self.observer.initialize(initial_state, self.P0)

    def update_and_get_delayed_input(self, t, u):
        self.u_buffer.append((t, u))

        if t < self.dead_time:
            u_delayed = np.zeros_like(u)  # u と同じ次元でゼロを返す
        elif len(self.u_buffer) >= 1:
            u_delayed = self.u_buffer[0][1]
        else:
            u_delayed = np.zeros_like(u)  # バッファが空でも同じ次元でゼロを返す

        # 古い入力を削除
        while len(self.u_buffer) > 0 and self.u_buffer[0][0] < t - self.dead_time:
            self.u_buffer.pop(0)

        return u_delayed
    
    def verify_jacobian(self, state, u):
        """
        状態遷移のヤコビアンの検証を行うメソッド
        :param state: 現在の状態ベクトル
        :param u: 現在の制御入力ベクトル
        """
        difference = self.controller.verify_state_transition_jacobian(state, u)
        max_difference = np.max(difference)
        if max_difference > 1e-4:
            print(f"Warning: Large difference detected in Jacobian calculation! Max difference: {max_difference:.6f}")

    def run(self):
        self.com_history = []  # COMの履歴
        self.cop_history = []  # COPの履歴
        
        previous_state = self.state  # 初期状態としての前の状態
        
    
        for i, ti in enumerate(self.time):
            # EKFを用いた状態推定を使用
            current_state = self.observer.get_state_estimate()

            u = self.controller.control(current_state, self.desired_state, ti)
            u_clip = np.clip(u, -500, 500)
            u_delayed = (
                self.update_and_get_delayed_input(ti, u_clip) if self.dead_time > 0.0 else u_clip
            )
            self.control_inputs.append(u)
            self.delayed_inputs.append(u_delayed)

            # Runge-Kuttaで新しい状態を計算
            next_state = self.runge_kutta_step(
                self.state_space.dynamics.update_state,
                self.state,
                ti,
                self.dt,
                u_delayed,
            )
            
            # state_dotを計算 (時間微分)
            self.state_dot = (next_state - previous_state) / self.dt
            previous_state = next_state

            self.state = self.normalize_state(next_state)
            self.states.append(self.state)

            # COMとCOPを計算
            com_x, com_y = self.state_space.dynamics.calculate_com(self.state)
            self.com_history.append((com_x, com_y))
            
            grf = self.state_space.dynamics.calculate_grf(self.state, self.state_dot)
            ut = self.state_space.dynamics.calculate_ut(self.state, self.state_dot)
            cop = self.state_space.dynamics.calculate_cop(grf, ut)
            self.cop_history.append((cop))

            # 差分履歴
            diff = np.abs(self.state[0] - self.desired_state[0]) + np.abs(
                self.state[2] - self.desired_state[2]
            )
            self.diff_history.append(diff)
            success_length = int(1 / self.dt)
            is_success = False
            if len(self.diff_history) > success_length:
                is_success = np.all(np.array(self.diff_history[-success_length:]) < np.deg2rad(5))
                if is_success and self.success_time is None:
                    self.success_time = ti

            # EKFで予測ステップを行う
            self.observer.predict(u_delayed)

            # 観測値を取得し、ノイズを加える場合の処理
            if self.state_space.C.shape[0] == 4:
                if self.add_measurement_noise:
                    y_k = self.state + np.random.multivariate_normal(
                        np.zeros(self.state.shape), self.R_obs
                    )
                else:
                    y_k = self.state
            elif self.state_space.C.shape[0] == 2:
                if self.add_measurement_noise:
                    y_k = self.state[[0, 2]] + np.random.multivariate_normal(
                        np.zeros(2), self.R_obs
                    )
                else:
                    y_k = self.state[[0, 2]]
            self.observed_states.append(y_k)

            # EKFで更新ステップを行う
            self.observer.update(y_k)
            self.estimated_states.append(self.observer.get_state_estimate())

            # 定期的にヤコビアンの検証を行う（例：100回に1回）
            if i % 100 == 0:
                self.verify_jacobian(current_state, u_delayed)


        # データをCSVに保存
        self.save_to_csv()

        return (
            np.array(self.states),
            np.array(self.estimated_states),
            np.array(self.observed_states),
            np.array(self.control_inputs),
            np.array(self.delayed_inputs),
            np.array(self.diff_history),
            self.success_time,
        )


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
