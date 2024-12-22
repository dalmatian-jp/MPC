import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model_check.visualize import MinimalVisualization
from model_check.EKF import ExtendedKalmanFilter
from model_check.dynamics import dynamics
from model_check.MPC import NonlinearMPCControllerCasADi
from model_check.morassodynamics import dynamics as morass_dynamics
import casadi as ca



def normalize_angle(theta):
    return theta % (2 * np.pi)

# ===== テスト用パラメータ =====
dt = 0.01
T = 5
time = np.arange(0, T, dt)
x_init = np.array([0.0873, 0.0, 0, 0]) # 初期状態例
# x_init = np.array([0.2618, 0, 0, 0]) 
x = x_init.copy()
u = np.zeros(2)
X = [x_init]
dyn  = morass_dynamics()
model_dynamics = dyn.model_dynamics
delay = 0.05
delay_steps = int(delay/dt)
observation_buffer = []
# 観測は真値に適当なノイズを加える
R = np.eye(4)* 0.000025
u = np.zeros(2)

#initialize EKF
f = dyn.next_state # (self, x, u,dt):
h = dyn.observation_function # (x):
f_jacobian = dyn.numerical_jacobian # (self, x, u, h=1e-5):
h_jacobian = dyn.observation_jacobian # (x):
obs = ExtendedKalmanFilter(f, h, f_jacobian, h_jacobian)
P0 = np.eye(len(x_init)) * 0.1  # 初期の誤差共分散行列
obs.initialize(x_init, P0)

# initialize MPC
controller =   NonlinearMPCControllerCasADi(dt)
S = controller.make_nlp(controller.make_RK4, controller.compute_stage_cost, controller.compute_terminal_cost,controller.make_f)
T = controller.make_integrater(controller.make_f)

x = x_init.copy()
X_true = [x_init]
X_est = [x_init.copy()]
input_history = []
P_est = [obs.get_covariance()]
X_true = [x_init.copy()]
nx = 4
nu = 2
N = 10
total = nx*(N+1)+nu*N
x_inital = ca.DM(x_init)
x0 = ca.DM.zeros(total)


# for i, t in enumerate(time[1:], start=1):
#     print("time:", t)

#     # 真値状態更新
#     x = dyn.next_state(x, u, dt)
#     X_true.append(x)
    
#     # 観測値生成
#     y = x + np.random.multivariate_normal(np.zeros(4), R)
#     print("y", y)
#     observation_buffer.append((t, y))
#     # print("observation_buffer", observation_buffer)
    
#     # 遅延観測取得
#     delayed_observation = None
#     if i >= delay_steps:
#         delayed_observation = observation_buffer[i - delay_steps][1]
#         # print(i - delay_steps-1)
#         print("delayed_observation", delayed_observation)

#     # EKF予測ステップ（現在入力uで予測）
#     obs.predict(u, t, dt)
    
#     if delayed_observation is not None:
#         # 過去の時点
#         delayed_index = i - delay_steps
        
#         # 過去の状態推定や共分散を復元 (事前に保存しておく必要あり)
#         x_past_est = X_est[delayed_index]
#         P_past_est = P_est[delayed_index]
#         obs.set_state_and_cov(x_past_est, P_past_est)
        
#         # 遅延観測でupdate
#         obs.update(delayed_observation)
        
        
#         # 過去から現在まで再予測
#         for back_step in range(delayed_index, i-1):
#             print("obs", obs.get_state_estimate())
#             t_back = time[back_step]
#             # print("back_step", back_step)
#             # print("u_back", input_history[back_step-1])
#             u_back = input_history[back_step-1]  # input_historyのインデックス調整必要
#             obs.predict(u_back, t_back, dt)

#     # 現在の状態推定を保存
#     X_est.append(obs.get_state_estimate())
#     P_est.append(obs.get_covariance())

#     current_est = obs.get_state_estimate()
    

#     if t < delay:
#         u = np.array([0.0, 0.0])
#     else:
#         current_est = ca.DM(current_est)
#         u, x0  = controller.compute_optimal_control(S,current_est,x0)
#         u = np.array(u).reshape(2)
#     u = np.clip(u, -500, 500)
#     input_history.append(u)
    
# # vis = MinimalVisualization(np.array(X_true), np.array(X_est), time, 0.78, 0.73)
# # vis.animate()

# # 3つのデータフレームを作成
# df_true = pd.DataFrame(X_true, columns=["theta1", "theta2", "theta1_dot", "theta2_dot"])
# df_est = pd.DataFrame(X_est, columns=["theta1_est", "theta2_est", "theta1_dot_est", "theta2_dot_est"])  # 列名を区別
# df_input = pd.DataFrame(input_history, columns=["u1", "u2"])

# # 列方向に結合
# combined_df = pd.concat([df_true, df_est, df_input], axis=1)
# combined_df.to_csv("combined.csv", index=False)

# X_true = np.array(X_true)
# X_est = np.array(X_est)
# plt.figure()
# plt.plot(time, X_true[:,0], label="true theta1")
# plt.plot(time, X_est[:,0], label="estimated theta1 (with delay)")
# plt.legend()
# plt.title("With Observer and Delay")
# plt.show()

# print("段階4完了：遅延観測付きで状態推定の挙動確認済み")



# モデル自由応答チェック
for t in time[1:]:
    x = dyn.next_state(x, u, dt)
    x[0] = normalize_angle(x[0])
    x[1] = normalize_angle(x[1])
    X.append(x)
X = np.array(X)
df = pd.DataFrame(X, columns=["theta1", "theta2", "theta1_dot", "theta2_dot"])
df.to_csv("model_check.csv", index=False)

visualiser = MinimalVisualization(X, X, time, 0.78, 0.73)
visualiser.animate(save_path="model_check.mp4")

print("段階1完了：モデルのみの挙動確認済み")

# # 観測は真値に適当なノイズを加える
# R = np.eye(4)* 0.000025
# u = np.zeros(2)

# #initialize EKF
# f = dyn.next_state # (self, x, u,dt):
# h = dyn.observation_function # (x):
# f_jacobian = dyn.numerical_jacobian # (self, x, u, h=1e-5):
# h_jacobian = dyn.observation_jacobian # (x):
# obs = ExtendedKalmanFilter(f, h, f_jacobian, h_jacobian)

# P0 = np.eye(len(x_init)) * 0.1  # 初期の誤差共分散行列
# obs.initialize(x_init, P0)

# X_true = [x_init]
# X_est = [x_init.copy()]

# P0 = np.eye(len(x_init)) * 0.1  # 初期の誤差共分散行列
# obs.initialize(x_init, P0)

# for t in time[1:]:
#     # 真値更新

#     x = dyn.next_state(x, u, dt)
#     X_true.append(x)
    
#     # 観測値生成
#     y = x + np.random.multivariate_normal(np.zeros(4), R)
    
#     # EKF予測・更新
#     obs.predict(u, t, dt)
#     obs.update(y)
#     X_est.append(obs.get_state_estimate())

# X_true = np.array(X_true)
# X_est = np.array(X_est)
# plt.figure()
# plt.plot(time, X_true[:,0], label="true theta1")
# plt.plot(time, X_est[:,0], label="estimated theta1")
# plt.legend()
# plt.title("With Observer")
# plt.show()

# print("段階2完了：オブザーバ付きで状態推定の挙動確認済み")




# for i, t in enumerate(time[1:], start=1):
#     # 真値状態更新
#     x = dyn.next_state(x, u, dt)
#     X_true.append(x)
#     input_history.append(u)  # 適用した入力を保持しておく
    
#     # 観測値生成
#     y = x + np.random.multivariate_normal(np.zeros(4), R)
#     observation_buffer.append((t, y))
    
#     # 遅延観測取得
#     delayed_observation = None
#     if i > delay_steps:
#         delayed_observation = observation_buffer[i - delay_steps][1]

#     # EKF予測ステップ（現在入力uで予測）
#     obs.predict(u, t, dt)
    
#     # 遅延観測がある場合は過去に戻って状態と共分散を復元しupdate
#     if delayed_observation is not None:
#         # 過去の時点
#         delayed_index = i - delay_steps
        
#         # 過去の状態推定や共分散を復元 (事前に保存しておく必要あり)
#         x_past_est = X_est[delayed_index]
#         P_past_est = P_est[delayed_index]
#         obs.set_state_and_cov(x_past_est, P_past_est)
        
#         # 遅延観測でupdate
#         obs.update(delayed_observation)
        
#         # 過去から現在まで再予測
#         for back_step in range(delayed_index+1, i+1):
#             t_back = time[back_step]
#             u_back = input_history[back_step-1]  # input_historyのインデックス調整必要
#             obs.predict(u_back, t_back, dt)
    
#     # 現在の状態推定を保存
#     X_est.append(obs.get_state_estimate())
#     P_est.append(obs.get_covariance())

    
# vis = MinimalVisualization(np.array(X_true), np.array(X_est), time, 0.78, 0.73)
# vis.animate()

# X_true = np.array(X_true)
# X_est = np.array(X_est)
# plt.figure()
# plt.plot(time, X_true[:,0], label="true theta1")
# plt.plot(time, X_est[:,0], label="estimated theta1 (with delay)")
# plt.legend()
# plt.title("With Observer and Delay")
# plt.show()

# print("段階3完了：遅延観測付きで状態推定の挙動確認済み")
