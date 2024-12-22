import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model_check.visualize import MinimalVisualization
from model_check.EKF import ExtendedKalmanFilter
from model_check.dynamics import dynamics
from model_check.linear_dynamics import DoubleInvertedPendulumDynamics



def normalize_angle(theta):
    return theta % (2 * np.pi)

# ===== テスト用パラメータ =====
dt = 0.01
T = 5.0
time = np.arange(0, T, dt)
x_init = np.array([np.pi, 0.0, 0.0, 0.0]) # 初期状態例
x = x_init.copy()
u = np.zeros(2)
X = [x_init]
L0 = 0.158
L1 = 0.78
L2 = 0.73
phi0 = np.radians(4)

dt = 0.01
dead_time = 0.05
add_measurement_noise = True
use_quantize = False
encoder_resolution = 144
controller_dt = 0.01
dyn  = DoubleInvertedPendulumDynamics(
        L1=1,
        L2=1,
        l1=L1 / 2,
        l2=L2 / 2,
        M1=1,
        M2=1,
        I1=0.35,
        I2=0.25,
        c1=0,
        c2=0,
        f1=0,
        f2=0,
        use_linearlized_dynamics=False,
    )


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
visualiser.animate()

print("段階1完了：モデルのみの挙動確認済み")
