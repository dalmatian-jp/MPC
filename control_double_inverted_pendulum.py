import multiprocessing as mp
import time

import control
import numpy as np

from controller.mpc import NonlinearMPCControllerCasADi
from dynamics.double_inverted_pendulum import DoubleInvertedPendulumDynamics
from dynamics.state_space import StateSpace
from estimator.kalman_filter import ExtendedKalmanFilter
from simulator.simulator import Simulator
from visualization.visualization import Visualization, visualize_roa

def run(initial_state, visualize=True):
    L0 = 0.158
    L1 = 0.78
    L2 = 0.73
    phi0 = np.radians(4)

    dt = 0.01
    dead_time = 0.2
    add_measurement_noise = False
    use_quantize = False
    encoder_resolution = 144
    controller_dt = 0.01
    dynamics = DoubleInvertedPendulumDynamics(
        initial_state,
        L0=L0,
        L1=L1,
        L2=L2,
        l1=L1 / 2,
        l2=L2 / 2,
        M1=11.41,
        M2=50.14,
        I1=0.35,
        I2=0.25,
        c1=0,
        c2=0,
        phi0=phi0,
        use_linearlized_dynamics=False,
    )
    desired_state = np.radians([0.0, 0.0, 0.0, 0.0])
    state_space = StateSpace(dynamics)

    Q_mpc = np.diag([5000,8000,7000,8500])
    R_mpc = np.diag([0.02,0.01])
    N = 10
    horizon_dt = 0.01
    controller = NonlinearMPCControllerCasADi(
        dynamics, state_space.A, state_space.B, Q_mpc, R_mpc, N, controller_dt, horizon_dt
    )

    # 状態遷移関数、観測関数、およびヤコビアンを定義
    f = dynamics.state_transition_function
    h = dynamics.observation_function
    F_jacobian = controller.state_transition_jacobian
    H_jacobian = dynamics.observation_jacobian

    # EKFを初期化
    observer = ExtendedKalmanFilter(f, h, F_jacobian, H_jacobian)

    simulator = Simulator(
        state_space,
        controller,
        observer,
        initial_state,
        desired_state,
        simulation_time=10.0,
        dt=dt,
        dead_time=dead_time,
        add_measurement_noise=add_measurement_noise,
        use_quantize=use_quantize,
        encoder_resolution=encoder_resolution,
    )
    start_time = time.time()
    (
        states,
        estimated_states,
        observed_states,
        control_inputs,
        delayed_inputs,
        diff_history,
        success_time,
    ) = simulator.run()
    end_time = time.time()
    processing_time = end_time - start_time

    if visualize:
        visualization = Visualization(
            "NonlinearMPCCasADi",
            states,
            estimated_states,
            observed_states,
            control_inputs,
            delayed_inputs,
            diff_history,
            success_time,
            simulator.time,
            L1,
            L2,
            0.0,  # f1 を削除
            0.0,  # f2 を削除
            initial_state,
            dt,
            controller_dt,
            dead_time,
            add_measurement_noise,
            use_quantize,
            encoder_resolution,
            save_dir="videos",
            save_format="mp4",
        )
        visualization.animate()

    return processing_time, success_time

def main():
    initial_state = np.array([0.0873, 0, 0, 0]) 
    run(initial_state)
   
if __name__ == "__main__":
    main()