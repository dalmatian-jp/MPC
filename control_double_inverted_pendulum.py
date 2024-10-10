import multiprocessing as mp
import time

import control
import numpy as np

from controller.mpc import NonlinearMPCControllerCasADi
from dynamics.double_inverted_pendulum import DoubleInvertedPendulumDynamics
from dynamics.state_space import StateSpace
from estimator.kalman_filter import KalmanFilter
from simulator.simulator import Simulator
from visualization.visualization import Visualization, visualize_roa

def run_roa(theta1, theta2, progress_list, total_trials):
    initial_state = np.array([theta1, 0, theta2, 0])
    processing_time, success_time = run(initial_state, visualize=False)
    progress_list.append(1)
    progress = len(progress_list)
    print(
        f"NonlinearMPCCasADi {progress}/{total_trials}, processing_time: {processing_time:.2f}, success_time: {success_time}, theta1: {theta1}, theta2: {theta2}"
    )
    return (theta1, theta2), processing_time, success_time

def run(initial_state, visualize=True):
    L1 = 2.0
    L2 = 2.0
    f1 = 0.0
    f2 = 0.0

    dt = 0.02
    dead_time = 0.0
    add_measurement_noise = False
    use_estimates = False
    use_quantize = False
    encoder_resolution = 144

    controller_dt = 0.02

    dynamics = DoubleInvertedPendulumDynamics(
        L1=L1,
        L2=L2,
        l1=1.0,
        l2=1.0,
        M1=1.0,
        M2=1.0,
        I1=1.0 / 3,
        I2=1.0 / 3,
        c1=0.3,
        c2=0.3,
        f1=f1,
        f2=f2,
        use_linearlized_dynamics=False,
    )
    desired_state = np.radians([0.0, 0.0, 0.0, 0.0])
    state_space = StateSpace(dynamics)

    Q_mpc = np.diag([100, 1, 100, 1])
    R_mpc = np.diag([0.01])
    N = 10
    horizon_dt = 0.1
    controller = NonlinearMPCControllerCasADi(
        dynamics, state_space.A, state_space.B, Q_mpc, R_mpc, N, controller_dt, horizon_dt
    )

    use_kalman_filter = True
    if use_kalman_filter:
        Q_kf = np.eye(4) * 1
        R_kf = np.eye(state_space.C.shape[0]) * 0.01
        observer = KalmanFilter(state_space.A, state_space.B, state_space.C, Q_kf, R_kf)
    else:
        desired_poles = np.array([-1 + 1j, -1 - 1j, -1.5 + 0.5j, -1.5 - 0.5j])
        L_do = np.eye(4) * 1
        observer = DisturbanceObserver(state_space.A, state_space.B, state_space.C, L_do)

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
        use_estimates=use_estimates,
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
    friction_compensations = controller.get_friction_compensations()

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
            f1,
            f2,
            initial_state,
            dt,
            controller_dt,
            dead_time,
            add_measurement_noise,
            use_estimates,
            use_quantize,
            encoder_resolution,
            friction_compensations,
            save_dir="videos",
            save_format="mp4",
        )
        visualization.animate()

    return processing_time, success_time

def main():
    run_once = True

    if run_once:
        initial_state = np.radians([180, 0, 180, 0])
        processing_time, success_time = run(initial_state)
    else:
        # ROAを求める
        theta1_range = np.linspace(0, np.pi, 36)
        theta2_range = np.linspace(0, np.pi, 36)
        total_trials = len(theta1_range) * len(theta2_range)

        successful_states = []
        unsuccessful_states = []

        processing_time_list = []
        success_time_list = []

        manager = mp.Manager()
        progress_list = manager.list()

        pool = mp.Pool(mp.cpu_count())

        results = []
        for theta1 in theta1_range:
            for theta2 in theta2_range:
                results.append(
                    pool.apply_async(
                        run_roa,
                        (
                            theta1,
                            theta2,
                            progress_list,
                            total_trials,
                        ),
                    )
                )

        pool.close()
        pool.join()

        for result in results:
            (theta1, theta2), processing_time, success_time = result.get()
            if success_time is not None:
                successful_states.append((theta1, theta2))
                success_time_list.append(success_time)
                processing_time_list.append(processing_time)
            else:
                unsuccessful_states.append((theta1, theta2))
                processing_time_list.append(processing_time)

        visualize_roa(
            "NonlinearMPCCasADi",
            total_trials,
            theta1_range,
            theta2_range,
            successful_states,
            unsuccessful_states,
            processing_time_list,
            success_time_list,
            show_plot=False,
            save_dir="roa_results_5deg",
        )

if __name__ == "__main__":
    main()