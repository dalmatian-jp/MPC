function objectives = paretoObjective(params, msobj, x0, u0, xf, pvstate, Ts, Duration, ekf, delay_s, distStd, Qekf, Rekf)
    Q = params(1:4);
    R = params(5:6);
    [rmse, totalEnergy] = simulateMPC_EKF(Q, msobj, x0, u0, xf, pvstate, Ts, Duration, ekf, delay_s, distStd, Qekf, Rekf, R);
    objectives = [rmse, totalEnergy];
end
