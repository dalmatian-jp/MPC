function objectives = paretoObjective_only_MPC(params, msobj, x0, u0, xf,p, pvstate, Ts, Duration, distStd)
    Q = params(1:4);
    R = params(5:6);
    [rmse, totalEnergy] = simulateMPC(Q, R,p, msobj, x0, u0, xf, pvstate, Ts, Duration, distStd);
    objectives = [rmse, totalEnergy];
end
