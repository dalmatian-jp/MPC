function [rmse, totalEnergy] = simulateMPC(Q, R,p, msobj, x0, u0, xf, pvstate, Ts, Duration, distStd)
    pvcost = [xf; Q(:); R(:); p];  % Rとpは固定
    simdata = getSimulationData(msobj);
    simdata.StageParameter = repmat(pvcost, p+1, 1);
    
    [~, onlinedata] = getCodeGenerationData(msobj, x0, u0, ...
        StateFcnParameter = pvstate, StageParameter = simdata.StageParameter);

    Nsteps = Duration / Ts;
    nx = length(x0);
    nmv = length(u0);
    rng default;

    xHistory = zeros(Nsteps + 1, length(x0));
    xHistory(1, :) = x0.';
    uHistory = zeros(Nsteps + 1, length(u0));
    uHistory(1, :) = u0.';
    uk = uHistory(1,:).';

    for k = 1:(Duration/Ts)
           xk = xHistory(k,:).';
           [uk, onlinedata] = nlmpcControllerMEX(xk, uk, onlinedata);
           XOUT = solveODE(xk, uk, pvstate, Ts);
           xHistory(k+1,:) = XOUT(end,:) +distStd * randn(1, nx);
           uHistory(k+1,:) = uk;
    end

    rmse = 0;
    totalEnergy = 0;
    for k = 1:Nsteps
        % 状態誤差の計算
        e = xHistory(k,:)' - xf;   % 状態偏差
        rmse = rmse + (e' * e);
        
    omega1 = xHistory(k, 3)';  % 角速度（状態ベクトル内の適切なインデックスに置き換え）
    omega2 = xHistory(k, 4)';
    totalEnergy = totalEnergy + abs(uHistory(k,1) * omega1) * Ts + abs(uHistory(k,2) * omega2) * Ts;
    end
    
    % 必要に応じてルートを取る（RMSEにする場合）
    rmse = sqrt(rmse / (Nsteps));
end
