function [rmse, totalEnergy] = simulateMPC_EKF(Q, msobj, x0, u0, xf, pvstate, Ts, Duration, ekf, delay_s, distStd,Qekf,Rekf, R)
    %% (1) コスト関数の設定
    pvcost = [xf; Q(:); R(:); 20];
    simdata = getSimulationData(msobj);
    simdata.StageParameter = repmat(pvcost, 21, 1);
    
    [~, onlinedata] = getCodeGenerationData(msobj, x0, u0, ...
        StateFcnParameter = pvstate, StageParameter = simdata.StageParameter);
    
    delaySteps = round(delay_s / Ts);
    Nsteps = Duration / Ts;
    nx =4;
    nmv = 2;
    rng(1);

    %% (2) EKFの初期状態設定（初期化は外部で済ませる）
    ekf.State = x0;
    if isempty(ekf.StateCovariance)
        ekf.StateCovariance = 1e-12*eye(nx); 
    end
    P0 = ekf.StateCovariance;

   %% シミュレーションループ
    xTrueHistory = zeros(Nsteps + 1, nx);
    xTrueHistory(1, :) = x0.';
    xEstHistory = zeros(Nsteps + 1, nx);
    xEstHistory(1,:) = x0.';
    zHistory = zeros(Nsteps + 1, nx);
    uHistory = zeros(Nsteps + 1, nmv);
    
    ekfStateAll = zeros(Nsteps+1, nx);
    ekfCovAll   = zeros(nx, nx, Nsteps+1);

    ekfStateAll(1,:) = x0.';
    ekfCovAll(:,:,1) = P0;

    timerVal = 0;

    %% EKFのみのシミュレーションループ
    for k = 1:Nsteps
        dist = distStd * randn(1, nx);
        pastIdx = k - delaySteps;  
        if pastIdx > 0
            z_k = xTrueHistory(pastIdx, :)';
            zHistory(pastIdx, :) = z_k.';
        else
            z_k = [];
        end
        
        tic
        [uMPC, onlinedata] = nlmpcControllerMEX(xEstHistory(k,:).', ...
                                                uHistory(k,:).', ...
                                                onlinedata);
        timerVal = timerVal + toc;
        uHistory(k+1,:) = uMPC.';
        
        ekf.State = xEstHistory(k, :)';
        predict(ekf, uHistory(k+1, :)');
    
        ekfStateAll(k+1,:)   = ekf.State.';
        ekfCovAll(:,:,k+1)   = ekf.StateCovariance;

        if pastIdx > 0
            xPast = ekfStateAll(pastIdx,:)';
            Ppast = ekfCovAll(:,:,pastIdx);
            ekf.State = xPast;
            ekf.StateCovariance = Ppast;

            correct(ekf, z_k);

            % 3) pastIdx+1 ~ k+1 の各ステップを再度 predict
            %    (制御入力は既に記録してある uHistory のものを順に使う)
            for reSimIdx = pastIdx : k
                uRe = uHistory(reSimIdx+1,:)';
                predict(ekf, uRe);
                % その都度 ekfStateAll / ekfCovAll を上書きする
                ekfStateAll(reSimIdx+1,:) = ekf.State.';
                ekfCovAll(:,:,reSimIdx+1) = ekf.StateCovariance;
            end
        end

        xEstHistory(k+1,:) = ekfStateAll(k+1,:);

        
        %--- (D) 実プラントの更新（制御なしの自然応答） ---
        options = odeset('Jacobian', @(t, x) twolinkStateJacFcnDiscrete(x, uMPC, pvstate, Ts), ...
                     'RelTol', 1e-4, 'AbsTol', 1e-6);
        ODEFUN = @(t, x) twolinkStateFcn(x, uMPC, pvstate);
        [~, XOUT] = ode23t(ODEFUN, [0 Ts], xTrueHistory(k, :));
        xTrueHistory(k + 1, :) = XOUT(end, :)+ dist;
    
    end
    %% (5) 最終状態誤差（目標状態との誤差）
    rmse = 0;
    totalEnergy = 0;
    for k = 1:Nsteps-delaySteps-3
        % 状態誤差の計算
        e = xTrueHistory(k,:)' - xf;   % 状態偏差
        rmse = rmse + (e' * e);
        
        % エネルギー消費の計算
        totalEnergy = totalEnergy + (uHistory(k,:) * uHistory(k,:)');
    end
    
    % 必要に応じてルートを取る（RMSEにする場合）
    rmse = sqrt(rmse / (Nsteps-delaySteps-3));

    
end