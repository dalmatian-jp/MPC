Value = [0.78; 0.39; 11.41; 
         0.35; 0.73; 0.365; 
         50.14; 0.25; 9.81];

Units = ["m"; "m"; "kg"; 
         "kg*m^2"; "m"; "m"; 
         "kg"; "kg*m^2"; "m/s^2"];

Param = [ ...
    "Length, L1", "Center of mass, s1", "Mass, m1", ...
    "Moment of Inertia, J1", "Length, L2", ...
    "Center of mass, s2", "Mass, m2", ...
    "Moment of Inertia, J2", "Gravity, g"];

ParamTable = array2table(Value, "RowNames", Param);
UnitsTable = table(categorical(Units), ...
                   VariableNames = "Units");
pvstate = ParamTable.Value;

nx  = 4;
nmv = 2;
p  = 20;                    % Prediction Horizon.
xf = [0; 0; 0; 0];          % Terminal State.
pvcost = [xf; Q; R; p];
x0 = [0.0873; 0; 0; 0];
%x0 = [0.2618; 0; 0; 0];
u0 = zeros(nmv,1);

Ts = 0.01;
Duration = 15;
Nsteps = Duration / Ts;
delay_s = 0.2;      
delaySteps = round(delay_s / Ts); 
Qekf = diag([0.01, 0.01, 0.01, 0.01]);  
Rekf = diag([0.1, 0.1, 0.01, 0.01]);    
distStd = 0.005;

%% EKFオブジェクトの作成
stateFcn = @(x,u) twolinkStateFcnRK4(x, u, pvstate, Ts);
measurementFcn = @(x) x;  % 観測モデル: 直接観測
stateJacFcn = @(x, u) twolinkStateJacFcnDiscreteRK4(x, u, pvstate, Ts);
MeasurementJacFcn = @(x) measurementJacobian(x);

% EKFオブジェクトの初期化
ekf = extendedKalmanFilter(...
    stateFcn, ...
    measurementFcn, ...
    x0, ...
    'ProcessNoise', Qekf, ...
    'MeasurementNoise', Rekf, ...
    'StateTransitionJacobianFcn', stateJacFcn, ...
    'MeasurementJacobianFcn', MeasurementJacFcn);

xTrueHistory = zeros(Nsteps + 1, nx);
xTrueHistory(1, :) = x0.';

xEstHistory = zeros(Nsteps + 1, nx);

zHistory = zeros(Nsteps + 1, nx);
uHistory = zeros(Nsteps + 1, nmv);

xDelayBuffer = repmat(x0, 1, delaySteps + 1);

%% EKFのみのシミュレーションループ
for k = 1:Nsteps
    %--- (A) センサノイズ付加観測（0.2秒遅延対応） ---
    pastIdx = k - delaySteps;  
    if pastIdx > 0
        z_k = xTrueHistory(pastIdx, :)' + chol(Rekf) * randn(nx, 1);
    end
    if k > delaySteps
        zHistory(k - delaySteps, :) = z_k.';  % 観測データも遅延分シフト
    end
    %--- (B) 遅延を考慮したEKFの予測と更新 ---
    x_delay = xDelayBuffer(:, 1);  % 最も古い状態
    ekf.State = x_delay;  % 遅延状態でEKFの状態を上書き

    for i = 1:delaySteps
        predict(ekf, u0);  % 遅延分の予測を追加
    end
    correct(ekf, z_k);

    %--- (C) EKFの推定値の記録（遅延が解消されてから記録） ---
    if k > delaySteps
        xEst = ekf.State;
        xEstHistory(k-delaySteps , :) = xEst.';
    end

    %--- (D) 実プラントの更新（制御なしの自然応答） ---
    options = odeset('RelTol', 1e-4, 'AbsTol', 1e-6);
    ODEFUN = @(t, x) twolinkStateFcn(x, u0, pvstate);
    [~, XOUT] = ode23t(ODEFUN, [0 Ts], xTrueHistory(k, :));
    xTrueHistory(k + 1, :) = XOUT(end, :);

    %--- (E) 遅延バッファの更新 ---
    xDelayBuffer(:, 1:end-1) = xDelayBuffer(:, 2:end);
    xDelayBuffer(:, end) = xTrueHistory(k + 1, :)';
end

%% 結果の可視化（EKFの性能確認）
figure;
for i = 1:nx
    subplot(nx, 1, i); hold on; grid on;
    plot((0:Nsteps) * Ts, xTrueHistory(:, i), 'k-', 'DisplayName', 'True State');
    plot((0:Nsteps) * Ts, zHistory(:, i), 'r.', 'DisplayName', 'Measured');
    plot((delaySteps:Nsteps) * Ts, xEstHistory(delaySteps + 1:end, i), 'b--', 'DisplayName', 'EKF Estimate');
    ylabel(['x(' num2str(i) ')']);

    if i == 1
        title('EKFの性能確認（制御入力なし）');
    end
    if i == nx
        xlabel('Time [s]');
    end
    legend();
end

fprintf("EKFのみでのシミュレーションが完了しました。\n");
