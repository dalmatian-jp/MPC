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
x0 = [0.2618; 0; 0; 0];
u0 = zeros(nmv,1);

Ts = 0.01;
Duration = 5;
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

%% (2) EKFの初期状態設定（初期化は外部で済ませる）
ekf.State = x0;  % 状態の初期化
if isempty(ekf.StateCovariance)
    ekf.StateCovariance = eye(nx)*1e-3; 
end
P0 = ekf.StateCovariance;  % 初期共分散保存

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
    %--- (A) センサノイズ付加観測（0.2秒遅延対応） ---
    pastIdx = k - delaySteps;  
    if pastIdx > 0
        z_k = xTrueHistory(pastIdx, :)' + chol(Rekf) * randn(nx, 1);
        zHistory(pastIdx, :) = z_k.';  % 観測データも遅延分シフト
    else
        z_k = [];  % まだ観測は届かない
    end
    
    tic
    uMPC = 0
    timerVal = timerVal + toc;
    uHistory(k+1,:) = uMPC.';

    predict(ekf, uHistory(k, :)');

    ekfStateAll(k+1,:)   = ekf.State.';
    ekfCovAll(:,:,k+1)   = ekf.StateCovariance;

    if pastIdx > 0
        % 1) EKF を "pastIdx" の状態に巻き戻す
        xPast = ekfStateAll(pastIdx,:)';
        Ppast = ekfCovAll(:,:,pastIdx);
        ekf.State = xPast;
        ekf.StateCovariance = Ppast;

        % 2) 遅延観測 z_k を用いて correct
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
    xTrueHistory(k + 1, :) = XOUT(end, :)+ distStd * randn(1, nx);

end
xTrueHistory(Nsteps - delaySteps + 1:end, :) = NaN;
xEstHistory(Nsteps - delaySteps + 1:end, :) = NaN;
zHistory(Nsteps - delaySteps + 1:end, :) = NaN;
uHistory(Nsteps - delaySteps + 1:end, :) = NaN;


%% 結果の可視化（EKFの性能確認）
figure;
for i = 1:nx
    subplot(nx, 1, i); hold on; grid on;
    plot((0:Nsteps) * Ts, xTrueHistory(:, i), 'k-', 'DisplayName', 'True State');
    plot((0:Nsteps) * Ts, zHistory(:, i), 'r.', 'DisplayName', 'Measured');
    plot((0:Nsteps) * Ts, xEstHistory(:, i), 'b--', 'DisplayName', 'EKF Estimate');
    ylabel(['x(' num2str(i) ')']);

    if i == 1
        title('EKFの性能確認（制御入力なし）');
    end
    if i == nx
        xlabel('Time [s]');
    end
end

fprintf("EKFのみでのシミュレーションが完了しました。\n");
