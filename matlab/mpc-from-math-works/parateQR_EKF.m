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

%Q  = [6900; 8000; 7000; 7500];%制限が2040で0.0873のとき
%Q  = [6900; 9000; 7000; 9500];%制限が3060で0.2618のとき
Q = [6675.6; 7852.6; 6185.5; 7620.6];%3060,0.0873
R  = [0.02; 0.01];            % Control Weighting Matrix.

nx  = 4;
nmv = 2;
p  = 20;                    % Prediction Horizon.
xf = [0; 0; 0; 0];          % Terminal State.
pvcost = [xf; Q; R; p];
x0 = [0.0873; 0; 0; 0];
%x0 = [0.2618; 0; 0; 0];
u0 = zeros(nmv,1);
rng(1);

Ts = 0.01;
Duration = 1;
delay_s = 0.1;      
delaySteps = round(delay_s / Ts); 
Nsteps = Duration / Ts;
Qekf = diag([0.01, 0.01, 0.01, 0.01]);  
Rekf = diag([0.1, 0.1, 0.01, 0.01]);    
distStd = 0.001;

%% MPCオブジェクト作成
[msobj, coredata, onlinedata] = setupNLMPC(p, nx, nmv, Ts, pvstate, pvcost, x0, u0);

%% EKFオブジェクトの作成
stateFcn = @(x,u) twolinkStateFcnRK4(x, u, pvstate, Ts);
measurementFcn = @(x) x;
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

%% parate opt

% パレート最適化用の関数ハンドル
objectiveFcn = @(params) paretoObjective(params, msobj, x0, u0, xf, pvstate, Ts, Duration, ekf, delay_s, distStd, Qekf, Rekf);

% 最適化の実行
[numberOfObjectives, ~] = deal(2, 0); % 二つの目的関数

% `gamultiobj` を使用した多目的最適化
opts = optimoptions('gamultiobj', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 50, ...
    'FunctionTolerance', 1e-6, ...
    'ParetoFraction', 0.35, ...
    'Display', 'iter', ...
    'PlotFcn', {@gaplotpareto, @gaplotscores}, ...
    'UseParallel', true); % 並列計算を有効にする場合


% 最適化の範囲を設定
lb = [3100, 7000, 6000, 7500, 0.018, 0.009];
ub = [6900, 9000, 8000, 9500, 0.022, 0.011];

% 最適化の実行
[optimalParams, optimalObjectives] = gamultiobj(objectiveFcn, 6, [], [], [], [], lb, ub, opts);

%%
% パレートフロンティアのプロット
figure;
scatter(optimalObjectives(:,1), optimalObjectives(:,2), 'filled');
xlabel('RMSE');
ylabel('Total Energy Consumption');
title('Pareto Front');
grid on;

% 最適なQとRのパラメータを表示
disp('Pareto Optimal Solutions:');
for i = 1:size(optimalParams, 1)
    fprintf('Solution %d:\n', i);
    fprintf('  Q = [%f; %f; %f; %f]\n', optimalParams(i, 1:4));
    fprintf('  R = [%f; %f]\n', optimalParams(i, 5:6));
    fprintf('  RMSE = %f, Total Energy = %f\n\n', optimalObjectives(i, 1), optimalObjectives(i, 2));
end

% 例として、最初のパレート最適解を使用
best_Q = optimalParams(1, 1:4).';
best_R = optimalParams(1, 5:6).';

fprintf("選択した最適なQ: [%f; %f; %f; %f]\n", best_Q);
fprintf("選択した最適なR: [%f; %f]\n", best_R);

%% シミュレーションループ
% 最適なQで再シミュレーション
pvcost = [xf; best_Q; best_R; 20];
simdata.StageParameter = repmat(pvcost, 21, 1);

ekf.State = x0;  % 状態の初期化
if isempty(ekf.StateCovariance)
    ekf.StateCovariance = eye(nx)*1e-12; 
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
    pastIdx = k - delaySteps;  
    if pastIdx > 0
        z_k = xTrueHistory(pastIdx, :)' ;
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
    xTrueHistory(k + 1, :) = XOUT(end, :)+ distStd * randn(1, nx);

end

%% --- 状態変数のプロット ---
figure;

for i = 1:nx
    subplot(nx + nmv, 1, i); hold on; grid on;
    plot((0:Nsteps) * Ts, xTrueHistory(:, i), 'k-', 'DisplayName', 'True State');
    plot((0:Nsteps) * Ts, zHistory(:, i), 'r.', 'DisplayName', 'Delayed Meas.');
    plot((0:Nsteps) * Ts, xEstHistory(:, i), 'b--', 'DisplayName', 'EKF Estimate');
    ylabel(['x(' num2str(i) ')']);
    
    if i == 1
        title('bayesopt+EKF');
    end
    if i == nx + nmv
        xlabel('Time [s]');
    end
    legend();
end

% --- 制御入力（トルク）のプロット ---
for j = 1:nmv
    subplot(nx + nmv, 1, nx + j); hold on; grid on;
    plot((0:Nsteps) * Ts, uHistory(:, j), 'm-', 'LineWidth', 1.5, 'DisplayName', ['Torque u_' num2str(j)]);
    ylabel(['u(' num2str(j) ')']);
    
    if j == 1
        title('Control Input (Torque)');
    end
    if j == nmv
        xlabel('Time [s]');
    end
    legend();
end

% --- 制御計算の合計時間を出力 ---
fprintf("制御計算に要した合計時間: %.3f [sec]\n", timerVal);
