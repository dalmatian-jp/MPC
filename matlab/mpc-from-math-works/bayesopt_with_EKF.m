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

msobj = nlmpcMultistage(p,nx,nmv);
msobj.Ts = Ts;
msobj.Model.StateFcn = "twolinkStateFcn";
msobj.Model.StateJacFcn = "twolinkStateJacFcn";
msobj.Model.ParameterLength = length(pvstate);

for k = 1:p+1
    msobj.Stages(k).CostFcn = "twolinkCostFcn";
    msobj.Stages(k).CostJacFcn = "twolinkCostJacFcn";
    msobj.Stages(k).ParameterLength = length(pvcost);
end

msobj.ManipulatedVariables(1).Min = -30;
msobj.ManipulatedVariables(1).Max =  30;
msobj.ManipulatedVariables(2).Min = -60;
msobj.ManipulatedVariables(2).Max =  60;

msobj.States(1).Min = -0.35;
msobj.States(1).Max =  0.53;
msobj.States(2).Min = -0.53;
msobj.States(2).Max =  0.87;

simdata = getSimulationData(msobj);
simdata.StateFcnParameter = pvstate;
simdata.StageParameter = repmat(pvcost, p+1, 1);

msobj.Optimization.Solver = "fmincon";
msobj.Optimization.SolverOptions.MaxIterations = 5000;          % 最大反復回数
msobj.Optimization.SolverOptions.MaxFunctionEvaluations = 1e6;
msobj.Optimization.SolverOptions.ConstraintTolerance = 1e-3;   % 制約許容誤差
msobj.Optimization.SolverOptions.StepTolerance = 1e-8;         % ステップ許容誤差
msobj.Optimization.SolverOptions.FunctionTolerance = 1e-9;     % 関数値許容誤差
msobj.Optimization.SolverOptions.OptimalityTolerance = 1e-10;   % 最適性許容誤差
msobj.Optimization.SolverOptions.Display = "iter";             % 反復情報を表示
msobj.Optimization.SolverOptions.ScaleProblem = true;
msobj.Optimization.SolverOptions.Algorithm = 'sqp'; 

[coredata, onlinedata] = getCodeGenerationData(msobj, x0, u0, ...
    StateFcnParameter = pvstate, StageParameter = simdata.StageParameter);
buildMEX(msobj, "nlmpcControllerMEX", coredata, onlinedata);

simdata.StageParameter = repmat(pvcost, p+1, 1);
onlinedata.StageParameter = simdata.StageParameter; % 更新


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

%%

% ベイズ最適化の変数設定
Q1 = optimizableVariable('Q1', [3100, 6900]);
Q2 = optimizableVariable('Q2', [7000, 9000]);
Q3 = optimizableVariable('Q3', [6000, 8000]);
Q4 = optimizableVariable('Q4', [7500, 9500]);

% 制御重みRの最適化変数（R1, R2）
R1 = optimizableVariable('R1', [0.018, 0.022]);  % 例: 制御入力1の重み
R2 = optimizableVariable('R2', [0.009, 0.011]);  % 例: 制御入力2の重み

% 最適化の目的関数
objectiveFcn = @(param) simulateMPC_EKF([param.Q1; param.Q2; param.Q3; param.Q4], ...
                                         msobj, x0, u0, xf, pvstate, Ts, Duration, ekf, delay_s,...
                                         [param.R1; param.R2]);
% ベイズ最適化の実行
results = bayesopt(objectiveFcn, [Q1, Q2, Q3, Q4, R1, R2], ...
    'UseParallel', true, ...
    'MaxObjectiveEvaluations', 40, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1, ...
    'PlotFcn', {@plotMinObjective});

% 最適なQの抽出
best_Q = [results.XAtMinObjective.Q1; results.XAtMinObjective.Q2; ...
          results.XAtMinObjective.Q3; results.XAtMinObjective.Q4];

best_R = [results.XAtMinObjective.R1; results.XAtMinObjective.R2];

fprintf("最適なQ: [%f, %f, %f, %f]\n", best_Q);
fprintf("最適なR: [%f, %f]\n", best_R);

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
    predict(ekf, uHistory(k, :)');

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
        title('bayesopt_with_EKF');
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
