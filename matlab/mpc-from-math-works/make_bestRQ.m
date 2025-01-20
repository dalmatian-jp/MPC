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

% Convert arrays to table
ParamTable = array2table(Value, "RowNames", Param);
UnitsTable = table(categorical(Units), ...
                   VariableNames = "Units");
pvstate = ParamTable.Value;

% Stage Parameters
Q = [6675.6; 7852.6; 6185.5; 7620.6];  % State Weight Matrix.
R  = [0.02; 0.01];            % Control Weighting Matrix.
p  = 20;                    % Prediction Horizon.
xf = [0; 0; 0; 0];          % Terminal State.
%x0 = [0.1746; 0; 0; 0];
x0 = [0.0873; 0; 0; 0];
u0 = zeros(nmv,1);
nx  = 4;
nmv = 2;
Duration = 3;
Ts = 0.01;
pvcost = [xf; Q; R; p];
rng('shuffle');
distStd = 0.005;
                
msobj = nlmpcMultistage(p, nx, nmv);
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
objectiveFcn = @(param) simulateMPC([param.Q1; param.Q2; param.Q3; param.Q4], ...
                                         msobj, x0, u0, xf, pvstate, Ts, Duration,distStd,...
                                         [param.R1; param.R2]);

% ベイズ最適化の実行
results = bayesopt(objectiveFcn, [Q1, Q2, Q3, Q4, R1, R2], ...
    'UseParallel', true, ...
    'MaxObjectiveEvaluations', 100, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1, ...
    'PlotFcn', {@plotMinObjective});

% 最適なQの抽出
best_Q = [results.XAtMinObjective.Q1; results.XAtMinObjective.Q2; ...
          results.XAtMinObjective.Q3; results.XAtMinObjective.Q4];

best_R = [results.XAtMinObjective.R1; results.XAtMinObjective.R2];

fprintf("最適なQ: [%f, %f, %f, %f]\n", best_Q);
fprintf("最適なR: [%f, %f]\n", best_R);

%% 最適なQRで最終シミュレーション
pvcost = [xf; best_Q; best_R; p];
simdata.StageParameter = repmat(pvcost, p+1, 1);

xHistory = x0.';
uHistory = u0.';
xk = xHistory1(1,:).';
uk = uHistory1(1,:).';

for k = 1:(Duration/Ts)
    xk = xHistory(k,:).';
    [uk, onlinedata] = nlmpcControllerMEX(xk, u0, onlinedata);
    ODEFUN = @(t,xk) twolinkStateFcn(xk,uk,pvstate);
    [TOUT, XOUT] = ode23t(ODEFUN, [0 Ts], xk);
    xHistory(k+1,:) = XOUT(end,:)+distStd * randn(1, nx);
    uHistory(k+1,:) = uk;
end

% 結果描画
figure;
helperPlotResults(xHistory, uHistory, Ts, xf, 'show');