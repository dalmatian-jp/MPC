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
Q  = [5000; 8000; 7000; 8500];  % State Weight Matrix.
R  = [0.02; 0.01];            % Control Weighting Matrix.

Q = [6658.801103; 7714.380876; 6533.144908; 8356.967656];
R = [0.02; 0.01];
p  = 20;                    % Prediction Horizon.
nx  = 4;
nmv = 2;
Duration = 3;
pvcost = [xf; Q; R; p];
Ts = 0.01;
xf = [0; 0; 0; 0];          % Terminal State.
x0 = [0.0873; 0; 0; 0];
%x0 = [0.2618; 0; 0; 0];
u0 = zeros(nmv,1);
distStd = 0;


                
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

% ベイズ最適化の変数設定
Q1 = optimizableVariable('Q1', [3100, 6900]);
Q2 = optimizableVariable('Q2', [7000, 9000]);
Q3 = optimizableVariable('Q3', [6000, 8000]);
Q4 = optimizableVariable('Q4', [7500, 9500]);

% 最適化の目的関数
objectiveFcn = @(Q) simulateMPC([Q.Q1; Q.Q2; Q.Q3; Q.Q4], msobj, x0, u0, xf, pvstate, Ts, Duration,distStd,R);

% ベイズ最適化の実行
results = bayesopt(objectiveFcn, [Q1, Q2, Q3, Q4], ...
    'UseParallel', true, ...
    'MaxObjectiveEvaluations', 1, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1, ...
    'PlotFcn', {@plotMinObjective});

% 最適なQの抽出
best_Q = [results.XAtMinObjective.Q1; results.XAtMinObjective.Q2; ...
          results.XAtMinObjective.Q3; results.XAtMinObjective.Q4];

fprintf("最適なQ: [%f, %f, %f, %f]\n", best_Q);
%%
% 最適なQで再シミュレーション
pvcost = [xf; best_Q; R; p];
simdata.StageParameter = repmat(pvcost, p+1, 1);

[coredata, onlinedata] = getCodeGenerationData(msobj, x0, u0, ...
    StateFcnParameter = pvstate, StageParameter = simdata.StageParameter);

buildMEX(msobj, "nlmpcControllerMEX", coredata, onlinedata);

xHistory1 = x0.';
uHistory1 = u0.';

xk = xHistory1(1,:).';
uk = uHistory1(1,:).';


for k = 1:(Duration/Ts)
    xk = xHistory1(k,:).';
    [uk, onlinedata] = nlmpcControllerMEX(xk, uk, onlinedata);
    ODEFUN = @(t,xk) twolinkStateFcn(xk, uk, pvstate);
    [~, XOUT] = ode23t(ODEFUN, [0 Ts], xk);
    xHistory1(k+1,:) = XOUT(end,:);
    uHistory1(k+1,:) = uk;
end

figure;
helperPlotResults(xHistory1, uHistory1, Ts, xf, 'show')
