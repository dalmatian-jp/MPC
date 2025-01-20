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

Q = [0;0;0;0];
R  = [0; 0];
nx  = 4;
nmv = 2;
p  = 20;
xf = [0; 0; 0; 0];
pvcost = [xf; Q; R; p];
x0 = [0.0873; 0; 0; 0];
%x0 = [0.2618; 0; 0; 0];
u0 = zeros(nmv,1);
rng default;
Ts = 0.01;
Duration = 1;
distStd = 0.00;

[msobj, coredata, onlinedata] = setupNLMPC(p, nx, nmv, Ts, pvstate, pvcost, x0, u0);

%% parate opt

% パレート最適化用の関数ハンドル
objectiveFcn = @(params) paretoObjective_only_MPC(params, msobj, x0, u0, xf,p, pvstate, Ts, Duration, distStd);

% `gamultiobj` を使用した多目的最適化
opts = optimoptions('gamultiobj', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 3, ...
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
[~, onlinedata] = getCodeGenerationData(msobj, x0, u0, ...
    StateFcnParameter = pvstate, StageParameter = simdata.StageParameter);

[~, ~, info1] = nlmpcmove(msobj, x0, u0, onlinedata);
[~, ~, info2] = nlmpcmoveCodeGeneration(coredata, x0, u0, onlinedata); 
[~, ~, info3] = nlmpcControllerMEX(x0, u0, onlinedata); 

xHistory1 = zeros(Duration/Ts + 1, nx);
uHistory1 = zeros(Duration/Ts + 1, nmv);
xHistory1(1,:) = x0.';
uHistory1(1,:) = u0.';

uk = uHistory1(1,:).';

for k = 1:(Duration/Ts)
   xk = xHistory1(k,:).';
   [uk, onlinedata] = nlmpcControllerMEX(xk, uk, onlinedata);
   XOUT = solveODE(xk, uk, pvstate, Ts);
   xHistory1(k+1,:) = XOUT(end,:) +distStd * randn(1, nx);
   uHistory1(k+1,:) = uk;
end

figure;
helperPlotResults(xHistory1, uHistory1, Ts, xf, 'show')