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
Q  = [5000; 8000; 7000; 9500];  % State Weight Matrix.
R  = [0.02; 0.01];            % Control Weighting Matrix.
p  = 20;                    % Prediction Horizon.
xf = [0; 0; 0; 0];          % Terminal State.
x0 = [0.2618; 0; 0; 0];
%x0 = [0.0873; 0; 0; 0];
nx  = 4;
nmv = 2;
Duration = 3;
Ts = 0.01;
pvcost = [xf; Q; R; p];
          
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

msobj.ManipulatedVariables(1).Min = -20;
msobj.ManipulatedVariables(1).Max =  20;
msobj.ManipulatedVariables(2).Min = -40;
msobj.ManipulatedVariables(2).Max =  40;

msobj.Optimization.Solver = "fmincon";
msobj.Optimization.SolverOptions.MaxIterations = 5000;          % 最大反復回数
msobj.Optimization.SolverOptions.MaxFunctionEvaluations = 1e6;
msobj.Optimization.SolverOptions.ConstraintTolerance = 1e-1;   % 制約許容誤差
msobj.Optimization.SolverOptions.StepTolerance = 1e-8;         % ステップ許容誤差
msobj.Optimization.SolverOptions.FunctionTolerance = 1e-9;     % 関数値許容誤差
msobj.Optimization.SolverOptions.OptimalityTolerance = 1e-10;   % 最適性許容誤差
msobj.Optimization.SolverOptions.Display = "iter";             % 反復情報を表示
msobj.Optimization.SolverOptions.ScaleProblem = true;
msobj.Optimization.SolverOptions.Algorithm = 'sqp'; 

simdata = getSimulationData(msobj);
simdata.StateFcnParameter = pvstate;
simdata.StageParameter = repmat(pvcost, p+1, 1);

R_values = {
    linspace(0.018, 0.022, 10);  % Q(1)
    linspace(0.009, 0.011, 10);  % Q(2)
};
min_error = inf;

[coredata, onlinedata] = getCodeGenerationData(msobj, x0, u0, ...
    StateFcnParameter = pvstate, StageParameter = simdata.StageParameter);
buildMEX(msobj, "nlmpcControllerMEX", coredata, onlinedata);

%% 最適なQの探索
for q1 = R_values{1}
    for q2 = R_values{2}
                R = [q1; q2];
                pvcost = [xf; Q; R; p];

                simdata.StageParameter = repmat(pvcost, p+1, 1);
                onlinedata.StageParameter = simdata.StageParameter; % 更新

                u0 = zeros(nmv,1);

                xHistory = x0.';
                uHistory = u0.';

                % シミュレーション実行
                for k = 1:(Duration/Ts)
                    xk = xHistory(k,:).';
                    [uk, onlinedata] = nlmpcControllerMEX(xk, u0, onlinedata);
                    ODEFUN = @(t,xk) twolinkStateFcn(xk,uk,pvstate);
                    [~, XOUT] = ode23t(ODEFUN, [0 Ts], xk);
                    xHistory(k+1,:) = XOUT(end,:);
                    uHistory(k+1,:) = uk;
                end

                % 誤差評価
                final_error = norm(xHistory(end,:)' - xf);
                if final_error < min_error
                    min_error = final_error;
                    best_R = R;
                end

                fprintf("R = [%d, %d], Error = %.4f\n", q1, q2, final_error);
    end
end

fprintf("最適なR: [%d, %d, %d, %d]\n", best_R);
fprintf("最小誤差: %.4f\n", min_error);

%% 最適なRで最終シミュレーション
pvcost = [xf; Q; best_R; p];
simdata.StageParameter = repmat(pvcost, p+1, 1);

xHistory = x0.';
uHistory = u0.';

for k = 1:(Duration/Ts)
    xk = xHistory(k,:).';
    [uk, onlinedata] = nlmpcControllerMEX(xk, u0, onlinedata);
    ODEFUN = @(t,xk) twolinkStateFcn(xk,uk,pvstate);
    [~, XOUT] = ode23t(ODEFUN, [0 Ts], xk);
    xHistory(k+1,:) = XOUT(end,:);
    uHistory(k+1,:) = uk;
end

% 結果描画
figure;
helperPlotResults(xHistory, uHistory, Ts, xf, 'show');