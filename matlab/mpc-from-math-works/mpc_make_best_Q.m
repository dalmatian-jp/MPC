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
p  = 20;                    % Prediction Horizon.
xf = [0; 0; 0; 0];          % Terminal State.
x0 = [0.0873; 0; 0; 0];
%x0 = [0.2618; 0; 0; 0];
u0 = zeros(nmv,1);

nx  = 4;
nmv = 2;
Duration = 3;
Ts = 0.01;

Q_values = {
   linspace(3100, 6900, 3);  % Q(1)
   linspace(7000, 9000, 3);  % Q(2)
   linspace(6000, 8000, 3);  % Q(3)
   linspace(7500, 9500, 3);  % Q(4)
};
min_error = inf;

                
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



%% 最適なQの探索
for q1 = Q_values{1}
    for q2 = Q_values{2}
        for q3 = Q_values{3}
            for q4 = Q_values{4}
                
                Q = [q1; q2; q3; q4];
                pvcost = [xf; Q; R; p];

                simdata.StageParameter = repmat(pvcost, p+1, 1);
                onlinedata.StageParameter = simdata.StageParameter; % 更新

                xHistory = x0.';
                uHistory = u0.';

                % シミュレーション実行
                for k = 1:(Duration/Ts)
                   xk = xHistory(k,:).';
                   [uk, onlinedata] = nlmpcControllerMEX(xk, uk, onlinedata);
                   ODEFUN = @(t,xk) twolinkStateFcn(xk,uk,pvstate);
                   [TOUT, XOUT] = ode23t(ODEFUN, [0 Ts], xHistory(k,:));
                   xHistory(k+1,:) = XOUT(end,:);
                   uHistory(k+1,:) = uk;
                end

                % 誤差評価
                final_error = norm(xHistory(end,:)' - xf);
                if final_error < min_error
                    min_error = final_error;
                    best_Q = Q;
                end

                fprintf("Q = [%d, %d, %d, %d], Error = %.4f\n", q1, q2, q3, q4, final_error);
            end
        end
    end
end

fprintf("最適なQ: [%d, %d, %d, %d]\n", best_Q);
fprintf("最小誤差: %.4f\n", min_error);

%% 最適なQで最終シミュレーション
pvcost = [xf; best_Q; R; p];
simdata.StageParameter = repmat(pvcost, p+1, 1);

xHistory1 = x0.';
uHistory1 = u0.';

xk = xHistory1(1,:).';
uk = uHistory1(1,:).';

for k = 1:(Duration/Ts)
   xk = xHistory1(k,:).';
   [uk, onlinedata] = nlmpcControllerMEX(xk, uk, onlinedata);
   ODEFUN = @(t,xk) twolinkStateFcn(xk,uk,pvstate);
   [TOUT, XOUT] = ode23t(ODEFUN, [0 Ts], xHistory1(k,:));
   xHistory1(k+1,:) = XOUT(end,:);
   uHistory1(k+1,:) = uk;
end

figure;
helperPlotResults(xHistory1, uHistory1, Ts, xf, 'show')