% Two-Link Robot Physical Parameter Values.
%Value = [0.9; 0.58; 28; 
%         9.21; 0.88; 0.32; 
%         53; 5.35; 9.81];

Value = [0.78; 0.39; 11.41; 
         0.35; 0.73; 0.365; 
         50.14; 0.25; 9.81];

%Value = [0.5;   % L1: 第1リンクの長さ [m]
 %        0.25;  % s1: 第1リンクの重心位置 [m]
  %       2.0;   % m1: 第1リンクの質量 [kg]
   %      0.05;  % J1: 第1リンクの慣性モーメント [kg·m²]
    %     0.4;   % L2: 第2リンクの長さ [m]
     %    0.15;  % s2: 第2リンクの重心位置 [m]
      %   1.5;   % m2: 第2リンクの質量 [kg]
       %  0.03;  % J2: 第2リンクの慣性モーメント [kg·m²]
        % 9.81]; % g: 重力加速度 [m/s²]



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
%Q  = [5000; 8000; 7000; 8500];  % State Weight Matrix.
R  = [0.02; 0.01];            % Control Weighting Matrix.
p  = 20;                    % Prediction Horizon.
xf = [0; 0; 0; 0];          % Terminal State.
x0 = [0.04; 0; 0; 0];
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

                simdata = getSimulationData(msobj);
                simdata.StateFcnParameter = pvstate;
                simdata.StageParameter = repmat(pvcost, p+1, 1);

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