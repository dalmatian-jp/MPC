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


Q=[4650.736905; 7414.887208; 7076.969769; 8378.248309];
R = [0.01964225195;0.009442641393];

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


%% 

[~, ~, info1] = nlmpcmove(msobj, x0, u0, onlinedata);
[~, ~, info2] = nlmpcmoveCodeGeneration(coredata, x0, u0, onlinedata); 
[~, ~, info3] = nlmpcControllerMEX(x0, u0, onlinedata); 

figure;
for ix = 1:nx
    subplot(3,2,ix); hold on; box on;
    plot(info1.Topt, info1.Xopt(:,ix), "-.");
    plot(info2.Topt, info2.Xopt(:,ix), "-x");
    plot(info3.Topt, info3.Xopt(:,ix), "-+");
    title(['State, x_' num2str(ix)]);
    xlabel("Time, s");
end

for imv = 1:nmv
    subplot(3,2,ix+imv); hold on; box on;
    plot(info1.Topt, info1.MVopt(:,imv), "-.");
    plot(info2.Topt, info2.MVopt(:,imv), "-x");
    plot(info3.Topt, info3.MVopt(:,imv), "-+");
    title(['Control, u_' num2str(imv)]);
    xlabel("Time, s");
end

legend("nlmpcmove", ...
    "nlmpcmoveCodeGeneration", ...
    "nlmpcControllerMEX")

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

error = simulateMPC_EKF(Q, msobj, x0, u0, xf, pvstate, Ts, Duration, ekf, delay_s, distStd,Qekf,Rekf, R);

