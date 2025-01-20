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
R  = [0.02; 0.01];

nx  = 4;
nmv = 2;
p  = 20;                    
xf = [0; 0; 0; 0];          
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

[msobj, coredata, onlinedata] = setupNLMPC(p, nx, nmv, Ts, pvstate, pvcost, x0, u0);
%% 
[~, ~, info1] = nlmpcmove(msobj, x0, u0, onlinedata);
[~, ~, info2] = nlmpcmoveCodeGeneration(coredata, x0, u0, onlinedata); 
[~, ~, info3] = nlmpcControllerMEX(x0, u0, onlinedata); 

%% EKFオブジェクトの作成
stateFcn = @(x,u,dist) twolinkStateFcnDiscrete(x, u, pvstate, Ts, dist);
measurementFcn = @(x) x;
MeasurementJacFcn = @(x) measurementJacobian(x);

% EKFオブジェクトの初期化
ekf = extendedKalmanFilter(...
    stateFcn, ...
    measurementFcn, ...
    x0, ...
    'ProcessNoise', Qekf, ...
    'MeasurementNoise', Rekf, ...
    'MeasurementJacobianFcn', MeasurementJacFcn);

%%
ekf.State = x0;
if isempty(ekf.StateCovariance)
    ekf.StateCovariance = 1e-12*eye(nx);
end
P0 = ekf.StateCovariance;

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
    dist =distStd * randn(1, nx);
    pastIdx = k - delaySteps;

    if pastIdx > 0
        z_k = xTrueHistory(pastIdx, :)';
        zHistory(pastIdx, :) = z_k.'; 
    else
        z_k = [];
    end

    tic
    [uMPC, onlinedata] = nlmpcControllerMEX(xEstHistory(k,:).',uHistory(k,:).',onlinedata);
    timerVal = timerVal + toc;
    uHistory(k+1,:) = uMPC.';

    ekf.State = xEstHistory(k, :)';
    predict(ekf, uMPC, dist);

    ekfStateAll(k+1,:)   = ekf.State.';
    ekfCovAll(:,:,k+1)   = ekf.StateCovariance;

    if pastIdx > 0
        xPast = ekfStateAll(pastIdx,:)';
        Ppast = ekfCovAll(:,:,pastIdx);
        ekf.State = xPast;
        ekf.StateCovariance = Ppast;

        correct(ekf, z_k);

        for reSimIdx = pastIdx : k
            uRe = uHistory(reSimIdx+1,:)';
            predict(ekf, uRe);
            ekfStateAll(reSimIdx+1,:) = ekf.State.';
            ekfCovAll(:,:,reSimIdx+1) = ekf.StateCovariance;
        end
    end

    xEstHistory(k+1,:) = ekfStateAll(k+1,:);
    XOUT = twolinkStateFcnDiscrete(xk, uMPC, pvstate, Ts, dist);
    xTrueHistory(k + 1, :) = XOUT(end, :) + dist;
end

%%  状態変数のプロット
figure;

for i = 1:nx
    subplot(nx + nmv, 1, i); hold on; grid on;
    plot((0:Nsteps) * Ts, xTrueHistory(:, i), 'k-', 'DisplayName', 'True State');
    plot((0:Nsteps) * Ts, zHistory(:, i), 'r.', 'DisplayName', 'Delayed Meas.');
    plot((0:Nsteps) * Ts, xEstHistory(:, i), 'b--', 'DisplayName', 'EKF Estimate');
    ylabel(['x(' num2str(i) ')']);
    
    if i == 1
        title('MPC_+ekf');
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
