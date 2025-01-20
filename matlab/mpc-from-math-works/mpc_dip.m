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
Q  = [6900; 9000; 7000; 9500];%制限が3060で0.2618のとき
Q = [6675.6; 7852.6; 6185.5; 7620.6];%3060,0.0873
R  = [0.02; 0.01];            % Control Weighting Matrix.

Q=[3100.000000; 7000.000000; 6000.000000; 7500.000000];
R = [0.022000; 0.009000];
nx  = 4;
nmv = 2;
p  = 20;                    % Prediction Horizon.
xf = [0; 0; 0; 0];          % Terminal State.
pvcost = [xf; Q; R; p];
x0 = [0.0873; 0; 0; 0];
%x0 = [0.2618; 0; 0; 0];
u0 = zeros(nmv,1);
distStd = 0.001;
rng default;
Ts = 0.01;
Duration = 1;

[msobj, coredata, onlinedata] = setupNLMPC(p, nx, nmv, Ts, pvstate, pvcost, x0, u0);
%%
[~, ~, info1] = nlmpcmove(msobj, x0, u0, onlinedata);
[~, ~, info2] = nlmpcmoveCodeGeneration(coredata, x0, u0, onlinedata); 
[~, ~, info3] = nlmpcControllerMEX(x0, u0, onlinedata); 

%%
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