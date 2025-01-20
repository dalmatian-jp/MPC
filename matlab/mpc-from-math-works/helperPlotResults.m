function helperPlotResults(xHistory, uHistory, Ts, xf, LegendControl)
% HELPERPLOTRESULTS
%
%   HELPERPLOTRESULTS is a helper function used for plotting the closed-loop
%   results of a controller in MATLAB and Simulink simulations.
%
%   HELPERPLOTRESULTS(SIMHISTORY) plots the states and control action
%   during a Simulink simulation, where SIMHISTORY is a
%   Simulink.SimulationOutput object.
%
%   HELPERPLOTRESULTS(XHISTORY, UHISTORY, TS) plots the states and
%   control action during a MATLAB simulation, where XHISTORY and UHISTORY are
%   tall matrices containing the states and control actions, and TS is a
%   scalar value for the sample time. 
%
%   HELPERPLOTRESULTS(XHISTORY, UHISTORY, TS, XF) plots the states and
%   control action during a MATLAB simulation. Additionally, it adds a new plot
%   for the desired robot position, XF.
%
%   HELPERPLOTRESULTS(__, LEGENDCONTROL) plots the closed-loop results and
%   sets the visibility of the legend using the input LEGENDCONTROL ("hide" | "show").

% Copyright 2023 The MathWorks, Inc.

arguments
    xHistory = []
    uHistory = []
    Ts       = []
    xf       = []
    LegendControl {mustBeMember(LegendControl,{'show','hide'})} = "hide"
end

if nargin == 1
    % Input arguments is a Simulink.SimulationOutput object for a single input
    simOut = xHistory;

    % Extract timeX, timeMV, xHistory, and uHistory from Simulink.SimulationOutput
    timeX = simOut.logsout.get("States").Values.Time;
    timeMv = simOut.logsout.get("Control Action").Values.Time;
    xHistory = simOut.logsout.get("States").extractTimetable.("States");
    uHistory = simOut.logsout.get("Control Action").extractTimetable.("Control Action");
else
    % Generate the time vector
    timeX = (0:(length(xHistory)-1))*Ts;
    timeMv = (0:(length(xHistory)-1))*Ts;
end

% Set number of states and control inputs
nx  = 4;
nmv = 2; 

% Define the state labels
states = {...
    'Joint Angle, ', '\theta_1';
    'Joint Angle, ', '\theta_2';
    'Joint Velocity, ', '\omega_1';
    'Joint Velocity, ', '\omega_2'};

control = {...
    'Torque, ', '\tau_1';
    'Torque, ', '\tau_2'};

% Plot the states
for ix = 1:nx
    % Create a subplot for each state
    subplot(3, 2, ix); hold on; box on;

    % Plot the state history
    plot(timeX, xHistory(:,ix), "DisplayName", states{ix,2});

    % Plot the desired state trajectory
    if ~isempty(xf)
        plot(timeX, ones(size(timeX))*xf(ix), "DisplayName", ['xf_' num2str(ix)]);
    end

    % Set the title and labels for the subplot
    legend(gca, LegendControl);
    title([states{ix,:}])
    xlabel("Time, s");
end

% Plot the control inputs
for imv = 1:nmv
    % Create a subplot for each control input
    subplot(3, 2, ix+imv); hold on; box on;

    % Plot the control input history
    plot(timeMv, uHistory(:,imv), "DisplayName", control{imv,2});

    % Set the title and labels for the subplot
    legend(gca, LegendControl);
    title([control{imv,:}])
    xlabel("Time, s");
end
end