function dxdt = twolinkMorassoStateFcn(x, tau, pvstate)

% States
q1 = x(1);
q2 = x(2);
dq1 = x(3);
dq2 = x(4);

% Retrieve physical parameters from pvstate
l1 = pvstate(1);    % Length of link 1 (m)
r1 = pvstate(2);    % Distance from joint 1 to center of mass of link 1 (m)
m1 = pvstate(3);    % Mass of link 1 (kg)
I1 = pvstate(4);    % Moment of inertia of link 1 (kg*m^2)
l2 = pvstate(5);    % Length of link 2 (m)
r2 = pvstate(6);    % Distance from joint 2 to center of mass of link 2 (m)
m2 = pvstate(7);    % Mass of link 2 (kg)
I2 = pvstate(8);    % Moment of inertia of link 2 (kg*m^2)
g  = pvstate(9);    % Acceleration due to gravity (m/s^2)

a = I1 + I2 + m1*r1^2 + m2*(l1^2 + r2^2);
b = m2*l1*r2;
d = I2 + m2*r2^2;

% Manipulator inertia matrix
M = zeros(2, 2);
M(1, 1) = a+2*b*cos(q2);
M(1, 2) = d + b*cos(q2);
M(2, 1) = M(1, 2);
M(2, 2) = d;

% Coriolis matrix
C = zeros(2,2);
C(1, 1) = -b*sin(q2)*dq2;
C(1, 2) = -b*sin(q2)*(dq1+dq2);
C(2, 1) =  b*sin(q2)*dq2;


% Gravity vector
G = zeros(2, 1);
G(1) = -g*((m1*r1+m2*l1)*sin(q1)+m2*r2*sin(q1+q2));
G(2) = -g*m2*r2*sin(q1+q2);

dxdt = [dq1; dq2; M \ (tau - C*[dq1; dq2] - G)];
end