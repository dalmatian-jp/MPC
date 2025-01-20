function [A,B] = twolinkStateJacFcn(x,tau,pvstate)

l1 = pvstate(1,:);
r1 = pvstate(2,:);
m1 = pvstate(3,:);
I1 = pvstate(4,:);
l2 = pvstate(5,:);
r2 = pvstate(6,:);
m2 = pvstate(7,:);
I2 = pvstate(8,:);
g  = pvstate(9,:);

% Control Input
tau1 = tau(1,:);
tau2 = tau(2,:);

% States
q1 = x(1,:);
q2 = x(2,:);
dq1 = x(3,:);
dq2 = x(4,:);

s1 = sin(q1);
s2 = sin(q2);
c1 = cos(q1);
c2 = cos(q2);
s12 = sin(q1+q2);
c12 = cos(q1+q2);

a = I1 + I2 + m1*r1^2 + m2*(l1^2 + r2^2);
b = m2*l1*r2;
d = I2 + m2*r2^2;

det = a*d - d^2 - b^2*cos(q2)^2;
ddet = b^2*sin(2*q2);

M11 = a+2*b*cos(q2);
M12 = d + b*cos(q2);
M22 = d;

jac = zeros(4,4);

jac32_1 = d*(tau1 - tau2);
jac32_2 = sin(q1)*d*g*(m1*r1-m2*l1);
jac32_3 = sin(q2)*(2*b*d*dq1*dq2 + b*d*dq2^2  + 1/2*d*l1*dq1^2);
jac32_4 = cos(q2)*(-b*tau2);
jac32_5 = sin(q2)*cos(q2)*1/2*b*l1*dq1^2;
jac32_6 = sin(q1 + q2)*cos(q2)*(-b*g*m2*r2);

djac32_3 = cos(q2)*(2*b*d*dq1*dq2 + b*d*dq2^2  + 1/2*d*l1*dq1^2);
djac32_4 = -sin(q2)*(-b*tau2);
djac32_5 = cos(2*q2)*1/2*b*l1*dq1^2;
djac32_6 = cos(q1 + 2*q2)*(-b*g*m2*r2);

jac31 = (d*g*(m1*r1-m2*l1)*cos(q1))/det;
jac32 = (-jac32_1*ddet  -jac32_2*ddet + (djac32_3*det - jac32_3*ddet) + (djac32_4*det - jac32_4*ddet) + (djac32_5*det - jac32_5*ddet) + (djac32_6*det - jac32_6*ddet))/det^2;
jac33 = (2*b*d*sin(q2)*dq2 + b*l1*dq1*(1 + s2*c2))/det;
jac34 = (2*b*d*s2*(dq1+dq2))/det;

jac42_1 = -d*tau1 -d*g*(m1*r1 - m2*l1)*sin(q1) + a*tau2;
jac42_2 = sin(q2)*(-2*d*b*dq1*dq2 - d*b*dq2^2 - 1/2*a*l1*dq1^2);
jac42_3 = sin(q1+q2)*(-d*g*m2*r2 + a*g*m2*r2);
jac42_4 = cos(q2)*(-b*tau1 + 2*b*tau2);
jac42_5 = sin(q2)*cos(q2)*(-2*b^2*dq1*dq2 -b^2*dq2^2 -b*l1*dq1^2);
jac42_6 = s1*c2*(-b*g*(m1*r1 - m2*l1));
jac42_7 = s12*c2*(b*g*m2*r2);

djac42_2 = cos(q2)*(-2*d*b*dq1*dq2 - d*b*dq2^2 - 1/2*a*l1*dq1^2);
djac42_3 = cos(q1+q2)*(-d*g*m2*r2 + a*g*m2*r2);
djac42_4 = -sin(q2)*(-b*tau1 + 2*b*tau2);
djac42_5 = cos(2*q2)*(-2*b^2*dq1*dq2 -b^2*dq2^2 -b*l1*dq1^2);
djac42_6 = c12*(-b*g*(m1*r1 - m2*l1));
djac42_7 = cos(q1+ 2*q2)*(b*g*m2*r2);

jac41 = (-M12*g*((m1*r1 - m2*l1)*cos(q1)+ m2*r2*cos(q1 + q2)) + M11*g*m2*r2*cos(q1+q2))/det;
jac42 = (-jac42_1*ddet + (djac42_2*det - jac42_2*ddet) + (djac42_3*det - jac42_3*ddet) + (djac42_4*det - jac42_4*ddet) + (djac42_5*det - jac42_5*ddet) + (djac42_6*det - jac42_6*ddet) + (djac42_7*det - jac42_7*ddet))/det^2;
jac43 = (-M12*2*b*sin(q2)*dq2 - M11*l1*sin(q2)*dq1)/det;
jac44 = (-M12*(2*b*s2*(dq1+dq2)))/det;

jac(1,3) = 1;
jac(2,4) = 1;
jac(3,1) = jac31;
jac(3,2) = jac32;
jac(3,3) = jac33;
jac(3,4) = jac34;
jac(4,1) = jac41;
jac(4,2) = jac42;
jac(4,3) = jac43;
jac(4,4) = jac44;

jacB  = zeros(4,2);

jacB(3,1) = M22/det;
jacB(3,2) = -M12/det;
jacB(4,1) = -M12/det;
jacB(4,2) = M11/det;

A = jac;
B = jacB;
end