function L = twolinkCostFcn(stage, x, u, pvcost)

% Extract Desired State
xf = pvcost(1:4);
Q  = diag(pvcost(5:8));                % State Weight Matrix
R  = diag(pvcost(9:10));    % Control Weight Matrix   
uf = zeros(2,1);

% Calculate Deviations
dx = x(:) - xf(:);
du = u(:) - uf(:);
    
% Quadratic Cost Function
L = dx' * Q * dx + du' * R * du;
end