function [Gx, Gmv] = twolinkCostJacFcn(stage, x, u, pvcost) %#codegen

    xf = pvcost(1:4);
    Q  = diag(pvcost(5:8));
    R  = diag(pvcost(9:10));
    uf = zeros(2,1);

    dx = x(:) - xf(:);
    du = u(:) - uf(:);

    % Gradients
    Gx = 2 * Q * dx;  % Gradient w.r.t State
    Gmv = 2 * R * du; % Gradient w.r.t Control Input
end
