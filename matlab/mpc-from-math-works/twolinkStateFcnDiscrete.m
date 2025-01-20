function XOUT = twolinkStateFcnDiscrete(xk, uMPC, pvstate, Ts, dist)
    %options = odeset('Jacobian', @(t, x) twolinkStateJacFcnDiscrete(x, uMPC, pvstate, Ts), ...
       %              'RelTol', 1e-4, 'AbsTol', 1e-6);
    ODEFUN = @(t, x) twolinkStateFcn(x, uMPC, pvstate);
    [~, XOUT] = ode23t(ODEFUN, [0 Ts], xk);
    XOUT = XOUT(end, :) + dist;
end
