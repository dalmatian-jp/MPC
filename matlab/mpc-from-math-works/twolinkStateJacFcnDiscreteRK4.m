function dfdx_d = twolinkStateJacFcnDiscreteRK4(x, u, params, Ts)
    % --- k1の計算 ---
    k1 = twolinkStateFcn(x, u, params);                % dx/dt at (x, u)
    A_k1 = twolinkStateJacFcn(x, u, params);          % ∂f/∂x at (x, u)

    % --- k2の計算 ---
    x_k2 = x + 0.5 * Ts * k1;                         % 中間状態
    k2 = twolinkStateFcn(x_k2, u, params);           % dx/dt at (x + Ts/2 * k1, u)
    A_k2 = twolinkStateJacFcn(x_k2, u, params);      % ∂f/∂x at (x + Ts/2 * k1, u)

    % --- k3の計算 ---
    x_k3 = x + 0.5 * Ts * k2;                         % 中間状態
    k3 = twolinkStateFcn(x_k3, u, params);           % dx/dt at (x + Ts/2 * k2, u)
    A_k3 = twolinkStateJacFcn(x_k3, u, params);      % ∂f/∂x at (x + Ts/2 * k2, u)

    % --- k4の計算 ---
    x_k4 = x + Ts * k3;                               % 最終状態
    A_k4 = twolinkStateJacFcn(x_k4, u, params);      % ∂f/∂x at (x + Ts * k3, u)

    % --- RK4に対応したヤコビアンの計算 ---
    dfdx_d = eye(length(x)) + (Ts / 6) * (A_k1 + 2 * A_k2 + 2 * A_k3 + A_k4);
end
