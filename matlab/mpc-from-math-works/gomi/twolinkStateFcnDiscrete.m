function xNext = twolinkStateFcnDiscrete(xk, uk, pvstate, Ts)
    % 例: オイラー離散化 (雑な方法ですが分かりやすい):
    % x(k+1) = x(k) + Ts * f( x(k), u(k) )
    f = twolinkStateFcn(xk, uk, pvstate);   % 連続時間の xdot
    xNext = xk + Ts * f;
end