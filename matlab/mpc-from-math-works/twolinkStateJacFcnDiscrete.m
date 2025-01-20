function dfdx_d = twolinkStateJacFcnDiscrete(x, u, params, Ts)
    % 連続時間のヤコビアンを取得
    dfdx_c = twolinkStateJacFcn(x, u, params);
    
    % 行列指数関数で離散化
    dfdx_d = expm(dfdx_c * Ts);
end