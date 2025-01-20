function xNext = twolinkStateFcnRK4(x, u, params, Ts)
    k1 = twolinkStateFcn(x, u, params);
    k2 = twolinkStateFcn(x + 0.5 * Ts * k1, u, params);
    k3 = twolinkStateFcn(x + 0.5 * Ts * k2, u, params);
    k4 = twolinkStateFcn(x + Ts * k3, u, params);

    xNext = x + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
end