function xdot = twolinkStateFcnWithDisturbance(x,u,pvstate,distStd)
    % twolinkStateFcnの出力に、外乱(ガウス白色雑音)を足す例
    % 「モデル外乱」として xdot に直接加えているが、
    %  場合によってはトルクや外力に加える方法もある
    xdot_nominal = twolinkStateFcn(x,u,pvstate);
    xdot_dist = xdot_nominal + distStd * randn(size(x)); % 同次元分足す
    xdot = xdot_dist;
end