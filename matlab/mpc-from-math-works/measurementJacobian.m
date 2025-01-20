function H = measurementJacobian(x)
    % z = x の場合、dz/dx = I
    nx = length(x);
    H = eye(nx);
end