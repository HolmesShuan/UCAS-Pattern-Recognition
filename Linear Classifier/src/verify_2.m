function [index,err] = verify_2(X, a, b)
    Y = X*a-b;
    index = find(abs(Y) <= 1e-12);
    err = sum(Y.^2);
end