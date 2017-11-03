function [index,err] = verify_2(X, a)
    Y = X*a;
    index = find(Y <= 0);
    err = length(index);
end