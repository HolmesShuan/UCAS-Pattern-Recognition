%%
% Info: Bayesian Decision
% Auth: Shuan
% Env : MatLab2016b and upper
%%
function [W,w,w_0] = decision_function(X, P_w, cases)
    if cases == 1
        Cov_Matrix = cov(X);
        mean_vector = mean(X);
        mean_vector = mean_vector';
        Sigma_inv = inv(Cov_Matrix);
        w = Sigma_inv*mean_vector;
        w_0 = mean_vector'*Sigma_inv*mean_vector/-2 + log(P_w); 
        W = 0;
    else
        Cov_Matrix = cov(X);
        mean_vector = mean(X);
        mean_vector = mean_vector';
        Sigma_inv = inv(Cov_Matrix);
        W = Sigma_inv/-2;
        w = Sigma_inv*mean_vector;
        w_0 = mean_vector'*Sigma_inv*mean_vector/-2 + log(det(Cov_Matrix))/-2 + log(P_w);
    end
end