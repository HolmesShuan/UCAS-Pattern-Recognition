%%
% EM : Expectation Maximization
% Auth : Shuan
% Env : Matlab2016b or higher
%%
clc;
clear;
close all;
warning off all;
%% init setting
y = textread('./data/data.txt');
N = size(y, 1);
Dim = size(y, 2); % feature vector dimension
K = 2; % submodel number
mu = zeros(Dim, K);
mu(1,1) = 0;
mu(2,1) = 2;
mu(1,2) = 4;
mu(2,2) = 0;
y = y';
Sigma = zeros(Dim, Dim, K);
alpha = ones(K,1)./K;
iter_max = 30;
for i = 1:K
    Sigma(:,:,i) = eye(Dim)*3;
end
%%
for iter = 1:iter_max
    for k = 1:K
        denominator = 0;
        for j = 1:N
            denominator = denominator + E_gamma_jk( y, j, k, mu, Sigma, alpha);
        end
        % Update Sigma
        Sigma_tmp = 0;
        for j = 1:N
            Sigma_tmp = Sigma_tmp + (y(:,j)-mu(:,k))*(y(:,j)-mu(:,k))'*E_gamma_jk( y, j, k, mu, Sigma, alpha);
        end
        Sigma(:,:,k) = Sigma_tmp / denominator;
        % Update mu
        mu_tmp = 0;
        for j = 1:N
            mu_tmp = mu_tmp + y(:,j)*E_gamma_jk( y, j, k, mu, Sigma, alpha);
        end
        mu(:,k) = mu_tmp / denominator;
        % Update alpha
        alpha(k) = denominator / N;
    end
end
%%
figure(1);
scatter(y(1,:), y(2,:), 'b');
t = title('Observed Distribution');
t.FontSize = 15;
%%
figure(2);
%rng default  % For reproducibility
n1 = N*alpha(1);
r = mvnrnd(mu(:,1)', Sigma(:,:,1), fix(n1));
scatter(r(:,1), r(:,2), 'b');
axis([-6 14 -4 12])
hold on;
n2 = N*alpha(2);
r = mvnrnd(mu(:,2)', Sigma(:,:,2), fix(n2));
scatter(r(:,1), r(:,2), 'r');
axis([-6 14 -4 12])
t = title(['Distribution Estimation ( iter = ' num2str(iter_max) ' )']);
t.FontSize = 15;

