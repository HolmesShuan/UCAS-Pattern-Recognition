%%
% Info: Bayesian Decision
% Auth: Shuan
% Env : MatLab2016b and upper
%%
clc;
clear;
%%
load('class_1.mat');
load('class_2.mat');
load('test_class_1.mat');
load('test_class_2.mat');
pca_dimension = 292;
M = size(class_1_matrix,1);
N = size(class_1_matrix,2);
Original_Feature_Space = zeros(M*2,N);
P_w = 0.5;
cases = 3; % Quadrastic
%% Zero mean
Original_Feature_Space(1:M,:) = class_1_matrix;
Original_Feature_Space(M+1:end,:) = class_2_matrix;
mean_vector = mean(Original_Feature_Space);
Original_Feature_Space = bsxfun(@minus, Original_Feature_Space, mean_vector);
%% PCA Feature dimension reduction
Cov_Matrix = cov(Original_Feature_Space);
[P, Lambda] = eig(Cov_Matrix);
P = P(:,1:pca_dimension);
New_Feature_Space = Original_Feature_Space*P;
%% ZCA Whiten, unit variance
% Lambda = Lambda(1:pca_dimension);
% P = bsxfun(@rdivide, P, sqrt(Lambda));
%% Create Model
class_1_matrix = New_Feature_Space(1:M,:);
class_2_matrix = New_Feature_Space(M+1:end,:);
[W_1,w_1,w_01] = decision_function(class_1_matrix, P_w, cases);
[W_2,w_2,w_02] = decision_function(class_2_matrix, P_w, cases);
%% Quadrastic Decision
X = test_class_1_matrix(:,:);
m = size(X,1);
g_1 = zeros(m);
g_2 = zeros(m);
for i = 1:m
    x = X(i,:);
    x = x*P;
    x = x';
    g_1(i) = QDF(x ,W_1, w_1, w_01);
    g_2(i) = QDF(x ,W_2, w_2, w_02);
end
Results = find(g_1>g_2);
Acc_1 = length(Results) / size(test_class_1_matrix,1);
fprintf('Class 1 ACC : %f.\n', Acc_1);
%%
X = test_class_2_matrix(:,:);
m = size(X,1);
g_1 = zeros(m);
g_2 = zeros(m);
for i = 1:m
    x = X(i,:);
    x = x*P;
    x = x';
    g_1(i) = QDF(x ,W_1, w_1, w_01);
    g_2(i) = QDF(x ,W_2, w_2, w_02);
end
Results = find(g_1<g_2);
Acc_2 = length(Results) / size(test_class_2_matrix,1);
fprintf('Class 2 ACC : %f.\n', Acc_2);
fprintf('Mean ACC : %f.\n', (Acc_2+Acc_1)/2);


