%%
% Info: Bayesian Decision
% Auth: Shuan
% Env : MatLab2016b and upper
%%
% clc;
clear;
%%
load('../Data/class_1.mat');
load('../Data/class_2.mat');
load('../Data/test_class_1.mat');
load('../Data/test_class_2.mat');
pca_dimension = 1000;
M = size(class_1_matrix,1);
N = size(class_1_matrix,2);
Original_Feature_Space = zeros(M*2,N);
P_w = 0.5;
cases = 1; % 2 : Quadrastic 1 : Linear
%% Zero mean
Original_Feature_Space(1:M,:) = class_1_matrix;
Original_Feature_Space(M+1:end,:) = class_2_matrix;
mean_vector = mean(Original_Feature_Space);
Original_Feature_Space = bsxfun(@minus, Original_Feature_Space, mean_vector);
%% PCA Feature dimension reduction
Cov_Matrix = cov(Original_Feature_Space);
[P, Lambda] = eig(Cov_Matrix);
P = P(:,1:pca_dimension);
%% ZCA Whiten, unit variance
% Lambda = diag(Lambda(1:pca_dimension,:))';
% P = bsxfun(@rdivide, P, sqrt(Lambda));
%% Create Model
New_Feature_Space = Original_Feature_Space*P;
class_1_matrix = New_Feature_Space(1:M,:);
class_2_matrix = New_Feature_Space(M+1:end,:);
[W_1,w_1,w_01] = decision_function(class_1_matrix, P_w, cases);
[W_2,w_2,w_02] = decision_function(class_2_matrix, P_w, cases);
%% Quadrastic Decision
if cases == 1
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
fprintf('Quadrastic Class 1 ACC : %f.\n', Acc_1);
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
fprintf('Quadrastic Class 2 ACC : %f.\n', Acc_2);
fprintf('Quadrastic Mean ACC : %f.\n', (Acc_2+Acc_1)/2);
%% Linear Decision
else
X = test_class_1_matrix(:,:);
m = size(X,1);
g_1 = zeros(m);
g_2 = zeros(m);
for i = 1:m
    x = X(i,:);
    x = x*P;
    x = x';
    g_1(i) = LDF(x ,w_1, w_01);
    g_2(i) = LDF(x ,w_2, w_02);
end
Results = find(g_1>g_2);
Acc_1 = length(Results) / size(test_class_1_matrix,1);
fprintf('Linear Class 1 ACC : %f.\n', Acc_1);
%%
X = test_class_2_matrix(:,:);
m = size(X,1);
g_1 = zeros(m);
g_2 = zeros(m);
for i = 1:m
    x = X(i,:);
    x = x*P;
    x = x';
    g_1(i) = LDF(x ,w_1, w_01);
    g_2(i) = LDF(x ,w_2, w_02);
end
Results = find(g_1<g_2);
Acc_2 = length(Results) / size(test_class_2_matrix,1);
fprintf('Linear Class 2 ACC : %f.\n', Acc_2);
fprintf('Linear Mean ACC : %f.\n', (Acc_2+Acc_1)/2);
end
 


