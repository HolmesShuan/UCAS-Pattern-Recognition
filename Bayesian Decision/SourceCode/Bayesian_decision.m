%%
% Info: Bayesian Decision
% Auth: Shuan
% Env : MatLab2016b and upper
%%
clc;
clear;
%% SETTING
P_w = 0.5;
cases = 1; % 2 : Quadrastic 1 : Linear
for pca_dimension = 32:32:1024
val_size = 500;
%%
load('../Data/class_1.mat');
load('../Data/class_2.mat');
load('../Data/test_class_1.mat');
load('../Data/test_class_2.mat');
val_cls1_matrix = class_1_matrix(1:val_size,:);
val_cls2_matrix = class_2_matrix(1:val_size,:);
class_1_matrix = class_1_matrix(val_size+1:end,:);
class_2_matrix = class_2_matrix(val_size+1:end,:);
%% Zero mean
M = size(class_1_matrix,1);
N = size(class_1_matrix,2);
Original_Feature_Space = zeros(M*2,N);
Original_Feature_Space(1:M,:) = class_1_matrix;
Original_Feature_Space(M+1:end,:) = class_2_matrix;
mean_vector = mean(Original_Feature_Space);
Original_Feature_Space = bsxfun(@minus, Original_Feature_Space, mean_vector);
%% PCA Feature dimension reduction
Cov_Matrix = cov(Original_Feature_Space);
[P, Lambda] = eig(Cov_Matrix);
P = P(:,1:pca_dimension);
%% ZCA Whiten, unit variance
Lambda = diag(Lambda(1:pca_dimension,:))';
P = bsxfun(@rdivide, P, sqrt(Lambda));
%% Create Model
New_Feature_Space = Original_Feature_Space*P;
class_1_matrix = New_Feature_Space(1:M,:);
class_2_matrix = New_Feature_Space(M+1:end,:);
[W_1,w_1,w_01] = decision_function(class_1_matrix, P_w, cases);
[W_2,w_2,w_02] = decision_function(class_2_matrix, P_w, cases);
%% VAL SET Quadrastic Decision
if cases == 2
X = val_cls1_matrix(:,:);
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
Acc_1 = length(Results) / size(val_cls1_matrix,1);
fprintf('VAL Quadrastic Class 1 ACC : %f.\n', Acc_1);
%% CLS 2
X = val_cls2_matrix(:,:);
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
Acc_2 = length(Results) / size(val_cls2_matrix,1);
fprintf('VAL Quadrastic Class 2 ACC : %f.\n', Acc_2);
fprintf('VAL Quadrastic Mean ACC : %f.\n', (Acc_2+Acc_1)/2);
%% VAL SET Linear Decision
else
X = val_cls1_matrix(:,:);
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
Acc_1 = length(Results) / size(val_cls1_matrix,1);
fprintf('VAL Linear Class 1 ACC : %f.\n', Acc_1);
%% CLS 2
X = val_cls2_matrix(:,:);
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
Acc_2 = length(Results) / size(val_cls2_matrix,1);
fprintf('VAL Linear Class 2 ACC : %f.\n', Acc_2);
fprintf('VAL Linear Mean ACC : %f.\n', (Acc_2+Acc_1)/2);
end
%% TEST SET Quadrastic Decision
if cases == 2
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
fprintf('TEST Quadrastic Class 1 ACC : %f.\n', Acc_1);
%% CLS 2
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
fprintf('TEST Quadrastic Class 2 ACC : %f.\n', Acc_2);
fprintf('TEST Quadrastic Mean ACC : %f.\n', (Acc_2+Acc_1)/2);
%% TEST SET Linear Decision
else % decision option
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
fprintf('TEST Linear Class 1 ACC : %f.\n', Acc_1);
%% CLS 2
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
fprintf('TEST Linear Class 2 ACC : %f.\n', Acc_2);
fprintf('TEST Linear Mean ACC : %f.\n', (Acc_2+Acc_1)/2);
end % decision option
end % PCA DIM for 

