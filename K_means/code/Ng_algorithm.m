%%
clc;
clear;
close all;
%%
load('./data/data_2.mat');
N = size(X,1);
W = zeros(N,N);
k = 30;
sigma = 1;
%% construct W
tic;
for i = 1:N
    kNN = k_nn(X, X(i,:), k);
    for j = 1:length(kNN)
        W(i,kNN(j)) = affinity(X(i,:), X(kNN(j),:), sigma); 
    end
end
W = (W + W')/2;
%% Normalized_Spectral_Clustering
k = 2;
idx = Normalized_Spectral_Clustering(W, k);
toc;
cls_1 = idx(1:100);
cls_2 = idx(101:200);
correct_cls_1_num = length(find(cls_1 == mode(cls_1)));
if mode(cls_1) == mode(cls_2)
    cls_2 = find(cls_2 ~= mode(cls_2));
    correct_cls_2_num = length(find(cls_2 == mode(cls_2)));
else
    correct_cls_2_num = length(find(cls_2 == mode(cls_2)));
end
fprintf('ACC : %f\n', (correct_cls_1_num+correct_cls_2_num)/N);
%%
for i = 1:k
    idx_vector = find(idx==i);
    plot(X(idx_vector,1),X(idx_vector,2),'x', 'LineWidth', 2, 'Display', ['CLS : ' num2str(i)]);
    hold on;
end
legend('show');