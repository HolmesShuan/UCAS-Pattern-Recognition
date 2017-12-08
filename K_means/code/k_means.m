%%
clc;
clear;
close all;
%%
load './data/data.mat'
%%
mu1 = [1, -1];
mu2 = [5.5, -4.5];
mu3 = [1, 4];
mu4 = [6, 4.5];
mu5 = [9, 0.0];
mu = [mu1;mu2;mu3;mu4;mu5];
%%
max_val = max(X);
min_val = min(X);
cls = 5;
dim = 2;
%%
max_iter = 25;
random_test = 25;
cls_samples_num = zeros(max_iter, cls);
loss = zeros(random_test, max_iter, 1);
for iter = 1:random_test % random test
center_points = zeros(cls, dim);
center_points(1,:) = min_val + max_val.*rand(1, dim);
center_points(2,:) = min_val + max_val.*rand(1, dim);
center_points(3,:) = min_val + max_val.*rand(1, dim);
center_points(4,:) = min_val + max_val.*rand(1, dim);
center_points(5,:) = min_val + max_val.*rand(1, dim);
res_matrix = zeros(cls, size(X,1), size(X,2));
%%
for i = 1:max_iter % iteration number
    for k = 1:cls 
        res_matrix(k,:,:) = bsxfun(@minus, X, center_points(k,:));
    end
    res_matrix = res_matrix.^2;
    distance_matrix = sum(res_matrix,3);
    [row, col] = find(distance_matrix==min(distance_matrix));
    for k = 1:cls 
        candidate_samples = find(row==k);
        if isempty(candidate_samples)
            cls_samples_num(i, k) = length(candidate_samples);
            continue;
        end
        tmp_matrix = X(candidate_samples,:);
        center_points(k,:) = mean(tmp_matrix);
        cls_samples_num(i, k) = length(candidate_samples);
    end
    for k = 1:cls 
        tmp_loss = bsxfun(@minus, mu, center_points(k,:));
        tmp_loss = min(tmp_loss.^2);
        loss(iter, i) = loss(iter, i) + sum(tmp_loss)/cls;
    end
end
%%
end
%% plot
figure();
for i = 1:random_test
    plot(loss(i, :), 'x-', 'LineWidth', 1.8);
    hold on;
end
title('Random Start Ponits Comparsion.');
xlabel('Iter');
ylabel('Mean Euclidean Loss');
set(gca,'Fontsize',15)
%%
% cls samples number
figure();
for i = 1:cls
    plot(cls_samples_num(:, i), '-x', 'LineWidth', 1.8, 'DisplayName',['cls: ' num2str(i)]);
    hold on;
end
legend('show')
title('Class Sample Number Comparsion.');
xlabel('Iter');
ylabel('Number');
set(gca,'Fontsize',15)
%%
% Show the data point
figure();
for k = 1:cls 
    res_matrix(k,:,:) = bsxfun(@minus, X, center_points(k,:));
end
res_matrix = res_matrix.^2;
distance_matrix = sum(res_matrix,3);
[row, col] = find(distance_matrix==min(distance_matrix));
for k = 1:cls 
    candidate_samples = find(row==k);
    if isempty(candidate_samples)
        cls_samples_num(i, k) = length(candidate_samples);
        continue;
    end
    tmp_matrix = X(candidate_samples,:);
    plot(tmp_matrix(:,1), tmp_matrix(:,2), '.'); 
    hold on;
end
