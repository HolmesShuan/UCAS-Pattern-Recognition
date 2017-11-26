%%
% Info: BP Algorithm
% Auth: Shuan
%%
clc;
clear;
close all;
%% network init
parameter_layer_number = 3;
hidden_nodes_number = [3,7,7,3]; % data_layer layer_1 layer_2 layer_3
assert(length(hidden_nodes_number) == parameter_layer_number+1);
%% solver init
max_iter = 5000;
show_iter = 20;
test_iter = 100;
base_lr = 0.0005;
weight_decay = 0.00005;
batch_size = 1;
lr_type = 'poly';
lr_list = get_lr(max_iter, base_lr, lr_type);
train_loss = zeros(max_iter,1);
train_acc = zeros(max_iter,1);
test_acc = zeros(max_iter/test_iter,1);
test_loss = zeros(max_iter/test_iter,1);
%% data init
val_size = 2; % validation dataset size = val_size*cls_num
cls_num = 3;
[val_data, val_label, train_data, train_label] = data_layer(val_size, cls_num);
%% weights init
for i = 1:parameter_layer_number
    W = weights_init(hidden_nodes_number(i), hidden_nodes_number(i+1));
    eval(['W',num2str(i),'=','W',';']);
end
%% Trainig 
for iter = 1:1:max_iter
    %% Test 
    if mod(iter, test_iter) == 0
        batch_data = val_data(:,:)';
        batch_label = val_label(:,:)';
        [top_data_1, ~, ~] = innerproduct_layer(W1, batch_data, [], 'forward');
        [top_data_1, ~] = tanh_layer(top_data_1, [], 'forward');
        [top_data_2, ~, ~] = innerproduct_layer(W2, top_data_1, [], 'forward');
        [top_data_2, ~] = tanh_layer(top_data_2, [], 'forward');
        [top_data_3, ~, ~] = innerproduct_layer(W3, top_data_2, [], 'forward');
        [top_data_3, ~] = sigmoid_layer(top_data_3, [], 'forward');
        [test_loss(iter/test_iter), ~] = Euclidean_loss_layer(top_data_3, batch_label);
        [test_acc(iter/test_iter)] = accuracy_layer(top_data_3, batch_label);
        fprintf('Loss : %f TEST Acc : %f\n', test_loss(iter/test_iter), test_acc(iter/test_iter));
    end
    sel_id = randperm(size(train_data,1), batch_size); % randomly select samples
    batch_data = train_data(sel_id,:)';
    batch_label = train_label(sel_id,:)';
%% Forward Propagation
    [top_data_1, ~, ~] = innerproduct_layer(W1, batch_data, [], 'forward');
    [top_data_1, ~] = tanh_layer(top_data_1, [], 'forward');
    [top_data_2, ~, ~] = innerproduct_layer(W2, top_data_1, [], 'forward');
    [top_data_2, ~] = tanh_layer(top_data_2, [], 'forward');
    [top_data_3, ~, ~] = innerproduct_layer(W3, top_data_2, [], 'forward');
    [top_data_3, ~] = sigmoid_layer(top_data_3, [], 'forward');
    [train_loss(iter), bottom_diff_1] = Euclidean_loss_layer(top_data_3, batch_label);
    [train_acc(iter)] = accuracy_layer(top_data_3, batch_label);
    if mod(iter, show_iter) == 0
        fprintf('iter : %d  Loss : %f TRAIN Accuracy : %f\n', iter, train_loss(iter), train_acc(iter));
    end
%% Backward Propagation
    [~, bottom_diff_1] = sigmoid_layer(top_data_3, bottom_diff_1, 'backward');
    [~, bottom_diff_2, weights_diff_1] = innerproduct_layer(W3, top_data_2, bottom_diff_1, 'backward');
    [~, bottom_diff_2] = tanh_layer(top_data_2, bottom_diff_2, 'backward');
    [~, bottom_diff_3, weights_diff_2] = innerproduct_layer(W2, top_data_1, bottom_diff_2, 'backward');
    [~, bottom_diff_3] = tanh_layer(top_data_1, bottom_diff_3, 'backward');
    [~, bottom_diff_4, weights_diff_3] = innerproduct_layer(W1, batch_data, bottom_diff_3, 'backward');
%% Parameter Update
    lr = lr_list(iter);
    W3 = W3 - lr*weights_diff_1 - weight_decay*W3/2;
    W2 = W2 - lr*weights_diff_2 - weight_decay*W2/2;
    W1 = W1 - lr*weights_diff_3 - weight_decay*W1/2;
end
disp(['Strategy : ' lr_type]);
%% Figure
% figure(1);
% scatter3(train_data(:,1),train_data(:,2),train_data(:,3),'b*');
%% save
save(['./test_acc_' lr_type '_batchsize_' num2str(batch_size) '.mat'], 'test_acc');
save(['./test_loss_' lr_type '_batchsize_' num2str(batch_size) '.mat'], 'test_loss');
save(['./train_acc_' lr_type '_batchsize_' num2str(batch_size) '.mat'], 'train_acc');
save(['./train_loss_' lr_type '_batchsize_' num2str(batch_size) '.mat'], 'train_loss');





