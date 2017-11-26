function [val_data, val_label, train_data, train_label] = data_layer(val_size, cls_num)
%%
assert(cls_num == 3);
[data_1, data_2, data_3] = get_data();
[data_num, data_dim] = size(data_1);
%%
assert(data_num > val_size);
val_data = zeros(cls_num*val_size, data_dim); % data(1:3) 
val_label = zeros(cls_num*val_size, cls_num); % label(1:3)
train_data = zeros(cls_num*(data_num-val_size), data_dim); % data(1:3)
train_label = zeros(cls_num*(data_num-val_size), cls_num); % label(1:3)
%%
val_data(1:val_size,1:data_dim) = data_1(1:val_size,:);
val_label(1:val_size,1) = 1;
val_label(1:val_size,2) = 0;
val_label(1:val_size,3) = 0;
val_data(val_size+1:2*val_size,1:data_dim) = data_2(1:val_size,:);
val_label(val_size+1:2*val_size,1) = 0;
val_label(val_size+1:2*val_size,2) = 1;
val_label(val_size+1:2*val_size,3) = 0;
val_data(2*val_size+1:3*val_size,1:data_dim) = data_3(1:val_size,:);
val_label(2*val_size+1:3*val_size,1) = 0;
val_label(2*val_size+1:3*val_size,2) = 0;
val_label(2*val_size+1:3*val_size,3) = 1;
%%
train_data(1:data_num-val_size,1:data_dim) = data_1(val_size+1:end,:);
train_label(1:data_num-val_size,1) = 1;
train_label(1:data_num-val_size,2) = 0;
train_label(1:data_num-val_size,3) = 0;
train_data(data_num-val_size+1:2*(data_num-val_size),1:data_dim) = data_2(val_size+1:end,:);
train_label(data_num-val_size+1:2*(data_num-val_size),1) = 0;
train_label(data_num-val_size+1:2*(data_num-val_size),2) = 1;
train_label(data_num-val_size+1:2*(data_num-val_size),3) = 0;
train_data(2*(data_num-val_size)+1:3*(data_num-val_size),1:data_dim) = data_3(val_size+1:end,:);
train_label(2*(data_num-val_size)+1:3*(data_num-val_size),1) = 0;
train_label(2*(data_num-val_size)+1:3*(data_num-val_size),2) = 0;
train_label(2*(data_num-val_size)+1:3*(data_num-val_size),3) = 1;
%% shuffle
order = randperm(cls_num*val_size);
val_data = val_data(order, :);
val_label = val_label(order, :);
order = randperm(3*(data_num-val_size));
train_data = train_data(order, :);
train_label = train_label(order, :);
end