function [val_data, train_data] = data_layer(val_size, cls_num)
%%
assert(cls_num == 3);
[data_1, data_2, data_3] = get_data();
[data_num, data_dim] = size(data_1);
%%
assert(data_num > val_size);
val_data = zeros(cls_num*val_size, data_dim+1); % data(1:3) + label(4)
train_data = zeros(cls_num*(data_num-val_size), data_dim+1); % data(1:3) + label(4)
%%
val_data(1:val_size,1:data_dim) = data_1(1:val_size,:);
val_data(1:val_size,data_dim+1) = 0;
val_data(val_size+1:2*val_size,1:data_dim) = data_2(1:val_size,:);
val_data(val_size+1:2*val_size,data_dim+1) = 1;
val_data(2*val_size+1:3*val_size,1:data_dim) = data_3(1:val_size,:);
val_data(2*val_size+1:3*val_size,data_dim+1) = 2;
%%
train_data(1:data_num-val_size,1:data_dim) = data_1(val_size+1:end,:);
train_data(1:data_num-val_size,data_dim+1) = 0;
train_data(data_num-val_size+1:2*(data_num-val_size),1:data_dim) = data_2(val_size+1:end,:);
train_data(data_num-val_size+1:2*(data_num-val_size),data_dim+1) = 1;
train_data(2*(data_num-val_size)+1:3*(data_num-val_size),1:data_dim) = data_3(val_size+1:end,:);
train_data(2*(data_num-val_size)+1:3*(data_num-val_size),data_dim+1) = 2;
%%
order = randperm(cls_num*val_size);
val_data = val_data(order, :);
order = randperm(3*(data_num-val_size));
train_data = train_data(order, :);
end