%%
% Info: Read raw data
% Auth: Shuan
% Env : MatLab2016b and upper
%%
clc;
clear;
%%
catagories = [3,7];
Width = 32;
Height = 32;
raw_data_path = './';
raw_data = {...
     [raw_data_path 'data_batch_1.mat'],...
     [raw_data_path 'data_batch_2.mat'],...
     [raw_data_path 'data_batch_3.mat'],...
     [raw_data_path 'data_batch_4.mat'],...
     [raw_data_path 'data_batch_5.mat'],...
     [raw_data_path 'test_batch.mat']
     };
class_1_matrix = zeros(5000, 3072);
class_2_matrix = zeros(5000, 3072);
test_class_1_matrix = zeros(1000, 3072);
test_class_2_matrix = zeros(1000, 3072);
index_class_1 = 1;
index_class_2 = 1;
%%
for i = 1:length(raw_data)-1
    load(raw_data{i});
    if ismember(catagories(1), labels)
        index = find(labels == catagories(1));
        for j = 1:length(index)
            class_1_matrix(index_class_1,:) = data(index(j),:);
            index_class_1 = index_class_1 + 1;
        end
    end
    if ismember(catagories(2), labels)
        index = find(labels == catagories(2));
        for j = 1:length(index)
            class_2_matrix(index_class_2,:) = data(index(j),:);
            index_class_2 = index_class_2 + 1;
        end
    end
end
%%
load(raw_data{length(raw_data)});
index = find(labels == catagories(1));
for j = 1:length(index)
    test_class_1_matrix(j,:) = data(index(j),:);
end
index = find(labels == catagories(2));
for j = 1:length(index)
    test_class_2_matrix(j,:) = data(index(j),:);
end
%% Selected Features : Y domain in YCBCR Space
class_1_matrix = RGB2YCBCR(class_1_matrix, Width, Height);
class_2_matrix = RGB2YCBCR(class_2_matrix, Width, Height);
test_class_1_matrix = RGB2YCBCR(test_class_1_matrix, Width, Height);
test_class_2_matrix = RGB2YCBCR(test_class_2_matrix, Width, Height);
%%
save('class_1.mat', 'class_1_matrix');
save('class_2.mat', 'class_2_matrix');
save('test_class_1.mat', 'test_class_1_matrix');
save('test_class_2.mat', 'test_class_2_matrix');

