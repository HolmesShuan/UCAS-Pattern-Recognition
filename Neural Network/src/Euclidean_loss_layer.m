function [loss, bottom_diff] = Euclidean_loss_layer(bottom_data, label)
    %% forward
    [D,N] = size(bottom_data);
    unit_vector = ones(1,D);
    tmpMatrix = bottom_data-label;
    tmpMatrix = tmpMatrix.*tmpMatrix;
    loss = sum(unit_vector*tmpMatrix)/2;
    loss = loss/N;
    %% backward
    bottom_diff = bottom_data - label;
end