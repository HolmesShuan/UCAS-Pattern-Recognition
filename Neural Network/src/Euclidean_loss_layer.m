function [loss, bottom_diff] = Euclidean_loss_layer(bottom_data, label)
    %% forward
    N = length(bottom_data);
    tmpvector = bottom_data-label';
    loss = (tmpvector*tmpvector')/2;
    loss = loss/N;
    %% backward
    bottom_diff = bottom_data - label';
end