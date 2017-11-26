function acc = accuracy_layer(bottom_data, label)
    %% forward
    N = size(bottom_data,2);
    tmpMatrix = bottom_data;
    tmpMatrix = bsxfun(@rdivide, tmpMatrix, max(bottom_data));
    tmpMatrix = floor(tmpMatrix);
    tmpMatrix = abs(tmpMatrix-label);
    ResultVector = sum(tmpMatrix);
    acc = length(find(ResultVector==0))/N;
end