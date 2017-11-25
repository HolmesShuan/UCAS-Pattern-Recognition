function acc = accuracy_layer(bottom_data, label)
    %% forward
    N = length(bottom_data);
    tmpvector = round(bottom_data');
    tmpvector = tmpvector-label;
    acc = length(find(tmpvector==0))/N;
end