function [top_data, bottom_diff] = sigmoid_layer(bottom_data, top_diff, state)
    switch(state)
        case 'forward'
            %% forward
            top_data = sigmoid(bottom_data);
            bottom_diff = [];
        case 'backward'
            %% backward
            bottom_diff = top_diff.*bottom_data.*(1-bottom_data);
            top_data = [];
        otherwise
            disp('WARNING : unknown layer statement.');
            top_data = bottom_data;
            bottom_diff = top_diff;
    end
end