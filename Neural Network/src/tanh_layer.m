function [top_data, bottom_diff] = tanh_layer(bottom_data, top_diff, state)
    switch(state)
        case 'forward'
            %% forward
            top_data = tanh(bottom_data);
            bottom_diff = [];
        case 'backward'
            %% backward
            bottom_diff = top_diff.*(1-bottom_data.^2);
            top_data = [];
        otherwise
            disp('WARNING : unknown layer statement.');
            top_data = bottom_data;
            bottom_diff = top_diff;
    end
end