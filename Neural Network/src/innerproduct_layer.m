function [top_data, bottom_diff, weights_diff] = innerproduct_layer(weights, bottom_data, top_diff, state)
    switch(state)
        case 'forward'
        %% forward 
            top_data = weights*bottom_data;
            bottom_diff = [];
            weights_diff = [];
        case 'backward'
        %% backward
            weights_diff = top_diff*bottom_data';
            bottom_diff = weights'*top_diff;
            top_data = [];
        otherwise
            disp('WARNING : unknown layer statement.');
            top_data = bottom_data;
            bottom_diff = top_diff;
            weights_diff = [];
    end
end