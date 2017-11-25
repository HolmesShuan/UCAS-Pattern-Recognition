function weights = weights_init(input_size, output_size, init_type, scale)
    if nargin==2
        weights = randn(output_size, input_size);
        return;
    end
    if nargin==3
        if init_type == 'Fixed'
            weights = ones(output_size, input_size);
            return;
        end
    end
    switch(init_type)
        case 'Gaussian'
            weights = randn(output_size, input_size);
        case 'Uniform'
            weights = rand(output_size, input_size);
        case 'Fixed'
            weights = scale*ones(output_size, input_size);
        otherwise
            % - default Gaussian
            weights = randn(output_size, input_size);
    end
end