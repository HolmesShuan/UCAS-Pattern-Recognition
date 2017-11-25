function lr = get_lr(max_iter, base_lr, lr_type)
%% parameter init
iter=1:max_iter;
gamma=0.0001;
power=0.75;
step_size=max_iter/3;
%%
switch(lr_type)
    case 'fixed'
        % - fixed: always return base_lr.
        lr = base_lr*ones(1,max_iter);
    case 'step'
        % - step: return base_lr * gamma ^ (floor(iter / step))
        lr = base_lr .* gamma.^(floor(iter./step_size));
    case 'exp'
        % - exp: return base_lr * gamma ^ iter
        lr = base_lr * gamma .^ iter;
    case 'inv'
        % - inv: return base_lr * (1 + gamma * iter) ^ (- power)
        lr = base_lr.*(1./(1+gamma.*iter).^power);
    case 'poly'
        % - poly: the effective learning rate follows a polynomial decay, to be
        % zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
        lr = base_lr *(1 - iter./max_iter) .^ (power);
    otherwise
        % - default : fixed 
        lr = base_lr*ones(1,max_iter);
end
end