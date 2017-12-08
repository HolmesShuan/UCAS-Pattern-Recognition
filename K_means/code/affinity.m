function w = affinity(x, y, sigma)
    w = exp(-(x-y)*(x-y)'/(2*sigma^2));
end