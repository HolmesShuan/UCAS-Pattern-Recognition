function idx = Normalized_Spectral_Clustering(W, k)
    d = sum(W,2);
    D = diag(d);
    L = D - W;
    d = d.^(-0.5);
    D = diag(d);
    L = D*L*D + 0.01*eye(size(W,1)); % avoid singular
    [V, ~] = eigs(L,k,'sm');
    N = size(V,1);
    for i = 1 : N
        V(i,:) = V(i,:) / norm(V(i,:)) ;
    end
    idx = kmeans(V, k);
end