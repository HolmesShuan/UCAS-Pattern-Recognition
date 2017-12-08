function knn_loc = k_nn(X, x, k)
   [n, dim] = size(X);
   assert(dim == size(x,2)); % same feature dimension
   assert(k<=n);
   distance_matrix = bsxfun(@minus, X, x);
   distance_matrix = distance_matrix.^2;
   distance_vector = sum(distance_matrix,2);
   sorted_vector = sort(distance_vector);
   knn_loc = zeros(k,1);
   for i = 1:k
    candidate = find(distance_vector == sorted_vector(i+1));
    assert(length(candidate) == 1);
    knn_loc(i) = candidate;
   end
end