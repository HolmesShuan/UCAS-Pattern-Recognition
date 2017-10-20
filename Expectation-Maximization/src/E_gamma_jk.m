function e_gamma_jk = E_gamma_jk( y, j, k, mu, Sigma, alpha)
    K = size(alpha, 1);
    denominator_E_gamma = 0;
    for i = 1:K
        denominator_E_gamma = denominator_E_gamma + alpha(i)*mvnpdf(y(:,j)',mu(:,i)',Sigma(:,:,i));
    end
    e_gamma_jk = alpha(k)*mvnpdf(y(:,j)',mu(:,k)',Sigma(:,:,k)) / denominator_E_gamma;
end
