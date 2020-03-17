% Function to create datasets
% n -> indicates the number of observations, examples, rows
% p -> indicates the number of variables, predictors
% y_func -> 1 to 5 indicates the choice of the function
% output -> n x (p+2) 

% Beta0 -> vector of the coefficients based on 
% s -> sparsity level
% beta_type -> type of sparsity
function [data, Beta0] = data_generate(n, p, s, beta_type, rho, mew)
    data = zeros(2*n,p+2);          % +1 for the intercept = 1
    data(:,1) = 1;                  % defining the intercept as 1
    Beta0 = beta_generate(p, s, beta_type);
    [x y] = meshgrid(1:p, 1:p);
    sigma = rho.^(abs(x-y));
    mu_X = zeros(1,p);
    X = mvnrnd(mu_X, sigma, 2*n);
    
%     mew = (Beta0'*sigma*Beta0)/snr_level;
%     disp(Beta0'*sigma*Beta0);
%     mew = (Beta0'*sigma*Beta0)*5;
%     disp(mew);
    variance_const = (Beta0'*sigma*Beta0)/mew;
%     variance_const = snr_level;
    
    mu_Y = X*Beta0;
    Y = normrnd(mu_Y, variance_const);
    data(:,2:p+1) = X;
    data(:,p+2) = Y;
end

% function [X, Y] = var_generate(n, p, Beta0)
%     [x y] = meshgrid(1:p, 1:p);
%     sigma = rho.^(abs(x-y));
%     mu = zeros(1,p);
%     X = mvnrnd(mu, sigma, n);
% end

% function Y = y_generate(n, p, X, Beta0, mew)
%     variance_const = Beta0'*sigma;
% end

function Beta0 = beta_generate(p, s, beta_type)
    Beta0 = zeros(p,1);
    if (beta_type == 0)
        Beta0 = 1*rand(p,1);
    elseif (beta_type == 1)
        dist = floor(p/s);
%         disp(dist);
        for i=1:p
            if mod(i,dist) == 0
                Beta0(i,1) = 1;
            end
        end
    elseif (beta_type == 2)
        for i=1:s
            Beta0(i,1) = 1;
        end
    elseif (beta_type == 3)
        for i=1:s
            Beta0(i,1) = 10 - (9.5/(s-1))*(i-1);
        end
    elseif (beta_type == 5)
        Beta0(1:s,1) = 1;
        for i=s+1:p
            Beta0(i,1) = 0.5^(i-s);
        end
    end
end
        
            
            
            
            