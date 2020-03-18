%% Creating the Dataset
% Dataset parameters
n          = 500;
p          = 50;
s          = 10;
beta_type  = 1;
rho        = 0;           % correlation levels - 0, 0.35 and 0.7
mew        = 5000;        % controls for the SNR level in the Y
folds      = 2;           % K value in K-fold cross validation    
max_iter   = 20;          % controls the number of iterations


global data
[data, Beta0] = data_generate_new(n, p, s, beta_type, rho, mew);
% current data_generate_new -> has 20% variance in Y, 20% variance in X
% 

%% CV Lasso eps-approx method

% Need to maintain a running vector of phi, ul, ll, mew_vectors
ul = zeros(max_iter+1,1);
ll = zeros(max_iter+1,p+1,folds);
phi = zeros(max_iter+1,folds);
time = zeros(max_iter-1,1);
obj_val = zeros(max_iter+1,1);
mew_mat = zeros(max_iter, max_iter, folds);

ul(1,:) = 0.01;
ul(2,:) = 10;

n_obs = size(data,1);                % no. of observations in the complete data
n_test = floor(n_obs/folds);         % no. of observations in the test data
index = 1:n_obs;
test_indices = zeros(folds, n_test);
train_indices = zeros(folds, n_obs-n_test);
for i=1:folds
    test_indices(i,:) = 1+(i-1)*n_test:n_test+(i-1)*n_test;
    train_indices(i,:) = setdiff(index, test_indices(i,:));
    [ll(1,:,i), phi(1,i)] = CVAdaptiveLasso1(ul(1,:), data, train_indices(i,:));
    [ll(2,:,i), phi(2,i)] = CVAdaptiveLasso1(ul(2,:), data, train_indices(i,:));
end


for iter=2:max_iter
    for i=1:folds
        [ll(iter,:,i), phi(iter,i)] = CVAdaptiveLasso1(ul(iter,:), data, train_indices(i,:));
    end
    [ul(iter+1,:), ll(iter+1,:,:), mew_mat(iter,1:iter,:), obj_val(iter+1,:), time(iter-1,:)] = CVAdaptiveLassoBilevelDempe(ul, ll, phi, data, iter, test_indices, train_indices);
end

Lambda(:,3) = ul(max_iter+1,:);
Beta(:,3) = ll(max_iter+1,2:end)';
%     optimalObjval_dempe = obj_val(max_iter+1,:);  
Time(:,3) = sum(time);