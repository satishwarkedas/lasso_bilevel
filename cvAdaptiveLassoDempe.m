%% CV Adaptive Lasso eps-approx method
function [optimalLambda, totalTime] = cvAdaptiveLassoDempe(data, folds, max_iter)

p = size(data, 2)-2;                 % no. of parameters/variables
n_obs = size(data,1);                % no. of observations in the complete data
n_test = floor(n_obs/folds);         % no. of observations in the test data
index = 1:n_obs;

% Need to maintain a running vector of phi, ul, ll, mew_vectors
ul = zeros(max_iter+1,1);
ll = zeros(max_iter+1,p+1,folds);
phi = zeros(max_iter+1,folds);
time = zeros(max_iter-1,1);
obj_val = zeros(max_iter+1,1);
mew_mat = zeros(max_iter, max_iter, folds);

ul(1,:) = 0.01;                     % first initial value of Lambda
ul(2,:) = 10;                       % second initial value of Lambda

test_indices = zeros(folds, n_test);
train_indices = zeros(folds, n_obs-n_test);
for i=1:folds
    test_indices(i,:) = 1+(i-1)*n_test:n_test+(i-1)*n_test;
    train_indices(i,:) = setdiff(index, test_indices(i,:));
    tStart1 = tic;
    [ll(1,:,i), phi(1,i)] = CVAdaptiveLasso1(ul(1,:), data, train_indices(i,:));
    [ll(2,:,i), phi(2,i)] = CVAdaptiveLasso1(ul(2,:), data, train_indices(i,:));
    tEnd1 = toc(tStart1);
end


for iter=2:max_iter
    tStart2 = tic;
    for i=1:folds
        [ll(iter,:,i), phi(iter,i)] = CVAdaptiveLasso1(ul(iter,:), data, train_indices(i,:));
    end
    tEnd2 = toc(tStart2);
    [ul(iter+1,:), ll(iter+1,:,:), mew_mat(iter,1:iter,:), obj_val(iter+1,:), time(iter-1,:)] = CVAdaptiveLassoBilevelDempe(ul, ll, phi, data, iter, test_indices, train_indices);
    time(iter-1,:) = time(iter-1, :) + tEnd2;
end

optimalLambda = ul(max_iter+1,:);
totalTime = sum(time)+tEnd1;

end