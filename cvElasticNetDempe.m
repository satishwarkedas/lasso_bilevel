%% CV Elastic Net eps-approx method
function [optimalLambda, optimalAlpha, optimalObjVal, totalTime] = cvElasticNetDempe(data, folds, max_iter)
    p = size(data, 2)-2;                 % no. of parameters/variables
    n_obs = size(data,1);                % no. of observations in the complete data
    n_test = floor(n_obs/folds);         % no. of observations in the test data
    index = 1:n_obs;

    % Need to maintain a running vector of phi, ul, ll, mew_vectors
    ul = zeros(max_iter+1,2);
    ll = zeros(max_iter+1,p+1,folds);
    phi = zeros(max_iter+1,folds);
    time = zeros(max_iter-1,1);
    obj_val = zeros(max_iter+1,1);
    mew_mat = zeros(max_iter, max_iter, folds);

    ul(1,:) = [100, 0];                     % first initial value of Lambda
    ul(2,:) = [0, 1];                       % second initial value of Lambda

    test_indices = zeros(folds, n_test);
    train_indices = zeros(folds, n_obs-n_test);
    for i=1:folds
        test_indices(i,:) = 1+(i-1)*n_test:n_test+(i-1)*n_test;
        train_indices(i,:) = setdiff(index, test_indices(i,:));
        tStart1 = tic;
        [ll(1,:,i), phi(1,i)] = CVElasticNet1(ul(1,:), data, train_indices(i,:));
        [ll(2,:,i), phi(2,i)] = CVElasticNet1(ul(2,:), data, train_indices(i,:));
        tEnd1 = toc(tStart1);
    end


    for iter=2:max_iter
        tStart2 = tic;
        for i=1:folds
            [ll(iter,:,i), phi(iter,i)] = CVElasticNet1(ul(iter,:), data, train_indices(i,:));
        end
        tEnd2 = toc(tStart2);
        [ul(iter+1,:), ll(iter+1,:,:), mew_mat(iter,1:iter,:), obj_val(iter+1,:), time(iter-1,:)] = CVElasticNetBilevelDempe(ul, ll, phi, data, iter, test_indices, train_indices);
    end
    disp([ul,phi,obj_val]);
    optimalLambda  = ul(max_iter+1,1);
    optimalAlpha   = ul(max_iter+1,2);
    optimalObjVal  = obj_val(iter+1,:);
    totalTime      = sum(time)+tEnd2+tEnd1;

end