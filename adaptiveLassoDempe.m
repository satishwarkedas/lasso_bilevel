% Adaptive Lasso eps-approx method
function [optimalLambda, optimalBeta, totalTime] = adaptiveLassoDempe(data, max_iter, split)
    p = size(data,2)-2;
    % Need to maintain a running vector of phi, ul, ll, mew_vectors
    ul = zeros(max_iter+1,1);
    ll = zeros(max_iter+1,p+1);
    phi = zeros(max_iter+1,1);
    time = zeros(max_iter-1,1);
    obj_val = zeros(max_iter+1,1);
    mew_mat = zeros(max_iter, max_iter);

    ul(1,:) = 0.01;
    ul(2,:) = 10;
    tStart1 = tic;
    [ll(1,:), phi(1,:)] = AdaptiveLasso1(ul(1,:), data, split);
    [ll(2,:), phi(2,:)] = AdaptiveLasso1(ul(2,:), data, split);
    tEnd1 = toc(tStart1);
    
    for iter=2:max_iter
        tStart2 = tic;
        [ll(iter,:), phi(iter,:)] = AdaptiveLasso1(ul(iter,:), data, split);
        tEnd2 = toc(tStart2);
        [ul(iter+1,:), ll(iter+1,:), mew_mat(iter,1:iter), obj_val(iter+1,:), time(iter-1,:)] = AdaptiveLassoBilevelDempe(ul, ll, mew_mat, phi, data, iter, split);
%         disp(ul(iter+1,:));
        time(iter-1,:) = time(iter-1,:) + tEnd2;
        disp(ul);
    end

    optimalLambda = ul(max_iter+1,:);
    optimalBeta = ll(max_iter+1,2:end)';
%     optimalObjval_dempe = obj_val(max_iter+1,:);  
    totalTime = sum(time) + tEnd1;
end