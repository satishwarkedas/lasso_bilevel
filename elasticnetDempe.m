% Elastic Net eps-approx method
function [optimalLambda, optimalAlpha, optimalBeta, totalTime] = elasticnetDempe(data, max_iter, split)
    p = size(data,2)-2;
    
    % Need to maintain a running vector of phi, ul, ll, mew_vectors
    ul = zeros(max_iter+1,2);
    ll = zeros(max_iter+1,p+1);
    phi = zeros(max_iter+1,1);
    time = zeros(max_iter-1,1);
    obj_val = zeros(max_iter+1,1);
    mew_mat = zeros(max_iter, max_iter);

    ul(1,:) = [100, 0];            % change this
    ul(2,:) = [0, 1];           % change this         
                    
    tStart1 = tic;
    [ll(1,:), phi(1,:)] = ElasticNet1(ul(1,:), data, split);
%     disp(ll(1,:));
    [ll(2,:), phi(2,:)] = ElasticNet1(ul(2,:), data, split);
%     disp(ll(2,:));
    tEnd1 = toc(tStart1);
    
    for iter=2:max_iter
        tStart2 = tic;
        [ll(iter,:), phi(iter,:)] = ElasticNet1(ul(iter,:), data, split);
        tEnd2 = toc(tStart2);
        [ul(iter+1,:), ll(iter+1,:), mew_mat(iter,1:iter), obj_val(iter+1,:), time(iter-1,:)] = ElasticNetBilevelDempe(ul, ll, mew_mat, phi, data, iter, split);
        time(iter-1,:) = time(iter-1,:) + tEnd2;
    end
    
    disp([ul,phi,obj_val]);
    optimalLambda = ul(max_iter+1,1);
    optimalAlpha  = ul(max_iter+1,2); 
    optimalBeta = ll(max_iter+1,2:end)';
%     optimalObjval_dempe = obj_val(max_iter+1,:);  
    totalTime = sum(time) + tEnd1;
end