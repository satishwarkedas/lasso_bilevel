<<<<<<< HEAD:tests/lassotest.m
function [Lambda, Beta, Time, Beta0] = lassotest(n, p, s, beta_type, rho, mew, max_iter, split)
    global data
    [data, Beta0] = data_generate(n, p, s, beta_type, rho, mew);
    
    Beta = zeros(p,3);
    Lambda = zeros(1,3);
    Time = zeros(1,3);
    
    % Lasso CV method in the MATLAB
    X = data(1:end,2:size(data,2)-1);
    Y = data(1:end,size(data,2));

    lambda = [0:0.01:10]; % can give these as arguments
    tStart = tic;
    [B, fitinfo] = lasso(X,Y,'CV',2,'Lambda',lambda);
%     [B, fitinfo] = lasso(X,Y,'CV',2);
    min_index = fitinfo.IndexMinMSE;
    Time(:,1) = toc(tStart)/2;
    Beta(:,1) = B(:,min_index);
    Lambda(:,1) = fitinfo.Lambda(:,min_index);
    
    
    % Lasso MIQP method
    start_vec = sign(corr(X,Y)).*(corr(X,Y)>median(corr(X,Y)));
    [result_new, var_names_new, Time(:,2)] = solveLassoBilevelMIQP(data, start_vec, split);
    Lambda(:,2) = result_new.x(end,:);
    Beta(:,2) = result_new.x(2:(size(data,2)-1),:);
    
    % Lasso eps-approx method
    % Need to maintain a running vector of phi, ul, ll, mew_vectors
    ul = zeros(max_iter+1,1);
    ll = zeros(max_iter+1,p+1);
    phi = zeros(max_iter+1,1);
    time = zeros(max_iter-1,1);
    obj_val = zeros(max_iter+1,1);
    mew_mat = zeros(max_iter, max_iter);

    ul(1,:) = 0.01;
    ul(2,:) = 10;
    [ll(1,:), phi(1,:)] = solveLasso1(ul(1,:), data, split);
    [ll(2,:), phi(2,:)] = solveLasso1(ul(2,:), data, split);
    
    for iter=2:max_iter
%         disp(iter);
        [ll(iter,:), phi(iter,:)] = solveLasso1(ul(iter,:), data, split);
    %     [ul(iter+1,:), ll(iter+1,:), mew_mat(iter,1:iter), obj_val(iter+1,:)] = solveLassoBilevelMIQP_Dempe(ul, ll', mew_mat, phi, data, iter);
        [ul(iter+1,:), ll(iter+1,:), mew_mat(iter,1:iter), obj_val(iter+1,:), time(iter-1,:)] = solveLassoBilevelDempe(ul, ll, mew_mat, phi, data, iter, split);
    end

    Lambda(:,3) = ul(max_iter+1,:);
    Beta(:,3) = ll(max_iter+1,2:end)';
%     optimalObjval_dempe = obj_val(max_iter+1,:);  
    Time(:,3) = sum(time);
end

=======
function [Lambda, Beta, Time, Beta0] = lassotest(n, p, s, beta_type, rho, mew, max_iter, split)
    global data
    [data, Beta0] = data_generate(n, p, s, beta_type, rho, mew);
    
    Beta = zeros(p,3);
    Lambda = zeros(1,3);
    Time = zeros(1,3);
    
    % Lasso CV method in the MATLAB
    X = data(1:end,2:size(data,2)-1);
    Y = data(1:end,size(data,2));

    lambda = [0:0.01:10]; % can give these as arguments
    tStart = tic;
    [B, fitinfo] = lasso(X,Y,'CV',2,'Lambda',lambda);
%     [B, fitinfo] = lasso(X,Y,'CV',2);
    min_index = fitinfo.IndexMinMSE;
    Time(:,1) = toc(tStart)/2;
    Beta(:,1) = B(:,min_index);
    Lambda(:,1) = fitinfo.Lambda(:,min_index);
    
    
    % Lasso MIQP method
    start_vec = sign(corr(X,Y)).*(corr(X,Y)>median(corr(X,Y)));
    [result_new, var_names_new, Time(:,2)] = solveLassoBilevelMIQP(data, start_vec, split);
    Lambda(:,2) = result_new.x(end,:);
    Beta(:,2) = result_new.x(2:(size(data,2)-1),:);
    
    % Lasso eps-approx method
    % Need to maintain a running vector of phi, ul, ll, mew_vectors
    ul = zeros(max_iter+1,1);
    ll = zeros(max_iter+1,p+1);
    phi = zeros(max_iter+1,1);
    time = zeros(max_iter-1,1);
    obj_val = zeros(max_iter+1,1);
    mew_mat = zeros(max_iter, max_iter);

    ul(1,:) = 0.01;
    ul(2,:) = 10;
    [ll(1,:), phi(1,:)] = solveLasso1(ul(1,:), data, split);
    [ll(2,:), phi(2,:)] = solveLasso1(ul(2,:), data, split);
    
    for iter=2:max_iter
%         disp(iter);
        [ll(iter,:), phi(iter,:)] = solveLasso1(ul(iter,:), data, split);
    %     [ul(iter+1,:), ll(iter+1,:), mew_mat(iter,1:iter), obj_val(iter+1,:)] = solveLassoBilevelMIQP_Dempe(ul, ll', mew_mat, phi, data, iter);
        [ul(iter+1,:), ll(iter+1,:), mew_mat(iter,1:iter), obj_val(iter+1,:), time(iter-1,:)] = solveLassoBilevelDempe(ul, ll, mew_mat, phi, data, iter, split);
    end

    Lambda(:,3) = ul(max_iter+1,:);
    Beta(:,3) = ll(max_iter+1,2:end)';
%     optimalObjval_dempe = obj_val(max_iter+1,:);  
    Time(:,3) = sum(time);
end

>>>>>>> c835512b3d131f76d7a91aec03d051c772c7ef55:lassotest.m
