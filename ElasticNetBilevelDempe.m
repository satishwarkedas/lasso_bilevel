function [ul_vec, ll_vec, mew_vec, objval, tEnd] = ElasticNetBilevelDempe(ul, ll, mew_mat, phi, data, iter, split)
    Y_train = data(1:split*size(data,1),size(data,2));
    X_train = data(1:split*size(data,1),1:size(data,2)-1);
    Y_test = data(split*size(data,1)+1:size(data,1),size(data,2));
    X_test = data(split*size(data,1)+1:size(data,1),1:size(data,2)-1);
    
    
    p = size(X_train,2)-1;
    lambda = ul(iter,1);
    alpha  = ul(iter,2);
    beta0 = ll(iter,1);
    beta = ll(iter,2:end)';
    eps = abs(beta);
    tau1 = 0;
    tau2 = 0;
    tau3 = 0;
    tau_mew = zeros(iter,1);
    mew_vec = zeros(iter,1);
    dec_vec_init = [lambda; alpha; beta0; beta; eps; tau1; tau2; tau3; tau_mew; mew_vec];
    dec_vec_size = length(dec_vec_init);
    
%     % Upper level objective
%     upper_objective = @(dec_vec) upperlevelobjective(Y_test, X_test, dec_vec, p);
    
    % Boundary constraints
    lb_dec_vec = [0;0;-Inf(p+1,1);zeros(p,1);-Inf;-Inf;-Inf;zeros(iter,1);zeros(iter,1)];
    ub_dec_vec = [Inf;1;Inf(dec_vec_size-2,1)];
    
    options = optimset('Algorithm','sqp');
    options = optimset('Display','off','TolX',1e-10,'TolFun',1e-10);
    
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    C = [];
    Ceq = [];
    
    % Linear equality constraints
    [Aeq,beq] = linearequalityconstraints(ul, phi, dec_vec_size, p, iter);
    % Linear inequality constraints
    [A,b] = linearinequalityconstraints(dec_vec_size, p);
    
    tStart = tic;
    [dec_vec_optimal, objval] = fmincon(@(dec_vec) upperlevelobjective(Y_test, X_test, dec_vec, p),...
                                        dec_vec_init,A,b,Aeq,beq,...
                                        lb_dec_vec,ub_dec_vec,...
                                        @(dec_vec) nonlinearconstraints(ul, phi, Y_train, X_train, dec_vec, p, iter),...
                                        options);
%     disp(dec_vec_optimal(1:2,:));
    
    tEnd = toc(tStart);
    
    ul_vec  = dec_vec_optimal(1:2,:)';
%     disp(ul_vec);
    ll_vec  = dec_vec_optimal(3:p+3,:)';
    mew_vec = dec_vec_optimal(2*p+6+iter:2*p+5+2*iter,:)';
end

function u_val = upperlevelobjective(Y_test, X_test, dec_vec, p)
    u_val = dec_vec(3:p+3,:)'*(X_test'*X_test)*dec_vec(3:p+3,:) - 2*Y_test'*X_test*dec_vec(3:p+3,:) + Y_test'*Y_test;
end

function [c,ceq] = nonlinearconstraints(ul, phi, Y_train, X_train, dec_vec, p, iter)
    n_obs = 2*size(Y_train,1);
    ul_iter = ul(1:iter,:);
    phi_iter = phi(1:iter,:);
    
%     lambda = dec_vec(1,:);
%     ll = dec_vec(2:p+2,:);
%     eps = dec_vec(p+3:2*p+2,:);
%     one = ones(size(eps));
%     tau1 = dec_vec(2*p+3,:);
%     tau2 = dec_vec(2*p+4,:);
%     tau_mew = dec_vec(2*p+5:2*p+4+iter,:);
%     mew_vec = dec_vec(2*p+5+iter:2*p+4+2*iter,:);
%     
    lambda  = dec_vec(1,:);
    alpha   = dec_vec(2,:);
    ll      = dec_vec(3:p+3,:);
    eps     = dec_vec(p+4:2*p+3,:);
    one     = ones(size(eps));
    tau1    = dec_vec(2*p+4,:);
    tau2    = dec_vec(2*p+5,:);
    tau3    = dec_vec(2*p+6,:);
    tau_mew = dec_vec(2*p+7:2*p+6+iter,:);
    mew_vec = dec_vec(2*p+7+iter:2*p+6+2*iter,:);

    c   = (1/n_obs)*(ll'*(X_train'*X_train)*ll - 2*Y_train'*X_train*ll + Y_train'*Y_train) ...
        + (lambda*(1-alpha)/2)*(ll(2:end,:)'*ll(2:end,:)) ...
        + lambda*alpha*(one'*eps) ... 
        - mew_vec'*phi_iter;
%     disp(c);
    ceq = tau_mew.*mew_vec;
end


function [Aeq, beq] = linearequalityconstraints(ul, phi, dec_vec_size, p, iter)
    Aeq = zeros(iter+3, dec_vec_size);
    beq = [zeros(iter,1);1;0;0];
    ul_iter_1 = ul(1:iter,1);
    ul_iter_2 = ul(1:iter,2);
%     disp([ul_iter_1, ul_iter_2, ul_iter_1.*ul_iter_2]);
    
    
    % Creating the first kkt constraints
    for i=1:iter
        Aeq(i,2*p+4) = 1;
        Aeq(i,2*p+5) = ul(i,1);
        Aeq(i,2*p+6) = ul(i,2);
        Aeq(i,2*p+6+i) = -1;
        beq(i,:) = phi(i,:);
    end
    
    Aeq(iter+1,2*p+7+iter:2*p+6+2*iter) = ones(1,iter);
    
    Aeq(iter+2,1) = -1;
    Aeq(iter+2,2*p+7+iter:2*p+6+2*iter) = ul_iter_1';
    
    Aeq(iter+3,2) = -1;
    Aeq(iter+3,2*p+7+iter:2*p+6+2*iter) = ul_iter_2';
    
    
end

function [A, b] = linearinequalityconstraints(dec_vec_size, p)
    A = zeros(2*p,dec_vec_size);
    b = zeros(2*p,1);
    for i=1:p
        A(i,3+i)     =  1;
        A(i,3+p+i)   = -1;
        
        A(p+i,3+i)   = -1;
        A(p+i,3+p+i) = -1;
    end
end