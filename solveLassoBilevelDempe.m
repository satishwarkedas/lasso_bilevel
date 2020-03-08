function [ul_vec, ll_vec, mew_vec, objval, tEnd] = solveLassoBilevelDempe(ul, ll, mew_mat, phi, data, iter)
    Y_train = data(1:size(data,1)/2,size(data,2));
    X_train = data(1:size(data,1)/2,1:size(data,2)-1);
    Y_test = data(size(data,1)/2+1:size(data,1),size(data,2));
    X_test = data(size(data,1)/2+1:size(data,1),1:size(data,2)-1);
    
    p = size(X_train,2)-1;
    lambda = ul(iter,1);
    beta0 = ll(iter,1);
    beta = ll(iter,2:end)';
    eps = abs(beta);
    tau1 = 0;
    tau2 = 0;
    tau_mew = zeros(iter,1);
    mew_vec = zeros(iter,1);
    dec_vec_init = [lambda; beta0; beta; eps; tau1; tau2; tau_mew; mew_vec];
    dec_vec_size = length(dec_vec_init);
    
    % Upper level objective
    upper_objective = @(dec_vec) upperlevelobjective(Y_test, X_test, dec_vec, p);
    
    % Boundary constraints
    lb_dec_vec = [0;-Inf(p+1,1);zeros(p,1);-Inf;-Inf;zeros(iter,1);zeros(iter,1)];
    ub_dec_vec = Inf(dec_vec_size,1);
    
    options = optimoptions('fmincon','Algorithm','sqp');
    options = optimset('Display','off');
    
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
    tEnd = toc(tStart);
    
    ul_vec  = dec_vec_optimal(1,:);
    ll_vec  = dec_vec_optimal(2:p+2,:)';
    mew_vec = dec_vec_optimal(2*p+5+iter:2*p+4+2*iter,:)';
end

function u_val = upperlevelobjective(Y_test, X_test, dec_vec, p)
    u_val = dec_vec(2:p+2,:)'*(X_test'*X_test)*dec_vec(2:p+2,:) - 2*Y_test'*X_test*dec_vec(2:p+2,:) + Y_test'*Y_test;
end

function [c,ceq] = nonlinearconstraints(ul, phi, Y_train, X_train, dec_vec, p, iter)
    ul_iter = ul(1:iter,:);
    phi_iter = phi(1:iter,:);
    
    lambda = dec_vec(1,:);
    ll = dec_vec(2:p+2,:);
    eps = dec_vec(p+3:2*p+2,:);
    one = ones(size(eps));
    tau1 = dec_vec(2*p+3,:);
    tau2 = dec_vec(2*p+4,:);
    tau_mew = dec_vec(2*p+5:2*p+4+iter,:);
    mew_vec = dec_vec(2*p+5+iter:2*p+4+2*iter,:);

    c = ll'*(X_train'*X_train)*ll - 2*Y_train'*X_train*ll + Y_train'*Y_train ...
        + lambda*(one'*eps) - mew_vec'*phi_iter;
%     disp('The current c value is');
%     disp(c);
    ceq = tau_mew.*mew_vec;
%     disp([c,ceq]);
end


function [Aeq, beq] = linearequalityconstraints(ul, phi, dec_vec_size, p, iter)
    Aeq = zeros(iter+2, dec_vec_size);
    beq = [zeros(iter,1);1;0];
    ul_iter = ul(1:iter,:);
    
    % Creating the first kkt constraints
    for i=1:iter
        Aeq(i,2*p+3) = 1;
        Aeq(i,2*p+4) = ul(i,:);
        Aeq(i,2*p+4+i) = -1;
        beq(i,:) = phi(i,:);
    end
    
    Aeq(iter+1,2*p+5+iter:2*p+4+2*iter) = ones(1,iter);
    
    Aeq(iter+2,1) = -1;
    Aeq(iter+2,2*p+5+iter:2*p+4+2*iter) = ul_iter';
    
    Aeq_1 = Aeq;
    beq_1 = beq;
end

function [A, b] = linearinequalityconstraints(dec_vec_size, p)
    A = zeros(2*p,dec_vec_size);
    b = zeros(2*p,1);
    for i=1:p
        A(i,2+i)     =  1;
        A(i,2+p+i)   = -1;
        
        A(p+i,2+i)   = -1;
        A(p+i,2+p+i) = -1;
    end
    
    A_1 = A;
    b_1 = b;
end
