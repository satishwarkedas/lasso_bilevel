function [ul_vec, ll_mat, mew_mat, objval, tEnd] = CVElasticNetBilevelDempe(ul, ll, phi, data, iter, test_ind, train_ind)
    folds = size(phi,2);
    p = size(data,2)-2;
    ul_vec = zeros(1,2);
    ll_mat = zeros(p+1,folds);
    mew_mat = zeros(iter, folds);


    n_train = size(train_ind,2);
    n_test = size(test_ind,2);
    

    lambda = ul(iter,1);
    alpha = ul(iter,2);
    beta = reshape(squeeze(ll(iter,:,:)),[],folds*(p+1))';
    eps = abs(reshape(squeeze(ll(iter,2:end,:)),[],folds*p))';
    tau1 = zeros(folds,1);
    tau2 = zeros(folds,1);
    tau3 = zeros(folds,1);
    tau_mew = zeros(iter*folds,1);
    mew_vec = zeros(iter*folds,1);
    dec_vec_init = [lambda; alpha; beta; eps; tau1; tau2; tau3; tau_mew; mew_vec];
    dec_vec_size = length(dec_vec_init);
    unit_dec_vec_size = (dec_vec_size-2)/folds;
%     
%     % Upper level objective
%     upper_objective = @(dec_vec) upperlevelobjective(data, dec_vec, test_ind, p, unit_dec_vec_size);
    
    % Boundary constraints
    lb_dec_vec = [0;0;repmat([-Inf(1,p+1),zeros(1,p),-Inf,-Inf,-Inf,zeros(1,iter),zeros(1,iter)],1,folds)'];
    ub_dec_vec = [Inf;1;Inf(dec_vec_size-2,1)];
    
    options = optimoptions('fmincon','Algorithm','sqp');
    options = optimset('Display','off');
    
    A = zeros(2*p*folds, dec_vec_size);
    b = zeros(2*p*folds, 1);
    Aeq = zeros((iter+3)*folds, dec_vec_size);
    beq = zeros((iter+3)*folds, 1);
    C = [];
    Ceq = [];
    
    for i=1:folds
        % Linear equality constraints
        [Aeq_i,beq((i-1)*(iter+3)+1:i*(iter+3),:)] = linearequalityconstraints(ul, phi(:,i), unit_dec_vec_size+2, p, iter);
        Aeq((i-1)*(iter+3)+1:i*(iter+3),1) = Aeq_i(:,1);
        Aeq((i-1)*(iter+3)+1:i*(iter+3),2) = Aeq_i(:,2);
        Aeq((i-1)*(iter+3)+1:i*(iter+3),(i-1)*unit_dec_vec_size+2:i*unit_dec_vec_size+1) = Aeq_i(:,3:end);
        
        % Linear inequality constraints
        [A_i,b((i-1)*(2*p)+1:i*(2*p),:)] = linearinequalityconstraints(unit_dec_vec_size+2, p);
        A((i-1)*(2*p)+1:i*(2*p),1) = A_i(:,1);
        A((i-1)*(2*p)+1:i*(2*p),2) = A_i(:,2);
        A((i-1)*(2*p)+1:i*(2*p),(i-1)*unit_dec_vec_size+2:i*unit_dec_vec_size+1) = A_i(:,3:end);
        
    end
    
    tStart = tic;
    [dec_vec_optimal, objval] = fmincon(@(dec_vec) upperlevelobjective(data, dec_vec, test_ind, p, unit_dec_vec_size),...
                                        dec_vec_init,A,b,Aeq,beq,...
                                        lb_dec_vec,ub_dec_vec,...
                                        @(dec_vec) nonlinearconstraints(ul, phi, data, dec_vec, train_ind, p, iter, unit_dec_vec_size),...
                                        options);
    tEnd = toc(tStart);
    
    ul_vec  = dec_vec_optimal(1:2,:)';
    for i=1:folds
        s_ind = (i-1)*unit_dec_vec_size+3;
        ll_mat(:,i) = dec_vec_optimal(s_ind:s_ind+p,:);
%         disp(s_ind+2*p+2+2*iter);
        mew_mat(:,i) = dec_vec_optimal(s_ind+2*p+3+iter:s_ind+2*p+2+2*iter,:);
    end
end

function u_val = upperlevelobjective(data, dec_vec, test_ind, p, unit_dec_vec_size)
    u_val = 0;
    for i=1:size(test_ind,1)
        s_ind = (i-1)*unit_dec_vec_size+3;
        X_test = data(test_ind,1:end-1);
        Y_test = data(test_ind,end);
        beta = dec_vec(s_ind:s_ind+p,:);
        u_val = u_val + beta'*(X_test'*X_test)*beta - 2*Y_test'*X_test*beta + Y_test'*Y_test;
    end
end

function [c,ceq] = nonlinearconstraints(ul, phi, data, dec_vec, train_ind, p, iter, unit_dec_vec_size)
    folds = size(train_ind,1);
    c = zeros(folds,1);
    ceq = zeros(folds*iter,1);
    lambda = dec_vec(1,:);
    alpha = dec_vec(2,:);
    for i=1:folds
        X_train = data(train_ind(i,:),1:end-1);
        Y_train = data(train_ind(i,:),end);
        
        n_obs = 2*size(Y_train,1);
        ul_iter = ul(1:iter,1);
        phi_iter = phi(1:iter,i);
    
        
        s_ind = (i-1)*unit_dec_vec_size;
        ll = dec_vec(s_ind+3:s_ind+p+3,:);
        eps = dec_vec(s_ind+p+4:s_ind+2*p+3,:);
        one = ones(size(eps));
        tau1 = dec_vec(s_ind+2*p+4,:);
        tau2 = dec_vec(s_ind+2*p+5,:);
        tau3 = dec_vec(s_ind+2*p+6,:);
        tau_mew = dec_vec(s_ind+2*p+7:s_ind+2*p+6+iter,:);
        mew_vec = dec_vec(s_ind+2*p+7+iter:s_ind+2*p+6+2*iter,:);

        c(i,1) = (1/n_obs)*(ll'*(X_train'*X_train)*ll - 2*Y_train'*X_train*ll + Y_train'*Y_train) ...
                + (lambda*(1-alpha)/2)*(ll(2:end,:)'*ll(2:end,:)) ...
                + lambda*alpha*(one'*eps) ... 
                - mew_vec'*phi_iter;

        ceq((i-1)*iter+1:i*iter,1) = tau_mew.*mew_vec;
    end
end


function [Aeq, beq] = linearequalityconstraints(ul, phi, dec_vec_size, p, iter)
    Aeq = zeros(iter+3, dec_vec_size);
    beq = [zeros(iter,1);1;0;0];
    ul_iter_1 = ul(1:iter,1);
    ul_iter_2 = ul(1:iter,2);
        
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