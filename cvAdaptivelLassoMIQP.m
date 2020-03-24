function [optimalLambda, totalTime, res] = cvAdaptivelLassoMIQP(data, folds, start_vec)

    problemName = 'Bilevel approach to CV for Adaptive LASSO using MIQP';

    param = size(data,2)-1;              % no. of parameters including intercept
    n_obs = size(data,1);                % no. of observations in the complete data
    n_test = floor(n_obs/folds);         % no. of observations in the test data
    n_cons = 8*(param-1)+1;              % no. of linear constraints per each training set
    
    % Model formulation in Gurobi
    % min x'*Q*x + c'*x + alpha
    % st
    %     A*x = b                         % '=', '<' or '>'  
    %     lb <= x <= ub
    %     x'*Qc*x + q'*x <= beta

    % Final Decision Vector
    % [beta1, beta2, ..., betaK, eps1, eps2, ..., epsK,
    % tau11, tau12, ..., tau1K, tau21, tau22, ..., tau2K
    % u1, u2, lambda]
    % Considering that the u vector will remain the same across runs
    
    % Decision vector per each training set as per MIQP formulation:
    % beta - size: param
    % eps - size: param-1
    % tau1 - size: param-1
    % tau2 - size: param-1
    % u - size: param-1
    % lambda - 1
    
    final_dec_vec_size = folds*(param+ 4*(param-1)) + 1;
    
    % Model variable names are stored in a cell array
    names = cell(final_dec_vec_size,1);
    var_type = '';
    lower_bound = zeros(final_dec_vec_size,1);
    [names, var_type, lower_bound] = gen_model_def(final_dec_vec_size, param, folds);
    
    % Generating the model values
    Q = zeros(final_dec_vec_size, final_dec_vec_size);
    c = zeros(final_dec_vec_size,1);
    alpha = size(param,1);

    A = zeros(folds*n_cons,final_dec_vec_size);
    b = zeros(folds*n_cons,1);
    sense_vec = '';
    
    
    for i=1:folds

        index = 1:n_obs;
        index_test = 1+(i-1)*n_test:n_test+(i-1)*n_test;
        index_train = setdiff(index, index_test);
        
        X_train = data(index_train, 1:end-1);
        Y_train = data(index_train, end);
        X_test  = data(index_test, 1:end-1);
        Y_test  = data(index_test, end);
        
        
        s_ind = 1+(i-1)*param;
        e_ind = param+(i-1)*param;
        s_row = 1+(i-1)*n_cons;
        e_row = n_cons+(i-1)*n_cons;
        
        % Generating the matrices necessary for modelling
        [Q(s_ind:e_ind,s_ind:e_ind), c(s_ind:e_ind,1), alpha(i,1)] = genobjmat(X_test, Y_test, param);
        [A_i, b(s_row:e_row,:), sense_vec(s_row:e_row,:)] = genlinconst(X_train, Y_train, param);
       
        % Assigning the linear constraints to the A matrix
        A(s_row:e_row, 1+(i-1)*param:param+(i-1)*param) ...
            = A_i(:, 1:param);
        A(s_row:e_row, folds*param+1+(i-1)*(param-1):(param-1)+folds*param+(i-1)*(param-1)) ...
            = A_i(:, param+1:param+(param-1));
        A(s_row:e_row, folds*(param+(param-1))+1+(i-1)*(param-1):(param-1)+folds*(param+(param-1))+(i-1)*(param-1)) ...
            = A_i(:, param+(param-1)+1:param+2*(param-1));
        A(s_row:e_row, folds*(param+2*(param-1))+1+(i-1)*(param-1):(param-1)+folds*(param+2*(param-1))+(i-1)*(param-1)) ...
            = A_i(:, param+2*(param-1)+1:param+3*(param-1));
        A(s_row:e_row, folds*(param+3*(param-1))+1+(i-1)*(param-1):(param-1)+folds*(param+3*(param-1))+(i-1)*(param-1)) ...
            = A_i(:, param+3*(param-1)+1:param+4*(param-1));
        A(s_row:e_row, end) = A_i(:, end);
        
    end
    
    
    % Assigning the model values to a model struct to be fed to Gurobi
    % objective function
    model.Q = sparse(Q');
    model.obj = c';
    model.objcon = sum(alpha);
    model.modelsense = 'min';
    model.varnames = names;
    model.vtype = var_type;

    % constraints
    model.A = sparse(A);
    model.rhs = b;
    model.sense = sense_vec;
    model.lb = lower_bound;
    
    % initial solution
    model.start = [nan*ones(final_dec_vec_size-2*param+1,1);start_vec;start_vec;nan];

    % writing the model into a mps format file
    gurobi_write(model,'cv_adaptivelasso_test1.mps');

    % setting the parameters
    params.outputflag = 0;
%     params.TimeLimit = 180;
%     params.MIPGap = 0.001;

    % result
    tStart = tic;
    result = gurobi(model,params);
    totalTime = toc(tStart);
%     disp(result);

    res = result;
    sol = result.x;
    optimalLambda = sol(end,:); 
%     sol = result.x;
%     lambda_opt = sol(5*param+1,:);
%     beta_opt = sol(1:param,:);
end

function [Q, c, alpha] = genobjmat(X_test, Y_test, param)
    Q = zeros(param, param);
    c = zeros(param, 1);

    Q(1:param, 1:param) = X_test'*X_test;
    c(1:param, 1) = -2*X_test'*Y_test;
    alpha = Y_test'*Y_test;  
end


function [A, b, sense_vec] = genlinconst(X_train, Y_train, param)
    row_dim = 8*(param-1)+1;
    col_dim = param + 4*(param-1) + 1;
    eps_weights = abs(inv(X_train'*X_train)*X_train'*Y_train);
%     disp(norm(eps_weights))
    n_obs = 2*size(Y_train,1);
    
    A = zeros(row_dim, col_dim);
    b = zeros(row_dim, 1);
    sense_vec = '';
    
    I_pos = eye(param-1);
    I_neg = -1*eye(param-1);
    
    % First FOC constraints
    % for beta_0 -> 
    % (1/n_obs)*2*(X_train(1)*X_train) x beta + 0 x eps + 1 x tau1(k) + (-1) x tau2k
    % + 0 x u + 0 x lambda = (1/n_obs)*2*X_train(1)*Y_train
    A_beta = (1/n_obs)*2*X_train(:,1)'*X_train;
    A_eps = zeros(param-1,1)';
    A_tau1 = zeros(param-1,1)';
    A_tau2 = zeros(param-1,1)';
    A_u = zeros(param-1,1)';
    A_lambda = 0;
        
    A(1,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
    b(1,:) = (1/n_obs)*2*X_train(:,1)'*Y_train;
    sense_vec(1,:) = '=';
   
    % (1/n_obs)*2*(X_train(k)'*X_train) x beta + 0 x eps + 1 x tau1(k) + (-1) x
    % tau2(k)
    % + 0 x u + 0 x lambda = (1/n_obs)*2*X_train(k)'*Y_train
    for k=2:param
        A_beta = (1/n_obs)*2*X_train(:,k)'*X_train;
        A_eps = zeros(param-1,1)';
        A_tau1 = I_pos(:,k-1)';
        A_tau2 = I_neg(:,k-1)';
        A_u = zeros(param-1,1)';
        A_lambda = 0;
        
        A(k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(k,:) = (1/n_obs)*2*X_train(:,k)'*Y_train;
        sense_vec(k,:) = '=';
    end
  
    
    % Second FOC constraints
    % for eps vector
    % 0 x beta + 0 x eps + (-1) x tau1(k) + (-1) x tau2(k)
    % + 0 x u + (1/eps_weight(k)) x 1 x lambda = 0
    for k=1:param-1
        A_beta = zeros(param,1)';
        A_eps = zeros(param-1,1)';
        A_tau1 = I_neg(:,k)';
        A_tau2 = I_neg(:,k)';
        A_u = zeros(param-1,1)';
        A_lambda = 1/eps_weights(k+1,:);
%         A_lambda = 1;
        
        A(param+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(param+k,:) = 0;
        sense_vec(param+k,:) = '=';
    end
    
    % Complementary Slackness MIP formulation - 1
    % tau1(k) <= M_1(k) x u(k)
    % 0 x beta + 0 x eps + 1 x tau1(k) + 0 x tau2(k) + (-M_1(k)) x u(k) +
    % 0 x lambda <= 0
    
    % M_1 made dependent on the OLS estimates
    M_1 = 2*(max(eps_weights));
    for k=1:param-1
        A_beta = zeros(param,1)';
        A_eps = zeros(param-1,1)';
        A_tau1 = I_pos(:,k)';
        A_tau2 = zeros(param-1,1)';
        A_u = M_1 * I_neg(:,k)';
        A_lambda = 0;
        
        A(param+(param-1)+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(param+(param-1)+k,:) = 0;
        sense_vec(param+(param-1)+k,:) = '<';
    end
    
    % Complementary Slackness MIP formulation - 2
    % tau2(k) + M_1(k) x u(k) <= M_1(k)
    % 0 x beta + 0 x eps + 0 x tau1(k) + 1 x tau2(k) + (M_1(k)) x u(k) +
    % 0 x lambda <= M_1(k)
    for k=1:param-1
        A_beta = zeros(param,1)';
        A_eps = zeros(param-1,1)';
        A_tau1 = zeros(param-1,1)';
        A_tau2 = I_pos(:,k)';
        A_u = M_1 * I_pos(:,k)';
        A_lambda = 0;
        
        A(param+2*(param-1)+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(param+2*(param-1)+k,:) = M_1;
        sense_vec(param+2*(param-1)+k,:) = '<';
    end
    
    % Complementary Slackness MIP formulation - 3
    % - beta(k) + eps(k) + M_2(k) x u(k) <= M_2(k)
    % (-1) x beta(k) + 1 x eps(k) + 0 x tau1(k) + 0 x tau2(k) + (M_2(k)) x u(k) +
    % 0 x lambda <= M_2(k)
    
    % M_2 made dependent on the OLS estimates
    M_2 = 2*(max(eps_weights));
    for k=1:param-1
        A_beta = [0,I_neg(:,k)'];
        A_eps = I_pos(:,k)';
        A_tau1 = zeros(param-1,1)';
        A_tau2 = zeros(param-1,1)';
        A_u = M_2 * I_pos(:,k)';
        A_lambda = 0;
        
        A(param+3*(param-1)+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(param+3*(param-1)+k,:) = M_2;
        sense_vec(param+3*(param-1)+k,:) = '<';
    end
    
    % Complementary Slackness MIP formulation - 4
    % beta(k) + eps(k) - M_2(k) x u(k) <= 
    % 1 x beta(k) + 1 x eps(k) + 0 x tau1(k) + 0 x tau2(k) + (- M_2(k)) x u(k) +
    % 0 x lambda <= 0
    for k=1:param-1
        A_beta = [0,I_pos(:,k)'];
        A_eps = I_pos(:,k)';
        A_tau1 = zeros(param-1,1)';
        A_tau2 = zeros(param-1,1)';
        A_u = M_2 * I_neg(:,k)';
        A_lambda = 0;
        
        A(param+4*(param-1)+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(param+4*(param-1)+k,:) = 0;
        sense_vec(param+4*(param-1)+k,:) = '<';
    end
    
    % beta and eps constraint - 1
    % beta(k) - eps(k) <= 0
    % (1) x beta(k) + (-1) x eps(k) + 0 x tau1(k) + 0 x tau2(k) + 0 x u(k) +
    % 0 x lambda <= 0
    for k=1:param-1
        A_beta = [0,I_pos(:,k)'];
        A_eps = I_neg(:,k)';
        A_tau1 = zeros(param-1,1)';
        A_tau2 = zeros(param-1,1)';
        A_u = zeros(param-1,1)';
        A_lambda = 0;
        
        A(param+5*(param-1)+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(param+5*(param-1)+k,:) = 0;
        sense_vec(param+5*(param-1)+k,:) = '<';
    end
    
    % beta and eps constraint - 2
    % - beta(k) - eps(k) <= 0
    % (-1) x beta(k) + (-1) x eps(k) + 0 x tau1(k) + 0 x tau2(k) + 0 x u(k) +
    % 0 x lambda <= 0
    for k=1:param-1
        A_beta = [0,I_neg(:,k)'];
        A_eps = I_neg(:,k)';
        A_tau1 = zeros(param-1,1)';
        A_tau2 = zeros(param-1,1)';
        A_u = zeros(param-1,1)';
        A_lambda = 0;
        
        A(param+6*(param-1)+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(param+6*(param-1)+k,:) = 0;
        sense_vec(param+6*(param-1)+k,:) = '<';
    end
end

function [names, var_type, lower_bound] = gen_model_def(final_dec_vec_size, param, folds)
    
    for i=1:folds 
        names((i-1)*param+1,1) = {join(['beta0', int2str(i)])};
        var_type((i-1)*param+1,:) = 'C';
        lower_bound((i-1)*param+1,:) = -Inf;
        
        for j=1:param-1
            % beta vector 
            names((i-1)*param+j+1,1) = {join(['beta', int2str(j), int2str(i)])};
            var_type((i-1)*param+j+1,:) = 'C';
            lower_bound((i-1)*param+j+1,:) = -Inf;
      
            % eps vector 
            names((i-1)*(param-1)+folds*param+j,1) = {join(['eps', int2str(j), int2str(i)])};
            var_type((i-1)*(param-1)+folds*param+j,:) = 'C';
            lower_bound((i-1)*(param-1)+folds*param+j,:) = 0;
        
            % tau1 vector 
            names((i-1)*(param-1)+folds*param+folds*(param-1)+j,1) = {join(['tau1', int2str(j), int2str(i)])};
            var_type((i-1)*(param-1)+folds*param+folds*(param-1)+j,:) = 'C';
            lower_bound((i-1)*(param-1)+folds*param+folds*(param-1)+j,:) = 0;
        
            % tau2 vector
            names((i-1)*(param-1)+folds*param+folds*2*(param-1)+j,1) = {join(['tau2', int2str(j), int2str(i)])};
            var_type((i-1)*(param-1)+folds*param+folds*2*(param-1)+j,:) = 'C';
            lower_bound((i-1)*(param-1)+folds*param+folds*2*(param-1)+j,:) = 0;
            
            % u vector
            names((i-1)*(param-1)+folds*param+folds*3*(param-1)+j,1) = {join(['u', int2str(j), int2str(i)])};
            var_type((i-1)*(param-1)+folds*param+folds*3*(param-1)+j,:) = 'B';
%             lower_bound((i-1)*(param-1)+folds*param+folds*3*(param-1)+j,:) = 0;
        
        end
        
    end
    
    % lambda 
    names(final_dec_vec_size,1) = {'lambda'};
    var_type(final_dec_vec_size,:) = 'C';
    lower_bound(final_dec_vec_size,:) = 0;

end