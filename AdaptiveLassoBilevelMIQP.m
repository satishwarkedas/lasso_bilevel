function [res, nam, tEnd] = AdaptiveLassoBilevelMIQP(data, start_vec, split)

    problemName = 'lasso L1 regularization based on the KKT formulation ';             % Test problem name

    param = size(data,2)-1;              % no. of parameters including intercept

    datapoints_train = [1:split*size(data,1)];
    datapoints_test = [split*size(data,1)+1:size(data,1)];
    data_trainX = data(datapoints_train,1:end-1);
    data_trainY = data(datapoints_train,end);
    data_testX = data(datapoints_test,1:end-1);
    data_testY = data(datapoints_test,end);

    % Model formulation in Gurobi
    % min x'*Q*x + c'*x + alpha
    % st
    %     A*x = b                         % '=', '<' or '>'  
    %     lb <= x <= ub
    %     x'*Qc*x + q'*x <= beta

    % Decision vector:
    % beta - size: param
    % eps - size: param-1
    % tau1 - size: param-1
    % tau2 - size: param-1
    % u - size: param-1
    % lambda - 1
    dec_vec_size = param + 4*(param-1) + 1;
    
    % Model variable names are stored in a cell array
    names = cell(dec_vec_size,1);
    var_type = '';
    lower_bound = zeros(dec_vec_size,1);
    names(1,1) = {'beta0'};
    var_type(1,:) = 'C';
    lower_bound(1,:) = -Inf;
    for i=1:param-1
        names(i+1,1) = {join(['beta', int2str(i)])};
        names(param+i,1) = {join(['eps', int2str(i)])};
        names(param + (param-1) +i,1) = {join(['tau1', int2str(i)])};
        names(param + 2*(param-1) +i,1) = {join(['tau2', int2str(i)])};
        names(param + 3*(param-1) +i,1) = {join(['u', int2str(i)])};
    
        var_type(i+1,:) = 'C';
        var_type(param+i,:) = 'C';
        var_type(param + (param-1) +i,:) = 'C';
        var_type(param + 2*(param-1) +i,:) = 'C';
        var_type(param + 3*(param-1) +i,:) = 'B';
    
        lower_bound(i+1,:) = -Inf;
    end
    names(dec_vec_size,1) = {'lambda'};
    var_type(dec_vec_size,:) = 'C';

    % Generating the matrices necessary for modelling
    [Q, c, alpha] = genobjmat(data_testX, data_testY, param);
    [A, b, sense_vec] = genlinconst(data_trainX, data_trainY, param);

    % Generating the model
    % objective function
    model.Q = sparse(Q');
    model.obj = c';
    model.objcon = alpha;
    model.modelsense = 'min';
    model.varnames = names;
    model.vtype = var_type;

    % constraints
    model.A = sparse(A);
    model.rhs = b;
    model.sense = sense_vec;
    model.lb = lower_bound;
    
    % initial solution
%     u_vec_pos = u_vec.*(u_vec>0);
    model.start = [nan*ones(4*param-3,1);start_vec;nan];

    % writing the model into a mps format file
    gurobi_write(model,'lasso_test1.mps');

    % setting the parameters
    params.outputflag = 0;
%     params.TimeLimit = 180;
%     params.MIPGap = 0.001;

    % result
    tStart = tic;
    result = gurobi(model,params);
    tEnd = toc(tStart);
%     disp(result);

    res = result;
    nam = names;
%     sol = result.x;
%     lambda_opt = sol(5*param+1,:);
%     beta_opt = sol(1:param,:);
end

function [Q, c, alpha] = genobjmat(data_testX, data_testY, param )
    row_dim = param + 4*(param-1) + 1;
    col_dim = param + 4*(param-1) + 1;
%     disp(row_dim);
    
    Q = zeros(row_dim, col_dim);
    c = zeros(row_dim, 1);
    
    X_test = data_testX;            % N x p matrix
    Y_test = data_testY;            % N x 1 matrix
    
    Q(1:param, 1:param) = X_test'*X_test;
    c(1:param, 1) = -2*X_test'*Y_test;
    alpha = Y_test'*Y_test;  
end


function [A, b, sense_vec] = genlinconst(data_trainX, data_trainY, param)
    row_dim = 8*(param-1)+1;
    col_dim = param + 4*(param-1) + 1;
    eps_weights = abs(inv(data_trainX'*data_trainX)*data_trainX'*data_trainY);
    n_obs = 2*size(data_trainY,1);
    
    A = zeros(row_dim, col_dim);
    b = zeros(row_dim, 1);
    sense_vec = '';
    
    X_train = data_trainX;
    Y_train = data_trainY;
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
    % 0 x beta + 0 x eps + (-1) x tau1(k) + (-1) x tau2k
    % + 0 x u + (1/eps_weight(k)) x 1 x lambda = 0
    for k=1:param-1
        A_beta = zeros(param,1)';
        A_eps = zeros(param-1,1)';
        A_tau1 = I_neg(:,k)';
        A_tau2 = I_neg(:,k)';
        A_u = zeros(param-1,1)';
        A_lambda = 1/eps_weights(k+1,:);
        
        A(param+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(param+k,:) = 0;
        sense_vec(param+k,:) = '=';
    end
    
    % Complementary Slackness MIP formulation - 1
    % tau1(k) <= M_1(k) x u(k)
    % 0 x beta + 0 x eps + 1 x tau1(k) + 0 x tau2(k) + (-M_1(k)) x u(k) +
    % 0 x lambda <= 0
    
    % define M = 20 here
    M_1 = 20;
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
    
    % define M = 1000 here
    M_2 = 20;
    for k=1:param-1
        A_beta = [0,I_neg(:,k)'];
        A_eps = I_pos(:,k)';
        A_tau1 = zeros(param-1,1)';
        A_tau2 = zeros(param-1,1)';
        A_u = M_1 * I_pos(:,k)';
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
        A_u = M_1 * I_neg(:,k)';
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
