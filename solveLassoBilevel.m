function [res, nam, tEnd] = solveLassoBilevel(data)

    problemName = 'lasso L1 regularization based on the KKT formulation ';             % Test problem name

    param = size(data,2)-1;              % no. of parameters including intercept


    datapoints_train = [1:0.5*size(data,1)];
    datapoints_test = [0.5*size(data,1)+1:size(data,1)];
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


    % Model variable names are stored in a cell array
    names = cell(5*param+1,1);
    var_type = '';
    lower_bound = zeros(5*param+1,1);
    for i=1:param
        names(i,1) = {join(['beta', int2str(i)])};
        names(param+i,1) = {join(['eps', int2str(i)])};
        names(2*param+i,1) = {join(['tau1', int2str(i)])};
        names(3*param+i,1) = {join(['tau2', int2str(i)])};
        names(4*param+i,1) = {join(['u', int2str(i)])};
    
        var_type(i,:) = 'C';
        var_type(param+i,:) = 'C';
        var_type(2*param+i,:) = 'C';
        var_type(3*param+i,:) = 'C';
        var_type(4*param+i,:) = 'B';
    
        lower_bound(i,:) = -10000;
    end
    names(5*param+1,1) = {'lambda'};
    var_type(5*param+1,:) = 'C';

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

    % writing the model into a mps format file
    gurobi_write(model,'lasso_test1.mps');

    % setting the parameters
    params.outputflag = 0;

    % result
    tStart = tic;
    result = gurobi(model,params);
    tEnd = toc(tStart);
    disp(result);

    res = result;
    nam = names;
%     sol = result.x;
%     lambda_opt = sol(5*param+1,:);
%     beta_opt = sol(1:param,:);
end

function [Q, c, alpha] = genobjmat(data_testX, data_testY, param )
    row_dim = 5*param + 1;
    col_dim = 5*param + 1;
    
    Q = zeros(row_dim, col_dim);
    c = zeros(row_dim, 1);
    
    X_test = data_testX;            % N x p matrix
    Y_test = data_testY;            % N x 1 matrix
    
    Q(1:param, 1:param) = X_test'*X_test;
    c(1:param, 1) = -2*X_test'*Y_test;
    alpha = Y_test'*Y_test;  
end


function [A, b, sense_vec] = genlinconst(data_trainX, data_trainY, param)
    row_dim = 8*param;
    col_dim = 5*param + 1;
    
    A = zeros(row_dim, col_dim);
    b = zeros(row_dim, 1);
    sense_vec = '';
    
    X_train = data_trainX;
    Y_train = data_trainY;
    I_pos = eye(param);
    I_neg = -1*eye(param);
    
    % First FOC constraints
    % 2*(X_train(k)'*X_train) x beta + 0 x eps + 1 x tau1(k) + (-1) x tau2k
    % + 0 x u + 0 x lambda = 2 X_train(k)'*Y_train
    for k=1:param
        A_beta = 2*X_train(:,k)'*X_train;
        A_eps = zeros(param,1)';
        A_tau1 = I_pos(:,k)';
        A_tau2 = I_neg(:,k)';
        A_u = zeros(param,1)';
        A_lambda = 0;
        
        A(k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(k,:) = 2*X_train(:,k)'*Y_train;
        sense_vec(k,:) = '=';
    end
    
    % Second FOC constraints
    % 0 x beta + 0 x eps + (-1) x tau1(k) + (-1) x tau2k
    % + 0 x u + 1 x lambda = 0
    for k=1:param
        A_beta = zeros(param,1)';
        A_eps = zeros(param,1)';
        A_tau1 = I_neg(:,k)';
        A_tau2 = I_neg(:,k)';
        A_u = zeros(param,1)';
        A_lambda = 1;
        
        A(param+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(param+k,:) = 0;
        sense_vec(param+k,:) = '=';
    end
    
    % Complementary Slackness MIP formulation - 1
    % tau1(k) <= M_1(k) x u(k)
    % 0 x beta + 0 x eps + 1 x tau1(k) + 0 x tau2(k) + (-M_1(k)) x u(k) +
    % 0 x lambda <= 0
    
    % define M = 10000 here
    M_1 = 20;
    for k=1:param
        A_beta = zeros(param,1)';
        A_eps = zeros(param,1)';
        A_tau1 = I_pos(:,k)';
        A_tau2 = zeros(param,1)';
        A_u = M_1 * I_neg(:,k)';
        A_lambda = 0;
        
        A(2*param+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(2*param+k,:) = 0;
        sense_vec(2*param+k,:) = '<';
    end
    
    % Complementary Slackness MIP formulation - 2
    % tau2(k) + M_1(k) x u(k) <= M_1(k)
    % 0 x beta + 0 x eps + 0 x tau1(k) + 1 x tau2(k) + (M_1(k)) x u(k) +
    % 0 x lambda <= M_1(k)
    for k=1:param
        A_beta = zeros(param,1)';
        A_eps = zeros(param,1)';
        A_tau1 = zeros(param,1)';
        A_tau2 = I_pos(:,k)';
        A_u = M_1 * I_pos(:,k)';
        A_lambda = 0;
        
        A(3*param+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(3*param+k,:) = M_1;
        sense_vec(3*param+k,:) = '<';
    end
    
    % Complementary Slackness MIP formulation - 3
    % - beta(k) + eps(k) + M_2(k) x u(k) <= M_2(k)
    % (-1) x beta(k) + 1 x eps(k) + 0 x tau1(k) + 0 x tau2(k) + (M_2(k)) x u(k) +
    % 0 x lambda <= M_2(k)
    
    % define M = 1000 here
    M_2 = 20;
    for k=1:param
        A_beta = I_neg(:,k)';
        A_eps = I_pos(:,k)';
        A_tau1 = zeros(param,1)';
        A_tau2 = zeros(param,1)';
        A_u = M_1 * I_pos(:,k)';
        A_lambda = 0;
        
        A(4*param+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(4*param+k,:) = M_2;
        sense_vec(4*param+k,:) = '<';
    end
    
    % Complementary Slackness MIP formulation - 4
    % beta(k) + eps(k) - M_2(k) x u(k) <= 
    % 1 x beta(k) + 1 x eps(k) + 0 x tau1(k) + 0 x tau2(k) + (- M_2(k)) x u(k) +
    % 0 x lambda <= 0
    for k=1:param
        A_beta = I_pos(:,k)';
        A_eps = I_pos(:,k)';
        A_tau1 = zeros(param,1)';
        A_tau2 = zeros(param,1)';
        A_u = M_1 * I_neg(:,k)';
        A_lambda = 0;
        
        A(5*param+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(5*param+k,:) = 0;
        sense_vec(5*param+k,:) = '<';
    end
    
    % beta and eps constraint - 1
    % beta(k) - eps(k) <= 0
    % (1) x beta(k) + (-1) x eps(k) + 0 x tau1(k) + 0 x tau2(k) + 0 x u(k) +
    % 0 x lambda <= 0
    for k=1:param
        A_beta = I_pos(:,k)';
        A_eps = I_neg(:,k)';
        A_tau1 = zeros(param,1)';
        A_tau2 = zeros(param,1)';
        A_u = zeros(param,1)';
        A_lambda = 0;
        
        A(6*param+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(6*param+k,:) = 0;
        sense_vec(6*param+k,:) = '<';
    end
    
    % beta and eps constraint - 2
    % - beta(k) - eps(k) <= 0
    % (-1) x beta(k) + (-1) x eps(k) + 0 x tau1(k) + 0 x tau2(k) + 0 x u(k) +
    % 0 x lambda <= 0
    for k=1:param
        A_beta = I_neg(:,k)';
        A_eps = I_neg(:,k)';
        A_tau1 = zeros(param,1)';
        A_tau2 = zeros(param,1)';
        A_u = zeros(param,1)';
        A_lambda = 0;
        
        A(7*param+k,:) = [A_beta, A_eps, A_tau1, A_tau2, A_u, A_lambda];
        b(7*param+k,:) = 0;
        sense_vec(7*param+k,:) = '<';
        
     
    end
end
