%% The LASSO test script
% The inputs are the dataset in the format below
% data(i,:)  -> [intercept, X_1_i, X_2_i, ..., X_n_i, Y_i]
% and corresponding lambda mapping based on group set G = {1,2,...,g}
% lambda_vec -> [a_intercept, a_X_1, a_X_2, ..., a_X_n] 
% with a_xx belongs to G
% The key results are as follows
% result     -> the Gurobi optimization solution: struct with multiple fields
% var_names  -> the decision vector [beta, eps, tau1, tau2, u, lambda]  
% optimalLambda -> the final lambda vector of dimension as no. of parameter
% optimalBeta   -> the final beta vector - order as the variable order
% optimalObjVal -> the objective value at the optimal solution

%% Creating the Dataset
% Dataset parameters
n          = 2000;
p          = 200;
s          = 20;
beta_type  = 1;
rho        = 0;    % correlation levels - 0, 0.35 and 0.7
mew        = 10;    % controls for the SNR level in the Y

global data
[data, Beta0] = data_generate(n, p, s, beta_type, rho, mew);

%% The implementation in MATLAB - Lasso
X = data(1:end,2:size(data,2)-1);
Y = data(1:end,size(data,2));

tStart = tic;
lambda = [0:0.0001:100];
% [B, fitinfo] = lasso(X,Y,'CV',2,'Lambda',lambda);
[B, fitinfo] = lasso(X,Y,'CV',2);
% [B, fitinfo] = lasso(X,Y);
tEnd = toc(tStart);

lassoPlot(B,fitinfo,'PlotType','CV');
legend('show') % Show legend

%% The First Formulation of the problem using only a single lambda
%  This implementation has lambda for all beta's including the intercept 
% [result_old, var_names, t_old] = solveLassoBilevel(data);
% optimalLambda_old = result_old.x(end,:);
% optimalBeta_old = result_old.x(2:(size(data,2)-1),:);
% optimalObjVal_old = result_old.objval;

%  This implementation has no lambda for beta_0 -> original LASSO
% 

start_vec = sign(corr(X,Y)).*(corr(X,Y)>median(corr(X,Y)));
[result_new, var_names_new, t_new] = solveLassoBilevelMIQP(data, start_vec);
[result_new_2, var_names_new_2, t_new_2] = solveLassoBilevelMIQP(flipud(data), start_vec);
optimalLambda_new = (result_new.x(end,:)+result_new_2.x(end,:))/2;
optimalBeta_new = result_new.x(2:(size(data,2)-1),:);
optimalObjVal_new = result_new.objval;
t_total_new = t_new+t_new_2;

%  This is the implementation for the original lasso when lambda is given
%  Search for optimal lambda and beta over a grid of lambda's

% [optimalLambda_normal,  optimalObjVal_normal, optimalBeta_normal, t_normal] = optimalLambda(data);

% norm(optimalBeta_normal(2:end,1)-Beta0)
% dot(optimalBeta_old,Beta0)/norm(optimalBeta_old)/norm(Beta0)
dot(optimalBeta_new,Beta0)/norm(optimalBeta_new)/norm(Beta0)
%% Creation of the Lambda vector
% Example lambda vector 
% lambda_vec = zeros(size(data,2)-1,1);
% for i=1:length(lambda_vec)
%     lambda_vec(i) = mod(i,3)+1;
% end

% for our current dataset let's group as intercept - 1; (x1,x2,x3 - 2; 
% (x2^2,x3^2,x4^2) - 3; (x1*x2, x2*x3, x3*x1) - 4
% test_lambda = [1,1,1,1,1,1,1,1,1,1];
% lambda_vec =  [1,2,3,4,5,6,7,8,9,10];
% dim_lambda = max(lambda_vec);


%% The Second Formulation of the problem using multiple lambda - Group
% lambda_vec = test_lambda;
% dim_lambda = max(lambda_vec);
% [result, var_names, Q, A, b, c] = solveLassoBilevel2(data, lambda_vec);
% optimalLambda = zeros(length(lambda_vec),1)';
% for i=1:length(optimalLambda)
%     optimalLambda(i) = result.x(end+lambda_vec(i)-dim_lambda,:);
% end
% optimalBeta = result.x(1:(size(data,2)-1),:);
% optimalObjVal = result.objval;
