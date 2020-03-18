%% Creating the Dataset
% Dataset parameters
n          = 500;
p          = 200;
s          = 10;
beta_type  = 1;
rho        = 0;         % correlation levels - 0, 0.35 and 0.7
mew        = 5000;        % controls for the SNR level in the Y
folds      = 2;         % K value in K-fold cross validation    

global data
[data, Beta0] = data_generate_new(n, p, s, beta_type, rho, mew);
% current data_generate_new -> has 20% variance in Y, 20% variance in X
% 


%% CV adaptive lasso test
X = data(:,1:end-1);
Y = data(:,end);
start_vec = sign(corr(X,Y)).*(corr(X,Y)>median(corr(X,Y)));
% [result_new, var_names_new, t_new, Q, b, A, c] = CVAdaptiveLassoBilevelMIQP1(data, start_vec(2:end,:), folds);
[result_new, var_names_new, t_new, Q, b, A, c] = CVAdaptiveLassoBilevelMIQP2(data, start_vec(2:end,:), folds);
% [result_new, var_names_new, t_new, Q, b, A, c] = CVLassoBilevelMIQP2(data, start_vec(2:end,:), folds);
optimalLambda_new = result_new.x(end,:);
optimalBeta_new = result_new.x(2:(size(data,2)-1),:);
optimalObjVal_new = result_new.objval;