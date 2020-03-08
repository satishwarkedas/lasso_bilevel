function [beta, fval] = solveLasso1(lambda, data)

problemName = 'lasso L1 regularization';             % Test problem name

ulDim=1;                         % Number of UL dimensions
llDim=size(data,2)-1;            % Number of LL dimensions

options = optimset('Algorithm','active-set'); % run active-set algorithm
options = optimset('Display','off','TolX',1e-10,'TolFun',1e-10);

llDimStart = zeros(1,2*llDim);
[betaPlus, fval] = fmincon(@(beta) problemFunction(lambda,beta), llDimStart,[],[],[],[],[],[], @(beta) problemConstraints(lambda,beta), options);
beta = betaPlus(1:end/2);

save('externalProblem');

function functionValue = problemFunction(lambda, betaPlus)

    %Upper level TP1 implemented
    global data
    nvars = length(betaPlus);
    beta = betaPlus(1:nvars/2);
    eps = betaPlus(nvars/2+1:end);
    datapoints_train = [1:0.5*size(data,1)]; %change this
    datapoints_test = [0.5*size(data,1)+1:size(data,1)]; %change this
    data_trainX = data(datapoints_train,1:end-1);
    data_trainY = data(datapoints_train,end);
    
    %1 Lasso regression
    dataPoints = length(data_trainY);
%     functionValue = 1/dataPoints*sum((data_trainY-data_trainX*beta').^2)+lambda*(sum(eps));
    functionValue = sum((data_trainY-data_trainX*beta').^2)+lambda*(sum(eps(2:end,1)));
        
function [inequalityConstrVals equalityConstrVals] = problemConstraints(lambda, betaPlus)

    %Upper level TP1 implemented
    global data
    nvars = length(betaPlus);
    beta = betaPlus(1:nvars/2);
    eps = betaPlus(nvars/2+1:end);
    datapoints_train = [1:0.5*size(data,1)];
    datapoints_test = [0.5*size(data,1)+1:size(data,1)];
    data_trainX = data(datapoints_train,1:end-1);
    data_trainY = data(datapoints_train,end);
    
  
    inequalityConstrVals1 = beta(2:end,1)-eps(2:end,1);
    inequalityConstrVals2 = -beta(2:end,1)-eps(2:end,1);
    
    inequalityConstrVals = [inequalityConstrVals1, inequalityConstrVals2];
    if lambda == 0
        inequalityConstrVals = 0*[inequalityConstrVals1, inequalityConstrVals2];
    end
    equalityConstrVals = [];