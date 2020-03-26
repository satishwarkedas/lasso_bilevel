function [beta, fval] = ElasticNet1(ul, data, split)

problemName = 'Elastic Net regularization';             % Test problem name


ulDim=2;                         % Number of UL dimensions
llDim=size(data,2)-1;            % Number of LL dimensions

%Upper level TP1 implemented
datapoints_train = [1:split*size(data,1)]; %change this
datapoints_test = [split*size(data,1)+1:size(data,1)]; %change this
data_trainX = data(datapoints_train,1:end-1);
data_trainY = data(datapoints_train,end);

options = optimset('Algorithm','active-set'); % run active-set algorithm
options = optimset('Display','off','TolX',1e-10,'TolFun',1e-10);

lambda = ul(:,1);
alpha  = ul(:,2);
llDimStart = zeros(1,2*llDim);
[betaPlus, fval] = fmincon(@(beta) problemFunction(lambda,alpha, beta, data_trainX, data_trainY), llDimStart,[],[],[],[],[],[], @(beta) problemConstraints(lambda,alpha,beta), options);
beta = betaPlus(1:end/2);

save('externalProblem');
end

function functionValue = problemFunction(lambda, alpha, betaPlus, data_trainX, data_trainY) 
    nvars = length(betaPlus);
    beta = betaPlus(1:nvars/2);
    eps = betaPlus(nvars/2+1:end);
    one = ones(size(eps(:,2:end)))';
    dataPoints = 2*length(data_trainY);
    functionValue = (1/dataPoints)*sum((data_trainY-data_trainX*beta').^2)+lambda*(1-alpha)/2*(beta(:,2:end)*beta(:,2:end)')+lambda*alpha*(eps(:,2:end)*one);
end
        
function [inequalityConstrVals equalityConstrVals] = problemConstraints(lambda, alpha, betaPlus)
    nvars = length(betaPlus);
    beta = betaPlus(1:nvars/2);
    eps = betaPlus(nvars/2+1:end);
    
  
    inequalityConstrVals1 = (beta(:,2:end)-eps(:,2:end))';
    inequalityConstrVals2 = (-beta(:,2:end)-eps(:,2:end))';
    
    inequalityConstrVals = [inequalityConstrVals1, inequalityConstrVals2];
    if lambda == 0
        inequalityConstrVals = 0*[inequalityConstrVals1, inequalityConstrVals2];
    end
    equalityConstrVals = [];
end