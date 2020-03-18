function [beta, fval] = AdaptiveLasso1(lambda, data, split)

problemName = 'lasso L1 regularization';             % Test problem name

ulDim=1;                         % Number of UL dimensions
llDim=size(data,2)-1;            % Number of LL dimensions

%Upper level TP1 implemented
datapoints_train = [1:split*size(data,1)]; %change this
datapoints_test = [split*size(data,1)+1:size(data,1)]; %change this
data_trainX = data(datapoints_train,1:end-1);
data_trainY = data(datapoints_train,end);

options = optimset('Algorithm','active-set'); % run active-set algorithm
options = optimset('Display','off','TolX',1e-10,'TolFun',1e-10);

llDimStart = zeros(1,2*llDim);
[betaPlus, fval] = fmincon(@(beta) problemFunction(lambda,beta, data_trainX, data_trainY), llDimStart,[],[],[],[],[],[], @(beta) problemConstraints(lambda,beta), options);
beta = betaPlus(1:end/2);

save('externalProblem');
end

function functionValue = problemFunction(lambda, betaPlus, data_trainX, data_trainY) 
    nvars = length(betaPlus);
    beta = betaPlus(1:nvars/2);
    eps = betaPlus(nvars/2+1:end);
    one = ones(size(eps))';
    eps_weights = abs(inv(data_trainX'*data_trainX)*data_trainX'*data_trainY);

    dataPoints = 2*length(data_trainY);
%     functionValue = 1/dataPoints*sum((data_trainY-data_trainX*beta').^2)+lambda*(sum(eps));
    functionValue = (1/dataPoints)*sum((data_trainY-data_trainX*beta').^2)+lambda*(eps(:,2:end)*(one(2:end,:)./eps_weights(2:end,:)));
end
        
function [inequalityConstrVals equalityConstrVals] = problemConstraints(lambda, betaPlus)
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