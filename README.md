# Bilevel Optimization Approaches for Tuning Parameter Selection in LASSO and its variants

MATLAB Codes 

### Major inputs and their format
data -> (X,y) where X(:,i) = [1, x0(i), x1(i), x2(i), ..., xp(i)]  
folds -> no. of CV folds  
max_iter -> maximum no. of iterations to be considered for lower-level approximations  

### datagenerate.m
Synthetic Datasets whose parameters are
- n - number of examples/observations
- p - no. of parameters/variables
- s - intended sparsity in the dataset - as defined by the number of parameters that are non-negative
- beta_type - 4 different types of beta vectors
- rho - correlation between the variables
- mew - controls for the Signal-to-Noise Ratio

### cvAdaptiveLassoDempe.m
[optimalLambda, totalTime] = cvAdaptiveLassoDempe(data, folds, max_iter)  
Uses the functions CVAdaptiveLasso1.m and CVAdaptiveLassoBilevelDempe.m; Refer to these functions below

### cvAdapativeLassoBilevelDempe.m
[ul_vec, ll_mat, mew_mat, objval, tEnd] = CVAdaptiveLassoBilevelDempe(ul, ll, phi, data, iter, test_ind, train_ind)  
outputs:
- ul_vec  -> the optimal upper-level vector after the current iteration (iter)
- ll_mat  -> the lower-level matrix of dimension ( (p+1) x folds) after the current iteration (iter)
- mew_mat -> the mew matrix of dimension (iter x folds) for the current iteration (iter)
inputs:
- ul      -> the list of upper-level vectors until the current iteration
- ll      -> the list of lower-level vectors until the current iteration
- phi     -> the list of phi-vectors corresponding to ul for each fold
- data    -> data in the format mentioned above
- iter    -> current iteration value
- test_ind -> matrix of test indices for all folds - dimension - folds x #testindices
- train_ind -> matrix of train indices for all folds - dimension - folds x #trainindices

### cvAdaptiveLasso1.m
[beta, fval] = cvAdaptiveLasso1(lambda, data, index_train)  
inputs:
- lambda -> a given Lambda value
- data -> data matrix in the format mentioned above
- index_train -> vector of row indices that are to be used for training, rest of the data will be considered for testing

### cvLassoMIQP.m/cvAdaptiveLassoMIQP.m
[optimalLambda, totalTime, res] = cvLassoMIQP/cvAdaptiveLassoMIQP(data, folds, start_vec)  
outputs:
- res -> the result of the MIQP solver of Gurobi (struct with multiple attributes)
- if no start_vec is provided, leave it blank; else needs to be the size - p x 1
- elements of start_vec can only take binary values -> corresponds to the signs of beta's (negative beta -> positive u)

### solveLassoBilevelMIQP.m/AdaptiveLassoBilevelMIQP.m  
Solve the Bi-level problem with lower level problem formulated using KKT conditions    
[optimalLambda, optimalBeta, totalTime] = AdaptiveLassoBilevelMIQP(data, split, start_vec)  
- split -> a value between 0 and 1; usage -> 0.7, 0.8: the first 70% and 80% of the data used for training, rest for testing

### solveLasso1/AdaptiveLasso1.m
[beta, fval] = solveLasso1/AdaptiveLasso1(lambda, data, splits)
- Given a lambda value, implement Original LASSO or Adaptive LASSO

### solveLassoBilevelDempe/AdaptiveLassoBilevelDempe.m
Solve the Bi-level problem with iterative lower-level mapping approximation scheme as proposed by Dempe et al.
