# Bilevel Optimization Approaches for Tuning Parameter Selection in LASSO and its variants

MATLAB Codes 

### Data input format
data -> (X,y) where X(:,i) = [1, x0(i), x1(i), x2(i), ..., xp(i)]

### datagenerate.m
Synthetic Datasets whose parameters are
- n - number of examples/observations
- p - no. of parameters/variables
- s - intended sparsity in the dataset - as defined by the number of parameters that are non-negative
- beta_type - 4 different types of beta vectors
- rho - correlation between the variables
- mew - controls for the Signal-to-Noise Ratio

### cvAdaptiveLassoDempe.m
[optimalLambda, totalTime] = cvAdaptiveLassoDempe(data, folds)
- Uses the functions CVAdaptiveLasso1.m and CVAdaptiveLassoBilevelDempe.m; Refer to these functions below

### cvAdaptiveLassoMIQP.m
[optimalLambda, totalTime] = cvAdaptiveLassoMIQP(data, folds, start_vec)
- if no start_vec is provided, leave it blank; else needs to be the size - px1
- elements of start_vec can only take binary values -> corresponds to the signs of beta's (negative beta -> positive u)

### cvAdapativeLassoBilevelDempe.m
[ul_vec, ll_mat, mew_mat, objval, tEnd] = CVAdaptiveLassoBilevelDempe(ul, ll, phi, data, iter, test_ind, train_ind)
\noutputs:
- ul_vec -> the optimal upper-level vector after the current iteration (iter)
- ll_mat -> the lower-level matrix of dimension ( (p+1) x folds) after the current iteration (iter)
- mew_mat -> the mew matrix of dimension (iter x folds) for the current iteration (iter)


### cvAdaptiveLasso1.m
[beta, fval] = solveLasso1/AdaptiveLasso1(lambda, data, 
- Given a lambda value, implement Original LASSO or Adaptive LASSO


### solveLassoBilevelMIQP/AdaptiveLassoBilevelMIQP.m
Solve the Bi-level problem with lower level problem formulated using KKT conditions

### solveLassoBilevelDempe/AdaptiveLassoBilevelDempe.m
Solve the Bi-level problem with iterative lower-level mapping approximation scheme as proposed by Dempe et al.
