# Bilevel Optimization Approaches for Tuning Parameter Selection in LASSO and its variants

MATLAB Codes 

## datagenerate.m
Synthetic Datasets whose parameters are
- n - number of examples/observations
- p - no. of parameters/variables
- s - intended sparsity in the dataset - as defined by the number of parameters that are non-negative
- beta_type - 4 different types of beta vectors
- rho - correlation between the variables
- mew - controls for the Signal-to-Noise Ratio

## solveLasso1/AdaptiveLasso1.m
Given a lambda value, implement Original LASSO or Adaptive LASSO

## solveLassoBilevelMIQP/AdaptiveLassoBilevelMIQP.m
Solve the Bi-level problem with lower level problem formulated using KKT conditions

## solveLassoBilevelDempe/AdaptiveLassoBilevelDempe.m
Solve the Bi-level problem with iterative lower-level mapping approximation scheme as proposed by Dempe et al.

# CVAdaptiveLassoBilevelMIQP2.m
Solve the K-fold cross validation approach to tuning parameter selection as Bilevel problem 
with lower level problem formulated using KKT conditions
