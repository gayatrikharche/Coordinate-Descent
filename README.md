# Coordinate Descent

## Abstract
This project explores coordinate descent optimization techniques for logistic regression in binary classification tasks using the wine dataset. It investigates two methods: Cyclic Coordinate Descent (CCD) and Cyclic Coordinate Descent with Random feature selection (CCDR). The project provides a framework for implementation and benchmarks these methods against standard logistic regression, analyzing their convergence patterns and computational efficiency.

## Methodology
1. **Standard Logistic Regression**: Prepares the wine dataset for binary classification, standardizes features, and initializes a logistic regression model without regularization. The model learns optimal parameters by minimizing the logistic loss function.

2. **Cyclic Coordinate Descent (CCD)**: Implements a systematic approach where model parameters are updated sequentially, using the gradient of the logistic loss function to refine parameters iteratively.

3. **Cyclic Coordinate Descent Random (CCDR)**: Introduces randomness by selecting a coordinate index for updates during each iteration, facilitating a stochastic optimization approach.

4. **Sparse Coordinate Descent**: Aims to achieve solutions with a limited number of non-zero coefficients, varying the number of selected coordinates during optimization.

## Results
The results indicate that CCD and CCDR effectively minimize the logistic loss function. The study provides insights into their convergence patterns and computational efficiency compared to the standard logistic regression approach.
