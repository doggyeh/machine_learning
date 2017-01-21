import numpy as np
def compute_cost(X, y, theta):
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y); # number of training examples

    # You need to return the following variables correctly
    J = 0;

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    J = sum(np.power(np.dot(X, theta).transpose()[0] - y, 2)) / (2*m)
    return J
