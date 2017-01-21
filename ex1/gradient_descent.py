import numpy as np
from compute_cost import compute_cost

def gradient_descent(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y); # number of training examples
    #J_history = np.zeros(num_iters)
    J_history = [compute_cost(X, y, theta)]

    for _ in xrange(num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        # ============================================================
        delta = np.dot(X, theta).transpose() - y
        delta = np.multiply(delta.transpose(), X).transpose()
        delta = np.dot(delta, np.ones((m, 1))) # sum
        theta = theta - alpha * delta / m

        # Save the cost J in every iteration
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history
