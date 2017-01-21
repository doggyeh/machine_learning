import numpy as np
import matplotlib.pyplot as plt
import sys

from mpl_toolkits.mplot3d import Axes3D
from plot import plot_data
from compute_cost import compute_cost
from gradient_descent import gradient_descent


print "======================= Part 2: Plotting ======================="
print 'Plotting Data ...'
data = np.loadtxt('ex1data1.txt', delimiter=',').T
X, y = data
m = len(y)
linear_plot = plt.figure()
ax1 = linear_plot.add_subplot(111)
ax1.plot(X, y, 'rx')

#plot_data(X, y)
#plt.plot(X, y, 'rx')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
#raw_input("Press the <ENTER> key to continue...")

print " =================== Part 3: Gradient descent ==================="
print 'Running Gradient Descent ...'
X = np.array([np.ones(m), X]).transpose() # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

# compute and display initial cost
print 'Initial cost is', compute_cost(X, y, theta)
# run gradient descent
theta, J_history = gradient_descent(X, y, theta, alpha, iterations);

# print theta to screen
print 'Theta found by gradient descent:\n', theta
print 'J_history=', J_history

# Plot the linear fit
ax1.plot(X[:,1], np.dot(X, theta), 'k-')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 10], theta)
print 'For population = 100,000, we predict a profit:', predict1*10000


print '============= Part 4: Visualizing J(theta_0, theta_1) ============= '
print 'Visualizing J(theta_0, theta_1) ...'

# Grid over which we will calculate J
theta0_vals = np.arange(-10, 10, 0.2)
theta1_vals = np.arange(-1, 4, 0.05)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = compute_cost(X, y, t)

# Surface plot
surface = plt.figure()
ax2 = surface.add_subplot(111, projection='3d')
ax2.plot_surface(theta0_vals, theta1_vals, J_vals)

ax2.set_xlabel('theta_0')
ax2.set_ylabel('theta_1')
ax2.set_zlabel('cost')


# Contour plot
contour_plot = plt.figure()
ax3 = linear_plot.add_subplot(111)
CS = plt.contour(theta0_vals, theta1_vals, J_vals)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.plot(theta[0], theta[1], 'rx')
plt.show()
