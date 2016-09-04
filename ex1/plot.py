import matplotlib.pyplot as plt
from numpy import loadtxt

#x, y = loadtxt('ex1data1.txt', delimiter=',', unpack=True)
x, y = loadtxt('ex1data1.txt', delimiter=',').T
print x

plt.plot(x, y, 'rx')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
