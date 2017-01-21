import matplotlib.pyplot as plt
from numpy import loadtxt

def plot_data(x, y):
    plt.plot(x, y, 'rx')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

def main():
    #x, y = loadtxt('ex1data1.txt', delimiter=',', unpack=True)
    x, y = loadtxt('ex1data1.txt', delimiter=',').T
    plot_data(x, y)

if __name__ == '__main__':
    main()
