import numpy as np
from scipy.optimize import curve_fit as cf
import matplotlib.pyplot as plt

# Helper Code to Generate Trivial Datasets
def generate_dataset1():
    x = [np.random.rand() for i in range(1000)]
    y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

def generate_dataset2():
    x = [np.random.rand() for i in range(1000)]
    y = [(0.5 - x[i])*(0.7 - x[i]) + 1 for i in range(1000)]
    return [np.array(x), np.array(y)]

[x1, y1] = generate_dataset1()
[x2, y2] = generate_dataset2()

# seeing the data before doing anything
plt.plot(x1, y1, "ro", label="1st data")
plt.plot(x2, y2, "bo", label= "2nd data")
plt.legend()
plt.show()


def linfunc(x, m, c):
    return m*x+c

pout, cout = cf(linfunc, x2, y2)
print(pout)
print(cout)

plt.plot(x2, y2, "ro", label="2nd data")
plt.plot(x2, linfunc(x2, pout[0], pout[1]), label= "Linear fit")
plt.legend()
plt.show()