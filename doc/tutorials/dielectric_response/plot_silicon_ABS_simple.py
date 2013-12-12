import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('absorption.csv')
plt.plot(data[:,0], data[:,2])
plt.show()
