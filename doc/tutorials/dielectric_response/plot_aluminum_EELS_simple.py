import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('eels_x_.csv', delimiter=' ')
plt.plot(data[:,0], data[:,2])
plt.show()
