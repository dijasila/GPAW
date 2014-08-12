import sys
import numpy as np
import matplotlib.pyplot as plt
a = np.loadtxt(sys.argv[1], delimiter=',').T
x = a[0]
for y in a[1:]:
    plt.plot(x, y, '-')
plt.show()
