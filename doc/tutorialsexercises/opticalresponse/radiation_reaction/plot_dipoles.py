# web-page: dipoles.png
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6 / 1.62))

data = np.loadtxt('dm_nad.dat')
ax.plot(data[:, 0], data[:, 4], label=r'Na$_2$')
data = np.loadtxt('dm_nad2.dat')
ax.plot(data[:, 0], data[:, 4], label=r'2Na$_2$')
plt.xlabel('Time')
plt.legend()
plt.ylabel(r'<z>')
plt.tight_layout()
plt.savefig('dipoles.png')
