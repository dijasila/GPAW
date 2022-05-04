import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

# Input file has format ([BandPath, ndarray, ndarray])
fil = 'spiral_1000pw18k.npy'
print(f'Plotting file: {fil}')

with open(fil, 'rb') as f:
    path = np.load(f, allow_pickle=True)[0]
    e = np.load(f)
    mT = np.load(f)

q, x, X = path.get_linear_kpoint_axis()

p = np.polyfit(q[:-2], e[:-2], 2)
e = (e - e[0]) * 1000
fig, ax1 = plt.subplots()
ax1.plot(q, e, c='r', marker='.')
ax1.set_xlabel('$|q|$', size=20)
ax1.set_xticks(x)
ax1.set_xticklabels(X, size=16)
ax1.set_yticks([0, -20, -40, -60])
ax1.set_yticklabels([0, -20, -40, -60], size=16)
ax1.set_ylabel('Energy [meV per atom]', size=16, c='r')
ax1.set_ylim([-70, 10])

ax2 = ax1.twinx()
mT = np.linalg.norm(mT, axis=-1)
ax2.plot(q, mT, 'b')
ax2.set_ylabel(r'Total magnetic moment [$\mu_B$]', size=16, c='b')
ax2.set_ylim([0, 1.4])
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax2.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], size=16)
plt.axis([0, max(q), None, None])
plt.tight_layout()
plt.show()
