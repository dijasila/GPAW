import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6 / 2 ** 0.5))
data = np.loadtxt('ag55.spec')
plt.plot(data[:, 0], data[:, 1], 'k')
plt.title(r'Absorption spectrum of $Ag_{55}$ with GLLB-SC potential')
plt.xlabel('eV')
plt.ylabel('Absorption (arbitary units)')
plt.savefig('fig1.png')
