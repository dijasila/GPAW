import numpy as np
import pylab as pl

a = np.loadtxt('anisotropy.dat')
pl.plot(a[:, 0], (a[:, 2] - a[0, 2]) * 1.0e6, '-o')

pl.xticks([0, np.pi / 2, np.pi], ['0', r'$\pi/2$', r'$\pi$'], size=20)
pl.yticks(size=16)
pl.xlabel(r'$\theta$', size=24)
pl.ylabel(r'$\mu eV$', size=24)
pl.axis([0, np.pi, None, None])
pl.tight_layout()
#pl.show()
pl.savefig('anisotropy.png')
