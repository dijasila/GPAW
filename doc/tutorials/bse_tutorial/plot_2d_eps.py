import pylab as pl
import numpy as np

pl.figure()

a = np.loadtxt('2d_eps.dat')
pl.plot(a[:, 0], a[:, 1], 'o-', c='b', ms=10, lw=2, label='Bare')
pl.plot(a[:, 0], a[:, 2], 'o-', c='r', ms=8, lw=2, label='Truncated')

pl.xlabel(r'$q$', size=28)
pl.ylabel(r'$\epsilon_{2D}$', size=28)
pl.xticks([0, 10], [r'$\Gamma$', r'$K$'], size=24)
pl.yticks(size=20)
pl.tight_layout()
pl.legend(loc='upper right')
pl.axis([0, a[-1, 0], 0.0, 9.9])

#pl.show()
pl.savefig('2d_eps.png')
