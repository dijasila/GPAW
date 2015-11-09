from ase import Atoms
from ase.io import read

from gpaw import GPAW
from gpaw.utilities.dos import LCAODOS, RestartLCAODOS, fold
from ase.units import Hartree

import numpy as np

name = 'HfS2'
calc = GPAW(name+'.gpw', txt=None)
atoms = read(name+'.gpw')
ef = calc.get_fermi_level()

dos = RestartLCAODOS(calc) 
energies, weights = dos.get_subspace_pdos(range(51))
e, w = fold(energies * Hartree, weights, 2000, 0.1)

e, m_s_pdos = dos.get_subspace_pdos([0,1])
e, m_s_pdos = fold(e * Hartree, m_s_pdos, 2000, 0.1)
e, m_p_pdos = dos.get_subspace_pdos([2,3,4])
e, m_p_pdos = fold(e * Hartree, m_p_pdos, 2000, 0.1)
e, m_d_pdos = dos.get_subspace_pdos([5,6,7,8,9])
e, m_d_pdos = fold(e * Hartree, m_d_pdos, 2000, 0.1)

e, x_s_pdos = dos.get_subspace_pdos([25])
e, x_s_pdos = fold(e * Hartree, x_s_pdos, 2000, 0.1)
e, x_p_pdos = dos.get_subspace_pdos([26,27,28])
e, x_p_pdos = fold(e * Hartree, x_p_pdos, 2000, 0.1)

w_max = []
for i in range(len(e)):
    if (-4.5 <= e[i]-ef <= 4.5):
        w_max.append(w[i])

w_max = np.asarray(w_max)

from pylab import plotfile, show, gca
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='normal', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

font = {'family' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)

plt.plot(e-ef, w, label=r'\textbf{Total}', c='k', lw=2, alpha=0.7)
plt.plot(e-ef, x_s_pdos, label=r'\textbf{X-$s$}', c='g', lw=2, alpha=0.7)
plt.plot(e-ef, x_p_pdos, label=r'\textbf{X-$p$}', c='b', lw=2, alpha=0.7)
plt.plot(e-ef, m_s_pdos, label=r'\textbf{M-$s$}', c='y', lw=2, alpha=0.7)
plt.plot(e-ef, m_p_pdos, label=r'\textbf{M-$p$}', c='c', lw=2, alpha=0.7)
plt.plot(e-ef, m_d_pdos, label=r'\textbf{M-$d$}', c='r', lw=2, alpha=0.7)

plt.axis(ymin=0., ymax=np.max(w_max), xmin=-4.5, xmax=4.5, )
plt.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
plt.ylabel(r'\textbf{DOS}')
plt.legend(loc=1)
plt.show()
