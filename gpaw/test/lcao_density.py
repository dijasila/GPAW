"""Test LCAO density calculation and conversion to grid.

Test that the density generated by the following three procedures is the same:

 * basis_functions.construct_density as used in a normal calculation
 * axpy used on the psit_nG as constructed by lcao_to_grid
 * axpy used on the Phit_MG[i] * Phit_MG[j] * rho[j, i], where Phit_MG
   are the actual basis functions on the grid, constructed using lcao_to_grid

TODO: non-gamma-point test

"""

import numpy as np
from ase.structure import molecule

from gpaw import GPAW, ConvergenceError
from gpaw.utilities.blas import axpy

system = molecule('H2O')
system.center(vacuum=2.5)

calc = GPAW(mode='lcao',
            #basis='dzp',
            maxiter=1)

system.set_calculator(calc)
try:
    system.get_potential_energy()
except ConvergenceError:
    pass

wfs = calc.wfs
kpt = wfs.kpt_u[0]
nt_G = calc.density.gd.zeros()
bfs = wfs.basis_functions
nao = wfs.setups.nao
f_n = kpt.f_n
rho_MM = np.zeros((nao, nao))
wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM, rho_MM)

bfs.construct_density(rho_MM, nt_G, -1)

nbands = wfs.bd.nbands
psit_nG = wfs.gd.zeros(nbands)
bfs.lcao_to_grid(kpt.C_nM, psit_nG, -1)

nt2_G = calc.density.gd.zeros()
for f, psit_G in zip(f_n, psit_nG):
    axpy(f, psit_G**2, nt2_G)

identity_MM = np.identity(nao)
Phit_MG = calc.wfs.gd.zeros(nao)
bfs.lcao_to_grid(identity_MM, Phit_MG, -1)

nt3_G = calc.density.gd.zeros()
for M1, Phit1_G in enumerate(Phit_MG):
    for M2, Phit2_G in enumerate(Phit_MG):
        nt3_G += rho_MM[M1, M2] * Phit1_G * Phit2_G


err1_G = nt2_G - nt_G
err2_G = nt3_G - nt_G

maxerr1 = np.abs(err1_G).max()
maxerr2 = np.abs(err2_G).max()

print 'err1', maxerr1
print 'err2', maxerr2

assert max(maxerr1, maxerr2) < 1e-15
