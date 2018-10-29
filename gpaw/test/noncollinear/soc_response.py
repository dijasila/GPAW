from gpaw.response.df import DielectricFunction
"""Self-consistent SOC with response."""
from unittest import SkipTest

import numpy as np
from ase.build import mx2

from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues
from gpaw.mpi import size
from gpaw.lfc import BasisFunctions


if size > 1:
    raise SkipTest()


def getspinorbitcalc(calc, soc=True):
    # Takes a collinear calculator with wfs
    # and returns a non-collinear calculator
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)

    atoms = calc.atoms
    params = calc.parameters
    params.update(experimental={'soc': soc,
                                'magmoms': [[0, 0, 0]] * len(atoms)},
                  fixdensity=True,
                  txt=None)
    socalc = GPAW(**params)
    socalc.initialize(atoms=atoms, reading=True)
    socalc.set_positions()
    # socalc.wfs.eigensolver.initialize(socalc.wfs)

    # socalc.initialize_positions(atoms)
    # socalc.hamiltonian.initialize()
    # wfs = socalc.wfs

    # basis_functions = BasisFunctions(wfs.gd,
    #                                  [setup.phit_j
    #                                   for setup in wfs.setups],
    #                                  wfs.kd, dtype=wfs.dtype,
    #                                  cut=True)
    # basis_functions.set_positions(socalc.spos_ac)
    # socalc.density.initialize_from_atomic_densities(basis_functions)
    # wfs.initialize_wave_functions_from_basis_functions(
    #     basis_functions, socalc.density, socalc.hamiltonian, socalc.spos_ac)
    # hamiltonian.update(density)

    # wfs.eigensolver.reset()
    # socalc.scf.reset()

    # Now set the wavefunctions
    for kpt, sokpt in zip(calc.wfs.kpt_u, socalc.wfs.kpt_u):

        sokpt.psit_nG[:] = 0
        print(sokpt.psit_nG.shape)
        for s in [0, 1]:
            sokpt.psit_nG[s::2, s, :] = kpt.psit_nG[:]
        sokpt.eps_n = np.repeat(kpt.eps_n, 2)
        sokpt.f_n = np.zeros_like(sokpt.eps_n)
        P = sokpt.P
        sokpt.psit.matrix_elements(socalc.wfs.pt, out=P)

    socalc.occupations.fermilevel = calc.occupations.fermilevel
    socalc.occupations.calculate(socalc.wfs)

    socalc.density.initialize_from_wavefunctions(socalc.wfs)
    socalc.density.update(socalc.wfs)
    socalc.hamiltonian.update(socalc.density)
    socalc.wfs.eigensolver.initialize(socalc.wfs)
    for sokpt in socalc.wfs.kpt_u:
        socalc.wfs.eigensolver.subspace_diagonalize(
            socalc.hamiltonian, socalc.wfs, sokpt)
    socalc.occupations.fixed_fermilevel = False
    socalc.occupations.calculate(socalc.wfs)
    socalc.occupations.fixed_fermilevel = True
    return socalc


def readcalc(calc):
    # Takes a collinear calculator with wfs
    # and returns a non-collinear calculator
    assert isinstance(calc, str)
    calc = GPAW(calc, txt=None)

    return calc


# a = mx2('MoS2')
# a.center(vacuum=3, axis=2)

# params = dict(mode='pw',
#               kpts={'size': (3, 3, 1),
#                     'gamma': True})

# # Selfconsistent:
# calc = GPAW(convergence={'bands': 28},
#             **params)
# a.calc = calc
# a.get_potential_energy()
# calc.write('mos2_soc_wfs.gpw', mode='all')

socalc = getspinorbitcalc('mos2_nosoc_wfs.gpw', soc=False)
socalc.write('mos2_soc_wfs.gpw')
socalc2 = readcalc('mos2_soc_wfs.gpw')
calc = readcalc('mos2_nosoc_wfs.gpw')


# Check that these values are the same
for kpt, sokpt, sokpt1 in zip(calc.wfs.kpt_u,
                              socalc.wfs.kpt_u,
                              socalc2.wfs.kpt_u):

    # print(dir(sokpt))
    # print(sokpt.k, sokpt.weight)
    f0_n = kpt.f_n
    f_n = sokpt.f_n
    f1_n = sokpt1.f_n
    eps0_n = kpt.eps_n
    eps_n = sokpt.eps_n
    eps1_n = sokpt1.eps_n
    print(eps0_n[0:5])
    print(eps_n[0:10:2])

    print(np.sum(f0_n / kpt.weight), np.sum(f_n / sokpt.weight))
    # print('max(f_n - f1_n)', np.abs(f_n - f1_n).max())
    # print('max(eps_n - eps1_n)', np.abs(eps_n - eps1_n).max())
    print('max(eps_n - eps0_n)', np.abs(eps_n[0::2] - eps0_n).max())

# df = DielectricFunction('mos2_soc_wfs.gpw')
# df.get_dielectric_function()
