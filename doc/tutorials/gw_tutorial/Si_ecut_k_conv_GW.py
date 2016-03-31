from ase.lattice import bulk

from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0

a = 5.431
atoms = bulk('Si', 'diamond', a=a)

for j, k in enumerate([4, 6, 8, 10]):
    calc = GPAW(mode=PW(350),
                kpts={'size': (k, k, k), 'gamma': True},
                dtype=complex,
                xc='LDA',
                occupations=FermiDirac(0.001),
                eigensolver='rmm-diis',
                txt='Si_groundstate.txt')

    atoms.set_calculator(calc)
    atoms.get_potential_energy()

    calc.diagonalize_full_hamiltonian()
    calc.write('Si_groundstate_%s.gpw' % k, mode='all')

    for i, ecut in enumerate([50, 100, 150, 200]):
        gw = G0W0(calc='Si_groundstate_%s.gpw' % k,
                  bands=(3, 5),
                  ecut=ecut,
                  kpts=[0],
                  filename='Si-g0w0_GW_k%s_ecut%s' % (k, ecut))

        result = gw.calculate()
