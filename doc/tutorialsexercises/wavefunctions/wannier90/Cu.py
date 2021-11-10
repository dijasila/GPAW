from ase.build import bulk
from gpaw import GPAW, FermiDirac, PW

a = bulk('Cu', 'fcc')

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.1),
            kpts=(12, 12, 12),
            txt='Cu_scf.txt')
a.calc = calc
a.get_potential_energy()

calc.fixed_density(
    kpts={'size': (4, 4, 4), 'gamma': True},
    nbands=30,
    symmetry='off',
    txt='Cu_nscf.txt',
    convergence={'bands': 20}).write('Cu.gpw', mode='all')
