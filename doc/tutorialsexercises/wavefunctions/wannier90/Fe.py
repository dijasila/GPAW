from ase.build import bulk
from gpaw import GPAW, FermiDirac, PW

a = bulk('Fe', 'bcc')
a.set_initial_magnetic_moments([-2.0])
a.set_cell([[1.434996, 1.434996, 1.434996],
            [-1.434996, 1.434996, 1.434996],
            [-1.434996, -1.434996, 1.434996]])
calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.01),
            kpts=(12, 12, 12),
            txt='Fe_scf.txt')
a.calc = calc
a.get_potential_energy()

calc.fixed_density(
    kpts={'size': (8, 8, 8), 'gamma': True},
    symmetry='off',
    nbands=40,
    convergence={'bands': 30},
    txt='Fe_nscf.txt').write('Fe.gpw', mode='all')
