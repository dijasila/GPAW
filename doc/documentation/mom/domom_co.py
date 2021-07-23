from ase.build import molecule
from ase.optimize import LBFGS
from ase.parallel import paropen
from gpaw import GPAW, LCAO
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.lcao.tools import excite
from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO


atoms = molecule('CO')
atoms.center(vacuum=5)

calc = GPAW(xc='PBE',
            mode=LCAO(force_complex_dtype=True),
            h=0.2,
            basis='dzp',
            spinpol=True,
            eigensolver='direct-min-lcao',
            occupations={'name': 'fixed-uniform'},
            mixer={'backend': 'no-mixing'},
            nbands='nao',
            symmetry='off',
            txt='co.txt')
atoms.calc = calc

# Ground-state calculation
E_gs = atoms.get_potential_energy()

# Prepare initial guess for complex pi* orbitals by taking
# linear combination of real pi*x and pi*y orbitals
lumo = 5  # lumo is pi*x or pi*y orbital
for kpt in calc.wfs.kpt_u:
    pp = kpt.C_nM[lumo] + 1.0j * kpt.C_nM[lumo + 1]
    pm = kpt.C_nM[lumo] - 1.0j * kpt.C_nM[lumo + 1]
    kpt.C_nM[lumo][:] = pm
    kpt.C_nM[lumo + 1][:] = pp

calc.set(eigensolver=DirectMinLCAO(searchdir_algo={'name': 'LSR1P'},
                                   linesearch_algo={'name': 'UnitStep'},
                                   need_init_orbs=False))

# Occupation numbers for sigma->pi* excited state:
# Remove one electron from homo (sigma) and add one electron to lumo (pi*)
f = excite(calc, 0, 0, spin=(0, 0))

# Prepare excited-state DO-MOM calculation
prepare_mom_calculation(calc, atoms, f)

opt = LBFGS(atoms, logfile='co.log')
opt.run(fmax=0.05)

d = atoms.get_distance(0, 1)

with paropen('co.log', 'a') as fd:
    print(f'Optimized C-O bond length of sigma->pi* state: {d:.2f} Å', file=fd)
    # https://doi.org/10.1007/978-1-4757-0961-2
    print('Experimental C-O bond length of sigma->pi* state: 1.24 Å', file=fd)
