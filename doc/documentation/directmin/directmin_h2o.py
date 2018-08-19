from gpaw import GPAW, LCAO, PoissonSolver
from gpaw.utilities import h2gpts
from gpaw.occupations import FermiDirac
from ase.build import molecule

sys = molecule('H2O')
sys.center(vacuum=4.5)

mode=LCAO()
xc='PBE'
calc = GPAW(xc=xc,
            basis='dzp',
            poissonsolver=PoissonSolver(relax='GS', eps=1.0e-16),  # important to use good eps
            mode=mode,
            occupations=FermiDirac(width=0.0, fixmagmom=True),
            spinpol=False
            )

calc.atoms = sys  # important to do this
calc.create_setups(mode, xc)  # important to do this
calc.set(nbands=calc.setups.nao)  # important to do this
calc.set(gpts=h2gpts(0.2, sys.get_cell(), idiv=8))

from gpaw.odd.lcao.oddvar import ODDvarLcao as ODD
opt = ODD(calc, g_tol=1.0e-4, memory_lbfgs=3)
sys.set_calculator(opt)
e = sys.get_potential_energy()
