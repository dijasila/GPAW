from gpaw import GPAW, LCAO
from gpaw import PoissonSolver as PS
from gpaw.utilities import h2gpts
from gpaw.occupations import FermiDirac
from ase.build import molecule
from gpaw.odd.lcao.oddvar import ODDvarLcao as ODD


sys = molecule('CH4')
sys.center(vacuum=5.0)

calc = GPAW(xc='PBE', basis='dzp',
            poissonsolver=PS(relax='GS', eps=1.0e-16),  # important to use good eps
            mode=LCAO(), nbands='nao', # all bands in calculations
            occupations=FermiDirac(width=0.0, fixmagmom=True),
            spinpol=False, gpts=h2gpts(0.2, sys.get_cell(), idiv=8),
            )

opt = ODD(calc, g_tol=1.0e-4, memory_lbfgs=3)
sys.set_calculator(opt)
e = sys.get_potential_energy()
