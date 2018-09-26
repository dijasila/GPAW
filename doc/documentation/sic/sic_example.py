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
            mode=LCAO(force_complex_dtype=True),  # need complex wfs for SIC
            occupations=FermiDirac(width=0.0, fixmagmom=True),
            spinpol=False, gpts=h2gpts(0.2, sys.get_cell(), idiv=8),
            nbands='nao' # important to use all bands
            )

opt = ODD(calc, odd='PZ_SIC',
          initial_orbitals='KS_PM',  # use Pipek-Mezey localization for initial guess
          g_tol=1.0e-3, beta=(0.5, 0.5))  # beta is a scaling factor

sys.set_calculator(opt)
e = sys.get_potential_energy()
