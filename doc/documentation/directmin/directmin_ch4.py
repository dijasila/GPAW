from gpaw import GPAW, LCAO, PoissonSolver
from gpaw.occupations import FermiDirac
from ase.build import molecule

sys = molecule('CH4')
sys.center(vacuum=4.5)

calc = GPAW(xc='PBE',
            basis='dzp',
            poissonsolver=PoissonSolver(relax='GS', eps=1.0e-16),  # important to use good eps
            mode=LCAO(), nbands='nao', # all bands in calculations
            occupations=FermiDirac(width=0.0, fixmagmom=True),
            spinpol=False
            )


from gpaw.odd.lcao.oddvar import ODDvarLcao as ODD

opt = ODD(calc, g_tol=1.0e-4, memory_lbfgs=3)
sys.set_calculator(opt)
e = sys.get_potential_energy()
