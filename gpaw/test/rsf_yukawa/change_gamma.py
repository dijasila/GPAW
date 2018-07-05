"""Check for tunability of gamma for yukawa potential."""
from ase import Atoms
from ase.units import kcal, mol
from gpaw import GPAW, setup_paths
from gpaw.mixer import Mixer
from gpaw.cluster import Cluster
from gpaw.poisson import PoissonSolver
from gpaw.eigensolvers import RMMDIIS
from gpaw.xc.hybrid import HybridXC
from gpaw.occupations import FermiDirac
from gpaw.test import equal, gen

if setup_paths[0] != '.':
    setup_paths.insert(0, '.')

for atom in ['C', 'O']:
    gen(atom, xcname='PBE', scalarrel=True, exx=True,
        yukawa_gamma=0.81, gpernode=149)

h = 0.30
co = Cluster(Atoms('CO', positions=[(0, 0, 0), (0, 0, 1.15)]))
co.minimal_box(4, h=h)

c = {'energy': 0.1, 'eigenstates': 3, 'density': 3}

# IP for CO using LCY-PBE with gamma=0.81 after
# R. Wuerdemann, M. Walter
# dx.doi.org/10.1021/acs.jctc.8b00238
IP = 14.31

xc = HybridXC('LCY_PBE', omega=0.81)

calc = GPAW(txt='CO.txt', xc=xc, convergence=c,
            eigensolver=RMMDIIS(), h=h,
            occupations=FermiDirac(width=0.0), spinpol=False)
co.set_calculator(calc)
co.get_potential_energy()
(eps_homo, eps_lumo) = calc.get_homo_lumo()
equal(eps_homo, -IP, 0.15)


