"""Check TDDFT ionizations with Yukawa potential."""
from ase import Atoms
from ase.structure import molecule
from ase.units import Hartree
from gpaw import GPAW, mpi
from gpaw.occupations import FermiDirac
from gpaw.test import equal
from gpaw.eigensolvers import RMM_DIIS
from gpaw.lrtddft import LrTDDFT
from gpaw.poisson import PoissonSolver
import _gpaw

newlibxc = _gpaw.lxcXCFuncNum('HYB_GGA_XC_LCY_PBE') is not None

h2o = Atoms(molecule('H2O'))
h2o.set_initial_magnetic_moments([2, -1, -1])
h2o.center(vacuum=3.0)
h2o_plus = Atoms(molecule('H2O'))
h2o_plus.set_initial_magnetic_moments([2, -0.5, -0.5])
h2o_plus.center(vacuum=3.0)


def get_paw():
    """Return calculator object."""
    c = {'energy': 0.001, 'eigenstates': 0.001, 'density': 0.001}
    return GPAW(convergence=c, eigensolver=RMM_DIIS(), xc='LCY_PBE(0.83)',
#            poissonsolver=PoissonSolver(use_charge_center=True),
        parallel={'domain': mpi.world.size}, gpts=(20, 20, 20),
        occupations=FermiDirac(width=0.0, fixmagmom=True))


calc = get_paw()
calc.set(txt='H2O_LCY_PBE_083.log')
calc_plus = get_paw()
calc_plus.set(txt='H2O_plus_LCY_PBE_083.log', charge=1)

h2o.set_calculator(calc)
e_h2o = h2o.get_potential_energy()
h2o_plus.set_calculator(calc_plus)
e_h2o_plus = h2o_plus.get_potential_energy()
e_ion = e_h2o_plus - e_h2o

print(e_ion, 12.62)
equal(e_ion, 12.62, 0.3)
lr = LrTDDFT(calc_plus, txt='LCY_TDDFT_H2O.log')
lr.diagonalize()

for i, ip_i in enumerate([14.74, 18.51]):
    ion_i = lr[i].get_energy() * Hartree + e_ion
    print(ion_i, ip_i)
    equal(ion_i, ip_i, 3.4)
