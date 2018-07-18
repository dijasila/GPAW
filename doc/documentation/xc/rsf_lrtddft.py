"""Check TDDFT ionizations with Yukawa potential."""
from ase import Atoms
from ase.structure import molecule
from ase.units import Hartree
from gpaw import GPAW, mpi
from gpaw.cluster import Cluster
from gpaw.occupations import FermiDirac
from gpaw.test import equal
from gpaw.eigensolvers import RMMDIIS
from gpaw.lrtddft import LrTDDFT

h2o = Cluster(Atoms(molecule('H2O')))
h2o.set_initial_magnetic_moments([2, -1, -1])
h2o.minimal_box(3.0, h=0.3)
h2o_plus = Cluster(Atoms(molecule('H2O')))
h2o_plus.set_initial_magnetic_moments([2, -0.5, -0.5])
h2o_plus.minimal_box(3.0, h=0.3)


def get_paw():
    """Return calculator object."""
    c = {'energy': 0.001, 'eigenstates': 0.001, 'density': 0.001}
    return GPAW(convergence=c, eigensolver=RMMDIIS(),
                xc='LCY_PBE:omega=0.83:unocc=True',
                parallel={'domain': mpi.world.size}, h=0.3,
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
equal(e_ion, 12.62, 0.1)
lr = LrTDDFT(calc_plus, txt='LCY_TDDFT_H2O.log', jend=4)
equal(lr.xc.omega, 0.83)
lr.write('LCY_TDDFT_H2O.ex.gz')
# reading is problematic with EXX on more than one core
if mpi.rank == 0:
    lr2 = LrTDDFT('LCY_TDDFT_H2O.ex.gz')
    lr2.diagonalize()
    equal(lr2.xc.omega, 0.83)

    for i, ip_i in enumerate([14.74, 18.51]):
        ion_i = lr2[i].get_energy() * Hartree + e_ion
        print(ion_i, ip_i)
        equal(ion_i, ip_i, 0.6)
