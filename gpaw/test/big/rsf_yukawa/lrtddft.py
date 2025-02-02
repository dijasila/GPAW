"""Check TDDFT ionizations with Yukawa potential."""
from ase.build import molecule
from ase.units import Hartree
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.occupations import FermiDirac
from gpaw.eigensolvers import RMMDIIS
from gpaw.lrtddft import LrTDDFT

# h2o = molecule('H2O')
# h2o.set_initial_magnetic_moments([2, -1, -1])
# adjust_cell(h2o, 3.0, h=0.3)
h2o_plus = molecule('H2O')
h2o_plus.set_initial_magnetic_moments([2, -0.5, -0.5])
adjust_cell(h2o_plus, 3.0, h=0.35)


def get_paw(**kwargs):
    """Return calculator object."""
    c = {'energy': 0.001, 'eigenstates': 0.001, 'density': 0.001}
    return GPAW(mode='fd', convergence=c, eigensolver=RMMDIIS(),
                xc='LCY-PBE:omega=0.83:unocc=True',
                parallel={'domain': world.size}, h=0.35,
                occupations=FermiDirac(width=0.0, fixmagmom=True),
                **kwargs)


# calc = get_paw(txt='H2O_LCY_PBE_083.log')
calc_plus = get_paw(txt='H2O_plus_LCY_PBE_083.log', charge=1)

# h2o.calc = calc
# e_h2o = h2o.get_potential_energy()
h2o_plus.calc = calc_plus
e_h2o_plus = h2o_plus.get_potential_energy()
# e_ion = e_h2o_plus - e_h2o
# e_ion = e_h2o_plus - e_h2o
e_ion = 12.5354477071  # from prior calculation

# print(e_ion, 12.62)
# equal(e_ion, 12.62, 0.1)
lr = LrTDDFT(calc_plus, txt='LCY_TDDFT_H2O.log',
             restrict={'istart': 3, 'jend': 4})
assert lr.xc.omega == 0.83
lr.write('LCY_TDDFT_H2O.ex.gz')
# reading is problematic with EXX on more than one core
if world.rank == 0:
    lr2 = LrTDDFT.read('LCY_TDDFT_H2O.ex.gz')
    lr2.diagonalize()
    assert lr2.xc.omega == 0.83

    # for i, ip_i in enumerate([14.74, 18.51]):  # 0.3 grid and all states
    for i, ip_i in enumerate([24.08]):
        ion_i = lr2[i].get_energy() * Hartree + e_ion
        print(ion_i, ip_i)
        assert abs(ion_i - ip_i) < 0.6
