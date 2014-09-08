from ase import Atoms
from gpaw import GPAW
from gpaw import PW
import numpy as np
from gpaw.test import equal

name = 'quartz'

alat = 5.032090
clat = alat * 1.09685337425
a1 = alat*np.array([1.0, 0.0, 0.0])
a2 = alat*np.array([-0.5, np.sqrt(3)/2., 0.0])
a3 = clat*np.array([0.0, 0.0, 1.])
cell_cv = np.array([a1,a2,a3])

symbols = ['Si', 'Si', 'Si','O', 'O', 'O','O', 'O','O']

spos_ac = np.array([(0.4778762817077312,  0.0000000000000000,  0.3333333333333333),
                    (0.0000000000000000,  0.4778762817077312,  0.6666666666666666),
                    (0.5221237182922689,  0.5221237182922689,  0.0000000000000000),
                    (0.4153075513810672,  0.2531339617721680,  0.2029892900232357),
                    (0.7468660382278319,  0.1621735896088991,  0.5363226233565690),
                    (0.8378264103911008,  0.5846924486189328,  0.8696559566899023),
                    (0.2531339617721680,  0.4153075513810672,  0.7970107099767644),
                    (0.1621735896088991,  0.7468660382278319,  0.4636773766434310),
                    (0.5846924486189328,  0.8378264103911008,  0.1303440433100977)])

atoms = Atoms(symbols=symbols,
              scaled_positions=spos_ac,
              cell=cell_cv,
              pbc=True
              )

## with fractional translations
calc = GPAW(mode=PW(),
            xc='LDA',
            kpts=(3, 3, 3),
            nbands=42,
            symmetry={'symmorphic': False},
            gpts=(20, 20, 24))

atoms.set_calculator(calc)
energy_fractrans = atoms.get_potential_energy()

assert(len(calc.wfs.kd.ibzk_kc) == 7)
assert(len(calc.wfs.kd.symmetry.op_scc) == 6)

## without fractional translations
calc = GPAW(mode=PW(),
            xc='LDA',
            kpts=(3,3,3),
            nbands = 42,
            gpts = (20,20,24),
           )
atoms.set_calculator(calc)
energy_no_fractrans = atoms.get_potential_energy()

assert(len(calc.wfs.kd.ibzk_kc) == 10)
assert(len(calc.wfs.kd.symmetry.op_scc) == 2)

equal(energy_fractrans, energy_no_fractrans, 1e-3)
