from ase import Atoms
from gpaw.new.ase_interface import GPAW

a = 2.5
k = 4


def test_afm_h_chain():
    h = Atoms('H',
              magmoms=[1],
              cell=[a, 0, 0],
              pbc=[1, 0, 0])
    h.center(vacuum=2.0, axis=(1, 2))
    h.calc = GPAW(mode={'name': 'pw',
                        'ecut': 300,
                        'qspiral': [0.5*0, 0, 0]},
                  magmoms=[[1, 0, 0]],
                  symmetry='off',
                  kpts=(2 * k, 1, 1))
    _ = h.get_potential_energy()
