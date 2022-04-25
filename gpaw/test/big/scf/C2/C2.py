from ase import Atoms
from gpaw import GPAW

m = Atoms(symbols='C2',
          magmoms=[-0.5, 0.5],
          positions=[
              [0., 0., -0.62000006],
              [0., 0., 0.62000006]])
m.center(vacuum=4.0)

calc = GPAW(h=0.18,
            xc='PBE',
            basis='dzp',
            occupations={'name': 'fermi-dirac', 'width': 0.0},
            txt='C2_conv2.txt')

m.calc = calc
e2 = m.get_potential_energy()
