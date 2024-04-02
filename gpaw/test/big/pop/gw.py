from pathlib import Path

from ase import Atoms
from gpaw import GPAW, PW
from gpaw.mpi import size
from gpaw.response.g0w0 import G0W0

c = 2.53574055
atoms = Atoms('ZrO2',
              scaled_positions=[(0, 0, 0),
                                (0.25, 0.25, 0.25),
                                (0.75, 0.75, 0.75)],
              cell=[(0, c, c), (c, 0, c), (c, c, 0)],
              pbc=True)


def gs(k: int, ecut1: float) -> None:
    atoms.calc = GPAW(mode=PW(ecut1),
                      kpts={'size': (k, k, k), 'gamma': True},
                      xc='PBEsol',
                      txt='gs.txt')
    atoms.get_potential_energy()
    atoms.calc.diagonalize_full_hamiltonian()
    atoms.calc.write('gs.gpw', mode='all')


def gw(ecut2: float) -> dict:
    gw = G0W0(calc='gs.gpw',
              relbands=(-3, 3),
              ecut=ecut2,
              frequencies={'type': 'nonlinear',
                           'domega0': 0.025,
                           'omega2': 10.0},
              eta=0.1,
              filename='gw',
              nblocks=size // 2)
    result = gw.calculate()
    return result


def main():
    if not Path('gs.gpw').is_file():
        gs(k=4, ecut1=600)
    gw(ecut2=350)


if __name__ == '__main__':
    main()
