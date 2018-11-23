from gpaw.wavefunctions.mode import Mode
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class TBMode(Mode):
    name = 'tb'

    def __call__(self):
        return TBWaveFunctions()


class TBWaveFunctions(LCAOWaveFunctions):
    pass


if __name__ == '__main__':
    from ase import Atoms
    from gpaw import GPAW
    a = Atoms('H')
    a.calc = GPAW(mode='tb')
    a.get_potential_energy()
