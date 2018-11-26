from gpaw.density import Density
from gpaw.hamiltonian import Hamiltonian
from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.wavefunctions.lcao import LCAOWaveFunctions
from gpaw.wavefunctions.mode import Mode


class TB(Mode):
    name = 'tb'
    interpolation = 1
    force_complex_dtype = False

    def __init__(self) -> None:
        pass

    def __call__(self, ksl, **kwargs) -> 'TBWaveFunctions':
        return TBWaveFunctions(ksl, **kwargs)


class TBWaveFunctions(LCAOWaveFunctions):
    mode = 'tb'


class TBEigenSolver(DirectLCAO):
    pass


class TBDensity(Density):
    pass


class TBHamiltonian(Hamiltonian):
    pass


if __name__ == '__main__':
    from ase import Atoms
    from gpaw import GPAW
    a = Atoms('H')
    a.calc = GPAW(mode='tb')
    a.get_potential_energy()
