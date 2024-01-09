from gpaw.new.calculation import DFTState
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian


class DirectOptimizer(Eigensolver):

    def iterate(self, state: DFTState, hamiltonian: Hamiltonian) -> float:
        pass