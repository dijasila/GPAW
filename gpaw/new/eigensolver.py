from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gpaw.new.calculation import DFTState


class Eigensolver(ABC):
    @abstractmethod
    def iterate(self, state: DFTState, hamiltonian) -> float:
        pass

    def reset(self) -> None:
        pass

    def update_to_canonical_orbitals(
        self,
        state: DFTState,
        hamiltonian
    ) -> None:
        pass
