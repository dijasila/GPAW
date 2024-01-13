from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gpaw.new.calculation import DFTState


class Eigensolver(ABC):
    @abstractmethod
    def iterate(self, state: DFTState, hamiltonian) -> float:
        pass
