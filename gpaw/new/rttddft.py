from __future__ import annotations

from typing import Generator, NamedTuple

from gpaw.new.ase_interface import ASECalculator
from gpaw.new.calculation import DFTState, DFTCalculation
from gpaw.new.pot_calc import PotentialCalculator


class TDAlgorithm:

    def propagate(self,
                  time_step: float,
                  state: DFTState,
                  pot_calc: PotentialCalculator):
        raise NotImplementedError()

    def get_description(self):
        return '%s' % self.__class__.__name__


class SICNAlgorithm(TDAlgorithm):

    def propagate(self,
                  time_step: float,
                  state: DFTState,
                  pot_calc: PotentialCalculator):
        pass


class RTTDDFTResult(NamedTuple):

    time: float


class RTTDDFT:
    def __init__(self,
                 state: DFTState,
                 pot_calc: PotentialCalculator,
                 propagator: TDAlgorithm | None = None):
        self.time = 0.0

        if propagator is None:
            propagator = SICNAlgorithm()

        self.state = state
        self.pot_calc = pot_calc
        self.propagator = propagator

    @classmethod
    def from_dft_calculation(cls,
                             calc: ASECalculator | DFTCalculation,
                             propagator: TDAlgorithm | None = None):

        if isinstance(calc, DFTCalculation):
            calculation = calc
        else:
            assert calc.calculation is not None
            calculation = calc.calculation

        state = calculation.state
        pot_calc = calculation.pot_calc

        return cls(state, pot_calc, propagator=propagator)

    def ipropagate(self,
                   time_step: float = 10.0,
                   maxiter: int = 2000,
                   ) -> Generator[RTTDDFTResult, None, None]:
        """Propagate the electronic system.

        Parameters
        ----------
        time_step
            Time step in attoseconds
        iterations
            Number of propagation steps
        """

        for iteration in range(maxiter):
            self.propagator.propagate(time_step, self.state, self.pot_calc)
            time = self.time + time_step
            self.time = time
            result = RTTDDFTResult(time=time)
            yield result
