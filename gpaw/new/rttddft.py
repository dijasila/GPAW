from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np
from numpy.linalg import solve

from gpaw.typing import Vector
from gpaw.new.ase_interface import ASECalculator
from gpaw.new.calculation import DFTState, DFTCalculation
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.tddft.units import asetime_to_autime, autime_to_asetime, au_to_eA


class TDAlgorithm:

    def propagate(self,
                  time_step: float,
                  state: DFTState,
                  pot_calc: PotentialCalculator,
                  hamiltonian):
        raise NotImplementedError()

    def get_description(self):
        return '%s' % self.__class__.__name__


def propagate_wave_functions_numpy(source_C_nM: np.ndarray,
                                   target_C_nM: np.ndarray,
                                   S_MM: np.ndarray,
                                   H_MM: np.ndarray,
                                   dt: float):
    SjH_MM = S_MM + (0.5j * dt) * H_MM
    target_C_nM[:] = source_C_nM @ SjH_MM.conj().T
    target_C_nM[:] = solve(SjH_MM.T, target_C_nM.T).T


class ECNAlgorithm(TDAlgorithm):

    def propagate(self,
                  time_step: float,
                  state: DFTState,
                  pot_calc: PotentialCalculator,
                  hamiltonian):
        matrix_calculator = hamiltonian.create_hamiltonian_matrix_calculator(
            state)
        for wfs in state.ibzwfs:
            H_MM = matrix_calculator.calculate_hamiltonian_matrix(wfs)

            # Phi_n <- U[H(t)] Phi_n
            propagate_wave_functions_numpy(wfs.C_nM.data, wfs.C_nM.data,
                                           wfs.S_MM,
                                           H_MM, time_step)
        # Update density
        state.density.update(pot_calc.nct_R, state.ibzwfs)

        # Calculate Hamiltonian H(t+dt) = H[n[Phi_n]]
        state.potential, state.vHt_x, _ = pot_calc.calculate(
            state.density, state.vHt_x)


class RTTDDFTResult(NamedTuple):

    """ Results are stored in atomic units, but displayed to the user in
    ASE units
    """

    time: float  # Time in atomic units
    dipolemoment: Vector  # Dipole moment in atomic units

    def __repr__(self):
        timestr = f'{self.time*autime_to_asetime:.3f}Å√(u/eV)'
        dmstr = ', '.join([f'{dm*au_to_eA:.3f}' for dm in self.dipolemoment])
        dmstr = f'[{dmstr}]'

        return (f'{self.__class__.__name__}: '
                f'(time: {timestr}, dipolemoment: {dmstr}eÅ)')


class RTTDDFT:
    def __init__(self,
                 state: DFTState,
                 pot_calc: PotentialCalculator,
                 hamiltonian,
                 propagator: TDAlgorithm | None = None):
        self.time = 0.0

        if propagator is None:
            propagator = ECNAlgorithm()

        self.state = state
        self.pot_calc = pot_calc
        self.propagator = propagator
        self.hamiltonian = hamiltonian

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
        hamiltonian = calculation.scf_loop.hamiltonian

        return cls(state, pot_calc, hamiltonian, propagator=propagator)

    def ipropagate(self,
                   time_step: float = 10.0,
                   maxiter: int = 2000,
                   ) -> Generator[RTTDDFTResult, None, None]:
        """Propagate the electronic system.

        Parameters
        ----------
        time_step
            Time step in ASE time units Å√(u/eV)
        iterations
            Number of propagation steps
        """

        time_step = time_step * asetime_to_autime

        for iteration in range(maxiter):
            self.propagator.propagate(time_step,
                                      state=self.state,
                                      pot_calc=self.pot_calc,
                                      hamiltonian=self.hamiltonian)
            time = self.time + time_step
            self.time = time
            dipolemoment = self.state.density.calculate_dipole_moment()
            result = RTTDDFTResult(time=time, dipolemoment=dipolemoment)
            yield result
