from __future__ import annotations

from abc import ABC
from typing import Generator, NamedTuple

import numpy as np
from numpy.linalg import solve

from ase.units import Bohr, Hartree

from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.external import ExternalPotential, ConstantElectricField
from gpaw.typing import Vector
from gpaw.mpi import world
from gpaw.new.ase_interface import ASECalculator
from gpaw.new.calculation import DFTState, DFTCalculation
from gpaw.new.fd.builder import FDHamiltonian
from gpaw.new.fd.pot_calc import UniformGridPotentialCalculator
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.lcao.hamiltonian import (HamiltonianMatrixCalculator,
                                       LCAOHamiltonian)
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.wave_functions import WaveFunctions
from gpaw.new.gpw import read_gpw
from gpaw.new.symmetry import Symmetries
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.pw.builder import PWHamiltonian
from gpaw.tddft.units import asetime_to_autime, autime_to_asetime, au_to_eA
from gpaw.utilities.timing import nulltimer


class TDAlgorithm:

    def kick(self,
             state: DFTState,
             pot_calc: PotentialCalculator,
             wf_kicker: WaveFunctionKicker):
        raise NotImplementedError()

    def propagate(self,
                  time_step: float,
                  state: DFTState,
                  pot_calc: PotentialCalculator,
                  wf_propagator: WaveFunctionPropagator):
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


class WaveFunctionKicker(ABC):
    """
    Takes care about applying delta-kick to wave functions

    Implementations are specific to parallelization scheme (Numpy) and
    type of wave functions (LCAO/FD)

    """

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: DFTState,
                 ext: ExternalPotential,
                 pot_calc: UniformGridPotentialCalculator):
        # XXX Maybe there is no point that this is separate
        # from WaveFunctionPropagator?
        raise NotImplementedError

    def kick(self,
             wfs: WaveFunctions,
             nkicks: int):
        raise NotImplementedError


class WaveFunctionPropagator(ABC):
    """
    Takes care about propagating wave functions

    Implementations are specific to parallelization scheme (Numpy) and
    type of wave functions (LCAO/FD)
    """

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: DFTState):
        raise NotImplementedError

    def propagate(self,
                  wfs: WaveFunctions,
                  time_step: float):
        raise NotImplementedError


class LCAONumpyKicker(WaveFunctionKicker):

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: DFTState,
                 ext: ExternalPotential,
                 pot_calc: UniformGridPotentialCalculator):
        assert isinstance(hamiltonian, LCAOHamiltonian)
        dm_operator_calc = hamiltonian.create_kick_matrix_calculator(
            state, ext, pot_calc)
        self.dm_calc = dm_operator_calc

    def kick(self,
             wfs: WaveFunctions,
             nkicks: int):
        assert isinstance(wfs, LCAOWaveFunctions)
        V_MM = self.dm_calc.calculate_matrix(wfs)

        # Phi_n <- U(0+, 0) Phi_n
        nkicks = 10
        for i in range(nkicks):
            propagate_wave_functions_numpy(wfs.C_nM.data, wfs.C_nM.data,
                                           wfs.S_MM.data,
                                           V_MM.data, 1 / nkicks)


class LCAONumpyPropagator(WaveFunctionPropagator):

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: DFTState):
        assert isinstance(hamiltonian, LCAOHamiltonian)
        ham_calc = hamiltonian.create_hamiltonian_matrix_calculator(state)
        self.ham_calc = ham_calc

    def propagate(self,
                  wfs: WaveFunctions,
                  time_step: float):
        assert isinstance(wfs, LCAOWaveFunctions)
        H_MM = self.ham_calc.calculate_matrix(wfs)

        # Phi_n <- U[H(t)] Phi_n
        propagate_wave_functions_numpy(wfs.C_nM.data, wfs.C_nM.data,
                                       wfs.S_MM.data,
                                       H_MM.data, time_step)


class FDNumpyKicker(WaveFunctionKicker):

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: DFTState,
                 ext: ExternalPotential,
                 pot_calc: UniformGridPotentialCalculator):
        from gpaw.utilities import unpack

        assert isinstance(hamiltonian, FDHamiltonian)
        vext_r = pot_calc.vbar_r.new()
        finegd = vext_r.desc._gd

        vext_r.data = ext.get_potential(finegd)
        vext_R = pot_calc.restrict(vext_r)

        nspins = state.ibzwfs.nspins

        W_aL = pot_calc.ghat_aLr.integrate(vext_r)

        assert state.ibzwfs.ibz.bz.gamma_only
        setups_a = state.ibzwfs.wfs_qs[0][0].setups

        dH_saii = [{a: unpack(setups_a[a].Delta_pL @ W_L)
                    for (a, W_L) in W_aL.items()}
                   for s in range(nspins)]

        self.vext_R = vext_R
        self.dH_saii = dH_saii

    def kick(self,
             wfs: WaveFunctions,
             nkicks: int):
        assert isinstance(wfs, PWFDWaveFunctions)
        assert isinstance(wfs.psit_nX, UniformGridFunctions)

        wfs.psit_nX.data += wfs.psit_nX.data * self.vext_R.data[None, ...]
        # TODO PAW stuff


class FDNumpyPropagator(WaveFunctionPropagator):

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: DFTState):
        """
        Parameters
        ----------
        hamiltonian
        """
        assert isinstance(hamiltonian, FDHamiltonian)

        self.timer = nulltimer
        self.preconditioner = None

        self.hamiltonian = hamiltonian
        self.vt_sR = state.potential.vt_sR

        # XXX Ugly hack due to having reused the CG solver
        self._wfs: PWFDWaveFunctions | None = None
        self._time_step: float | None = None

    @property
    def time_step(self) -> float:
        # XXX This is an ugly hack that I will remove after rewriting the
        # conjugate gradient solver
        if self._time_step is None:
            raise RuntimeError('One needs to run propagate before something '
                               'that uses time step')
        return self._time_step

    @property
    def wfs(self) -> PWFDWaveFunctions:
        # XXX This is an ugly hack that I will remove after rewriting the
        # conjugate gradient solver
        if self._wfs is None:
            raise RuntimeError('One needs to run propagate before something '
                               'that uses wfs')
        return self._wfs

    def propagate(self,
                  wfs: WaveFunctions,
                  time_step: float):
        assert isinstance(wfs, PWFDWaveFunctions)
        assert isinstance(wfs.psit_nX, UniformGridFunctions)
        psit_nR = wfs.psit_nX

        self._time_step = time_step
        copy_psit_nR = psit_nR.new()
        copy_psit_nR.data[:] = psit_nR.data

        # Update the projector function overlap integrals
        wfs.pt_aiX.integrate(psit_nR, wfs.P_ani)

        # Empty arrays
        rhs_nR = psit_nR.new()
        init_guess_nR = psit_nR.new()
        hpsit_nR = psit_nR.new()
        spsit_nR = psit_nR.new()
        sinvhpsit_nR = psit_nR.new()

        # Calculate right-hand side of equation
        # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
        self.apply_hamiltonian(wfs, hpsit_nR)
        self.apply_overlap_operator(wfs, spsit_nR)

        rhs_nR.data[:] = spsit_nR.data - 0.5j * time_step * hpsit_nR.data

        # Calculate (1 - i S^(-1) H dt) psit(t), which is an
        # initial guess for the conjugate gradient solver
        wfs.psit_nX.data[:] = hpsit_nR.data
        self.apply_inverse_overlap_operator(wfs, sinvhpsit_nR)
        init_guess_nR.data[:] = (copy_psit_nR.data -
                                 1j * time_step * sinvhpsit_nR.data)

        from gpaw.tddft.solvers.cscg import CSCG
        solver = CSCG()
        solver.initialize(psit_nR.desc._gd, nulltimer)
        # Solve A x = b where A is (S + i H dt/2) and b = rhs_kpt.psit_nG
        # A needs to implement the function dot, which operates
        # on wave functions
        psit_nR.data[:] = init_guess_nR.data
        solver.solve(self, copy_psit_nR.data, rhs_nR.data)
        wfs.psit_nX.data[:] = copy_psit_nR.data

        wfs.pt_aiX.integrate(wfs.psit_nX, wfs.P_ani)

    def dot(self, psit_nR: np.ndarray, out_nR: np.ndarray):
        """Applies the propagator matrix to the given wavefunctions.

        (S + i H dt/2 ) psi

        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result ( S + i H dt/2 ) psi

        """
        assert isinstance(self.wfs.psit_nX, UniformGridFunctions)

        self.timer.start('Apply time-dependent operators')
        # Update the projector function overlap integrals
        self.wfs.psit_nX.data[:] = psit_nR  # XXX very ugly hack
        self.wfs.pt_aiX.integrate(self.wfs.psit_nX, self.wfs.P_ani)
        hpsit_nR = self.wfs.psit_nX.new()
        spsit_nR = self.wfs.psit_nX.new()

        self.apply_hamiltonian(self.wfs, hpsit_nR)
        self.apply_overlap_operator(self.wfs, spsit_nR)
        self.timer.stop('Apply time-dependent operators')

        out_nR[:] = spsit_nR.data + 0.5j * self.time_step * hpsit_nR.data

    def apply_preconditioner(self, psi, psin):
        """Solves preconditioner equation.

        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result

        """
        self.timer.start('Solve TDDFT preconditioner')
        if self.preconditioner is not None:
            self.preconditioner.apply(self.kpt, psi, psin)
        else:
            psin[:] = psi
        self.timer.stop('Solve TDDFT preconditioner')

    def apply_hamiltonian(self,
                          wfs: PWFDWaveFunctions,
                          out_nR: UniformGridFunctions | None = None):
        """
        Apply the hamiltonian on wave functions

        Parameters
        ----------
        wfs
            Wave functions
        out_nR
            Result, i.e. hamiltonian acting on wavefunctions
            If None then the wave functions are overwritten (result
            is written into wfs.psit_nX)
        """
        assert isinstance(wfs.psit_nX, UniformGridFunctions)

        if out_nR is None:
            out_nR = wfs.psit_nX
        else:
            out_nR.data[:] = wfs.psit_nX.data

        self.hamiltonian.apply(
            self.vt_sR, wfs.psit_nX, out_nR, wfs.spin)
        self._wfs = wfs  # XXX temporary hack

    @staticmethod
    def apply_overlap_operator(
            wfs: PWFDWaveFunctions,
            out_nR: UniformGridFunctions | None = None):
        """
        Apply the overlap operator wave functions on the wave functions:

        ^  ~         ~             ~ a       a   a
        S |ψ (t)〉= |ψ (t)〉+  Σ  |p  (t)〉ΔO   P
            n         n       aij   i        ij  nj

        Parameters
        ----------
        wfs
            Wave functions
        out_nR
            Result, i.e. overlap operator acting on wavefunctions
            If None then the wave functions are overwritten (result
            is written into wfs.psit_nX)
        """
        assert isinstance(wfs.psit_nX, UniformGridFunctions)

        if out_nR is None:
            out_nR = wfs.psit_nX
        else:
            out_nR.data[:] = wfs.psit_nX.data

        P2_ani = wfs.P_ani.new()
        dS = wfs.setups.overlap_correction
        dS(wfs.P_ani, out_ani=P2_ani)  # P2 is ΔO_ij @ P_nj
        wfs.pt_aiX.add_to(out_nR, P2_ani)

    @staticmethod
    def apply_inverse_overlap_operator(
            wfs: PWFDWaveFunctions,
            out_nR: UniformGridFunctions | None = None):
        """
        Apply the approximate inverse overlap operator on the wave functions:

        ^ -1  ~         ~             ~ a       a   a
        S    |ψ (t)〉= |ψ (t)〉+  Σ  |p  (t)〉ΔC   P
               n         n       aij   i        ij  nj

        Parameters
        ----------
        wfs
            Wave functions
        out_nR
            Result, i.e. inverse overlap operator acting on wavefunctions
            If None then the wave functions are overwritten (result
            is written into wfs.psit_nX)
        """
        assert isinstance(wfs.psit_nX, UniformGridFunctions)

        if out_nR is None:
            out_nR = wfs.psit_nX
        else:
            out_nR.data[:] = wfs.psit_nX.data

        P2_ani = wfs.P_ani.new()
        dSinv = wfs.setups.inverse_overlap_correction
        dSinv(wfs.P_ani, out_ani=P2_ani)  # P2 is ΔC_ij @ P_nj
        wfs.pt_aiX.add_to(out_nR, P2_ani)


class ECNAlgorithm(TDAlgorithm):

    def kick(self,
             state: DFTState,
             pot_calc: PotentialCalculator,
             wf_kicker: WaveFunctionKicker):
        """Propagate wavefunctions by delta-kick.

        ::

                                 0+
                     ^        -1 /                     ^        -1
          U(0+, 0) = T exp[-iS   | δ(τ) V   (r) dτ ] = T exp[-iS  V   (r)]
                                 /       ext                       ext
                                 0

        (1) Calculate propagator U(0+, 0)
        (2) Update wavefunctions ψ_n(0+) ← U(0+, 0) ψ_n(0)
        (3) Update density and hamiltonian H(0+)

        Parameters
        ----------
        state
            Current state of the wave functions, that is to be updated
        pot_calc
            The potential calculator
        wf_kicker
            Object that describes how to kick. Contains the dipole moment
            operator and a field strength
        """
        for wfs in state.ibzwfs:
            wf_kicker.kick(wfs, 10)

        # Update density
        state.density.update(pot_calc.nct_R, state.ibzwfs)

        # Calculate Hamiltonian H(t+dt) = H[n[Phi_n]]
        state.potential, state.vHt_x, _ = pot_calc.calculate(
            state.density, state.vHt_x)

    def propagate(self,
                  time_step: float,
                  state: DFTState,
                  pot_calc: PotentialCalculator,
                  wf_propagator: WaveFunctionPropagator):
        """ One explicit Crank-Nicolson propagation step, i.e.

        (1) Calculate propagator U[H(t)]
        (2) Update wavefunctions ψ_n(t+dt) ← U[H(t)] ψ_n(t)
        (3) Update density and hamiltonian H(t+dt)
        """
        for wfs in state.ibzwfs:
            wf_propagator.propagate(wfs, time_step)

        # Update density
        state.density.update(pot_calc.nct_R, state.ibzwfs)

        # Calculate Hamiltonian H(t+dt) = H[n[Phi_n]]
        state.potential, state.vHt_x, _ = pot_calc.calculate(
            state.density, state.vHt_x)


class RTTDDFTHistory:

    kick_strength: Vector | None  # Kick strength in atomic units
    niter: int  # Number of propagation steps
    time: float  # Simulation time in atomic units

    def __init__(self):
        """Object that keeps track of the RT-TDDFT history, that is

        - Has a kick been performed?
        - The number of propagation states performed
        """
        self.kick_strength = None
        self.niter = 0
        self.time = 0.0

    def absorption_kick(self,
                        kick_strength: Vector):
        """ Store the kick strength in history

        At most one kick can be done, and it must happen before any
        propagation steps

        Parameters
        ----------
        kick_strength
            Strength of the kick in atomic units
        """
        assert self.niter == 0, 'Cannot kick if already propagated'
        assert self.kick_strength is None, 'Cannot kick if already kicked'
        self.kick_strength = np.array(kick_strength, dtype=float)

    def propagate(self,
                  time_step: float) -> float:
        """ Increment the number of propagation steps and simulation time
        in history

        Parameters
        ----------
        time_step
            Time step in atomic units

        Returns
        -------
        The new simulation time in atomic units
        """
        self.niter += 1
        self.time += time_step

        return self.time

    def todict(self):
        absorption_kick = self.absorption_kick
        if absorption_kick is not None:
            absorption_kick = absorption_kick.tolist()
        return {'niter': self.niter, 'time': self.time,
                'absorption_kick': absorption_kick}


class RTTDDFTResult(NamedTuple):

    """ Results are stored in atomic units, but displayed to the user in
    ASE units
    """

    time: float  # Time in atomic units
    dipolemoment: Vector  # Dipole moment in atomic units

    def __repr__(self):
        timestr = f'{self.time * autime_to_asetime:.3f} Å√(u/eV)'
        dmstr = ', '.join([f'{dm * au_to_eA:10.4g}'
                           for dm in self.dipolemoment])
        dmstr = f'[{dmstr}]'

        return (f'{self.__class__.__name__}: '
                f'(time: {timestr}, dipolemoment: {dmstr} eÅ)')


class RTTDDFT:
    def __init__(self,
                 state: DFTState,
                 pot_calc: PotentialCalculator,
                 hamiltonian,
                 history: RTTDDFTHistory,
                 td_algorithm: TDAlgorithm | None = None):
        if td_algorithm is None:
            td_algorithm = ECNAlgorithm()

        # Disable symmetries, ie keep only identity operation
        # I suppose this should be done in the kick, as the kick breaks
        # the symmetry
        symmetry = state.ibzwfs.ibz.symmetries.symmetry
        natoms = symmetry.a_sa.shape[1]
        symmetry.op_scc = np.eye(3)[None, ...]
        symmetry.ft_sc = np.zeros((1, 3))
        symmetry.a_sa = np.arange(natoms)[None, ...]
        state.ibzwfs.ibz.symmetries = Symmetries(symmetry)

        self.state = state
        self.pot_calc = pot_calc
        self.td_algorithm = td_algorithm
        self.hamiltonian = hamiltonian
        self.history = history

        self.kick_ext: ExternalPotential | None = None

        if isinstance(hamiltonian, LCAOHamiltonian):
            self.calculate_dipole_moment = self._calculate_dipole_moment_lcao
            self.mode = 'lcao'
        elif isinstance(hamiltonian, FDHamiltonian):
            self.calculate_dipole_moment = self._calculate_dipole_moment
            self.mode = 'fd'
        elif isinstance(hamiltonian, PWHamiltonian):
            raise NotImplementedError('PW TDDFT is not implemented')
        else:
            raise ValueError(f'I don\'t know {hamiltonian} '
                             f'({type(hamiltonian)})')
        # Dipole moment operators in each Cartesian direction
        # Only usable for LCAO
        # TODO is there even a point in caching these? I don't think it saves
        # much time
        self.dm_operator_c: list[HamiltonianMatrixCalculator] | None = None

        self.timer = nulltimer
        self.log = print

    @classmethod
    def from_dft_calculation(cls,
                             calc: ASECalculator | DFTCalculation,
                             td_algorithm: TDAlgorithm | None = None):

        if isinstance(calc, DFTCalculation):
            calculation = calc
        else:
            assert calc.calculation is not None
            calculation = calc.calculation

        state = calculation.state
        pot_calc = calculation.pot_calc
        hamiltonian = calculation.scf_loop.hamiltonian
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian, td_algorithm=td_algorithm,
                   history=history)

    @classmethod
    def from_dft_file(cls,
                      filepath: str,
                      td_algorithm: TDAlgorithm | None = None):
        _, calculation, params, builder = read_gpw(filepath,
                                                   '-',
                                                   {'world': world},
                                                   dtype=complex)

        state = calculation.state
        pot_calc = calculation.pot_calc
        hamiltonian = builder.create_hamiltonian_operator()
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian, td_algorithm=td_algorithm,
                   history=history)

    def absorption_kick(self,
                        kick_strength: Vector):
        """Kick with a weak electric field.

        Parameters
        ----------
        kick_strength
            Strength of the kick in atomic units
        """
        with self.timer('Kick'):
            kick_strength = np.array(kick_strength, dtype=float)
            self.history.absorption_kick(kick_strength)

            magnitude = np.sqrt(np.sum(kick_strength**2))
            direction = kick_strength / magnitude
            dirstr = [f'{d:.4f}' for d in direction]

            self.log('----  Applying absorption kick')
            self.log(f'----  Magnitude: {magnitude:.8f} Hartree/Bohr')
            self.log(f'----  Direction: {dirstr}')

            # Create hamiltonian object for absorption kick
            cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)

            # Propagate kick
            return self.kick(cef)

    def kick(self,
             ext: ExternalPotential):
        """Kick with any external potential.

        Note that unless this function is called by absorption_kick, the kick
        is not logged in history

        Parameters
        ----------
        ext
            External potential
        """
        with self.timer('Kick'):
            self.log('----  Applying kick')
            self.log(f'----  {ext}')

            cls: type[WaveFunctionKicker]
            if self.mode == 'lcao':
                cls = LCAONumpyKicker
            elif self.mode == 'fd':
                cls = FDNumpyKicker
            else:
                raise RuntimeError(f'Mode {self.mode} is unexpected')

            assert isinstance(self.pot_calc, UniformGridPotentialCalculator)
            wf_kicker = cls(self.hamiltonian, self.state, ext, self.pot_calc)
            self.kick_ext = ext

            # Propagate kick
            self.td_algorithm.kick(state=self.state,
                                   pot_calc=self.pot_calc,
                                   wf_kicker=wf_kicker)
            dipolemoment_xv = [
                self.calculate_dipole_moment(wfs)  # type: ignore
                for wfs in self.state.ibzwfs]
            dipolemoment_v = np.sum(dipolemoment_xv, axis=0)
            result = RTTDDFTResult(time=0, dipolemoment=dipolemoment_v)
            return result

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
            cls: type[WaveFunctionPropagator]
            if self.mode == 'lcao':
                cls = LCAONumpyPropagator
            elif self.mode == 'fd':
                cls = FDNumpyPropagator
            else:
                raise RuntimeError(f'Mode {self.mode} is unexpected')

            wf_propagator = cls(self.hamiltonian, self.state)

            self.td_algorithm.propagate(time_step,
                                        state=self.state,
                                        pot_calc=self.pot_calc,
                                        wf_propagator=wf_propagator)
            time = self.history.propagate(time_step)
            dipolemoment_xv = [
                self.calculate_dipole_moment(wfs)  # type: ignore
                for wfs in self.state.ibzwfs]
            dipolemoment_v = np.sum(dipolemoment_xv, axis=0)
            result = RTTDDFTResult(time=time, dipolemoment=dipolemoment_v)
            yield result

    def _calculate_dipole_moment(self, wfs: WaveFunctions) -> np.ndarray:
        dipolemoment_v = self.state.density.calculate_dipole_moment(
            self.pot_calc.fracpos_ac)

        return dipolemoment_v

    def _calculate_dipole_moment_lcao(self,
                                      wfs: LCAOWaveFunctions) -> np.ndarray:
        """ Calculates the dipole moment

        The dipole moment is calculated as the expectation value of the
        dipole moment operator, i.e. the trace of it times the density matrix::

          d = - Σ  ρ   d
                μν  μν  νμ

        """
        assert isinstance(wfs, LCAOWaveFunctions)
        if self.dm_operator_c is None:
            self.dm_operator_c = []

            # Create external potentials in each direction
            ext_c = [ConstantElectricField(Hartree / Bohr, dir)
                     for dir in np.eye(3)]
            dm_operator_c = [self.hamiltonian.create_kick_matrix_calculator(
                self.state, ext, self.pot_calc) for ext in ext_c]
            self.dm_operator_c = dm_operator_c

        dm_v = np.zeros(3)
        for c, dm_operator in enumerate(self.dm_operator_c):
            rho_MM = wfs.calculate_density_matrix()
            dm_MM = dm_operator.calculate_matrix(wfs)
            dm = - np.einsum('MN,NM->', rho_MM, dm_MM.data)
            assert np.abs(dm.imag) < 1e-20
            dm_v[c] = dm.real

        return dm_v
