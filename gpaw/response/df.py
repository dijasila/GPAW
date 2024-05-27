from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys

import numpy as np
from ase.units import Hartree, Bohr

import gpaw.mpi as mpi

from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.dyson import DysonEquation
from gpaw.response.density_kernels import get_density_xc_kernel
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
from gpaw.response.chi0_data import Chi0Data
from gpaw.response.pair import get_gs_and_context

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpaw.response.frequencies import FrequencyDescriptor
    from gpaw.response.pair_functions import SingleQPWDescriptor


@dataclass
class Chi0DysonEquations:
    chi0: Chi0Data
    df: 'DielectricFunctionCalculator'

    def __post_init__(self):
        self.gs = self.df.gs
        self.context = self.df.context
        self.coulomb = self.df.coulomb
        # When inverting the Dyson equation, we distribute frequencies globally
        blockdist = self.chi0.body.blockdist.new_distributor(nblocks='max')
        self.wblocks = Blocks1D(blockdist.blockcomm, len(self.chi0.wd))

    @staticmethod
    def _normalize(direction):
        if isinstance(direction, str):
            d_v = {'x': [1, 0, 0],
                   'y': [0, 1, 0],
                   'z': [0, 0, 1]}[direction]
        else:
            d_v = direction
        d_v = np.asarray(d_v) / np.linalg.norm(d_v)
        return d_v

    def get_chi0_wGG(self, direction='x'):
        chi0 = self.chi0
        chi0_wGG = chi0.body.get_distributed_frequencies_array().copy()
        if chi0.qpd.optical_limit:
            # Project head and wings along the input direction
            d_v = self._normalize(direction)
            W_w = self.wblocks.myslice
            chi0_wGG[:, 0] = np.dot(d_v, chi0.chi0_WxvG[W_w, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0.chi0_WxvG[W_w, 1])
            chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0.chi0_Wvv[W_w], d_v).T)
        return chi0_wGG

    def get_Kxc_GG(self, *, xc, chi0_wGG, **kwargs):
        """Get adiabatic xc kernel (TDDFT).

        Choose between ALDA, Bootstrap and LRalpha (long-range kernel), where
        alpha is a user specified parameter (for example xc='LR0.25')."""
        return get_density_xc_kernel(
            self.chi0.qpd, self.gs, self.context,
            functional=xc, chi0_wGG=chi0_wGG, **kwargs)

    def get_coulomb_scaled_kernel(self, xc='RPA', **xckwargs):
        """Get the Hxc kernel rescaled by the bare Coulomb potential V(q).

        Calculates
        ˷
        K(q) = V^(-1/2)(q) K_Hxc(q) V^(-1/2)(q),

        where V(q) is the bare Coulomb potential and

        K_Hxc(q) = K_H(q) + K_xc(q),

        where the Hartree kernel itself might be truncated.
        """
        qpd = self.chi0.qpd
        if self.coulomb.truncation is None:
            V_G = self.coulomb.V(qpd)
            K_GG = np.eye(len(V_G), dtype=complex)
        else:
            coulomb = self.coulomb.new(truncation=None)
            V_G = coulomb.V(qpd)
            Vtrunc_G = self.coulomb.V(qpd)
            K_GG = np.diag(Vtrunc_G / V_G)
        if xc != 'RPA':
            Kxc_GG = self.get_Kxc_GG(xc=xc, **xckwargs)
            sqrtV_G = V_G**0.5
            K_GG += Kxc_GG / sqrtV_G / sqrtV_G[:, np.newaxis]
        return V_G, K_GG

    @staticmethod
    def invert_dyson_like_equation(in_wGG, K_GG, reuse_buffer=True):
        """Generalized Dyson equation invertion.

        Calculates

        B(q,ω) = [1 - A(q,ω) K(q)]⁻¹ A(q,ω)

        while possibly storing the output B(q,ω) in the input A(q,ω) buffer.
        """
        if reuse_buffer:
            out_wGG = in_wGG
        else:
            out_wGG = np.zeros_like(in_wGG)
        for w, in_GG in enumerate(in_wGG):
            out_wGG[w] = DysonEquation(in_GG, in_GG @ K_GG).invert()
        return out_wGG

    def rpa_density_response(self, direction='x', qinf_v=None):
        """Calculate the RPA susceptibility for (semi-)finite q."""
        # Extract χ₀(q,ω)
        qpd = self.chi0.qpd
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        if qpd.optical_limit:
            # Restore the q-dependence of the head and wings in the q→0 limit
            assert qinf_v is not None and np.linalg.norm(qinf_v) > 0.
            d_v = self._normalize(direction)
            chi0_wGG[:, 1:, 0] *= np.dot(qinf_v, d_v)
            chi0_wGG[:, 0, 1:] *= np.dot(qinf_v, d_v)
            chi0_wGG[:, 0, 0] *= np.dot(qinf_v, d_v)**2
        # Invert Dyson equation, χ(q,ω) = [1 - χ₀(q,ω) V(q)]⁻¹ χ₀(q,ω)
        V_GG = self.coulomb.kernel(qpd, q_v=qinf_v)
        chi_wGG = self.invert_dyson_like_equation(chi0_wGG, V_GG)
        return qpd, chi_wGG, self.wblocks

    def inverse_dielectric_function(self, *args, **kwargs):
        """Calculate V^(1/2) χ V^(1/2), from which ε⁻¹(q,ω) is constructed."""
        return InverseDielectricFunction.from_chi0_dyson_eqs(
            self, *self.calculate_Vchi_symm(*args, **kwargs))

    def calculate_Vchi_symm(self, xc='RPA', direction='x', **xckwargs):
        """Calculate V^(1/2) χ V^(1/2).

        Starting from the TDDFT Dyson equation

        χ(q,ω) = χ₀(q,ω) + χ₀(q,ω) K_Hxc(q,ω) χ(q,ω),                (1)

        the Coulomb scaled susceptibility,
        ˷
        χ(q,ω) = V^(1/2)(q) χ(q,ω) V^(1/2)(q)

        can be calculated from the Dyson-like equation
        ˷        ˷         ˷       ˷      ˷
        χ(q,ω) = χ₀(q,ω) + χ₀(q,ω) K(q,ω) χ(q,ω)                     (2)

        where
        ˷
        K(q,ω) = V^(-1/2)(q) K_Hxc(q,ω) V^(-1/2)(q).

        Here V(q) refers to the bare Coulomb potential. It should be emphasized
        that invertion of (2) rather than (1) is not merely a rescaling
        excercise. In the optical q → 0 limit, the Coulomb kernel V(q) diverges
        as 1/|G+q|² while the Kohn-Sham susceptibility χ₀(q,ω) vanishes as
        |G+q|². Treating V^(1/2)(q) χ₀(q,ω) V^(1/2)(q) as a single variable,
        the effects of this cancellation can be treated accurately within k.p
        perturbation theory.
        """
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        V_G, K_GG = self.get_coulomb_scaled_kernel(
            xc=xc, chi0_wGG=chi0_wGG, **xckwargs)
        # Calculate V^(1/2)(q) χ₀(q,ω) V^(1/2)(q)
        sqrtV_G = V_G**0.5
        Vchi0_symm_wGG = chi0_wGG  # reuse buffer
        for w, chi0_GG in enumerate(chi0_wGG):
            Vchi0_symm_wGG[w] = chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
        # Invert Dyson equation
        Vchi_symm_wGG = self.invert_dyson_like_equation(
            Vchi0_symm_wGG, K_GG, reuse_buffer=False)
        return Vchi0_symm_wGG, Vchi_symm_wGG, V_G

    def dielectric_function(self, *args, **kwargs):
        """Construct the dielectric function as ε(q,ω) = 1 - V(q) P(q,ω)."""
        V_G = self.coulomb.V(self.chi0.qpd)
        if abs(V_G[0]) > 1e-8:
            return self._dielectric_function(*args, **kwargs)
        else:
            return self._modified_dielectric_function(*args, **kwargs)

    def _dielectric_function(self, *args, **kwargs):
        """Calculate ε(q,ω) = 1 - V(q) P(q,ω) literally."""
        V_GG = self.coulomb.kernel(self.chi0.qpd)
        P_wGG = self.polarizability_operator(*args, **kwargs)
        nG = len(V_GG)
        eps_wGG = P_wGG  # reuse buffer
        for w, P_GG in enumerate(P_wGG):
            eps_wGG[w] = np.eye(nG) - V_GG @ P_GG
        return DielectricFunctionData.from_chi0_dyson_eqs(self, eps_wGG)

    def _modified_dielectric_function(self, xc='RPA', *args, **kwargs):
        """Calculate ε(q,ω) using modified response functions.

        When using a modified Coulomb potential

        ˍ      ( 0       for G = 0
        V(q) = <
               ( V(q)    for G > 0

        one can introduce the local-field corrected dielectric function
        ˍ                 ˍ
        ϵ(q,ω) = 1 - V(q) P(q,ω)

        defined so as to yield the correct macroscopic dielectric function
        ε_M(q,ω) for G=G'=0, including all local-field effects according to the
        unmodified Coulomb potential V(q) [Rev. Mod. Phys. 74, 601 (2002)].
                                             ˍ
        The modified polarizability operator P(q,ω) is itself given by the
        Dyson-like equation
        ˍ                        ˍ    ˍ
        P(q,ω) = P(q,ω) + P(q,ω) V(q) P(q,ω).                        (3)

        In the special of RPA, where P(q,ω) = χ₀(q,ω), one may notice that the
        Dyson-like equation (3) is exactly identical to the TDDFT Dyson
        equation (1) when replacing the Hartree kernel with the modified
        Coulomb interaction:
                  ˍ
        K_H(q) -> V(q).
                                                                  ˍ
        We may thus reuse that functionality to calculate V^(1/2) P V^(1/2)
                   ˍ
        from which ϵ(q,ω) can be constructed.
        """
        assert xc == 'RPA'
        VP_symm_wGG, VPbar_symm_wGG, _ = self.calculate_Vchi_symm(
            xc=xc, *args, **kwargs)
        return ModifiedDielectricFunction.from_chi0_dyson_eqs(
            self, VP_symm_wGG, VPbar_symm_wGG)

    def polarizability_operator(self, xc='RPA', direction='x', **xckwargs):
        """Calculate the polarizability operator P(q,ω).

        Depending on the theory (RPA, TDDFT, MBPT etc.), the polarizability
        operator is approximated in various ways see e.g.
        [Rev. Mod. Phys. 74, 601 (2002)].

        In RPA:
            P(q,ω) = χ₀(q,ω)

        In TDDFT:
            P(q,ω) = [1 - χ₀(q,ω) K_xc(q,ω)]⁻¹ χ₀(q,ω)
        """
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        if xc == 'RPA':
            return chi0_wGG
        # TDDFT (in adiabatic approximations to the kernel)
        assert not self.chi0.qpd.optical_limit, \
            'For TDDFT in the q→0 limit, use the inverse dielectric function'
        Kxc_GG = self.get_Kxc_GG(xc=xc, chi0_wGG=chi0_wGG, **xckwargs)
        return self.invert_dyson_like_equation(chi0_wGG, Kxc_GG)


@dataclass
class DielectricFunctionBase(ABC):
    """Base class for the dielectric function ε(q,ω)."""
    cd: CellDescriptor
    qpd: SingleQPWDescriptor
    wd: FrequencyDescriptor
    wblocks: Blocks1D

    @classmethod
    def from_chi0_dyson_eqs(cls, chi0_dyson_eqs, *args, **kwargs):
        cd = CellDescriptor.from_gs(chi0_dyson_eqs.gs)
        chi0 = chi0_dyson_eqs.chi0
        return cls(cd, chi0.qpd, chi0.wd, chi0_dyson_eqs.wblocks,
                   *args, **kwargs)

    @abstractmethod
    def macroscopic_dielectric_function(self) -> ScalarResponseFunctionSet:
        """Get the macroscopic dielectric function ε_M(q,ω)."""

    def polarizability(self):
        """Get the macroscopic polarizability α_M(q,ω).

        Calculates the macroscopic polarizability

        α_M(q,ω) = Λ/(4π) (ε_M(q,ω) - 1),

        where Λ is the nonperiodic hypervolume of the unit cell.
        """
        _, eps0_w, eps_w = self.macroscopic_dielectric_function().arrays
        L = self.cd.nonperiodic_hypervolume
        alpha0_w = L / (4 * np.pi) * (eps0_w - 1.0)
        alpha_w = L / (4 * np.pi) * (eps_w - 1.0)
        return ScalarResponseFunctionSet(self.wd, alpha0_w, alpha_w)

    def eels_spectrum(self):
        """Get the macroscopic EELS spectrum.

        Here, we define the EELS spectrum to be the spectral part of the
        inverse dielectric function. For the macroscopic component,

        EELS(ω) = - Im[1/ε_M(q,ω)].

        We use the equivalent expression to give also the EELS spectrum
        without local-field effects.
        """
        _, eps0_w, eps_w = self.macroscopic_dielectric_function().arrays
        eels0_W = -(1. / eps0_w).imag
        eels_W = -(1. / eps_w).imag
        return ScalarResponseFunctionSet(self.wd, eels0_W, eels_W)


@dataclass
class InverseDielectricFunction(DielectricFunctionBase):
    """Data class for the inverse dielectric function ε⁻¹(q,ω).

    The inverse dielectric function characterizes the longitudinal response

    V (q,ω) = ε⁻¹(q,ω) V (q,ω),
     tot                ext

    where the induced potential due to the electronic system is given by Vχ,

    ε⁻¹(q,ω) = 1 + V(q) χ(q,ω).

    In this data class, ε⁻¹ is cast in terms if its symmetrized representation
    ˷
    ε⁻¹(q,ω) = V^(-1/2)(q) ε⁻¹(q,ω) V^(1/2)(q),

    that is, in terms of V^(1/2)(q) χ(q,ω) V^(1/2)(q).

    Please remark that V(q) here refers to the bare Coulomb potential
    irregardless of whether χ(q,ω) was determined using a truncated analogue.
    """
    Vchi0_symm_wGG: np.ndarray  # V^(1/2)(q) χ₀(q,ω) V^(1/2)(q)
    Vchi_symm_wGG: np.ndarray
    V_G: np.ndarray

    def _get_macroscopic_component(self, in_wGG):
        return self.wblocks.all_gather(in_wGG[:, 0, 0])

    def macroscopic_components(self):
        Vchi0_W = self._get_macroscopic_component(self.Vchi0_symm_wGG)
        Vchi_W = self._get_macroscopic_component(self.Vchi_symm_wGG)
        return Vchi0_W, Vchi_W

    def macroscopic_dielectric_function(self):
        """Get the macroscopic dielectric function ε_M(q,ω).

        Calculates ε_M(q,ω) given by

           1
        ‾‾‾‾‾‾‾‾ = ε⁻¹(q,ω)
        ε_M(q,ω)    00

        In addition to the many-body dielectric function, we also calculate the
        dielectric function in the independent-particle random-phase
        approximation, that is, using the RPA dielectric function ε = 1 - Vχ₀
        and neglecting local field effects [Rev. Mod. Phys. 74, 601 (2002)].
        """
        Vchi0_W, Vchi_W = self.macroscopic_components()
        eps0_W = 1. - Vchi0_W
        eps_W = 1. / (1. + Vchi_W)
        return ScalarResponseFunctionSet(self.wd, eps0_W, eps_W)

    def dynamic_susceptibility(self):
        """Get the macroscopic component of χ(q,ω)."""
        Vchi0_W, Vchi_W = self.macroscopic_components()
        V0 = self.V_G[0]  # Macroscopic Coulomb potential (4π/q²)
        return ScalarResponseFunctionSet(self.wd, Vchi0_W / V0, Vchi_W / V0)


@dataclass
class DielectricFunctionData(DielectricFunctionBase):
    """Data class for the dielectric function ε(q,ω).

    The dielectric function is written in terms of the Coulomb potential V and
    polarizability operator P [Rev. Mod. Phys. 74, 601 (2002)],

    ε(q,ω) = 1 - V(q) P(q,ω),

    and represented in a plane-wave basis.

    Please remark that the Coulomb potential may have been interchanged with
    its truncated analogue.
    """
    eps_wGG: np.ndarray

    def macroscopic_dielectric_function(self):
        """Get the macroscopic dielectric function ε_M(q,ω)."""
        # Ignoring local field effects
        eps0_W = self.wblocks.all_gather(self.eps_wGG[:, 0, 0])

        # Accouting for local field effects
        eps_w = np.zeros((self.wblocks.nlocal,), complex)
        for w, eps_GG in enumerate(self.eps_wGG):
            eps_w[w] = 1 / np.linalg.inv(eps_GG)[0, 0]
        eps_W = self.wblocks.all_gather(eps_w)

        return ScalarResponseFunctionSet(self.wd, eps0_W, eps_W)


@dataclass
class ModifiedDielectricFunction(DielectricFunctionBase):
    """Data class for the local-field corrected dielectric function.

    The field corrected dielectric function,
    ˍ                 ˍ
    ϵ(q,ω) = 1 - V(q) P(q,ω),
                                               ˍ
    is here represented in terms of V^(1/2)(q) P(q,ω) V^(1/2)(q).
    """
    VP_symm_wGG: np.ndarray  # V^(1/2) P V^(1/2)
    VPbar_symm_wGG: np.ndarray

    def macroscopic_dielectric_function(self):
        """Get the macroscopic dielectric function ε_M(q,ω).

        By design,
                   ˍ
        ε_M(q,ω) = ϵ (q,ω)
                    00

        when accounting for local field effects.
        """
        # Without and with local-field effects
        eps0_W = self.wblocks.all_gather(1. - self.VP_symm_wGG[:, 0, 0])
        eps_W = self.wblocks.all_gather(1. - self.VPbar_symm_wGG[:, 0, 0])
        return ScalarResponseFunctionSet(self.wd, eps0_W, eps_W)


@dataclass
class CellDescriptor:
    cell_cv: np.ndarray
    pbc_c: np.ndarray

    @classmethod
    def from_gs(cls, gs):
        return cls(gs.gd.cell_cv, gs.pbc)

    @property
    def nonperiodic_hypervolume(self):
        return nonperiodic_hypervolume(self.cell_cv, self.pbc_c)


def nonperiodic_hypervolume(cell_cv, pbc_c):
    """Get the hypervolume of the cell along nonperiodic directions.

    Returns the hypervolume Λ in units of Å, where

    Λ = 1        in 3D
    Λ = L        in 2D, where L is the out-of-plane cell vector length
    Λ = A        in 1D, where A is the transverse cell area
    Λ = V        in 0D, where V is the cell volume
    """
    if pbc_c.all():
        return 1.
    else:
        if sum(pbc_c) > 0:
            # In 1D and 2D, we assume the cartesian representation of the unit
            # cell to be block diagonal, separating the periodic and
            # nonperiodic cell vectors in different blocks.
            assert np.allclose(cell_cv[~pbc_c][:, pbc_c], 0.) and \
                np.allclose(cell_cv[pbc_c][:, ~pbc_c], 0.), \
                "In 1D and 2D, please put the periodic/nonperiodic axis " \
                "along a cartesian component"
        V = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
        return V * Bohr**sum(~pbc_c)  # Bohr -> Å


class DielectricFunctionCalculator:
    def __init__(self, chi0calc: Chi0Calculator, coulomb: CoulombKernel):
        self.chi0calc = chi0calc
        self.coulomb = coulomb

        self.gs = chi0calc.gs
        self.context = chi0calc.context

        self._chi0cache: dict = {}

    def calculate_chi0(self, q_c: list | np.ndarray):
        """Calculates the response function.

        Calculate the response function for a specific momentum.

        q_c: [float, float, float]
            The momentum wavevector.
        """

        # We cache the computed data since chi0 may otherwise be redundantly
        # calculated e.g. if the user calculates multiple directions.
        #
        # May be called multiple times with same q_c, and we want to
        # be able to recognize previous seen values of q_c.
        # We do this by rounding and converting to string with fixed
        # precision (so not very elegant).
        q_key = [f'{q:.10f}' for q in q_c]
        key = tuple(q_key)

        if key not in self._chi0cache:
            # We assume that the caller will trigger this multiple
            # times with the same qpoint, then several times with
            # another qpoint, etc.  If that's true, then we
            # need to cache no more than one qpoint at a time.
            # Thus to save memory, we clear the cache here.
            #
            # This should be replaced with something more reliable,
            # such as having the caller manage things more explicitly.
            #
            # See https://gitlab.com/gpaw/gpaw/-/issues/662
            #
            # In conclusion, delete the cache now:
            self._chi0cache.clear()

            # cache Chi0Data from gpaw.response.chi0_data
            self._chi0cache[key] = Chi0DysonEquations(
                self.chi0calc.calculate(q_c), self)
            self.context.write_timer()

        return self._chi0cache[key]

    def get_dielectric_function_new(self, q_c=[0, 0, 0], direction='x',
                                    **xckwargs):
        return self.calculate_chi0(q_c).dielectric_function(
            direction=direction, **xckwargs)

    def get_inverse_dielectric_function(self, q_c=[0, 0, 0], direction='x',
                                        **xckwargs):
        return self.calculate_chi0(q_c).inverse_dielectric_function(
            direction=direction, **xckwargs)

    def get_rpa_density_response(self, q_c, *, direction, qinf_v=None):
        # Used by the QEH code
        return self.calculate_chi0(q_c).rpa_density_response(
            direction=direction, qinf_v=qinf_v)


class DielectricFunction(DielectricFunctionCalculator):
    """This class defines dielectric function related physical quantities."""

    def __init__(self, calc, *,
                 frequencies=None,
                 ecut=50,
                 hilbert=True,
                 nbands=None, eta=0.2,
                 intraband=True, nblocks=1, world=mpi.world, txt=sys.stdout,
                 truncation=None, disable_point_group=False,
                 disable_time_reversal=False,
                 integrationmode=None, rate=0.0,
                 eshift: float | None = None):
        """Creates a DielectricFunction object.

        calc: str
            The ground-state calculation file that the linear response
            calculation is based on.
        frequencies:
            Input parameters for frequency_grid.
            Can be an array of frequencies to evaluate the response function at
            or dictionary of parameters for build-in nonlinear grid
            (see :ref:`frequency grid`).
        ecut: float
            Plane-wave cut-off.
        hilbert: bool
            Use hilbert transform.
        nbands: int
            Number of bands from calculation.
        eta: float
            Broadening parameter.
        intraband: bool
            Include intraband transitions.
        world: comm
            mpi communicator.
        nblocks: int
            Split matrices in nblocks blocks and distribute them G-vectors or
            frequencies over processes.
        txt: str
            Output file.
        truncation: str or None
            None for no truncation.
            '2D' for standard analytical truncation scheme.
            Non-periodic directions are determined from k-point grid
        eshift: float
            Shift unoccupied bands
        """
        gs, context = get_gs_and_context(calc, txt, world, timer=None)
        wd = get_frequency_descriptor(frequencies, gs=gs, nbands=nbands)

        chi0calc = Chi0Calculator(
            gs, context, nblocks=nblocks,
            wd=wd,
            ecut=ecut, nbands=nbands, eta=eta,
            hilbert=hilbert,
            intraband=intraband,
            disable_point_group=disable_point_group,
            disable_time_reversal=disable_time_reversal,
            integrationmode=integrationmode,
            rate=rate, eshift=eshift
        )
        coulomb = CoulombKernel.from_gs(gs, truncation=truncation)

        super().__init__(chi0calc, coulomb)

    def get_frequencies(self) -> np.ndarray:
        """Return frequencies (in eV) that the χ is evaluated on."""
        return self.chi0calc.wd.omega_w * Hartree

    def get_dynamic_susceptibility(self, *args, xc='ALDA',
                                   filename='chiM_w.csv',
                                   **kwargs):
        dynsus = self.get_inverse_dielectric_function(
            *args, xc=xc, **kwargs).dynamic_susceptibility()
        if filename:
            dynsus.write(filename)
        return dynsus.unpack()

    def get_dielectric_function(self, *args, filename='df.csv', **kwargs):
        """Calculate the dielectric function.

        Generates a file 'df.csv', unless filename is set to None.

        Returns
        -------
        df_NLFC_w: np.ndarray
            Dielectric function without local field corrections.
        df_LFC_w: np.ndarray
            Dielectric functio with local field corrections.
        """
        df = self.get_dielectric_function_new(
            *args, **kwargs).macroscopic_dielectric_function()
        if filename:
            df.write(filename)
        return df.unpack()

    def get_eels_spectrum(self, *args, filename='eels.csv', **kwargs):
        """Calculate the macroscopic EELS spectrum.

        Generates a file 'eels.csv', unless filename is set to None.

        Returns
        -------
        eels0_w: np.ndarray
            Spectrum in the independent-particle random-phase approximation.
        eels_w: np.ndarray
            Fully screened EELS spectrum.
        """
        eels = self.get_inverse_dielectric_function(
            *args, **kwargs).eels_spectrum()
        if filename:
            eels.write(filename)
        return eels.unpack()

    def get_polarizability(self, *args, filename='polarizability.csv',
                           **kwargs):
        """Calculate the macroscopic polarizability.

        Generate a file 'polarizability.csv', unless filename is set to None.

        Returns:
        --------
        alpha0_w: np.ndarray
            Polarizability calculated without local-field corrections
        alpha_w: np.ndarray
            Polarizability calculated with local-field corrections.
        """
        pol = self.get_dielectric_function_new(
            *args, **kwargs).polarizability()
        if filename:
            pol.write(filename)
        return pol.unpack()

    def get_macroscopic_dielectric_constant(self, xc='RPA', direction='x'):
        """Calculate the macroscopic dielectric constant.

        The macroscopic dielectric constant is defined as the real part of the
        dielectric function in the static limit.

        Returns:
        --------
        eps0: float
            Dielectric constant without local field corrections.
        eps: float
            Dielectric constant with local field correction. (RPA, ALDA)
        """
        df = self.get_dielectric_function_new(
            xc=xc, direction=direction).macroscopic_dielectric_function()
        return df.static_limit.real


# ----- Serialized dataclasses and IO ----- #


@dataclass
class ScalarResponseFunctionSet:
    """A set of scalar response functions rf₀(ω) and rf(ω)."""
    wd: FrequencyDescriptor
    rf0_w: np.ndarray
    rf_w: np.ndarray

    @property
    def arrays(self):
        return self.wd.omega_w * Hartree, self.rf0_w, self.rf_w

    def unpack(self):
        # Legacy feature to support old DielectricFunction output format
        # ... to be deprecated ...
        return self.rf0_w, self.rf_w

    def write(self, filename):
        if mpi.rank == 0:
            write_response_function(filename, *self.arrays)

    @property
    def static_limit(self):
        """Return the value of the response functions in the static limit."""
        w0 = np.argmin(np.abs(self.wd.omega_w))
        assert abs(self.wd.omega_w[w0]) < 1e-8
        return np.array([self.rf0_w[w0], self.rf_w[w0]])


def write_response_function(filename, omega_w, rf0_w, rf_w):
    with open(filename, 'w') as fd:
        for omega, rf0, rf in zip(omega_w, rf0_w, rf_w):
            if rf0_w.dtype == complex:
                print('%.6f, %.6f, %.6f, %.6f, %.6f' %
                      (omega, rf0.real, rf0.imag, rf.real, rf.imag),
                      file=fd)
            else:
                print(f'{omega:.6f}, {rf0:.6f}, {rf:.6f}', file=fd)


def read_response_function(filename):
    """Read a stored response function file"""
    d = np.loadtxt(filename, delimiter=',')
    omega_w = np.array(d[:, 0], float)

    if d.shape[1] == 3:
        # Real response function
        rf0_w = np.array(d[:, 1], float)
        rf_w = np.array(d[:, 2], float)
    elif d.shape[1] == 5:
        rf0_w = np.array(d[:, 1], complex)
        rf0_w.imag = d[:, 2]
        rf_w = np.array(d[:, 3], complex)
        rf_w.imag = d[:, 4]
    else:
        raise ValueError(f'Unexpected array dimension {d.shape}')

    return omega_w, rf0_w, rf_w
