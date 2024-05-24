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


"""
On the notation in this module.

When calculating properties such as the dielectric function, EELS spectrum and
polarizability there are many inherent subtleties relating to (ir)reducible
representations and inclusion of local-field effects. For the reciprocal space
representation of the Coulomb potential, we use the following notation

v or v(q): The bare Coulomb interaction, 4π/|G+q|²

V or V(q): The specified Coulomb interaction. Will usually be either the bare
           interaction or a truncated version hereof.
ˍ    ˍ
V or V(q): The modified Coulomb interaction. Equal to V(q) for finite
           reciprocal wave vectors G > 0, but modified to exclude long-range
           interactions, that is, equal to 0 for G = 0.
"""


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
        """Get the Hxc kernel rescaled by the bare Coulomb potential v(q).

        Calculates
        ˷
        K(q) = v^(-1/2)(q) K_Hxc(q) v^(-1/2)(q),

        where v(q) is the bare Coulomb potential and

        K_Hxc(q) = V(q) + K_xc(q),

        where the Hartree/Coulomb kernel V(q) itself might be truncated.
        """
        qpd = self.chi0.qpd
        if self.coulomb.truncation is None:
            v_G = self.coulomb.V(qpd)  # bare Coulomb interaction
            K_GG = np.eye(len(v_G), dtype=complex)
        else:
            bare_coulomb = self.coulomb.new(truncation=None)
            v_G = bare_coulomb.V(qpd)
            V_G = self.coulomb.V(qpd)
            K_GG = np.diag(V_G / v_G)
        if xc != 'RPA':
            Kxc_GG = self.get_Kxc_GG(xc=xc, **xckwargs)
            sqrtv_G = v_G**0.5
            K_GG += Kxc_GG / sqrtv_G / sqrtv_G[:, np.newaxis]
        return v_G, K_GG

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
        """Calculate v^(1/2) χ v^(1/2), from which ε⁻¹(q,ω) is constructed."""
        return InverseDielectricFunction.from_chi0_dyson_eqs(
            self, *self.calculate_vchi_symm(*args, **kwargs))

    def calculate_vchi_symm(self, xc='RPA', direction='x', **xckwargs):
        """Calculate v^(1/2) χ v^(1/2).

        Starting from the TDDFT Dyson equation

        χ(q,ω) = χ₀(q,ω) + χ₀(q,ω) K_Hxc(q,ω) χ(q,ω),                (1)

        the Coulomb scaled susceptibility,
        ˷
        χ(q,ω) = v^(1/2)(q) χ(q,ω) v^(1/2)(q)

        can be calculated from the Dyson-like equation
        ˷        ˷         ˷       ˷      ˷
        χ(q,ω) = χ₀(q,ω) + χ₀(q,ω) K(q,ω) χ(q,ω)                     (2)

        where
        ˷
        K(q,ω) = v^(-1/2)(q) K_Hxc(q,ω) v^(-1/2)(q).

        Here v(q) refers to the bare Coulomb potential. It should be emphasized
        that invertion of (2) rather than (1) is not merely a rescaling
        excercise. In the optical q → 0 limit, the Coulomb kernel v(q) diverges
        as 1/|G+q|² while the Kohn-Sham susceptibility χ₀(q,ω) vanishes as
        |G+q|². Treating v^(1/2)(q) χ₀(q,ω) v^(1/2)(q) as a single variable,
        the effects of this cancellation can be treated accurately within k.p
        perturbation theory.
        """
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        v_G, K_GG = self.get_coulomb_scaled_kernel(
            xc=xc, chi0_wGG=chi0_wGG, **xckwargs)
        # Calculate v^(1/2)(q) χ₀(q,ω) v^(1/2)(q)
        sqrtv_G = v_G**0.5
        vchi0_symm_wGG = chi0_wGG  # reuse buffer
        for w, chi0_GG in enumerate(chi0_wGG):
            vchi0_symm_wGG[w] = chi0_GG * sqrtv_G * sqrtv_G[:, np.newaxis]
        # Invert Dyson equation
        vchi_symm_wGG = self.invert_dyson_like_equation(
            vchi0_symm_wGG, K_GG, reuse_buffer=False)
        return vchi0_symm_wGG, vchi_symm_wGG, v_G

    def dielectric_matrix(self, *args, **kwargs):
        """Construct the dielectric function ε(q,ω)."""
        V_G = self.coulomb.V(self.chi0.qpd)
        if abs(V_G[0]) > 1e-8:
            return self.customized_dielectric_function(*args, **kwargs)
        else:
            return self._modified_dielectric_function(*args, **kwargs)

    def customized_dielectric_function(self, *args, **kwargs):
        """Calculate Ε(q,ω) = 1 - V(q) P(q,ω)."""
        V_GG = self.coulomb.kernel(self.chi0.qpd)
        P_wGG = self.polarizability_operator(*args, **kwargs)
        nG = len(V_GG)
        eps_wGG = P_wGG  # reuse buffer
        for w, P_GG in enumerate(P_wGG):
            eps_wGG[w] = np.eye(nG) - V_GG @ P_GG
        return CustomizableDielectricFunction.from_chi0_dyson_eqs(
            self, eps_wGG)

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
        equation (1) when replacing the Hartree-exchange-correlation kernel
        with the modified Coulomb interaction:
                    ˍ
        K_Hxc(q) -> V(q).
                                                                  ˍ
        We may thus reuse that functionality to calculate V^(1/2) P V^(1/2)
                   ˍ
        from which ϵ(q,ω) can be constructed.
        """
        assert xc == 'RPA'
        VP_symm_wGG, VPbar_symm_wGG, _ = self.calculate_vchi_symm(
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
        # WARNING: The TDDFT implementation seems to be invalid in the optical
        # limit... Namely, the Coulomb interaction V(q) is only well-defined
        # in products of V^(1/2) χ₀ V^(1/2), why a literal evaluation of
        # V(q) P(q,ω) does not seem sensible. Furthermore, one should use the
        # χ₀ body when calculating P(q,ω) and not usual version with head and
        # wings which are only well defined up to factors of V(q).
        Kxc_GG = self.get_Kxc_GG(xc=xc, chi0_wGG=chi0_wGG, **xckwargs)
        return self.invert_dyson_like_equation(chi0_wGG, Kxc_GG)


@dataclass
class DielectricFunctionRelatedData:
    qpd: SingleQPWDescriptor
    wd: FrequencyDescriptor
    wblocks: Blocks1D

    @classmethod
    def from_chi0_dyson_eqs(cls, chi0_dyson_eqs, *args, **kwargs):
        chi0 = chi0_dyson_eqs.chi0
        return cls(chi0.qpd, chi0.wd, chi0_dyson_eqs.wblocks, *args, **kwargs)


@dataclass
class InverseDielectricFunction(DielectricFunctionRelatedData):
    """Data class for the inverse dielectric function ε⁻¹(q,ω).

    The inverse dielectric function characterizes the longitudinal response

    V (q,ω) = ε⁻¹(q,ω) V (q,ω),
     tot                ext

    where the induced potential due to the electronic system is given by vχ,

    ε⁻¹(q,ω) = 1 + v(q) χ(q,ω).

    In this data class, ε⁻¹ is cast in terms if its symmetrized representation
    ˷
    ε⁻¹(q,ω) = v^(-1/2)(q) ε⁻¹(q,ω) v^(1/2)(q),

    that is, in terms of v^(1/2)(q) χ(q,ω) v^(1/2)(q).

    Please remark that v(q) here refers to the bare Coulomb potential
    irregardless of whether χ(q,ω) was determined using a truncated analogue.
    """
    vchi0_symm_wGG: np.ndarray  # v^(1/2)(q) χ₀(q,ω) v^(1/2)(q)
    vchi_symm_wGG: np.ndarray
    v_G: np.ndarray

    def _get_macroscopic_component(self, in_wGG):
        return self.wblocks.all_gather(in_wGG[:, 0, 0])

    def macroscopic_components(self):
        vchi0_W = self._get_macroscopic_component(self.vchi0_symm_wGG)
        vchi_W = self._get_macroscopic_component(self.vchi_symm_wGG)
        return vchi0_W, vchi_W

    def macroscopic_dielectric_function(self):
        """Get the macroscopic dielectric function.

        Calculates

                       1
        ε (q,ω) =  ‾‾‾‾‾‾‾‾
         M         ε⁻¹(q,ω)
                    00

        along with the macroscopic dielectric function in the independent-
        particle random-phase approximation [Rev. Mod. Phys. 74, 601 (2002)],

         IP
        ε (q,ω) = 1 - v(q) χ⁰(q,ω)
         M                  00

        that is, neglecting local field effects entirely.
        """
        vchi0_W, vchi_W = self.macroscopic_components()
        eps0_W = 1 - vchi0_W
        eps_W = 1 / (1 + vchi_W)
        return ScalarResponseFunctionSet(self.wd, eps0_W, eps_W)

    def dynamic_susceptibility(self):
        """Get the macroscopic components of χ(q,ω) and χ₀(q,ω)."""
        vchi0_W, vchi_W = self.macroscopic_components()
        v0 = self.v_G[0]  # Macroscopic Coulomb potential (4π/q²)
        return ScalarResponseFunctionSet(self.wd, vchi0_W / v0, vchi_W / v0)

    def eels_spectrum(self):
        """Get the macroscopic EELS spectrum.

        The spectrum is defined as

                                          1
        EELS(q,ω) ≡ -Im ε⁻¹(q,ω) = -Im ‾‾‾‾‾‾‾.
                         00            ε (q,ω)
                                        M

        In addition to the many-body spectrum, we also calculate the
        EELS spectrum in the independent-particle random-phase approximation,
        here defined as

                          1
        EELS₀(ω) = -Im ‾‾‾‾‾‾‾.
                        IP
                       ε (q,ω)
                        M
        """
        vchi0_W, vchi_W = self.macroscopic_components()
        eels0_W = -(1. / (1. - vchi0_W)).imag
        eels_W = -vchi_W.imag
        return ScalarResponseFunctionSet(self.wd, eels0_W, eels_W)


class DielectricFunctionBase(DielectricFunctionRelatedData, ABC):
    """Base class for the dielectric function ε(q,ω)."""

    @abstractmethod
    def dielectric_function(self) -> ScalarResponseFunctionSet:
        """Get the macroscopic dielectric function ε_M(q,ω)."""

    def polarizability(self, L: float):
        """Get the macroscopic polarizability α_M(q,ω).

        Calculates the macroscopic polarizability

        α_M(q,ω) = Λ/(4π) (ε_M(q,ω) - 1),

        where Λ (given as input L) is the nonperiodic hypervolume of the unit
        cell.
        """
        df = self.dielectric_function()
        alpha0_w = L / (4 * np.pi) * (df.rf0_w - 1.0)
        alpha_w = L / (4 * np.pi) * (df.rf_w - 1.0)
        return ScalarResponseFunctionSet(self.wd, alpha0_w, alpha_w)


@dataclass
class CustomizableDielectricFunction(DielectricFunctionBase):
    """Data class for customized dielectric functions Ε(q,ω).

    Ε(q,ω) is customizable in the sense that bare Coulomb interaction v(q) is
    replaced with the an arbitrary interaction V(q) in the formula for the
    dielectric function,

    Ε(q,ω) = 1 - V(q) P(q,ω),

    where P is the polarizability operator [Rev. Mod. Phys. 74, 601 (2002)].
    Thus, for any truncated or otherwise cusomized interaction V(q) ≠ v(q),
    Ε(q,ω) ≠ ε(q,ω) and Ε⁻¹(q,ω) ≠ ε⁻¹(q,ω).
    """
    eps_wGG: np.ndarray

    def dielectric_function(self):  # rename to macroscopic? XXX
        """Get the macroscopic customized dielectric function Ε_M(q,ω).

        We define the macroscopic customized dielectric function as

                       1
        Ε_M(q,ω) =  ‾‾‾‾‾‾‾‾
                    Ε⁻¹(q,ω)
                     00

        such that Ε_M(q,ω) = ε_M(q,ω) for V(q) = v(q).
        """
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

    def dielectric_function(self):
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


def nonperiodic_hypervolume(gs):
    """Get the hypervolume of the cell along nonperiodic directions.

    Returns the hypervolume Λ in units of Å, where

    Λ = 1        in 3D
    Λ = L        in 2D, where L is the out-of-plane cell vector length
    Λ = A        in 1D, where A is the transverse cell area
    Λ = V        in 0D, where V is the cell volume
    """
    cell_cv = gs.gd.cell_cv
    pbc_c = gs.pbc
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

    def _new_dynamic_susceptibility(self, *args, **kwargs):
        return self.get_inverse_dielectric_function(
            *args, **kwargs).dynamic_susceptibility()

    def _new_dielectric_function(self, *args, **kwargs):
        return self.get_inverse_dielectric_function(
            *args, **kwargs).macroscopic_dielectric_function()

    def _new_eels_spectrum(self, *args, **kwargs):
        return self.get_inverse_dielectric_function(
            *args, **kwargs).eels_spectrum()

    def _new_polarizability(self, *args, **kwargs):
        return self.get_dielectric_matrix(
            *args, **kwargs).polarizability(
                L=nonperiodic_hypervolume(self.gs))

    def get_dielectric_matrix(self, q_c=[0, 0, 0], direction='x', **xckwargs):
        return self.calculate_chi0(q_c).dielectric_matrix(
            direction=direction, **xckwargs)

    def get_literal_dielectric_function(self, *args, **kwargs):
        """Calculate the dielectric function ε(q,ω) = 1 - v(q) P(q,ω)."""
        return self.get_customized_dielectric_function(
            truncation=None, *args, **kwargs)

    def get_customized_dielectric_function(self, *, truncation: str | None,
                                           q_c=[0, 0, 0], direction='x',
                                           **xckwargs):
        # NB: ignores self.coulomb while this still exists XXX
        chi0_dyson_equation = self.calculate_chi0(q_c)
        chi0_dyson_equation.coulomb = chi0_dyson_equation.coulomb.new(
            truncation=truncation)
        return chi0_dyson_equation.customized_dielectric_function(
            direction=direction, **xckwargs)

    def get_inverse_dielectric_function(self, q_c=[0, 0, 0], direction='x',
                                        **xckwargs):
        return self.calculate_chi0(q_c).inverse_dielectric_function(
            direction=direction, **xckwargs)

    def get_rpa_density_response(self, q_c, *, direction, qinf_v=None):
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
        dynsus = self._new_dynamic_susceptibility(*args, xc=xc, **kwargs)
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
        df = self._new_dielectric_function(*args, **kwargs)
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
        eels = self._new_eels_spectrum(*args, **kwargs)
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
        pol = self._new_polarizability(*args, **kwargs)
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
        df = self._new_dielectric_function(xc=xc, direction=direction)
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
