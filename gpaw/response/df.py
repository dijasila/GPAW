from __future__ import annotations
from dataclasses import dataclass
from math import pi
import sys

import numpy as np
from ase.units import Hartree, Bohr

import gpaw.mpi as mpi

from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.density_kernels import get_density_xc_kernel
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
from gpaw.response.chi0_data import Chi0Data
from gpaw.response.pair import get_gs_and_context

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpaw.response.frequencies import FrequencyDescriptor


@dataclass
class Chi0DysonEquation:
    chi0: Chi0Data
    df: 'DielectricFunctionCalculator'

    def __post_init__(self):
        self.gs = self.df.gs
        self.context = self.df.context
        self.coulomb = self.df.coulomb
        self.blocks1d = self.df.blocks1d

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
            W_w = self.blocks1d.myslice
            chi0_wGG[:, 0] = np.dot(d_v, chi0.chi0_WxvG[W_w, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0.chi0_WxvG[W_w, 1])
            chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0.chi0_Wvv[W_w], d_v).T)
        return chi0_wGG

    def rpa_density_response(self, direction='x', qinf_v=None):
        """Calculate the RPA susceptibility for (semi-)finite q."""
        qpd = self.chi0.qpd
        V_G = self.coulomb.V(qpd, q_v=qinf_v)
        V_GG = np.diag(V_G)
        nG = len(V_G)

        # Extract χ₀(q,ω)
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        if qpd.optical_limit:
            # Restore the q-dependence of the head and wings in the q→0 limit
            assert qinf_v is not None and np.linalg.norm(qinf_v) > 0.
            d_v = self._normalize(direction)
            chi0_wGG[:, 1:, 0] *= np.dot(qinf_v, d_v)
            chi0_wGG[:, 0, 1:] *= np.dot(qinf_v, d_v)
            chi0_wGG[:, 0, 0] *= np.dot(qinf_v, d_v)**2

        # Invert Dyson equation
        chi_wGG = np.zeros_like(chi0_wGG)
        for w, chi0_GG in enumerate(chi0_wGG):
            xi_GG = chi0_GG @ V_GG
            enhancement_GG = np.linalg.inv(np.eye(nG) - xi_GG)
            chi_wGG[w] = enhancement_GG @ chi0_GG

        return qpd, chi_wGG

    def Vchi(self, xc='RPA', direction='x', **xckwargs):
        """Returns qpd, chi0 and chi0 in v^1/2 chi v^1/2 format.

        The truncated Coulomb interaction is included as
        v^-1/2 v_t v^-1/2. This is in order to conform with
        the head and wings of chi0, which is treated specially for q=0.
        """
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        qpd = self.chi0.qpd

        coulomb_bare = CoulombKernel.from_gs(self.gs, truncation=None)
        V_G = coulomb_bare.V(qpd)  # np.ndarray
        sqrtV_G = V_G**0.5

        nG = len(sqrtV_G)

        Vtrunc_G = self.coulomb.V(qpd)

        if self.coulomb.truncation is None:
            K_GG = np.eye(nG, dtype=complex)
        else:
            K_GG = np.diag(Vtrunc_G / V_G)

        if xc != 'RPA':
            Kxc_GG = get_density_xc_kernel(qpd,
                                           self.gs, self.context,
                                           functional=xc,
                                           chi0_wGG=chi0_wGG,
                                           **xckwargs)
            K_GG += Kxc_GG / sqrtV_G / sqrtV_G[:, np.newaxis]

        # Invert Dyson eq.
        chi_wGG = []
        for chi0_GG in chi0_wGG:
            """v^1/2 chi0 V^1/2"""
            chi0_GG[:] = chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
            chi_GG = np.dot(np.linalg.inv(np.eye(nG) -
                                          np.dot(chi0_GG, K_GG)),
                            chi0_GG)
            chi_wGG.append(chi_GG)

        if len(chi_wGG):
            chi_wGG = np.array(chi_wGG)
        else:
            chi_wGG = np.zeros((0, nG, nG), complex)

        return InverseDielectricFunction(
            self, chi0_wGG, np.array(chi_wGG), V_G)

    def dielectric_matrix(self, xc='RPA', direction='x', symmetric=True,
                          **xckwargs):
        r"""Returns the symmetrized dielectric matrix.

        ::

            \tilde\epsilon_GG' = v^{-1/2}_G \epsilon_GG' v^{1/2}_G',

        where::

            epsilon_GG' = 1 - v_G * P_GG' and P_GG'

        is the polarization.

        ::

            In RPA:   P = chi^0
            In TDDFT: P = (1 - chi^0 * f_xc)^{-1} chi^0

        in addition to RPA one can use the kernels, ALDA, Bootstrap and
        LRalpha (long-range kerne), where alpha is a user specified parameter
        (for example xc='LR0.25')

        The head of the inverse symmetrized dielectric matrix is equal
        to the head of the inverse dielectric matrix (inverse dielectric
        function)"""
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        qpd = self.chi0.qpd

        K_G = self.coulomb.sqrtV(qpd)
        nG = len(K_G)

        if xc != 'RPA':
            Kxc_GG = get_density_xc_kernel(qpd,
                                           self.gs, self.context,
                                           functional=xc,
                                           chi0_wGG=chi0_wGG,
                                           **xckwargs)

        for chi0_GG in chi0_wGG:
            if xc == 'RPA':
                P_GG = chi0_GG
            else:
                P_GG = np.dot(np.linalg.inv(np.eye(nG) -
                                            np.dot(chi0_GG, Kxc_GG)),
                              chi0_GG)
            if symmetric:
                e_GG = np.eye(nG) - P_GG * K_G * K_G[:, np.newaxis]
            else:
                K_GG = (K_G**2 * np.ones([nG, nG])).T
                e_GG = np.eye(nG) - P_GG * K_GG

            # Reuse the chi0_wGG buffer for the output dielectric matrix
            chi0_GG[:] = e_GG
        return DielectricMatrixData(self, e_wGG=chi0_wGG)


@dataclass
class InverseDielectricFunction:
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
    """
    dyson: Chi0DysonEquation
    Vchi0_symm_wGG: np.ndarray  # V^(1/2)(q) χ₀(q,ω) V^(1/2)(q)
    Vchi_symm_wGG: np.ndarray
    V_G: np.ndarray

    def __post_init__(self):
        # Very ugly this... XXX
        self.qpd = self.dyson.chi0.qpd
        self.wd = self.dyson.chi0.wd
        self.wblocks = self.dyson.df.blocks1d

    def _get_macroscopic_component(self, in_wGG):
        return self.wblocks.all_gather(in_wGG[:, 0, 0])

    def macroscopic_components(self):
        Vchi0_W = self._get_macroscopic_component(self.Vchi0_symm_wGG)
        Vchi_W = self._get_macroscopic_component(self.Vchi_symm_wGG)
        return Vchi0_W, Vchi_W

    def dynamic_susceptibility(self):
        """Get the macroscopic component of χ(q,ω)."""
        Vchi0_W, Vchi_W = self.macroscopic_components()
        V0 = self.V_G[0]  # Macroscopic Coulomb potential (4π/q²)
        return ScalarResponseFunctionSet(self.wd, Vchi0_W / V0, Vchi_W / V0)

    def eels_spectrum(self):
        """Get the macroscopic EELS spectrum.

        Here, we define the EELS spectrum to be the spectral part of the
        inverse dielectric function. In the plane-wave representation,

        EELS(G+q,ω) = -Im ε⁻¹(G+q,ω) = -Im V(G+q) χ(G+q,ω),

        where ε⁻¹(G+q,ω) denotes the G'th diagonal element.

        In addition to the many-body spectrum, we also calculate the
        macroscopic EELS spectrum in the independent-particle random-phase
        approximation, that is, using the RPA dielectric function ε = 1 - Vχ₀
        and neglecting local field effects [Rev. Mod. Phys. 74, 601 (2002)]:

        EELS₀(ω) = -Im 1 / (1 - V(q) χ₀(q,ω)).
        """
        Vchi0_W, Vchi_W = self.macroscopic_components()
        eels0_W = -(1. / (1. - Vchi0_W)).imag
        eels_W = -Vchi_W.imag
        return ScalarResponseFunctionSet(self.wd, eels0_W, eels_W)


@dataclass
class DielectricMatrixData:
    dyson: Chi0DysonEquation
    e_wGG: np.ndarray

    def unpack(self):
        # Kinda ugly still... XXX
        return self.dyson.chi0.qpd, self.e_wGG

    def dielectric_function(self):
        e_wGG = self.e_wGG
        df_NLFC_w = np.zeros(len(e_wGG), dtype=complex)
        df_LFC_w = np.zeros(len(e_wGG), dtype=complex)

        for w, e_GG in enumerate(e_wGG):
            df_NLFC_w[w] = e_GG[0, 0]
            df_LFC_w[w] = 1 / np.linalg.inv(e_GG)[0, 0]

        df_NLFC_w = self.dyson.df.collect(df_NLFC_w)
        df_LFC_w = self.dyson.df.collect(df_LFC_w)

        return ScalarResponseFunctionSet(self.dyson.df.wd, df_NLFC_w, df_LFC_w)


class DielectricFunctionCalculator:
    def __init__(self, wd: FrequencyDescriptor,
                 chi0calc: Chi0Calculator, truncation: str | None):
        from gpaw.response.pw_parallelization import Blocks1D
        self.wd = wd

        self.chi0calc = chi0calc

        self.coulomb = CoulombKernel.from_gs(self.gs, truncation=truncation)

        # context: ResponseContext object from gpaw.response.context
        self.context = chi0calc.context

        # context.comm : _Communicator object from gpaw.mpi
        self.blocks1d = Blocks1D(self.context.comm, len(self.wd))

        self._chi0cache: dict = {}

    @property
    def gs(self):
        # gs: ResponseGroundStateAdapter from gpaw.response.groundstate
        return self.chi0calc.gs

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
            self._chi0cache[key] = Chi0DysonEquation(
                self.chi0calc.calculate(q_c), self)
            self.context.write_timer()

        return self._chi0cache[key]

    def collect(self, a_w: np.ndarray) -> np.ndarray:
        # combines array from sub-processes into one.
        return self.blocks1d.all_gather(a_w)

    def _new_chi(self, xc='RPA', q_c=[0, 0, 0], **kwargs):
        return self.calculate_chi0(q_c).Vchi(xc=xc, **kwargs)

    def _new_dynamic_susceptibility(self, xc='ALDA', **kwargs):
        chi = self._new_chi(xc=xc, **kwargs)
        return chi.dynamic_susceptibility()

    def _new_dielectric_function(self, *args, **kwargs):
        dm = self._new_dielectric_matrix(*args, **kwargs)
        return dm.dielectric_function()

    def _new_dielectric_matrix(self, xc='RPA', q_c=[0, 0, 0], **kwargs):
        chi0 = self.calculate_chi0(q_c)
        return chi0.dielectric_matrix(xc=xc, **kwargs)

    def _new_eels_spectrum(self, xc='RPA', q_c=[0, 0, 0], direction='x'):
        chi = self._new_chi(xc=xc, q_c=q_c, direction=direction)
        return chi.eels_spectrum()

    def _new_polarizability(self, xc='RPA', direction='x', q_c=[0, 0, 0],
                            **kwargs):
        r"""Calculate the polarizability alpha.
        In 3D the imaginary part of the polarizability is related to the
        dielectric function by Im(eps_M) = 4 pi * Im(alpha). In systems
        with reduced dimensionality the converged value of alpha is
        independent of the cell volume. This is not the case for eps_M,
        which is ill-defined. A truncated Coulomb kernel will always give
        eps_M = 1.0, whereas the polarizability maintains its structure.

        By default, generate a file 'polarizability.csv'. The five columns are:
        frequency (eV), Real(alpha0), Imag(alpha0), Real(alpha), Imag(alpha)
        alpha0 is the result without local field effects and the
        dimension of alpha is \AA to the power of non-periodic directions
        """

        # gs: ResponseGroundStateAdapter from gpaw.response.groundstate
        # gd: GridDescriptor object from gpaw.grid_descriptor
        cell_cv = self.gs.gd.cell_cv

        # pbc_c: np.ndarray of type bool. Describes periodic directions.
        pbc_c = self.gs.pbc

        if pbc_c.all():
            V = 1.0
        else:
            V = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))

        if not self.coulomb.truncation:
            """Standard expression for the polarizability"""
            df = self._new_dielectric_function(
                xc=xc, q_c=q_c, direction=direction, **kwargs)
            alpha_w = V * (df.rf_w - 1.0) / (4 * pi)
            alpha0_w = V * (df.rf0_w - 1.0) / (4 * pi)
        else:
            # Since eps_M = 1.0 for a truncated Coulomb interaction, it does
            # not make sense to apply it here. Instead one should define the
            # polarizability by
            #
            #     alpha * eps_M^{-1} = -L / (4 * pi) * <v_ind>
            #
            # where <v_ind> = 4 * pi * \chi / q^2 is the averaged induced
            # potential (relative to the strength of the  external potential).
            # With the bare Coulomb potential, this expression is equivalent to
            # the standard one. In a 2D system \chi should be calculated with a
            # truncated Coulomb potential and eps_M = 1.0

            self.context.print('Using truncated Coulomb interaction')
            chi = self._new_chi(xc=xc, q_c=q_c, direction=direction, **kwargs)

            alpha_w = -V / (4 * pi) * chi.Vchi_symm_wGG[:, 0, 0]
            alpha0_w = -V / (4 * pi) * chi.Vchi0_symm_wGG[:, 0, 0]

            alpha_w = self.collect(alpha_w)
            alpha0_w = self.collect(alpha0_w)

        # Convert to external units
        hypervol = Bohr**sum(~pbc_c)
        alpha0_w *= hypervol
        alpha_w *= hypervol

        return ScalarResponseFunctionSet(self.wd, alpha0_w, alpha_w)

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

        super().__init__(wd=wd, chi0calc=chi0calc, truncation=truncation)

    def get_frequencies(self) -> np.ndarray:
        """ Return frequencies that Chi is evaluated on"""
        return self.wd.omega_w * Hartree

    def get_dynamic_susceptibility(self, *args, filename='chiM_w.csv',
                                   **kwargs):
        dynsus = self._new_dynamic_susceptibility(*args, **kwargs)
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

    def get_dielectric_matrix(self, *args, **kwargs):
        return self._new_dielectric_matrix(*args, **kwargs).unpack()

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
