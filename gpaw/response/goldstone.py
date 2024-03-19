from abc import abstractmethod
from functools import partial

import numpy as np

from gpaw.response.dyson import HXCScaling, DysonEquation


class GoldstoneScaling(HXCScaling):
    """Scale the Dyson equation to fulfill a Goldstone condition."""

    def _calculate_scaling(self, dyson_equations):
        """Calculate scaling coefficient λ."""
        self.check_descriptors(dyson_equations)

        # Find the frequency to determine the scaling from and identify where
        # the Dyson equation in question is distributed
        wgs = self.find_goldstone_frequency(
            dyson_equations.zd.omega_w)
        wblocks = dyson_equations.zblocks
        rgs, mywgs = wblocks.find_global_index(wgs)

        # Let the rank which holds the Goldstone frequency find and broadcast λ
        lambdbuf = np.empty(1, dtype=float)
        if wblocks.blockcomm.rank == rgs:
            lambdbuf[:] = self.find_goldstone_scaling(dyson_equations[mywgs])
        wblocks.blockcomm.broadcast(lambdbuf, rgs)
        lambd = lambdbuf[0]

        return lambd

    @staticmethod
    def check_descriptors(dyson_equations):
        if not (dyson_equations.qpd.optical_limit and
                dyson_equations.spincomponent in ['+-', '-+']):
            raise ValueError(
                'The Goldstone condition only applies to χ^(+-)(q=0).')

    @abstractmethod
    def find_goldstone_frequency(self, omega_w):
        """Determine frequency index for the Goldstone condition."""

    @abstractmethod
    def find_goldstone_scaling(self, dyson_equation: DysonEquation) -> float:
        """Calculate the Goldstone scaling parameter λ."""


class FMGoldstoneScaling(GoldstoneScaling):
    """Fulfil ferromagnetic Goldstone condition."""

    @staticmethod
    def find_goldstone_frequency(omega_w):
        """Ferromagnetic Goldstone condition is based on χ^(+-)(ω=0)."""
        wgs = np.abs(omega_w).argmin()
        assert abs(omega_w[wgs]) < 1.e-8, \
            "Frequency grid needs to include ω=0."
        return wgs

    @staticmethod
    def find_goldstone_scaling(dyson_equation):
        return find_fm_goldstone_scaling(dyson_equation)


class AFMGoldstoneScaling(GoldstoneScaling):
    """Fulfil antiferromagnetic Goldstone condition."""

    @staticmethod
    def find_goldstone_frequency(omega_w):
        """Antiferromagnetic Goldstone condition is based on ω->0^+."""
        # Set ω<=0. to np.inf
        omega1_w = np.where(omega_w < 1.e-8, np.inf, omega_w)
        # Sort for the two smallest positive frequencies
        omega2_w = np.partition(omega1_w, 1)
        # Find original index of second smallest positive frequency
        wgs = np.abs(omega_w - omega2_w[1]).argmin()
        return wgs

    @staticmethod
    def find_goldstone_scaling(dyson_equation):
        return find_afm_goldstone_scaling(dyson_equation)


def find_fm_goldstone_scaling(dyson_equation):
    """Find goldstone scaling of the kernel by ensuring that the
    macroscopic inverse enhancement function has a root in (q=0, omega=0)."""
    fnct = partial(calculate_macroscopic_kappa,
                   dyson_equation=dyson_equation)

    def is_converged(kappaM):
        return abs(kappaM) < 1e-7

    return minimize_function_of_lambd(fnct, is_converged)


def find_afm_goldstone_scaling(dyson_equation):
    """Find goldstone scaling of the kernel by ensuring that the
    macroscopic magnon spectrum vanishes at q=0. for finite frequencies."""
    fnct = partial(calculate_macroscopic_spectrum,
                   dyson_equation=dyson_equation)

    def is_converged(SM):
        return 0. < SM < 1.e-7

    return minimize_function_of_lambd(fnct, is_converged)


def minimize_function_of_lambd(fnct, is_converged):
    """Minimize |f(λ)|, where the scaling parameter λ~1 at the minimum.

    |f(λ)| is minimized iteratively, assuming that f(λ) is continuous and
    monotonically decreasing with λ for λ∊]0, 10[.
    """
    lambd = 1.  # initial guess for the scaling parameter
    value = fnct(lambd)
    lambd_incr = 0.1 * np.sign(value)  # Increase λ to decrease f(λ)
    while not is_converged(value) or abs(lambd_incr) > 1.e-7:
        # Update λ
        lambd += lambd_incr
        if lambd <= 0.0 or lambd >= 10.:
            raise Exception(f'Found an invalid λ-value of {lambd:.4f}')
        # Update value and refine increment, if we have passed f(λ)=0
        value = fnct(lambd)
        if np.sign(value) != np.sign(lambd_incr):
            lambd_incr *= -0.2
    return lambd


def calculate_macroscopic_kappa(lambd, dyson_equation):
    """Invert dyson equation and calculate the inverse enhancement function."""
    chi_GG = dyson_equation.invert(lambd=lambd)
    return (dyson_equation.chiks_GG[0, 0] / chi_GG[0, 0]).real


def calculate_macroscopic_spectrum(lambd, dyson_equation):
    """Invert dyson equation and extract the macroscopic spectrum."""
    chi_GG = dyson_equation.invert(lambd=lambd)
    return - chi_GG[0, 0].imag / np.pi
