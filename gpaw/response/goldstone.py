from abc import abstractmethod

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

        # Let rgs find and broadcast λ
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
        return find_fm_goldstone_frequency(omega_w)

    @staticmethod
    def find_goldstone_scaling(dyson_equation):
        return find_fm_goldstone_scaling(dyson_equation)


class AFMGoldstoneScaling(GoldstoneScaling):
    """Fulfil antiferromagnetic Goldstone condition."""

    @staticmethod
    def find_goldstone_frequency(omega_w):
        return find_afm_goldstone_frequency(omega_w)

    @staticmethod
    def find_goldstone_scaling(dyson_equation):
        return find_afm_goldstone_scaling(dyson_equation)


def find_fm_goldstone_frequency(omega_w):
    """Find omega=0. as the fm Goldstone frequency."""
    wgs = np.abs(omega_w).argmin()
    if not np.allclose(omega_w[wgs], 0., atol=1.e-8):
        raise ValueError("Frequency grid needs to include"
                         + " omega=0. to allow 'fm' Goldstone scaling")

    return wgs


def find_afm_goldstone_frequency(omega_w):
    """Find the second smallest positive frequency
    as the afm Goldstone frequency."""
    # Set omega=0. and negative frequencies to np.inf
    omega1_w = np.where(omega_w < 1.e-8, np.inf, omega_w)
    # Sort for the two smallest positive frequencies
    omega2_w = np.partition(omega1_w, 1)
    # Find original index of second smallest positive frequency
    wgs = np.abs(omega_w - omega2_w[1]).argmin()

    return wgs


def find_fm_goldstone_scaling(dyson_equation):
    """Find goldstone scaling of the kernel by ensuring that the
    macroscopic inverse enhancement function has a root in (q=0, omega=0)."""
    fxcs = 1.
    kappaM = calculate_macroscopic_kappa(fxcs, dyson_equation)
    # If kappaM > 0, increase scaling (recall: kappaM ~ 1 - Kxc Re{chi_0})
    scaling_incr = 0.1 * np.sign(kappaM)
    while abs(kappaM) > 1.e-7 and abs(scaling_incr) > 1.e-7:
        fxcs += scaling_incr
        if fxcs <= 0.0 or fxcs >= 10.:
            raise Exception('Found an invalid fxc_scaling of %.4f' % fxcs)

        kappaM = calculate_macroscopic_kappa(fxcs, dyson_equation)

        # If kappaM changes sign, change sign and refine increment
        if np.sign(kappaM) != np.sign(scaling_incr):
            scaling_incr *= -0.2

    return fxcs


def find_afm_goldstone_scaling(dyson_equation):
    """Find goldstone scaling of the kernel by ensuring that the
    macroscopic magnon spectrum vanishes at q=0. for finite frequencies."""
    fxcs = 1.
    SM = calculate_macroscopic_spectrum(fxcs, dyson_equation)
    # If SM > 0., increase the scaling. If SM < 0., decrease the scaling.
    scaling_incr = 0.1 * np.sign(SM)
    while (SM < 0. or SM > 1.e-7) or abs(scaling_incr) > 1.e-7:
        fxcs += scaling_incr
        if fxcs <= 0. or fxcs >= 10.:
            raise Exception('Found an invalid fxc_scaling of %.4f' % fxcs)

        SM = calculate_macroscopic_spectrum(fxcs, dyson_equation)

        # If chi changes sign, change sign and refine increment
        if np.sign(SM) != np.sign(scaling_incr):
            scaling_incr *= -0.2

    return fxcs


def calculate_macroscopic_kappa(fxcs, dyson_equation):
    """Invert dyson equation and calculate the inverse enhancement function."""
    chi_GG = dyson_equation.invert(lambd=fxcs)
    return (dyson_equation.chiks_GG[0, 0] / chi_GG[0, 0]).real


def calculate_macroscopic_spectrum(fxcs, dyson_equation):
    """Invert dyson equation and extract the macroscopic spectrum."""
    chi_GG = dyson_equation.invert(lambd=fxcs)
    return - chi_GG[0, 0].imag / np.pi
