import pytest

import numpy as np

from ase.units import Bohr

from gpaw import GPAW

from gpaw.sphere.integrate import integrate_lebedev, radial_trapz


def generate_analytical_integrals():
    """Return pair of functions that (1) evaluates a given f(r) (2) integrates

    r_cut
    /
    | r^2 dr f(r)
    /
    0

    analytically together with a relative tolerance.
    """

    def linear(r):
        f = 2. - r
        f[r > 2.] = 0.
        return f

    def integrate_linear(rcut):
        if rcut > 2.:
            return 4. / 3.
        else:
            return (2 / 3. - rcut / 4.) * rcut**3.

    def gaussian(r):
        return np.exp(-r**2 / 4)

    def integrate_gaussian(rcut):
        from scipy.special import erf
        return 2 * np.sqrt(np.pi) * erf(rcut / 2) \
            - 2 * rcut * np.exp(-rcut**2 / 4)

    def lorentzian(r):
        return 1 / (r**2 + 4)

    def integrate_lorentzian(rcut):
        return rcut - 2. * np.arctan(rcut / 2)

    return [
        # Exact numerical integration with linear interpolation
        (linear, integrate_linear, 1e-8),
        # Approximate numerical integration
        (gaussian, integrate_gaussian, 1e-5),
        (lorentzian, integrate_lorentzian, 1e-5)
    ]


@pytest.mark.parametrize('analytical_integral',
                         generate_analytical_integrals())
def test_radial_trapz(analytical_integral):
    func, integrate, rtol = analytical_integral
    # Create radial grid and evaluate the function
    r_g = np.linspace(0., 3.0, 250)
    f_g = func(r_g)

    # Compute the radial integral for a number of different cutoffs:
    for ncut in np.arange(100, 260, 10):
        rcut = r_g[ncut - 1]
        ref = integrate(rcut)
        result = radial_trapz(f_g[:ncut], r_g[:ncut])
        assert abs(result - ref) <= 1e-8 + rtol * ref


def test_fe_augmentation_sphere(gpw_files):
    # Extract the spherical grid information from the iron fixture
    calc = GPAW(gpw_files['fe_pw_wfs'], txt=None)
    setup = calc.setups[0]
    rgd = setup.rgd
    Y_nL = setup.xc_correction.Y_nL

    # Create a function which is unity over the entire grid
    f_ng = np.ones((Y_nL.shape[0], rgd.N))

    # Integrate f(r) with different cutoffs, to check that the volume is
    # correctly recovered
    for rcut in np.linspace(0.1 / Bohr, np.max(rgd.r_g), 100):
        ref = 4 * np.pi * rcut**3. / 3.
        # Integrate angular components, then radial
        f_g = integrate_lebedev(f_ng)
        vol = rgd.integrate_trapz(f_g, rcut=rcut)
        assert abs(vol - ref) <= 1e-8 + 1e-6 * ref

        # Integrate radial components, then angular
        f_n = rgd.integrate_trapz(f_ng, rcut=rcut)
        vol = integrate_lebedev(f_n)
        assert abs(vol - ref) <= 1e-8 + 1e-6 * ref
