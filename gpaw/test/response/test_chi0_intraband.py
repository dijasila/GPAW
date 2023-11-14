import numpy as np
import pytest

from gpaw import GPAW, PW
from gpaw.test import findpeak
from gpaw.response.df import DielectricFunction
from ase.build import bulk
from ase.units import Bohr, Hartree


@pytest.mark.response
@pytest.mark.slow
def test_chi0_intraband(in_tmp_dir, gpw_files):
    """Comparing the plasmon peaks found in bulk sodium for two different
    atomic structures. Testing for idential plasmon peaks. Not using
    physical sodium cell."""
    intraband_spinpaired = gpw_files['intraband_spinpaired_fulldiag']
    intraband_spinpolarized = gpw_files['intraband_spinpolarized_fulldiag']

    df1 = DielectricFunction(intraband_spinpaired,
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.03},
                             ecut=10,
                             rate=0.1,
                             integrationmode='tetrahedron integration',
                             txt=None)

    df1NLFCx, df1LFCx = df1.get_dielectric_function(direction='x')
    df1NLFCy, df1LFCy = df1.get_dielectric_function(direction='y')
    df1NLFCz, df1LFCz = df1.get_dielectric_function(direction='z')
    chi0_drude = df1.chi0calc.chi0_opt_ext_calc.drude_calc.calculate(
        df1.wd, 0.1)
    wp1 = chi0_drude.plasmafreq_vv[0, 0]**0.5

    df2 = DielectricFunction(intraband_spinpaired,
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.03},
                             ecut=10,
                             rate=0.1,
                             integrationmode=None,
                             txt=None)
    df2NLFCx, df2LFCx = df2.get_dielectric_function(direction='x')
    df2NLFCy, df2LFCy = df2.get_dielectric_function(direction='y')
    df2NLFCz, df2LFCz = df2.get_dielectric_function(direction='z')
    chi0_drude = df2.chi0calc.chi0_opt_ext_calc.drude_calc.calculate(
        df2.wd, 0.1)
    wp2 = chi0_drude.plasmafreq_vv[0, 0]**0.5

    df3 = DielectricFunction(intraband_spinpolarized,
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.03},
                             ecut=10,
                             rate=0.1,
                             integrationmode='tetrahedron integration',
                             txt=None)

    df3NLFCx, df3LFCx = df3.get_dielectric_function(direction='x')
    df3NLFCy, df3LFCy = df3.get_dielectric_function(direction='y')
    df3NLFCz, df3LFCz = df3.get_dielectric_function(direction='z')
    chi0_drude = df3.chi0calc.chi0_opt_ext_calc.drude_calc.calculate(
        df3.wd, 0.1)
    wp3 = chi0_drude.plasmafreq_vv[0, 0]**0.5

    df4 = DielectricFunction(intraband_spinpolarized,
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.03},
                             ecut=10,
                             rate=0.1,
                             integrationmode=None,
                             txt=None)

    df4NLFCx, df4LFCx = df4.get_dielectric_function(direction='x')
    df4NLFCy, df4LFCy = df4.get_dielectric_function(direction='y')
    df4NLFCz, df4LFCz = df4.get_dielectric_function(direction='z')
    chi0_drude = df4.chi0calc.chi0_opt_ext_calc.drude_calc.calculate(
        df4.wd, 0.1)
    wp4 = chi0_drude.plasmafreq_vv[0, 0]**0.5

    # Compare plasmon frequencies and intensities
    w_w = df1.wd.omega_w
    # frequency grids must be the same
    w_w2 = df2.wd.omega_w
    w_w3 = df3.wd.omega_w
    w_w4 = df4.wd.omega_w
    assert np.allclose(w_w, w_w2, atol=1e-5, rtol=1e-4)
    assert np.allclose(w_w2, w_w3, atol=1e-5, rtol=1e-4)
    assert np.allclose(w_w3, w_w4, atol=1e-5, rtol=1e-4)

    # Analytical Drude result
    n = 1 / (df1.gs.volume * Bohr**-3)

    wp = np.sqrt(4 * np.pi * n)

    # From https://doi.org/10.1021/jp810808h
    wpref = 5.71 / Hartree

    # spin paired matches spin polar - tetra
    assert wp1 == pytest.approx(wp3, abs=1e-2)
    # spin paired matches spin polar - none
    assert wp2 == pytest.approx(wp4, abs=1e-2)
    # Use larger margin when comparing to Drude
    assert wp1 == pytest.approx(wp, abs=0.5)
    # Use larger margin when comparing to Drude
    assert wp2 == pytest.approx(wp, abs=0.5)
    # paired tetra match paper
    assert wp1 == pytest.approx(wpref, abs=0.1)
    # paired none match paper
    assert wp2 == pytest.approx(wpref, abs=0.1)

    # w_x equal for paired & polarized tetra
    w1, I1 = findpeak(w_w, -(1. / df1LFCx).imag)
    w3, I3 = findpeak(w_w, -(1. / df3LFCx).imag)
    assert w1 == pytest.approx(w3, abs=1e-2)
    assert I1 == pytest.approx(I3, abs=1e-1)

    # w_x equal for paired & polarized none
    w2, I2 = findpeak(w_w, -(1. / df2LFCx).imag)
    w4, I4 = findpeak(w_w, -(1. / df4LFCx).imag)
    assert w2 == pytest.approx(w4, abs=1e-2)
    assert I2 == pytest.approx(I4, abs=1e-1)

    # w_y equal for paired & polarized tetra
    w1, I1 = findpeak(w_w, -(1. / df1LFCy).imag)
    w3, I3 = findpeak(w_w, -(1. / df3LFCy).imag)
    assert w1 == pytest.approx(w3, abs=1e-2)
    assert I1 == pytest.approx(I3, abs=1e-1)

    # w_y equal for paired & polarized none
    w2, I2 = findpeak(w_w, -(1. / df2LFCy).imag)
    w4, I4 = findpeak(w_w, -(1. / df4LFCy).imag)
    assert w2 == pytest.approx(w4, abs=1e-2)
    assert I2 == pytest.approx(I4, abs=1e-1)

    # w_z equal for paired & polarized tetra
    w1, I1 = findpeak(w_w, -(1. / df1LFCz).imag)
    w3, I3 = findpeak(w_w, -(1. / df3LFCz).imag)
    assert w1 == pytest.approx(w3, abs=1e-2)
    assert I1 == pytest.approx(I3, abs=1e-1)

    # w_z equal for paired & polarized none
    w2, I2 = findpeak(w_w, -(1. / df2LFCz).imag)
    w4, I4 = findpeak(w_w, -(1. / df4LFCz).imag)
    assert w2 == pytest.approx(w4, abs=1e-2)
    assert I2 == pytest.approx(I4, abs=1e-1)
