import pytest

import numpy as np

from ase.units import Bohr

from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.sphere import integrate_lebedev, integrate_radial_grid


@pytest.mark.response
def test_fe_augmentation_sphere(gpw_files):
    # Create ground state adapter from the iron fixture
    context = ResponseContext()
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files['fe_pw_wfs'],
                                                  context=context)

    # Extract the spherical grid information
    pawdata = gs.pawdatasets[0]
    rgd = pawdata.rgd
    Y_nL = pawdata.xc_correction.Y_nL

    # Create a function which is unity over the entire grid
    f_ng = np.ones((Y_nL.shape[0], rgd.N))

    # Integrate f(r) with different cutoffs, to check that the volume is
    # correctly recovered
    for rcut in np.linspace(0.1 / Bohr, 3.0 / Bohr, 100):
        ref = 4 * np.pi * rcut**3. / 3.
        # Integrate angular components, then radial
        f_g = integrate_lebedev(f_ng)
        vol = integrate_radial_grid(f_g, rgd.r_g, rcut=rcut)
        assert abs(vol - ref) <= 1e-8 + 1e-6 * ref

        # Integrate radial components, then angular
        f_n = integrate_radial_grid(f_ng.T, rgd.r_g, rcut=rcut)
        vol = integrate_lebedev(f_n)
        assert abs(vol - ref) <= 1e-8 + 1e-6 * ref
