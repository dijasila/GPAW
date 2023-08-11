import pytest

import numpy as np

from ase.units import Ha

from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.response.paw import (calculate_pair_density_correction,
                               calculate_matrix_element_correction)

from gpaw.sphere.rshe import calculate_reduced_rshe


@pytest.mark.response
@pytest.mark.serial
def test_paw_correction_consistency(gpw_files):
    """Test consistency of the pair density PAW corrections."""
    context = ResponseContext()
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files['fe_pw'],
                                                  context=context)

    # Set up plane-wave basis
    ecut = 50  # eV
    qpd = SingleQPWDescriptor.from_q([0., 0., 0.5],  # N-point
                                     ecut / Ha, gs.gd)
    qG_Gv = qpd.get_reciprocal_vectors(add_q=True)

    # Calculate ordinary pair density corrections
    pawdata = gs.pawdatasets[0]
    Q1_Gii = calculate_pair_density_correction(qG_Gv, pawdata=pawdata)

    # Calculate pair density as a generalized matrix element
    # Expand unity in real spherical harmonics
    rgd = pawdata.xc_correction.rgd
    Y_nL = pawdata.xc_correction.Y_nL
    f_ng = np.ones((Y_nL.shape[0], rgd.N))
    rshe, _ = calculate_reduced_rshe(rgd, f_ng, Y_nL, lmax=0)
    # Calculate correction
    Q2_Gii = calculate_matrix_element_correction(qG_Gv, pawdata, rshe)

    assert Q2_Gii == pytest.approx(Q1_Gii)
