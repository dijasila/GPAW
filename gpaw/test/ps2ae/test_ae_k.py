import pytest
from ase.units import Bohr

from gpaw import GPAW
from gpaw.utilities.ps2ae import PS2AE


def test_ae_k(gpw_files):
    """Test normalization of non gamma-point wave functions."""
    calc = GPAW(gpw_files['bcc_li_fd_wfs'])

    if 0:
        if 1:
            converter = PS2AE(calc)
            ae = converter.get_wave_function(n=0, k=1, ae=not True)
            norm = converter.gd.integrate((ae * ae.conj()).real) * Bohr**3
        #assert norm == pytest.approx(1.0, abs=3e-5)
        #ae1 = calc.get_pseudo_wave_function(kpt=1, periodic=False)
        #ae2 = calc.get_pseudo_wave_function(kpt=1, periodic=True)
        print(ae.shape)
        import matplotlib.pyplot as plt
        plt.plot(ae[:, 0,0].real)
        plt.plot(ae[:, 0,0].imag)
        plt.show()

    else:
        ae = calc.calculation.state.ibzwfs.get_all_electron_wave_function(
            0, kpt=1)
        print(ae)
        import matplotlib.pyplot as plt
        x, y = ae.xy(..., 0, 0)
        plt.plot(x, y.real)
        plt.plot(x, y.imag)
        ae = calc.calculation.state.ibzwfs.wfs_qs[1][0].psit_nX[0]
        x, y = ae.xy(..., 0, 0)
        plt.plot(x, y.real)
        plt.plot(x, y.imag)
        plt.show()
        assert ae.norm2() == pytest.approx(1.0, abs=3e-5)
