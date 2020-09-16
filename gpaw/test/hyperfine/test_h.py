from gpaw.hyperfine import hyper, paw_correction
from gpaw.setup import create_setup
from gpaw import GPAW


def test_h(gpw_file):
    calc = GPAW(gpw_file['h_pw'])
    dens = calc.density
    nt_sR = dens.nt_sg
    hyper(nt_sR[0] - nt_sR[1], dens.finegd, calc.spoc_ac)


def t():
    s = create_setup('H')
    paw_correction(1, s)
