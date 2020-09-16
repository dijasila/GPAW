from gpaw.hyperfine import hyper, paw_correction
from gpaw.setup import create_setup
from gpaw import GPAW


def test_h(gpw_files):
    calc = GPAW(gpw_files['h_pw'])
    dens = calc.density
    nt_sR = dens.nt_sG
    hyper(nt_sR[0] - nt_sR[1], dens.gd,
          calc.atoms.get_scaled_positions())
    paw_correction(calc.density.D_asp[0], calc.wfs.setups[0])


def t():
    s = create_setup('H')
    paw_correction(1, s)
