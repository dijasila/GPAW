from gpaw.hyperfine import hyper, paw_correction
from gpaw.setup import create_setup
from gpaw import GPAW
from gpaw.utilities import unpack2


def atest_h(gpw_files):
    calc = GPAW(gpw_files['h_pw'])
    dens = calc.density
    nt_sR = dens.nt_sG
    W1, W2 = hyper(nt_sR[0] - nt_sR[1], dens.gd,
                   calc.atoms.get_scaled_positions())
    print(W1)
    print(W2)
    D_asp = calc.density.D_asp
    paw_correction(unpack2(D_asp[0][0] - D_asp[0][1]), calc.wfs.setups[0])


def test_o2(gpw_files):
    calc = GPAW(gpw_files['o2_pw'])
    dens = calc.density
    nt_sR = dens.nt_sG
    W1, W2 = hyper(nt_sR[0] - nt_sR[1], dens.gd,
                   calc.atoms.get_scaled_positions())
    print(W1)
    print(W2)
    D_asp = calc.density.D_asp
    paw_correction(unpack2(D_asp[0][0] - D_asp[0][1]), calc.wfs.setups[0])


def t():
    s = create_setup('H')
    paw_correction(1, s)
