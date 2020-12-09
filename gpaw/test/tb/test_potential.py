from gpaw.tb.mode import calculate_potential
from gpaw.setup import create_setup
from gpaw.xc import XC


def test_vt():
    vt = calculate_potential(create_setup('H'), XC('LDA'))
