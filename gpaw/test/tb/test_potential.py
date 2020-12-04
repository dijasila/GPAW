from gpaw.tb.wavefunctions import pseudo_potential
from gpaw.setup import create_setup
from gpaw.xc import XC


def test_vt():
    pseudo_potential(create_setup('H'), XC('LDA'))
