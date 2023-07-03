"""Make sure we get a warning when mode is not supplied."""
from contextlib import nullcontext
from ase.build import molecule
from gpaw.calculator import GPAW as OldGPAW
from gpaw.new.ase_interface import GPAW as NewGPAW
from gpaw.new.input_parameters import DeprecatedParameterWarning
import pytest


@pytest.mark.ci
@pytest.mark.parametrize('new', [True, False])
def test_no_mode_supplied(new: bool) -> None:
    if new:
        GPAW, warning_catcher = (NewGPAW,
                                 pytest.warns(DeprecatedParameterWarning))
    else:
        GPAW, warning_catcher = OldGPAW, nullcontext()
    a = 6.0
    hydrogen = molecule('H2', cell=[a, a, a])
    hydrogen.center()
    with warning_catcher:
        hydrogen.calc = GPAW()
        print('{}: {} eV'.format(hydrogen, hydrogen.get_potential_energy()))
