"""
Test for whether both the old and new calculators turn positive in subclass/
instance checks against *ase.calculators.calculator.BaseCalculator*.
"""
import pytest
from ase.calculators.calculator import BaseCalculator
from gpaw import GPAW as make_calculator
from gpaw.calculator import GPAW as OldGPAW
from gpaw.new.ase_interface import ASECalculator as NewGPAW


@pytest.mark.ci
@pytest.mark.parametrize('new', [True, False])
def test_class_relations(monkeypatch, new: bool) -> None:
    if new:
        monkeypatch.setenv('GPAW_NEW', 'true')
        GPAW = NewGPAW
    else:
        monkeypatch.delenv('GPAW_NEW', raising=False)
        GPAW = OldGPAW
    calc = make_calculator(mode='fd')
    assert isinstance(calc, GPAW)
    assert isinstance(calc, BaseCalculator)
    assert issubclass(type(calc), BaseCalculator)
