"""Test CLI-dipole script."""
import pytest
from gpaw.utilities.dipole import main


@pytest.mark.serial
@pytest.mark.parametrize(
    'mode',
    ['pw', 'lcao', 'fd'])
def test_dipole(gpw_files, mode):
    args = [str(gpw_files[f'h2_{mode}_wfs'])]
    main(args)
