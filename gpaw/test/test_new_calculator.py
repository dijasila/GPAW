# General modules
import sys
from pathlib import Path
import pytest

# Script modules
from ase.build import bulk
from gpaw import GPAW, PW


# ---------- Actual tests ---------- #


@pytest.mark.ci
def test_new_calculator(in_tmp_dir):
    """Test the GPAW.new() method."""

    # ---------- Inputs ---------- #

    params = dict(
        mode=PW(200),
        xc='LDA',
        nbands=8,
        kpts={'size': (4, 4, 4), 'gamma': True})

    modification_m = [
        dict(mode='fd'),
        dict(xc='PBE'),
        dict(nbands=10),
        dict(kpts={'size': (4, 4, 4)}),
        dict(kpts={'size': (3, 3, 3)}, xc='PBE'),
    ]

    # ---------- Script ---------- #

    atoms = bulk('Si')

    calc = GPAW(**params, txt='calc0.txt')
    atoms.calc = calc

    for m, modification in enumerate(modification_m):
        if m == 0:
            # Don't give a new txt file
            atoms.calc = calc.new(**modification)
            check_file_handles(calc, atoms)
        else:
            txt = f'calc{m}.txt'
            atoms.calc = calc.new(**modification, txt=txt)
            check_file_handles(calc, atoms, txt=txt)

        check_calc(atoms, params, modification, world=calc.world)


# ---------- Test functionality ---------- #


def check_file_handles(calc, atoms, txt=None):
    assert calc.log.world.rank == atoms.calc.log.world.rank

    if atoms.calc.log.world.rank == 0:
        # We never want to reuse the output file
        assert atoms.calc.log._fd is not calc.log._fd

        if txt is None:
            # When no txt is specified, the new calculator should log its
            # output in stdout
            assert atoms.calc.log._fd is sys.stdout
        else:
            # Check that the new calculator log file handle was updated
            # appropriately
            assert Path(atoms.calc.log._fd.name).name == txt


def check_calc(atoms, params, modification, *, world):
    desired_params = params.copy()
    desired_params.update(modification)

    for param, value in desired_params.items():
        assert atoms.calc.parameters[param] == value

    # Check that the communicator is reused
    assert atoms.calc.world is world
