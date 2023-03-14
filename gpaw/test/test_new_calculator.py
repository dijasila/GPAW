from ase.build import bulk

from gpaw import GPAW, PW


# ---------- Actual tests ---------- #


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

    calc = GPAW(**params)
    atoms.calc = calc

    for modification in modification_m:
        atoms.calc = calc.new(**modification)
        check_calc(atoms, params, modification)


# ---------- Test functionality ---------- #


def check_calc(atoms, params, modification):
    desired_params = params.copy()
    desired_params.update(modification)

    for param, value in desired_params.items():
        assert atoms.calc.parameters[param] == value
