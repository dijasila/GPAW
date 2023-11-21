import pytest
from ase.data import atomic_numbers

from gpaw.atom.generator2 import DatasetGenerationError, generate

XC = "PBE"


def generate_setup(name, proj, x, type, nderiv0):
    splitname = name.split("_")
    element = splitname[0]

    gen = generate(
        element,
        atomic_numbers[element],
        proj,
        x[:-1],
        x[-1],
        None,
        nderiv0,
        xc=XC,
        pseudize=type,
        scalar_relativistic=True,
        ecut=None,
    )

    if not gen.check_all():
        raise DatasetGenerationError


setups_data = {
    "C": (
        "2s,s,2p,p,d",
        [1.2, 1.4, 1.1],
        ("orthonormal", (4, 50, {"inv_gpts": [2, 1.5]}, "nc")),
        2,
    ),
    "Al": (
        "3s,s,3p,p,d,F",
        [2.12, 1.9],
        ("orthonormal", (4, 50, {"inv_gpts": [2.0, 1.5]}, "nc")),
        4,
    ),
    "Ni": (
        "3s,4s,3p,4p,3d,d,F",
        [2.12, 2.12 * 0.9],
        ("orthonormal", (4, 6, {"rc_perc": [0.85, 0.9, 0.95]}, "poly")),
        4,
    ),
    "Ni_10": (
        "4s,s,4p,p,3d,d,F",
        [2.0, 2.2, 2.0 * 0.9],
        ("orthonormal", (4, 6, {"inv_gpts": [10, 2]}, "poly")),
        4,
    ),
}


@pytest.mark.serial
def test_generate_setup():
    for symb, par in setups_data.items():
        generate_setup(
            symb, par[0], par[1], par[2], par[3]
        )
