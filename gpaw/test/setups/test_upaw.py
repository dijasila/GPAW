import pytest
import numpy as np
import pickle

from ase.data import atomic_numbers
from gpaw.atom.generator2 import DatasetGenerationError, generate

XC = "PBE"


def generate_setup(name, proj, x, type, nderiv0, label, ecut):
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
        ecut=ecut,
    )

    if not gen.check_all():
        raise DatasetGenerationError

    data = {"pt": [], "phit": [], "phi": []}
    for waves in gen.waves_l:
        data["pt"].append(waves.pt_ng)
        data["phi"].append(waves.phi_ng)
        data["phit"].append(waves.phit_ng)

    with open(name + f"_{label}" + ".pkl", "rb") as f:
        my_data = pickle.load(f)

    for waves in my_data:
        for old_l, new_l in zip(my_data[waves], data[waves]):
            for old, new in zip(old_l, new_l):
                x = np.argmax(np.abs(old - new))
                print(waves, old[x], new[x], np.abs(old - new)[x])
                assert old == pytest.approx(new, 0.15)


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
        f"3s,4s,3p,4p,3d,d,F",
        [2.12, 2.12 * 0.9],
        ("orthonormal", (4, 6, {"rc_perc": [0.85, 0.9, 0.95]}, "poly")),
        4,
    ),
    "Ni_10": (
        f"4s,s,4p,p,3d,d,F",
        [2.0, 2.2, 2.0 * 0.9],
        ("orthonormal", (4, 6, {"inv_gpts": [10, 2]}, "poly")),
        4,
    ),
}


@pytest.mark.serial
def test_generate_setup():
    for symb, par in setups_data.items():
        generate_setup(
            symb, par[0], par[1], par[2], par[3], f"upaw", ecut=None
        )
