"""Test Hirshfeld for spin/no spin consistency."""
import pytest
from ase import Atom
from ase.parallel import parprint

from gpaw import GPAW, FermiDirac
from gpaw.analyse.hirshfeld import HirshfeldPartitioning
from gpaw.cluster import Cluster
from gpaw.test import equal


@pytest.mark.later
def test_vdw_H_Hirshfeld():
    h = 0.25
    box = 3

    atoms = Cluster()
    atoms.append(Atom('H'))
    atoms.minimal_box(box)

    volumes = []
    for spinpol in [False, True]:
        calc = GPAW(mode='fd',
                    h=h,
                    occupations=FermiDirac(0.1, fixmagmom=spinpol),
                    spinpol=spinpol)
        calc.calculate(atoms)
        vol = HirshfeldPartitioning(calc).get_effective_volume_ratios()
        volumes.append(vol)
    parprint(volumes)
    equal(volumes[0][0], volumes[1][0], 4e-9)
