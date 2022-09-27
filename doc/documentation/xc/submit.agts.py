from pathlib import Path

import numpy as np
from ase.data.s22 import get_interaction_energy_cc
from gpaw.utilities import compiled_with_libvdwxc
from gpaw.xc.libvdwxc import libvdwxc_has_pfft
from myqueue.workflow import run


def workflow():
    with run(script='s22_set.py', cores=8, tmax='1d'):
        run(function=check_s22)
    run(script='hydrogen_atom.py', cores=16)
    if compiled_with_libvdwxc():
        run(script='libvdwxc-example.py')
        if libvdwxc_has_pfft():
            run(script='libvdwxc-pfft-example.py', cores=8)


def check_s22():
    E = []
    E0 = []
    for line in Path('energies_TS09.dat').read_text().splitlines():
        if line.startswith('#'):
            continue
        name, *_, energy = line.split()
        e = -float(energy)
        e0 = get_interaction_energy_cc(name)
        print(name, e, e0, e - e0)
        E.append(e)
        E0.append(e0)
    dE = np.array(E) - E0
    print((dE**2).mean()**0.5)
    print(dE.mean())


# check_s22()
