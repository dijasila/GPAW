from pathlib import Path
from typing import Union

import numpy as np


def read_bader_charges(filename: Union[str, Path] = 'ACF.dat') -> np.ndarray:
    path = Path(filename)
    charges = []
    with path.open() as fd:
        for line in fd:
            words = line.split()
            if len(words) == 7:
                charges.append(float(words[4]))
    return np.array(charges)


if __name__ == '__main__':
    import subprocess
    import sys
    from ase.io import write
    from ase.units import Bohr
    from gpaw import GPAW
    from gpaw.utilities.ps2ae import PS2AE

    calc = GPAW(sys.argv[1])
    converter = PS2AE(calc, grid_spacing=0.05)
    density = converter.get_pseudo_density()
    ne = density.sum() * converter.dv
    print(ne, 'electrons')
    write('density.cube', calc.atoms, data=density * Bohr**3)
    subprocess.run('bader -p all_atom density.cube'.split())
    charges = read_bader_charges()
    for setup, charge in zip(calc.density.setups, charges):
        charge -= setup.Nv
        print(f'{setup.symbol:2} {charge:10.6f}')
