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
    from gpaw.new.ase_interface import GPAW

    calc = GPAW(sys.argv[1])
    dens = calc.calculation.densities()
    n_sR = dens.all_electron_densities(grid_spacing=0.05)
    ne = n_sR.integrate().sum()
    print(ne, 'electrons')
    # NOTE: Ignoring ASE's hint for **kwargs in write() because it is wrong:
    write('density.cube',
          calc.atoms,
          data=n_sR.data.sum(axis=0) * Bohr**3)  # type: ignore
    subprocess.run('bader -p all_atom density.cube'.split())
    charges = calc.atoms.numbers - read_bader_charges()
    for symbol, charge in zip(calc.atoms.symbols, charges):
        print(f'{symbol:2} {charge:10.6f} |e|')
