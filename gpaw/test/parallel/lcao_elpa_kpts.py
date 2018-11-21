from gpaw import GPAW
from ase.build import bulk
import numpy as np

atoms = bulk('Al')
calc = GPAW(mode='lcao', basis='sz(dzp)',
            kpts=[4, 4, 4],
            parallel=dict(sl_auto=True),
            experimental=dict(elpa={})
)
atoms.calc = calc
atoms.get_potential_energy()
