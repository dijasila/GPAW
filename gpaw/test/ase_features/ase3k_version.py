from gpaw import __ase_version_required__
from ase import __version__

assert ([int(v) for v in __version__.split('.')] >=
        [int(v) for v in __ase_version_required__.split('.')])
