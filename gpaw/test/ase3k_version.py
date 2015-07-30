from gpaw.version import ase_required_version
from ase.version import version_base as ase_version

assert ([int(v) for v in ase_version.split('.')] >=
        [int(v) for v in ase_required_version.split('.')])
