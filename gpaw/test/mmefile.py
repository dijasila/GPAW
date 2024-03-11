from pathlib import Path

from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.test.gpwfile import (Locked, response_band_cutoff,
                               world_temporary_lock)


class MMEFiles:
    """Create files that store momentum matrix elements."""

    def __init__(self, path: Path, gpw_files):
        self.gpw_files = gpw_files
        self.path = path

        self.mme_files = {}
        for file in path.glob('*.npy'):
            self.mme_files[file.name[:-4]] = file

    def __getitem__(self, name: str) -> Path:
        if name in self.mme_files:
            return self.mme_files[name]

        mmepath = self.path / (name + '.npz')

        lockfile = self.path / f'{name}.lock'

        for _attempt in range(60):  # ~60s timeout
            files_exist = 0
            if world.rank == 0:
                files_exist = int(mmepath.exists())
            files_exist = world.sum_scalar(files_exist)

            if files_exist:
                self.mme_files[name] = mmepath
                return self.mme_files[name]

            try:
                with world_temporary_lock(lockfile):
                    calc = GPAW(self.gpw_files[name])
                    nb = response_band_cutoff[
                        name if not name.endswith('_spinpol') else name[:-8]]
                    nlodata = make_nlodata(calc, ni=0, nf=nb, comm=world)
                    work_path = mmepath.with_suffix('.tmp.npz')
                    nlodata.write(work_path)

                    # By now files should exist *and* be fully written, by us.
                    # Rename them to the final intended paths:
                    if world.rank == 0:
                        work_path.rename(mmepath)

            except Locked:
                import time
                time.sleep(1)

        raise RuntimeError(f'MME fixture generation takes too long: {name}.  '
                           'Consider using pytest --cache-clear if there are '
                           'stale lockfiles, else write faster tests.')
