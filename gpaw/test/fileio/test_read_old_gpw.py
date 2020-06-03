import os
from pathlib import Path
from gpaw import GPAW


def test_fileio_read_old_gpw():
    dir = os.environ.get('GPW_TEST_FILES')
    if dir:
        for f in Path(dir).glob('*.gpw'):
            print(f)
            GPAW(str(f), txt=None)
