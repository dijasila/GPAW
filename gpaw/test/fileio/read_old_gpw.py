import os
from pathlib import Path
from gpaw import GPAW

dir = os.environ.get('GPW_TEST_FILES')
if dir:
    for f in Path(dir).glob('*.gpw'):
        print(f)
        calc = GPAW(str(f), txt=None)
