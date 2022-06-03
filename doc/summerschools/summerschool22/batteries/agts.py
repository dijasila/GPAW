import os
import shutil
from pathlib import Path
from myqueue.workflow import run


def workflow():
    if os.getenv('AGTS_FILES'):
        dir = Path(os.getenv('AGTS_FILES'))
        for file in [Path('lifepo4_wo_li.traj'),
                     Path('NEB_init.traj')]:
            if not file.is_file():
                shutil.copyfile(dir / file, file)

    r1 = run(script='batteries1.py', tmax='2h')
    r2 = run(script='batteries2.py', tmax='3h')
    with r1, r2:
        run(script='batteries3.py', tmax='1h', cores=8)
