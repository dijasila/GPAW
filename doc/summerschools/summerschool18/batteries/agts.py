import os
import shutil
from pathlib import Path
from myqueue.task import task


def create_tasks():
    if os.getenv('AGTS_FILES'):
        dir = Path(os.getenv('AGTS_FILES'))
        for file in [Path('lifepo4_wo_li.traj'),
                     Path('NEB_init.traj')]:
            if not file.is_file():
                shutil.copyfile(dir / file, file)

    t1 = task('batteries1.py', tmax='1h')
    t2 = task('batteries2.py', tmax='3h')
    t3 = task('batteries3.py', tmax='1h', cores=8, deps=[t1, t2])
    return [t1, t2, t3]
