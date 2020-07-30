import os
import shutil
from pathlib import Path
from myqueue.task import task


def create_tasks():
    if os.getenv('AGTS_FILES'):
        dir = Path(os.getenv('AGTS_FILES'))
        file = Path('organometal.master.db')
        if not file.is_file():
            shutil.copyfile(dir / file, file)
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task(nbrun, args=['machinelearning.py'], tmax='8h')]
