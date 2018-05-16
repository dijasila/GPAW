from myqueue.job import Job


def workflow():
    return [
        task('molecules.py+0@1:3h'),
        task('molecules.py+1@1:3h'),
        task('check.py', deps=['molecules.py+0', 'molecules.py+1'])]
