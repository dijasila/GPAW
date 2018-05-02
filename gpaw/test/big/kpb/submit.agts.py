from myqueue.job import Job


def workflow():
    return [
        Job('molecules.py+0@1x3h'),
        Job('molecules.py+1@1x3h'),
        Job('check.py', deps=['molecules.py+0', 'molecules.py+1'])]
