from q2.job import Job


def workflow():
    return [
        Job('molecules.py+0@1x3m'),
        Job('molecules.py+1@1x3m'),
        Job('check.py', deps=['molecules.py+0', 'molecules.py+1'])]
