from q2.job import Job


def workflow():
    return [
        Job('run.py+16@16x10m'),
        Job('run.py+8a@8x40m'),
        Job('run.py+8b@8x40m'),
        Job('run.py+4a@4x20m'),
        Job('run.py+4b@4x20m'),
        Job('run.py+1@1x5m'),
        Job('analyse.py@1x10s',
            deps=['run.py+16', 'run.py+8a', 'run.py+8b',
                  'run.py+4a', 'run.py+4b', 'run.py+1'])]
