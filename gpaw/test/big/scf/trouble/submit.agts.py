from myqueue.job import Job


def workflow():
    return [Job('run.py+16@16:10h'),
            Job('run.py+8a@8:40h'),
            Job('run.py+8b@8:40h'),
            Job('run.py+4a@4:20h'),
            Job('run.py+4b@4:20h'),
            Job('run.py+1@1:5h'),
            Job('analyse.py@1:10m',
                deps=['run.py+16', 'run.py+8a', 'run.py+8b',
                      'run.py+4a', 'run.py+4b', 'run.py+1'])]
