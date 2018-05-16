from myqueue.job import Job


def workflow():
    return [Job('tpss.py@8:1h')]
