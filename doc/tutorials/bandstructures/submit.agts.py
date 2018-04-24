from myqueue.job import Job


def workflow():
    return [
        Job('bandstructure.py@1x5m')]
