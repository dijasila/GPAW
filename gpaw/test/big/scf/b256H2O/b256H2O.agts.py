from myqueue.job import Job


def workflow():
    return [
        Job('b256H2O.py A@4x5m'),
        Job('b256H2O.py B@4x5m')]
