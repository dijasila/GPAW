from myqueue.job import Job


def workflow():
    return [
        Job('atomize.py@1:30m'),
        Job('relax.py@1:30m', deps=['atomize.py'])]
