from myqueue.job import Job


def workflow():
    return [
        Job('atomize.py@1x30m'),
        Job('relax.py@1x30m', deps=['atomize.py'])]
