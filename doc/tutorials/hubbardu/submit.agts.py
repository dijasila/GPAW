from myqueue.job import Job


def workflow():
    return [
        Job('nio.py'),
        Job('n.py'),
        Job('check.py', deps=['n.py'])]
