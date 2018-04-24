from myqueue.job import Job


def workflow():
    return [
        Job('diffusion1.py@4x10m'),
        Job('neb.py@6x1h', deps=['diffusion1.py']),
        Job('check.py', deps=['neb.py'])]
