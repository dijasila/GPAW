from myqueue.job import Job


def workflow():
    return [
        Job('diffusion1.py@4x10s'),
        Job('neb.py@6x1m', deps=['diffusion1.py']),
        Job('check.py', deps=['neb.py'])]
