from myqueue.job import Job


def workflow():
    return [
        Job('diffusion1.py@4:10m'),
        Job('neb.py@6:1h', deps=['diffusion1.py']),
        Job('check.py', deps=['neb.py'])]
