from q2.job import Job


def workflow():
    return [
        Job('diffusion1.py@4x10s'),
        Job('neb.py@12x1m', deps=['diffusion1.py']),
        Job('check.py', deps=['neb.py'])]
