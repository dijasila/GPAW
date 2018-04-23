from myqueue.job import Job


def workflow():
    return [
        Job('ethanol_in_water.py@4x10s'),
        Job('check.py', deps=['ethanol_in_water.py'])]
