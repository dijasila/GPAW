from q2.job import Job


def workflow():
    return [
        Job('tpss.py@8x1m')]
