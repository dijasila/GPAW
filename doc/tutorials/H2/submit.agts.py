from q2.job import Job


def workflow():
    return [
        Job('atomize.py@1x30s'),
        Job('relax.py@1x30s', deps=['atomize.py'])]
