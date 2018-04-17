from q2.job import Job


def workflow():
    return [
        Job('bandstructure.py@1x5s')]
