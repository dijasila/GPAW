from q2.job import Job


def workflow():
    return [
        Job('al111.py'),
        Job('stm.py', deps=['al111.py'])]
