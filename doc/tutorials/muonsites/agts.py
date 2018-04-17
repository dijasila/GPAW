from q2.job import Job


def workflow():
    return [
        Job('mnsi.py'),
        Job('plot2d.py', deps=['mnsi.py'])]
