from q2.job import Job


def workflow():
    return [
        Job('calculate.py@1x1m'),
        Job('plot.py', deps=['calculate.py'])]
