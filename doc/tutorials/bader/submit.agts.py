from q2.job import Job


def workflow():
    return [
        Job('h2o.py'),
        Job('bader.py', deps=['h2o.py']),
        Job('plot.py', deps=['bader.py'])]
