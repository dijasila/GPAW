from myqueue.job import Job


def workflow():
    return [
        Job('calculate.py@1:1h'),
        Job('plot.py', deps=['calculate.py'])]
