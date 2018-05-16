from myqueue.job import Job


def workflow():
    return [
        Job('calculate.py@8:1h'),
        Job('postprocess.py@8:10s', deps=['calculate.py']),
        Job('plot.py', deps=['postprocess.py'])]
