from q2.job import Job


def workflow():
    return [
        Job('calculate.py@8x1m'),
        Job('postprocess.py@8x10s', deps=['calculate.py']),
        Job('plot.py', deps=['postprocess.py'])]
