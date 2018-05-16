from myqueue.job import Job


def workflow():
    return [
        task('calculate.py@8:1h'),
        task('postprocess.py@8:10s', deps=['calculate.py']),
        task('plot.py', deps=['postprocess.py'])]
