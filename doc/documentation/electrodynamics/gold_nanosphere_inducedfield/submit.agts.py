from myqueue.job import Job


def workflow():
    return [
        task('calculate.py@1:1h'),
        task('plot.py', deps=['calculate.py'])]
