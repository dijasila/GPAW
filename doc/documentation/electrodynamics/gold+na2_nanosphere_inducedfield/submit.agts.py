from myqueue.task import task


def workflow():
    return [
        task('calculate.py@8:1h'),
        task('postprocess.py@8:10s', deps=['calculate.py']),
        task('plot.py', deps=['postprocess.py'])]
