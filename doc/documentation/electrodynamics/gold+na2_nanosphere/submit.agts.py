from myqueue.task import task


def workflow():
    return [
        task('calculate.py@8:1h'),
        task('plot_geom.py', deps=['calculate.py']),
        task('plot.py', deps=['calculate.py'])]
