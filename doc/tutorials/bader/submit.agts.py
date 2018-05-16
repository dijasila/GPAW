from myqueue.task import task


def workflow():
    return [
        task('h2o.py'),
        task('bader.py', deps='h2o.py'),
        task('plot.py', deps='bader.py')]
