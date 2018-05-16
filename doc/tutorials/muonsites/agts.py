from myqueue.task import task


def workflow():
    return [
        task('mnsi.py'),
        task('plot2d.py', deps=['mnsi.py'])]
