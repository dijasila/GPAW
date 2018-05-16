from myqueue.job import Job


def workflow():
    return [
        task('mnsi.py'),
        task('plot2d.py', deps=['mnsi.py'])]
