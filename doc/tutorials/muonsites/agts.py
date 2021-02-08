from myqueue.workflow import run


def workflow():
    return [
        task('mnsi.py'),
        task('plot2d.py', deps='mnsi.py')]
