from myqueue.workflow import run


def workflow():
    return [
        task('lcaotddft.py@4:40m')]
