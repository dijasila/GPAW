from myqueue.task import task


def workflow():
    return [
        task('al111.py'),
        task('stm.py', deps='al111.py')]
