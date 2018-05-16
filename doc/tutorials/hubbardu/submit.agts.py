from myqueue.job import Job


def workflow():
    return [
        task('nio.py'),
        task('n.py'),
        task('check.py', deps=['n.py'])]
