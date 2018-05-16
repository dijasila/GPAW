from myqueue.job import Job


def workflow():
    return [
        task('diffusion1.py@4:10m'),
        task('neb.py@6:1h', deps=['diffusion1.py']),
        task('check.py', deps=['neb.py'])]
