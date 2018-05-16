from myqueue.task import task


def workflow():
    return [
        task('gaps.py'),
        task('eos.py@4:10h'),
        task('plot_a.py', deps=['eos.py'])]
