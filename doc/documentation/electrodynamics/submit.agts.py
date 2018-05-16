from myqueue.task import task


def workflow():
    return [
        task('plot_permittivity.py')]
