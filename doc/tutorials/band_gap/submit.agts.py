from myqueue.task import task


def workflow():
    return [
        task('gllbsc_band_gap.py@1:30m')]
