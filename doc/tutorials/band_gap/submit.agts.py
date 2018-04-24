from myqueue.job import Job


def workflow():
    return [
        Job('gllbsc_band_gap.py@1x30m')]
