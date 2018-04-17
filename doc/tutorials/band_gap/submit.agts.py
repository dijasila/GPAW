from q2.job import Job


def workflow():
    return [
        Job('gllbsc_band_gap.py@1x30s')]
