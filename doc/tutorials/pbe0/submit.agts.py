from q2.job import Job


def workflow():
    return [
        Job('gaps.py'),
        Job('eos.py@4x10m'),
        Job('plot_a.py', deps=['eos.py'])]
