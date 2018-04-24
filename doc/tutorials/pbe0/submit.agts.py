from myqueue.job import Job


def workflow():
    return [
        Job('gaps.py'),
        Job('eos.py@4x10h'),
        Job('plot_a.py', deps=['eos.py'])]
