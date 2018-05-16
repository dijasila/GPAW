from myqueue.job import Job


def workflow():
    return [
        Job('gaps.py'),
        Job('eos.py@4:10h'),
        Job('plot_a.py', deps=['eos.py'])]
