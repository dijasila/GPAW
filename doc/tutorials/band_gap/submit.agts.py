from q2.job import Job


def workflow():
    return [
        Job('gllbsc_band_gap.py@1x30s')]

def agts(queue):
    queue.add('gllbsc_band_gap.py', ncpus=1, walltime=30)
