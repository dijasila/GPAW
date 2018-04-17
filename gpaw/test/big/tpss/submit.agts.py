from q2.job import Job


def workflow():
    return [
        Job('tpss.py@8x1m')]

def agts(queue):
    queue.add('tpss.py', ncpus=8, walltime=60)

