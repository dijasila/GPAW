from q2.job import Job


def workflow():
    return [
        Job('dscf.py@8x13m')]

def agts(queue):
    queue.add('dscf.py', ncpus=8, walltime=13 * 60)
