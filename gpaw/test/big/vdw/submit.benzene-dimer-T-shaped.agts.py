from q2.job import Job


def workflow():
    return [
        Job('benzene-dimer-T-shaped.py@48x20m')]

def agts(queue):
    queue.add('benzene-dimer-T-shaped.py', ncpus=48, walltime=20 * 60)
