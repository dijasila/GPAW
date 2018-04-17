from q2.job import Job


def workflow():
    return [
        Job('C2.py@4x1m')]

def agts(queue):
    run = queue.add('C2.py', ncpus=4, walltime=60, deps=[])
