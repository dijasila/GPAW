from q2.job import Job


def workflow():
    return [
        Job('b256H2O.py A@4x5m'),
        Job('b256H2O.py B@4x5m')]

def agts(queue):
    runA = queue.add('b256H2O.py A', ncpus=4, walltime=5*60, deps=[])
    runB = queue.add('b256H2O.py B', ncpus=4, walltime=5*60, deps=[])
