from q2.job import Job


def workflow():
    return [
        Job('na2o4.py@4x2m')]

def agts(queue):
    run = queue.add('na2o4.py', ncpus=4, walltime=2 * 60, deps=[])
