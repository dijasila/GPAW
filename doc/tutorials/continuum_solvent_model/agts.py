from q2.job import Job


def workflow():
    return [
        Job('ethanol_in_water.py@4x10s'),
        Job('check.py', deps=['ethanol_in_water.py'])]

def agts(queue):
    h2o = queue.add('ethanol_in_water.py', ncpus=4, walltime=10)
    queue.add('check.py', deps=h2o)
