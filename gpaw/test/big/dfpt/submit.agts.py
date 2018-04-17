from q2.job import Job


def workflow():
    return [
        Job('Si_AH.py@1x2m')]

def agts(queue):
    calc = queue.add('Si_AH.py', walltime=120, ncpus=1)
    return
    queue.add('Si_bandstructure.py', walltime=12 * 60, ncpus=8, deps=calc)
