from q2.job import Job


def workflow():
    return [
        Job('molecules.py 0@1x3m'),
        Job('molecules.py 1@1x3m'),
        Job('check.py', deps=['molecules.py 0', 'molecules.py 1'])]

def agts(queue):
    molecules = [queue.add('molecules.py %d' % i,
                           ncpus=1,
                           walltime=3*60)
                 for i in range(2)]
    queue.add('check.py', deps=molecules)
 
