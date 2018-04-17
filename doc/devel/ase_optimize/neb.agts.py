# Creates: neb-emt.csv, neb-gpaw.csv
from q2.job import Job


def workflow():
    return [
        Job('neb.agts.py@12x15m')]

def agts(queue):
    queue.add('neb.agts.py',
              walltime=15 * 60,
              ncpus=12,
              creates=['neb-emt.csv', 'neb-gpaw.csv'])

if __name__ == '__main__':
    from ase.optimize.test.neb import *

