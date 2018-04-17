# Creates: C5H12-gpaw.csv
from q2.job import Job


def workflow():
    return [
        Job('C5H12.agts.py@8x25s')]

def agts(queue):
    queue.add('C5H12.agts.py',
              walltime=25,
              ncpus=8,
              creates=['C5H12-gpaw.csv'])

if __name__ == "__main__":
    from ase.optimize.test.C5H12 import *
