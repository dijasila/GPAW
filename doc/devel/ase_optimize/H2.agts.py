# Creates: H2-emt.csv, H2-gpaw.csv
from q2.job import Job


def workflow():
    return [
        Job('H2.agts.py@8x25s')]

def agts(queue):
    queue.add('H2.agts.py',
              walltime=25,
              ncpus=8,
              creates=['H2-emt.csv', 'H2-gpaw.csv'])

if __name__ == "__main__":
    from ase.optimize.test.H2 import *
