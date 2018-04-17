# Creates: nanoparticle.csv
from q2.job import Job


def workflow():
    return [
        Job('nanoparticle.agts.py@8x2m')]

def agts(queue):
    queue.add('nanoparticle.agts.py',
              walltime=2 * 60 + 15,
              ncpus=8,
              creates=['nanoparticle.csv'])

if __name__ == "__main__":
    from ase.optimize.test.nanoparticle import *
