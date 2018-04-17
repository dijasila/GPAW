# Creates: Cu_bulk.csv
from q2.job import Job


def workflow():
    return [
        Job('Cu_bulk.agts.py')]

def agts(queue):
    queue.add('Cu_bulk.agts.py',
              creates=['Cu_bulk.csv'])

if __name__ == "__main__":
    from ase.optimize.test.Cu_bulk import *
