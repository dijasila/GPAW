from q2.job import Job


def workflow():
    return [
        Job('g21gpaw.py 0@1x40m'),
        Job('g21gpaw.py 1@1x40m'),
        Job('g21gpaw.py 2@1x40m'),
        Job('g21gpaw.py 3@1x40m'),
        Job('analyse.py', deps=['g21gpaw.py 0', 'g21gpaw.py 1', 'g21gpaw.py 2', 'g21gpaw.py 3'])]

def agts(queue):
    # generate = queue.add('generate.py', ncpus=1, walltime=20)
    G = [queue.add('g21gpaw.py %d' % i, walltime=40 * 60)
         for i in range(4)]
    queue.add('analyse.py', deps=G)
