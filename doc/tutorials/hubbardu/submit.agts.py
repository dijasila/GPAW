from q2.job import Job


def workflow():
    return [
        Job('nio.py'),
        Job('n.py'),
        Job('check.py', deps=['n.py'])]

def agts(queue):
    queue.add('nio.py')
    n = queue.add('n.py')
    queue.add('check.py', deps=n, creates='gaps.csv')

