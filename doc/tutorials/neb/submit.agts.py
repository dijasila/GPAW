from q2.job import Job


def workflow():
    return [
        Job('diffusion1.py@4x10s'),
        Job('neb.py@12x1m', deps=['diffusion1.py']),
        Job('check.py', deps=['neb.py'])]

def agts(queue):
    d = queue.add('diffusion1.py', ncpus=4, walltime=10)
    n = queue.add('neb.py', deps=d, ncpus=12, walltime=60)
    queue.add('check.py', deps=n)

