from q2.job import Job


def workflow():
    return [
        Job('ruslab.py@8x10m'),
        Job('ruslab.py H@8x10m'),
        Job('ruslab.py N@8x10m'),
        Job('ruslab.py O@16x15m'),
        Job('molecules.py@8x20s'),
        Job('results.py', deps=['ruslab.py', 'ruslab.py H', 'ruslab.py N', 'ruslab.py O', 'molecules.py'])]

def agts(queue):
    jobs = [
        queue.add('ruslab.py', walltime=10 * 60, ncpus=8),
        queue.add('ruslab.py H', walltime=10 * 60, ncpus=8),
        queue.add('ruslab.py N', walltime=10 * 60, ncpus=8),
        queue.add('ruslab.py O', walltime=15 * 60, ncpus=16),
        queue.add('molecules.py', walltime=20, ncpus=8)]
    queue.add('results.py', ncpus=1, deps=jobs)
