def agts(queue):
    jobs = [
        queue.add('ruslab.py', walltime=10 * 60, ncpus=8),
        queue.add('ruslab.py H', walltime=10 * 60, ncpus=8),
        queue.add('ruslab.py N', walltime=10 * 60, ncpus=8),
        queue.add('ruslab.py O', walltime=15 * 60, ncpus=16),
        queue.add('molecules.py', walltime=20, ncpus=8)]
    queue.add('results.py', ncpus=1, deps=jobs)
