from q2.job import Job


def workflow():
    return [
        Job('fc_butadiene.py@1x30s')]

def agts(queue):
    queue.add('fc_butadiene.py', walltime=30)
