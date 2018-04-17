from q2.job import Job


def workflow():
    return [
        Job('graphene.py@8x15s')]

def agts(queue):
    queue.add('graphene.py', walltime=15, ncpus=8)
