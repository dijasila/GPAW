from q2.job import Job


def workflow():
    return [
        Job('qmmm.py@8x15s')]

def agts(queue):
    queue.add('qmmm.py', ncpus=8)
