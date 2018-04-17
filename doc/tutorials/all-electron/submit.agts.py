from q2.job import Job


def workflow():
    return [
        Job('NaCl.py@1x30s')]

def agts(queue):
    queue.add('NaCl.py', ncpus=1, walltime=30, creates=['all_electron.csv'])

