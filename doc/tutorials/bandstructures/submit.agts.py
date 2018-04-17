from q2.job import Job


def workflow():
    return [
        Job('bandstructure.py@1x5s')]

def agts(queue):
    queue.add('bandstructure.py', ncpus=1, walltime=5,
              creates=['bandstructure.png'])
    
