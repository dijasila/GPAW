from q2.job import Job


def workflow():
    return [
        Job('gaps.py'),
        Job('eos.py@4x10m'),
        Job('plot_a.py', deps=['eos.py'])]

def agts(queue):
    queue.add('gaps.py', creates='si-gaps.csv')
    eos = queue.add('eos.py', ncpus=4, walltime=600)
    queue.add('plot_a.py', deps=eos, creates='si-a.png')
