from q2.job import Job


def workflow():
    return [
        Job('CO.py@1x10s'),
        Job('CO2cube.py@1x10s', deps=['CO.py']),
        Job('CO2plt.py@1x10s', deps=['CO.py'])]

def agts(queue):
    d = queue.add('CO.py', ncpus=1, walltime=10)
    queue.add('CO2cube.py', deps=d, ncpus=1, walltime=10)
    queue.add('CO2plt.py', deps=d, ncpus=1, walltime=10)
