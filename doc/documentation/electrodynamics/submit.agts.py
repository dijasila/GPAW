from q2.job import Job


def workflow():
    return [
        Job('plot_permittivity.py')]

def agts(queue):
    queue.add('plot_permittivity.py',
              creates=['Au.yml.png'])
