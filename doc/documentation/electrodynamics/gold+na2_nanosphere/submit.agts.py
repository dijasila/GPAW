from q2.job import Job


def workflow():
    return [
        Job('calculate.py@8x1m'),
        Job('plot_geom.py', deps=['calculate.py']),
        Job('plot.py', deps=['calculate.py'])]

def agts(queue):
    c1 = queue.add('calculate.py',
                   ncpus=8,
                   walltime=60)

    queue.add('plot_geom.py',
              deps=c1,
              creates=['geom.png'])

    queue.add('plot.py',
              deps=[c1],
              creates=['hybrid.png'])
