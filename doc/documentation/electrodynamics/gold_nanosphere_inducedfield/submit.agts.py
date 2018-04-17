from q2.job import Job


def workflow():
    return [
        Job('calculate.py@1x1m'),
        Job('plot.py', deps=['calculate.py'])]

def agts(queue):
    c1 = queue.add('calculate.py',
                   walltime=60)

    queue.add('plot.py',
              deps=c1,
              creates=['field.ind_Ffe.png'])
