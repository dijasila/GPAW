from q2.job import Job


def workflow():
    return [
        Job('al111.py'),
        Job('stm.py', deps=['al111.py'])]

def agts(queue):
    al = queue.add('al111.py')
    queue.add('stm.py', deps=al,
              creates=['2d.png', '2d_I.png', 'line.png', 'dIdV.png'])
