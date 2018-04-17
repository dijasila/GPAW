from q2.job import Job


def workflow():
    return [
        Job('dipole.py@4x1m'),
        Job('pwdipole.py'),
        Job('plot.py', deps=['dipole.py', 'pwdipole.py']),
        Job('check.py', deps=['dipole.py', 'pwdipole.py'])]

def agts(queue):
    d = [queue.add('dipole.py', ncpus=4, walltime=60),
         queue.add('pwdipole.py')]
    queue.add('plot.py', deps=d,
              creates=['zero.png', 'periodic.png', 'corrected.png',
                       'pwcorrected.png', 'slab.png'])
    queue.add('check.py', deps=d)
