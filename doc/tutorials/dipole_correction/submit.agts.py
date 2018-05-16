from myqueue.job import Job


def workflow():
    return [
        Job('dipole.py@4:1h'),
        Job('pwdipole.py'),
        Job('plot.py', deps=['dipole.py', 'pwdipole.py']),
        Job('check.py', deps=['dipole.py', 'pwdipole.py'])]
