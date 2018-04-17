from q2.job import Job


def workflow():
    return [
        Job('dipole.py@4x1m'),
        Job('pwdipole.py'),
        Job('plot.py', deps=['dipole.py', 'pwdipole.py']),
        Job('check.py', deps=['dipole.py', 'pwdipole.py'])]
