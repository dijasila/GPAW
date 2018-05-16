from myqueue.task import task


def workflow():
    return [
        task('dipole.py@4:1h'),
        task('pwdipole.py'),
        task('plot.py', deps=['dipole.py', 'pwdipole.py']),
        task('check.py', deps=['dipole.py', 'pwdipole.py'])]
