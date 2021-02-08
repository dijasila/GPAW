from myqueue.workflow import run


def workflow():
    return [
        task('dipole.py@4:1h'),
        task('pwdipole.py@4:5m'),
        task('plot.py', deps='dipole.py,pwdipole.py'),
        task('check.py', deps='dipole.py,pwdipole.py')]
