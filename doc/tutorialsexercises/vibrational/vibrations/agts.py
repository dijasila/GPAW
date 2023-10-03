from ase.vibrations import Vibrations
from myqueue.workflow import run
from ase.io import read


def workflow():
    with run(script='vibH2O.py', cores=8, tmax='15m'):
        run(function=check)


def check():
    vib = Vibrations(read('h2o.txt'))
    freqs = vib.get_frequencies(method='frederiksen')
    assert abs(freqs[-3:] - [1557.6, 3643.8, 3759.4]).max() < 0.1


if __name__ == '__main__':
    check()
