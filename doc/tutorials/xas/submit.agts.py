# Creates: xas_h2o_spectrum.png, h2o_xas_box.png
from q2.job import Job


def workflow():
    return [
        Job('setups.py'),
        Job('run.py@8x25s', deps=['setups.py']),
        Job('dks.py@8x25s', deps=['setups.py']),
        Job('h2o_xas_box1.py@8x25s', deps=['setups.py']),
        Job('submit.agts.py', deps=['run.py', 'dks.py', 'h2o_xas_box1.py'])]

def agts(queue):
    setups = queue.add('setups.py')
    run = queue.add('run.py', ncpus=8, walltime=25, deps=[setups])
    dks = queue.add('dks.py', ncpus=8, walltime=25, deps=[setups])
    box = queue.add('h2o_xas_box1.py', ncpus=8, walltime=25, deps=[setups])
    queue.add('submit.agts.py', deps=[run, dks, box],
              creates=['xas_h2o_spectrum.png', 'h2o_xas_box.png'])


if __name__ == '__main__':
    from gpaw.test import equal
    exec(open('plot.py').read())
    e_dks = float(open('dks.py.output').readline().split()[2])
    equal(e_dks, 532.508, 0.001)
    exec(open('h2o_xas_box2.py').read())
