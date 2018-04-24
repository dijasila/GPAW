# Creates: xas_h2o_spectrum.png, h2o_xas_box.png


def workflow():
    from myqueue.job import Job
    return [
        Job('setups.py'),
        Job('run.py@8x25m', deps=['setups.py']),
        Job('dks.py@8x25m', deps=['setups.py']),
        Job('h2o_xas_box1.py@8x25m', deps=['setups.py']),
        Job('submit.agts.py', deps=['run.py', 'dks.py', 'h2o_xas_box1.py'])]


if __name__ == '__main__':
    from gpaw.test import equal
    exec(open('plot.py').read())
    e_dks = float(open('dks.py.output').readline().split()[2])
    equal(e_dks, 532.508, 0.001)
    exec(open('h2o_xas_box2.py').read())
