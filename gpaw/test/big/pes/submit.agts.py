from myqueue.job import Job


def workflow():
    return [
        Job('PES_CO.py@8x1m'),
        Job('PES_H2O.py@8x1m'),
        Job('PES_NH3.py@8x55s'),
        Job('PES_plot.py@1x5s',
            deps=['PES_CO.py', 'PES_H2O.py', 'PES_NH3.py'])]
