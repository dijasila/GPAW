from myqueue.job import Job


def workflow():
    return [Job('PES_CO.py@8:1h'),
            Job('PES_H2O.py@8:1h'),
            Job('PES_NH3.py@8:55m'),
            Job('PES_plot.py@1:5m',
                deps=['PES_CO.py', 'PES_H2O.py', 'PES_NH3.py'])]
