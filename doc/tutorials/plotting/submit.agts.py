from myqueue.job import Job


def workflow():
    return [
        Job('CO.py@1x10s'),
        Job('CO2cube.py@1x10s', deps=['CO.py']),
        Job('CO2plt.py@1x10s', deps=['CO.py'])]
