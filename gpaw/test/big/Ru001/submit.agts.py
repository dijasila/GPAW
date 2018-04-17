from q2.job import Job


def workflow():
    return [
        Job('ruslab.py@8x10m'),
        Job('ruslab.py H@8x10m'),
        Job('ruslab.py N@8x10m'),
        Job('ruslab.py O@16x15m'),
        Job('molecules.py@8x20s'),
        Job('results.py',
            deps=['ruslab.py', 'ruslab.py H', 'ruslab.py N', 'ruslab.py O',
                  'molecules.py'])]
