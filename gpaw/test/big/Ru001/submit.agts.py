from myqueue.job import Job


def workflow():
    return [
        Job('ruslab.py@8x10h'),
        Job('ruslab.py+H@8x10h'),
        Job('ruslab.py+N@8x10h'),
        Job('ruslab.py+O@16x15h'),
        Job('molecules.py@8x20m'),
        Job('results.py',
            deps=['ruslab.py', 'ruslab.py+H', 'ruslab.py+N', 'ruslab.py+O',
                  'molecules.py'])]
