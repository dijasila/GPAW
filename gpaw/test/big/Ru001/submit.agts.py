from myqueue.job import Job


def workflow():
    return [
        Job('ruslab.py@8:10h'),
        Job('ruslab.py+H@8:10h'),
        Job('ruslab.py+N@8:10h'),
        Job('ruslab.py+O@16:15h'),
        Job('molecules.py@8:20m'),
        Job('results.py',
            deps=['ruslab.py', 'ruslab.py+H', 'ruslab.py+N', 'ruslab.py+O',
                  'molecules.py'])]
