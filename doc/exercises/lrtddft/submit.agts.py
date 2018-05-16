from myqueue.job import Job


def workflow():
    return [
        Job('Na2TDDFT.py@2:1h'),
        Job('part2.py', deps=['Na2TDDFT.py']),
        Job('ground_state.py@8:15s'),
        Job('spectrum.py', deps=['ground_state.py'])]
